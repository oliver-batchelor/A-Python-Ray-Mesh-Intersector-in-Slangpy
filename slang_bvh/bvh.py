from dataclasses import dataclass
from pathlib import Path
import torch

from tensordict import TensorClass
import slangtorch
import trimesh


from slang_bvh.cuda_lib import radix_sort

  
slang_path = Path(__file__).parent / 'slang'
module = slangtorch.loadModule(slang_path / 'slang_bvh.slang', 
                               extraCudaFlags=["-diag-suppress=177"])


def round_up(x, multiple):
  return (x + multiple - 1) // multiple


class Mesh:
  vertices: torch.Tensor # N, 3  float xyz
  indices: torch.Tensor  # N, 3  int index

  def __init__(self, vertices:torch.Tensor, indices:torch.Tensor, device:str):
    self.vertices = vertices
    self.indices = indices
    self.device = device

  @property
  def primitive_num(self):
    return self.indices.shape[0]
  
  def aabb(self):  
    aabb = torch.empty((self.primitive_num, 6), dtype=torch.float, device=self.vertices.device)
    module.triangle_aabb(vertices=self.vertices, indices=self.indices, aabb=aabb
      ).launchRaw(blockSize=(256, 1, 1), gridSize=(round_up(self.primitive_num, 256), 1, 1))
    return aabb   

  @staticmethod
  def load_trimesh(path:Path, device:str) -> 'Mesh':
    mesh = trimesh.load(path)

    return Mesh(
      vertices=torch.from_numpy(mesh.vertices).to(dtype=torch.float32, device=device),
      indices=torch.from_numpy(mesh.faces).to(dtype=torch.int32, device=device),
      device=device
    )

  def __repr__(self):
    return f"Mesh(n={self.primitive_num})"
  
  def __str__(self):
    return self.__repr__()

class BVH(TensorClass):
  info: torch.Tensor
  aabb: torch.Tensor
  construction_info: torch.Tensor


  def count_children(self) -> torch.Tensor:
    """Counts number of children for each node in the BVH.
    Returns tensor of size (N,) where N is number of nodes."""
    
    # Extract left and right child indices from self.info
    left_children = self.info[:, 0]  # first column contains left child
    right_children = self.info[:, 1]  # second column contains right child
    
    # Count non-zero children (0 indicates no child)
    child_count = (left_children != 0).to(torch.int) + (right_children != 0).to(torch.int)
    
    return child_count
  
  def visitations(self) -> torch.Tensor:
    return self.internal.construction_info[:, 1]

  @property
  def internal(self) -> 'BVH':
    internal_nodes = self.num_elements - 1
    return self[internal_nodes:]
         
  @staticmethod 
  def init(num_elems:int, device:str) -> 'BVH':
    num_nodes = num_elems + num_elems - 1

    return BVH(
      info = torch.empty((num_nodes, 3), dtype=torch.int, device=device),
      aabb = torch.empty((num_nodes, 6), dtype=torch.float, device=device),
      construction_info = torch.empty((num_nodes, 2), dtype=torch.int, device=device),
      batch_size=(num_nodes, ),
    )
  
  @property
  def num_elements(self):
    return (self.info.shape[0] + 1) // 2
  
  @property
  def num_nodes(self):
    return self.info.shape[0]


@dataclass
class MeshBVH: 

  mesh:Mesh
  bvh:BVH

  @property
  def device(self):
    return self.mesh.device

  def intersect(self, ray_origins:torch.Tensor, ray_directions:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert ray_origins.dim() == 3 and ray_origins.shape[2] == 3, f"Expected 3D tensor with shape (N, 3), got {ray_origins.shape}"
    h, w = ray_origins.shape[:2]

    hit_idx = torch.empty((h, w), dtype=torch.int, device=self.device)
    hit_t = torch.empty((h, w), dtype=torch.float, device=self.device)

    module.intersect_triangles(ray_origins=ray_origins, ray_directions=ray_directions,
                  bvh_info=self.bvh.info, bvh_aabb=self.bvh.aabb,
                  vertices=self.mesh.vertices, indices=self.mesh.indices,
                  hit_idx=hit_idx, hit_t=hit_t
      ).launchRaw(blockSize=(16, 16, 1), gridSize=(round_up(h, 16), round_up(w, 16), 1))
    
    return hit_t, hit_idx



def morton_codes(points:torch.Tensor) -> torch.Tensor:

  num_elems = points.shape[0]
  codes = torch.zeros((num_elems,), dtype=torch.int, device=points.device)
  module.morton_codes(points=points, morton_codes=codes
    ).launchRaw(blockSize=(256, 1, 1), gridSize=(round_up(num_elems, 256), 1, 1))

  return codes.view(dtype=torch.uint32)


def aabb_centroids(aabb:torch.Tensor) -> torch.Tensor:
  lower, upper = aabb[:, 0:3], aabb[:, 3:6]
  centroids = (lower + upper) * 0.5
  return centroids

@torch.compile
def normalized_centroids(aabb:torch.Tensor) -> torch.Tensor:
  centroids = aabb_centroids(aabb)

  min_extent = aabb[:, 0:3].min(0).values
  max_extent = aabb[:, 3:6].max(0).values

  normalized_centroids = (centroids - min_extent) / (max_extent - min_extent)
  return normalized_centroids 

def morton_codes_aabb(aabb:torch.Tensor) -> torch.Tensor:
  centroids = normalized_centroids(aabb)
  return morton_codes(centroids)


class BVHBuilder:
  def __init__(self, device:torch.device):
    self.device = device  


  def build(self, mesh:Mesh) -> BVH:

      num_elems = mesh.primitive_num
      ele_aabb = mesh.aabb()

      spatial_codes = morton_codes_aabb(ele_aabb)
      sorted_codes, primitive_indices = radix_sort(spatial_codes)
      
      block_size = 256

      bvh:BVH = BVH.init(num_elems, device=self.device)
      module.build_hierarchy(
          num_elements=num_elems,
          ele_aabb=ele_aabb,

          sorted_codes=sorted_codes.view(dtype=torch.int32),
          primitive_indices=primitive_indices,

          bvh_info=bvh.info, 
          bvh_aabb=bvh.aabb, 
          bvh_construction_infos=bvh.construction_info
        ).launchRaw(blockSize=(block_size, 1, 1), gridSize=(round_up(num_elems, block_size), 1, 1))

      module.bounding_boxes(num_elements=num_elems,
                     bvh_info=bvh.info,
                     bvh_aabb=bvh.aabb,
                     bvh_construction_infos=bvh.construction_info
        ).launchRaw(blockSize=(block_size, 1, 1), gridSize=(round_up(num_elems, block_size), 1, 1))

      return MeshBVH(mesh, bvh)

  def iterative_bounding_boxes(self, bvh:BVH) -> None:
      
      num_elems = bvh.num_elements
      num_internal = num_elems - 1

      # bounding_boxes
      tree_heights = torch.zeros((num_internal, 1), dtype=torch.int, device=self.device)
      module.get_bvh_height(num_elements=num_elems, 
                            bvh_construction_infos=bvh.construction_info, 
                            tree_heights=tree_heights
        ).launchRaw(blockSize=(256, 1, 1), gridSize=(round_up(num_internal, 256), 1, 1))


      tree_height_max = tree_heights.max()
      for i in range(tree_height_max + 1):
          height = tree_height_max - i 
          print(f"height: {height} {(tree_heights == height).sum()}")

          module.get_bbox(num_elements=num_elems, 
                          expected_height=height,
                              
                          bvh_info=bvh.info, 
                          bvh_aabb=bvh.aabb, 
                          tree_heights = tree_heights,
                          bvh_construction_infos=bvh.construction_info
              ).launchRaw(blockSize=(256, 1, 1), gridSize=(round_up(num_internal, 256), 1, 1))


      torch.cuda.synchronize()

      