from dataclasses import dataclass
from pathlib import Path
import torch

from tensordict import TensorClass
import slangtorch
import trimesh

from slang_bvh.util.radix_sort import radix_sort_pairs
  
slang_path = Path(__file__).parent / 'slang'
module = slangtorch.loadModule(slang_path / 'slang_bvh.slang', 
                               extraCudaFlags=["-diag-suppress=177"])




def round_up(x, multiple):
  return (x + multiple - 1) // multiple


class Mesh(TensorClass):
  vertices: torch.Tensor # N, 3  float xyz
  indices: torch.Tensor  # N, 3  int index

  @property
  def primitive_num(self):
    return self.indices.shape[0]
  
  def aabb(self):  
    aabb = torch.zeros((self.primitive_num, 6), dtype=torch.float, device=self.vertices.device)
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

         
  @staticmethod 
  def init(num_elems:int, device:str) -> 'BVH':
    num_nodes = num_elems + num_elems - 1

    return BVH(
      info = torch.zeros((num_nodes, 3), dtype=torch.int, device=device),
      aabb = torch.zeros((num_nodes, 6), dtype=torch.float, device=device),
      construction_info = torch.zeros((num_nodes, 2), dtype=torch.int, device=device)
    )


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

    module.intersect(ray_origins=ray_origins, ray_directions=ray_directions,
                  bvh_info=self.bvh.info, bvh_aabb=self.bvh.aabb,
                  vertices=self.mesh.vertices, indices=self.mesh.indices,
                  hit_idx=hit_idx, hit_t=hit_t
      ).launchRaw(blockSize=(16, 16, 1), gridSize=(round_up(h, 16), round_up(w, 16), 1))
    
    return hit_t, hit_idx


def aabb_morton_codes(aabb:torch.Tensor) -> torch.Tensor:
  min_extent = aabb[:, 0:3].min(0).values.tolist()
  max_extent = aabb[:, 3:6].max(0).values.tolist()
  
  num_elems = aabb.shape[0]
  codes = torch.zeros((num_elems, 2), dtype=torch.int, device=aabb.device)
  module.morton_codes_aabb(min_extent=min_extent, max_extent=max_extent, 
                            aabb=aabb, morton_codes=codes
    ).launchRaw(blockSize=(256, 1, 1), gridSize=(round_up(num_elems, 256), 1, 1))

  return codes



def aabb_centroids(aabb:torch.Tensor) -> torch.Tensor:
  lower, upper = aabb[:, 0:3], aabb[:, 3:6]
  centroids = (lower + upper) * 0.5
  return centroids



class BVHBuilder:
  def __init__(self, device:torch.device):
    self.device = device  


  def build(self, mesh:Mesh) -> BVH:

      num_elems = mesh.primitive_num
      prim_idx = torch.arange(0, num_elems, dtype=torch.int32, device=mesh.device)

      ele_aabb = mesh.aabb()
      spatial_codes = aabb_morton_codes(ele_aabb)
      radix_sort_pairs(spatial_codes)
      
      block_size = 256

      # hierarchy
      bvh:BVH = BVH.init(num_elems, device=self.device)
      torch.cuda.synchronize()

      module.hierarchy(num_elements=num_elems, ele_primitive_idx=prim_idx, ele_aabb=ele_aabb,
                          sorted_codes=spatial_codes, 
                          bvh_info=bvh.info, 
                          bvh_aabb=bvh.aabb, 
                          bvh_construction_infos=bvh.construction_info
        ).launchRaw(blockSize=(block_size, 1, 1), gridSize=(round_up(num_elems, block_size), 1, 1))
      torch.cuda.synchronize()

      module.bounding_boxes(num_elements=num_elems,
                     bvh_info=bvh.info,
                     bvh_aabb=bvh.aabb,
                     bvh_construction_infos=bvh.construction_info
        ).launchRaw(blockSize=(block_size, 1, 1), gridSize=(round_up(num_elems, block_size), 1, 1))
      torch.cuda.synchronize()


      return MeshBVH(mesh, bvh)

  def iterative_bounding_boxes(self, bvh:BVH) -> None:
      
      num_elems = bvh.info.shape[0]
      # bounding_boxes
      tree_heights = torch.zeros((num_elems, 1), dtype=torch.int, device=self.device)
      module.get_bvh_height(g_num_elements=num_elems, 
                                 bvh_info=bvh.info,
                                 bvh_aabb=bvh.aabb, 
                                 bvh_construction_infos=bvh.construction_info, 
                                 tree_heights=tree_heights
        ).launchRaw(blockSize=(256, 1, 1), gridSize=(round_up(num_elems, 256), 1, 1))


      tree_height_max = tree_heights.max()
      for i in range(tree_height_max):
          module.get_bbox(g_num_elements=num_elems, 
                              expected_height=int(i+1),
                              bvh_info=bvh.info, 
                              bvh_aabb=bvh.aabb, 
                              bvh_construction_infos=bvh.construction_info
              ).launchRaw(blockSize=(256, 1, 1), gridSize=(round_up(num_elems, 256), 1, 1))


      module.set_root(bvh_info=bvh.info, bvh_aabb=bvh.aabb
        ).launchRaw(blockSize=(1, 1, 1), gridSize=(1, 1, 1)) 
      