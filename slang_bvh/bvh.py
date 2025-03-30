from dataclasses import dataclass
from pathlib import Path
import torch

from tensordict import TensorClass
import slangtorch
import trimesh
  
slang_path = Path(__file__).parent / 'slang'
module = slangtorch.loadModule(slang_path / 'slang_bvh.slang', 
                               extraCudaFlags=["-diag-suppress=177"],
                               extraSlangFlags=["-O3"])


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

class BVH(TensorClass):
  info: torch.Tensor
  aabb: torch.Tensor
  construction_info: torch.Tensor

         
  @staticmethod 
  def init(n:int, device:str) -> 'BVH':
    return BVH(
      info = torch.zeros((n, 3), dtype=torch.int, device=device),
      aabb = torch.zeros((n, 6), dtype=torch.float, device=device),
      construction_info = torch.zeros((n, 2), dtype=torch.int, device=device)
    )


@dataclass
class MeshBVH: 

  mesh:Mesh
  bvh:BVH

  @property
  def device(self):
    return self.mesh.device

  def intersect(self, ray_origins:torch.Tensor, ray_directions:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    num_rays = ray_origins.shape[0]
    hit_idx = torch.zeros((num_rays,), dtype=torch.int, device=self.device)
    hit_t = torch.zeros((num_rays,), dtype=torch.float, device=self.device)

    module.intersect(num_rays=int(num_rays), ray_origins=ray_origins, ray_directions=ray_directions,
                  bvh_info=self.bvh.info, bvh_aabb=self.bvh.aabb,
                  vert=self.mesh.vertices, v_indx=self.mesh.indices,
                  hit_idx=hit_idx, hit_t=hit_t
      ).launchRaw(blockSize=(256, 1, 1), gridSize=(round_up(num_rays, 256), 1, 1))
    
    return hit_t, hit_idx



class BVHBuilder:
  def __init__(self, device:torch.device):
    self.device = device  


  def build(self, mesh:Mesh) -> BVH:

      num_elems = mesh.primitive_num
      prim_idx = torch.arange(0, num_elems, dtype=torch.int32, device=mesh.device)

      ele_aabb = mesh.aabb()
      min_extent = ele_aabb[:, 0:3].min(0).values.tolist()
      max_extent = ele_aabb[:, 3:6].max(0).values.tolist()

      
      morton_codes = torch.zeros((num_elems, 2), dtype=torch.int, device=self.device)
      module.morton_codes_aabb(num_elements=num_elems, min_extent=min_extent, max_extent=max_extent, 
                               aabb=ele_aabb, morton_codes=morton_codes
        ).launchRaw(blockSize=(256, 1, 1), gridSize=(round_up(num_elems, 256), 1, 1))


      #--------------------------------------------------
      # radix sort part
      morton_codes_pingpong = torch.zeros((num_elems, 2), dtype=torch.int, device=self.device)
      module.radix_sort(g_num_elements=num_elems, 
                             g_elements_in=morton_codes, g_elements_out=morton_codes_pingpong
        ).launchRaw(blockSize=(256, 1, 1), gridSize=(1, 1, 1))


      # hierarchy
      bvh:BVH = BVH.init(num_elems + num_elems - 1, device=self.device)

      module.hierarchy(g_num_elements=num_elems, ele_primitiveIdx=prim_idx, ele_aabb=ele_aabb,
                          g_sorted_morton_codes=morton_codes, 
                          bvh_info=bvh.info, 
                          bvh_aabb=bvh.aabb, 
                          bvh_construction_infos=bvh.construction_info
        ).launchRaw(blockSize=(256, 1, 1), gridSize=(round_up(num_elems, 256), 1, 1))
      torch.cuda.synchronize()


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
      

      return MeshBVH(mesh, bvh)
