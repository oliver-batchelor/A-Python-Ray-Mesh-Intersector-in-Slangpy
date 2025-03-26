import torch

from tensordict import TensorClass
import slangtorch
  

class Mesh(TensorClass):
  vertices: torch.Tensor # N, 3  float xyz
  indices: torch.Tensor  # N, 3  int index

  @property
  def primitive_num(self):
    return self.indices.shape[0]
  
  def aabb(self):  
    aabb = torch.zeros((self.primitive_num, 6), dtype=torch.float, device=self.device)

    # Invoke normally
    self.module.triangle_aabb(vertices=self.vertices, indices=self.indices, ele_aabb=aabb)\
        .launchRaw(blockSize=(256, 1, 1), gridSize=((self.primitive_num+255)//256, 1, 1))
    
    return aabb   


class BVH(TensorClass):
  info: torch.Tensor
  aabb: torch.Tensor
  construction_info: torch.Tensor

  mesh: Mesh
         
  @staticmethod 
  def init(n:int, device:str) -> 'BVH':
    return BVH(
      info = torch.zeros((n, 3), dtype=torch.int, device=device),
      aabb = torch.zeros((n, 6), dtype=torch.float, device=device),
      construction_info = torch.zeros((n, 2), dtype=torch.int, device=device)
    )


  def intersect(self, ray_origins:torch.Tensor, ray_directions:torch.Tensor):
    num_rays = ray_origins.shape[0]
    hit = torch.zeros((num_rays, 1), dtype=torch.int, device=self.device)
    hit_pos_map = torch.zeros((num_rays, 3), dtype=torch.float, device=self.device)

    self.module.intersect(num_rays=int(num_rays), rays_o=ray_origins, rays_d=ray_directions,
                  g_lbvh_info=self.info, g_lbvh_aabb=self.aabb,
                  vert=self.mesh.vertices, v_indx=self.mesh.indices,
                  hit_map=hit, hit_pos_map=hit_pos_map
      ).launchRaw(blockSize=(256, 1, 1), gridSize=((num_rays+255)//256, 1, 1))




class BVHBuilder:
  def __init__(self, device:torch.device):
    self.device = device  
    self.module = slangtorch.loadModule(device, 'slang_bvh/slang_bvh.slang') 


  def build(self, mesh:Mesh):
      #first part, get element and bbox---------------

      num_elems = mesh.primitive_num
      prim_idx = torch.arange(num_elems, type=torch.int32, device=mesh.device)

      ele_aabb = mesh.aabb()
      extent_min = ele_aabb[:, 0:3].min()
      extent_max = ele_aabb[:, 3:6].max()

    
      
      morton_codes_ele = torch.zeros((num_elems, 2), dtype=torch.int, device=self.device)
      self.module.morton_codes(num_elems=num_elems, extent_min=extent_min, extent_max=extent_max
                               , ele_aabb=ele_aabb, morton_codes_ele=morton_codes_ele
        ).launchRaw(blockSize=(256, 1, 1), gridSize=((num_elems+255)//256, 1, 1))

      #--------------------------------------------------
      # radix sort part
      morton_codes_ele_pingpong = torch.zeros((num_elems, 2), dtype=torch.int, device=self.device)
      self.module.radix_sort(g_num_elems=num_elems, 
                             g_elements_in=morton_codes_ele, g_elements_out=morton_codes_ele_pingpong
        ).launchRaw(blockSize=(256, 1, 1), gridSize=((num_elems+255)//256, 1, 1))


      #--------------------------------------------------
      # hierarchy
      bvh = BVH.init(num_elems + num_elems - 1, dtype=torch.int, device=self.device)

      self.module.hierarchy(g_num_elems=num_elems, prim_idx=prim_idx, ele_aabb=ele_aabb,
                          g_sorted_morton_codes=morton_codes_ele, 
                          g_lbvh_info=bvh.info, g_lbvh_aabb=bvh.aabb, 
                          g_lbvh_construction_infos=bvh.construction_info
        ).launchRaw(blockSize=(256, 1, 1), gridSize=((num_elems+255)//256, 1, 1))

      #--------------------------------------------------
      # bounding_boxes
      #'''
      tree_heights = torch.zeros((num_elems, 1), dtype=torch.int, device=self.device)
      self.module.get_bvh_height(g_num_elems=num_elems, 
                                 g_lbvh_info=bvh.info,
                                 g_lbvh_aabb=bvh.aabb, 
                                 g_lbvh_construction_infos=bvh.construction_info, 
                                 tree_heights=tree_heights
        ).launchRaw(blockSize=(256, 1, 1), gridSize=((num_elems+255)//256, 1, 1))

      tree_height_max = tree_heights.max()
      for i in range(tree_height_max):
          self.module.get_bbox(g_num_elems=num_elems, 
                              expected_height=int(i+1),
                              g_lbvh_info=bvh.info, 
                              g_lbvh_aabb=bvh.aabb, 
                              g_lbvh_construction_infos=bvh.construction_info
              ).launchRaw(blockSize=(256, 1, 1), gridSize=((num_elems+255)//256, 1, 1))

      self.module.set_root(g_lbvh_info=bvh.info, g_lbvh_aabb=bvh.aabb
        ).launchRaw(blockSize=(1, 1, 1), gridSize=(1, 1, 1)) 
      
      return bvh
