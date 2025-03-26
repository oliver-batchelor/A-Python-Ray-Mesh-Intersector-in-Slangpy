from pathlib import Path
import torch
import trimesh
import pyexr
import slangpy as spy
import time
import csv
import numpy as np

import slang_bvh

#load obj
mesh = trimesh.load('./models/bunny.obj')

vrt = torch.from_numpy(mesh.vertices).cuda().float()
v_ind = torch.from_numpy(mesh.faces).cuda().int()

device = spy.create_device(include_paths=[
          (Path(__file__).parent / 'slang_bvh').absolute(),
  ])

bvh = slang_bvh.Module(device)

def generate_rays():
  y, x = torch.meshgrid([torch.linspace(1, -1, 800), 
                        torch.linspace(-1, 1, 800)], indexing='ij')
  z = -torch.ones_like(x)
  ray_directions = torch.stack([x, y, z], dim=-1).cuda()
  ray_origins = torch.Tensor([0, 0.1, 0.3]).cuda().broadcast_to(ray_directions.shape)

  ray_origins = ray_origins.contiguous().reshape(-1,3)
  ray_directions = ray_directions.contiguous().reshape(-1,3)

  return ray_origins, ray_directions


# get bvh tree
start_time = time.time()
LBVHNode_info, LBVHNode_aabb = bvh.build(vrt, v_ind)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"GPU bvh build finished in: {elapsed_time} s")

print("bvh build over!")

start_time = time.time()


end_time = time.time()

elapsed_time = end_time - start_time
print("ray query time:", elapsed_time, "s")

# drawing result
#locs = hit.repeat(1,3)
locs = hit_pos_map
pyexr.write(f'./color.exr', locs.reshape(800,800,3).cpu().numpy())
