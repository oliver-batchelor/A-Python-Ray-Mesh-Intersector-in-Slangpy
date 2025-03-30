import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
import pyexr
import time

from .bvh import BVHBuilder, Mesh, MeshBVH


def generate_rays(width, height, device:torch.device):
  y, x = torch.meshgrid([torch.linspace(1, -1, height, device=device), 
                        torch.linspace(-1, 1, width, device=device)], indexing='ij')
  z = -torch.ones_like(x)

  ray_directions = torch.stack([x, y, z], dim=-1)
  ray_origins = torch.tensor([-0.025, 0.05, 0.125], device=device).broadcast_to(ray_directions.shape)

  ray_origins = ray_origins.reshape(-1,3)
  ray_directions = F.normalize(ray_directions.reshape(-1,3))

  return ray_origins.contiguous(), ray_directions.contiguous()


def main():

  project_root = Path(__file__).parent.parent

  parser = argparse.ArgumentParser()
  parser.add_argument('--model', type=str, default=str(project_root  / 'models' / 'bunny.obj'), help='path to the model')
  parser.add_argument('--output', type=str, default='./color.exr', help='output path')

  parser.add_argument('--device', type=torch.device, default='cuda:0', help='device to use')

  args = parser.parse_args()

  mesh = Mesh.load_trimesh(args.model, device=args.device)
  builder = BVHBuilder(device=args.device)



  # get bvh tree
  start_time = time.time()
  bvh:MeshBVH = builder.build(mesh)

  end_time = time.time()
  elapsed_time = end_time - start_time
  print(f"GPU bvh build finished in: {elapsed_time} s")

  print("bvh build over!")
  image_size = (800, 800)
  ray_origins, ray_directions = generate_rays(image_size[0], image_size[1], device=args.device)

  torch.cuda.synchronize()

  start_time = time.time()
  hit_t, hit_idx = bvh.intersect(ray_origins, ray_directions)
  end_time = time.time()

  elapsed_time = end_time - start_time
  print("ray query time:", elapsed_time, "s")

  mask = hit_idx >= 0
  print(mask.sum())
  
  hit_pos = torch.zeros_like(ray_origins)
  hit_pos[mask] = ray_origins[mask] + ray_directions[mask] * hit_t[mask].unsqueeze(-1)

  pyexr.write(args.output, hit_pos.reshape(image_size[1], image_size[0], 3).cpu().numpy())


if __name__ == "__main__":
  main()