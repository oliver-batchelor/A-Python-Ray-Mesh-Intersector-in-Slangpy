import argparse
from pathlib import Path
import cv2
import torch
import torch.nn.functional as F
import pyexr
import time

from .bvh import BVHBuilder, Mesh, MeshBVH


def generate_rays(width, height, device:torch.device):
  aspect = width / height


  y, x = torch.meshgrid([torch.linspace(1, -1, height, device=device), 
                        torch.linspace(-aspect, aspect, width, device=device)], indexing='ij')
  z = -torch.ones_like(x)

  ray_directions = torch.stack([x, y, z], dim=-1)
  ray_origins = torch.tensor([-0.025, 4.05, 6.325], device=device).broadcast_to(ray_directions.shape)

  ray_directions = F.normalize(ray_directions, dim=-1)

  return ray_origins.contiguous(), ray_directions.contiguous()


def prim_colors(hit_idx:torch.Tensor, mesh:Mesh):
  h, w = hit_idx.shape
  hit_colors = torch.zeros(h, w, 3, device=hit_idx.device, dtype=torch.uint8)
  colors = torch.rand(mesh.primitive_num, 3, device=hit_idx.device)
  colors = (colors * 255).to(torch.uint8)

  mask = hit_idx >= 0
  hit_colors[mask] = colors[hit_idx[mask]]
  return hit_colors

def pos_colors(mask:torch.Tensor, hit_pos:torch.Tensor):

  min_pos = hit_pos[mask].min(dim=0).values
  max_pos = hit_pos[mask].max(dim=0).values
  extent = max_pos - min_pos

  h, w = hit_pos.shape[:2]
  color_image = torch.zeros((h, w, 3), device=hit_pos.device, dtype=torch.float32)
  color_image[mask] = (hit_pos[mask] - min_pos.unsqueeze(0)) / extent.unsqueeze(0)

  color_image = (color_image * 255).clamp(0, 255).to(torch.uint8)
  return color_image


def display(image:torch.Tensor, title:str='image'):
  cv2.imshow(title, cv2.cvtColor(image.cpu().numpy(), cv2.COLOR_RGB2BGR))
  cv2.waitKey(0)



def main():

  project_root = Path(__file__).parent.parent

  parser = argparse.ArgumentParser()
  parser.add_argument('--model', type=str, default=str(project_root  / 'models' / 'bunny.obj'), help='path to the model')
  parser.add_argument('--output', type=str, default='./color.exr', help='output path')

  parser.add_argument('--device', type=torch.device, default='cuda:0', help='device to use')

  args = parser.parse_args()

  mesh = Mesh.load_trimesh(args.model, device=args.device)
  print(f"Loaded mesh {mesh} from {args.model}")
  builder = BVHBuilder(device=args.device)

  image_size = (1920, 1024)
  ray_origins, ray_directions = generate_rays(image_size[0], image_size[1], device=args.device)


  while True:

    # get bvh tree
    start_time = time.time()
    bvh:MeshBVH = builder.build(mesh)

    torch.cuda.synchronize()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"GPU bvh build finished in: {elapsed_time} s")

    torch.cuda.synchronize()


    start_time = time.time()
    hit_t, hit_idx = bvh.intersect(ray_origins, ray_directions)


    torch.cuda.synchronize()
    end_time = time.time()


    elapsed_time = end_time - start_time
    print("ray query time:", elapsed_time, "s")

    mask = hit_idx >= 0

    hit_pos = torch.zeros_like(ray_origins)
    hit_pos[mask] = ray_origins[mask] + ray_directions[mask] * hit_t[mask].unsqueeze(-1)

    image = pos_colors(mask, hit_pos)
    display(image, title='hit_pos')

    # prim_image = prim_colors(hit_idx, mesh)
    # display(prim_image, title='prim_colors')

  # pyexr.write(args.output, hit_pos.reshape(image_size[1], image_size[0], 3).cpu().numpy())


if __name__ == "__main__":
  main()