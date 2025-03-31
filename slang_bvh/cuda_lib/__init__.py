from beartype.typing import Tuple
import torch
from torch.utils.cpp_extension import load
from pathlib import Path

sources = [str(Path(__file__).parent  / filename)
            for filename in 
          ["radix_sort_pairs.cu", "segmented_sort_pairs.cu", "module.cpp"]]

cuda_lib = load("cuda_lib", sources=sources, verbose=True)

def check_cuda(name, arg):
  assert arg.is_cuda, f"{name}: device must be a cuda device, got {arg.device}"

def segmented_sort_pairs(keys: torch.Tensor,
                         values:torch.Tensor,
                         start_offsets: torch.Tensor,
                         end_offsets:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
  for name, value in dict(keys=keys, values=values, start_offsets=start_offsets, end_offset=end_offsets).items():
      check_cuda(name, value)
    
  return cuda_lib.segmented_sort_pairs(keys, values, start_offsets, end_offsets)

def radix_sort_pairs(keys:torch.Tensor, values:torch.Tensor, start_bit=0, end_bit=None):
  check_cuda("keys", keys)
  check_cuda("values", values)

  if end_bit is None:
    end_bit = -1

  return cuda_lib.radix_sort_pairs(keys, values, start_bit, end_bit)


def radix_argsort(keys:torch.Tensor):
  idx = torch.arange(keys.shape[0], dtype=torch.int32, device=keys.device)
  _, idx = radix_sort_pairs(keys, idx)
  return idx
  

def radix_sort(keys:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  idx = torch.arange(keys.shape[0], dtype=torch.int32, device=keys.device)
  return radix_sort_pairs(keys, idx)

__all__ = ["radix_sort_pairs", "segmented_sort_pairs", "radix_argsort"]




