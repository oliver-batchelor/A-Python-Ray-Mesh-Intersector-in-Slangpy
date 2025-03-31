from slang_bvh.cuda_lib import segmented_sort_pairs
from slang_bvh.cuda_lib import radix_sort_pairs
import torch


def test():
  k = torch.randint(100, (20,), dtype=torch.uint32, device="cuda")
  v = torch.arange(20, dtype=torch.int32, device="cuda")

  start_offsets = torch.tensor([0, 8, 16], dtype=torch.long, device="cuda")
  end_offsets = torch.tensor([8, 16, 20], dtype=torch.long, device="cuda")

  k1, v1 = segmented_sort_pairs(k, v, start_offsets, end_offsets)
  print(k1)
  print(v1)

  k2, v2 = radix_sort_pairs(k, v)
  print(k2)
  print(v2)  

if __name__ == "__main__":
  test()
