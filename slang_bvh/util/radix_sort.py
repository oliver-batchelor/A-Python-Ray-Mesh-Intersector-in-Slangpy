from pathlib import Path
import torch
import slangtorch

file_path = Path(__file__).parent 
module = slangtorch.loadModule(file_path / 'radix_sort.slang')

def radix_sort_pairs(pairs:torch.Tensor):
  assert pairs.dim() == 2 and pairs.shape[1] == 2, f"Expected 2D tensor with shape (N, 2), got {pairs.shape}"
  num_elems = pairs.shape[0]

  double_buffer = torch.zeros((num_elems, 2), dtype=torch.int, device=pairs.device)
  module.radix_sort(num_elements=num_elems, 
                          g_elements_in=pairs, g_elements_out=double_buffer
    ).launchRaw(blockSize=(256, 1, 1), gridSize=(1, 1, 1))
  
  return pairs