#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cub/cub.cuh>


// for our workload segmented sort (non radix) seems faster
template <typename K, typename V>
void sort_helper(
  K *d_keys, V *d_values, 
  K *d_keys_out, V *d_values_out,
  int num_items, 
  int64_t *d_start_offset, int64_t *d_end_offset, 
  int num_segments) 
{
  size_t temp_storage_bytes = 0;
  auto stream = at::cuda::getCurrentCUDAStream();

  cub::DeviceSegmentedSort::SortPairs(nullptr, temp_storage_bytes,
      d_keys, d_keys_out, d_values, d_values_out,
      num_items, num_segments, d_start_offset, d_end_offset, stream);

  auto temp_storage = torch::empty({int64_t(temp_storage_bytes)}, 
    torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));

  cub::DeviceSegmentedSort::SortPairs(temp_storage.data_ptr<uint8_t>(), temp_storage_bytes,
      d_keys, d_keys_out, d_values, d_values_out,
      num_items, num_segments, d_start_offset, d_end_offset, stream);
}

// Template dispatch helper for segmented sort with int32 values
template<typename K>
std::pair<torch::Tensor, torch::Tensor> segmented_sort_dispatch(
  const torch::Tensor& keys, const torch::Tensor& values,
  const torch::Tensor& start_offset, const torch::Tensor& end_offset) 
{
  auto keys_out = torch::empty_like(keys);
  auto values_out = torch::empty_like(values);
    
  sort_helper<K, int32_t>(
    keys.data_ptr<K>(), values.data_ptr<int32_t>(), 
    keys_out.data_ptr<K>(), values_out.data_ptr<int32_t>(),
    keys.size(0), 
    start_offset.data_ptr<int64_t>(), end_offset.data_ptr<int64_t>(),
    start_offset.size(0));

  return std::make_pair(keys_out, values_out);
}

std::pair<torch::Tensor, torch::Tensor> segmented_sort_pairs(
  const torch::Tensor keys, const torch::Tensor values,
  const torch::Tensor start_offset, const torch::Tensor end_offset) 
{
  assert (keys.dim() == 1 && values.dim() == 1), "keys and values must be 1D";
  assert (keys.size(0) == values.size(0)), "keys and values must have the same size";

  assert (start_offset.dim() == 1 && end_offset.dim() == 1 && start_offset.size(0) == end_offset.size(0)), 
    "start_offset and end_offset must be 1D and have the same size"; 

  assert (start_offset.scalar_type() == torch::kInt64 
    && end_offset.scalar_type() == torch::kInt64), "start_offset/end_offset must be int64";
  
  // Check for value type
  assert (values.scalar_type() == torch::kInt32), "values must be int32";
  
  // Dispatch based on key type
  if (keys.scalar_type() == torch::kUInt32) {
    return segmented_sort_dispatch<uint32_t>(keys, values, start_offset, end_offset);
  } else if (keys.scalar_type() == torch::kUInt16) {
    return segmented_sort_dispatch<uint16_t>(keys, values, start_offset, end_offset);
  } else { 
    throw std::runtime_error("Keys must be either uint16 or uint32");
  }
}

