
// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
uint expandBits(uint v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
uint morton3D(float3 p) {
  var int_p = uint3(clamp(p * 1024.0f, 0.0f, 1023.0f));

  uint x = expandBits(int_p.x);
  uint y = expandBits(int_p.y);
  uint z = expandBits(int_p.z);
  return (x << 2) | (y << 1) | z;
}

[AutoPyBindCUDA]
[CUDAKernel]
void morton_codes(float3 min_extent, float3 max_extent,
                  int num_elements, 

                  TensorView<float> ele_aabb,
                  TensorView<int> morton_codes_ele) 
{
    // Get the 'global' index of this thread.
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    uint idx = dispatchIdx.x;
    if (idx >= num_elements) return;

    var min_bound = float3(ele_aabb[idx, 0], ele_aabb[idx, 1], ele_aabb[idx, 2]);
    var max_bound = float3(ele_aabb[idx, 3], ele_aabb[idx, 4], ele_aabb[idx, 5]);

    // calculate center
    float3 center = (min_bound + 0.5 * (max_bound - min_bound));
    // map to unit cube
    float3 normalized_centre = (center - min_extent) / (max_extent - min_extent);
    // assign morton code
    uint mortonCode = morton3D(normalized_centre);

    morton_codes_ele[idx, 0] = int(mortonCode);
    morton_codes_ele[idx, 1] = int(idx);
}