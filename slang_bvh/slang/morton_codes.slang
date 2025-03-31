
// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
uint expandBits32(uint v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

uint64_t expandBits64(uint64_t v) {
    v &= 0x1fffff;
    v = (v | (v << 32)) & 0x1f00000000ffffULL;
    v = (v | (v << 16)) & 0x1f0000ff0000ffULL;
    v = (v | (v << 8)) & 0x100f00f00f00f00fULL;
    v = (v | (v << 4)) & 0x10c30c30c30c30c3ULL;
    v = (v | (v << 2)) & 0x1249249249249249ULL;
    return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
uint morton32(float3 p) {
  var int_p = uint3(clamp(p * 1024.0f, 0.0f, 1023.0f));

  uint x = expandBits32(int_p.x);
  uint y = expandBits32(int_p.y);
  uint z = expandBits32(int_p.z);
  return (x << 2) | (y << 1) | z;
}

uint64_t morton64(float3 p) {
  const float max_float = pow(2, 20);
  float3 clamped = clamp(p, 0.0, 1.0f);

  uint64_t x = expandBits64(uint64_t(clamped.x * max_float));
  uint64_t y = expandBits64(uint64_t(clamped.y * max_float));
  uint64_t z = expandBits64(uint64_t(clamped.z * max_float));
  return (x << 2) | (y << 1) | z;
}



[AutoPyBindCUDA]
[CUDAKernel]
void morton_codes_aabb(float3 min_extent, float3 max_extent,

                  TensorView<float> aabb,
                  TensorView<int> morton_codes) 
{
    // Get the 'global' index of this thread.
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    uint idx = dispatchIdx.x;
    if (idx >= aabb.size(0)) return;

    var min_bound = float3(aabb[idx, 0], aabb[idx, 1], aabb[idx, 2]);
    var max_bound = float3(aabb[idx, 3], aabb[idx, 4], aabb[idx, 5]);

    // calculate center
    float3 center = (min_bound + 0.5 * (max_bound - min_bound));
    // map to unit cube
    float3 normalized_centre = (center - min_extent) / (max_extent - min_extent);
    // assign morton code
    uint mortonCode = morton32(normalized_centre);

    morton_codes[idx, 0] = int(mortonCode);
    morton_codes[idx, 1] = int(idx);
}

[AutoPyBindCUDA]
[CUDAKernel]
void morton_codes(TensorView<float> points,
                  TensorView<int> morton_codes) 
{
    // Get the 'global' index of this thread.
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    uint idx = dispatchIdx.x;
    if (idx >= points.size(0)) return;

    var point = float3(points[idx, 0], points[idx, 1], points[idx, 2]);
    morton_codes[idx] = int(morton32(point));
}