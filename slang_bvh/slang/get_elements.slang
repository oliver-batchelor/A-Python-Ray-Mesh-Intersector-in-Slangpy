



[AutoPyBindCUDA]
[CUDAKernel]
void triangle_aabb(
    TensorView<float> vertices, TensorView<int> indices,
    TensorView<int> prim_idx, TensorView<float> aabb) {
    // Get the 'global' index of this thread.
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    if (dispatchIdx.x >= indices.size(0)) return;
    int primitiveIndex = dispatchIdx.x;
    float3 minV = float3(1e9, 1e9, 1e9);
    float3 maxV = float3(-1e9, -1e9, -1e9);
    
    for (int i = 0; i < 3; i++) {
        int v_i = indices[primitiveIndex, i];
        float3 v = float3(vertices[v_i, 0], vertices[v_i, 1], vertices[v_i, 2]);

        minV = min(minV, v);
        maxV = max(maxV, v);
    }

    prim_idx[primitiveIndex] = primitiveIndex;
    for (int i = 0; i < 3; i++) {
      aabb[primitiveIndex, i] = minV[i];
      aabb[primitiveIndex, i + 3] = maxV[i];
    }
}


int findMSB(uint val) {
    if (val == 0) return -1;
    int msb = 31;  // 32 uint
    while (!bool((val >> msb) & 1)) {
        msb--;
    }
    return msb;
}
uint expandBits(uint v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
uint morton3D(float x, float y, float z) {
    x = min(max(x * 1024.0f, 0.0f), 1023.0f);
    y = min(max(y * 1024.0f, 0.0f), 1023.0f);
    z = min(max(z * 1024.0f, 0.0f), 1023.0f);
    uint xx = expandBits(uint(x));
    uint yy = expandBits(uint(y));
    uint zz = expandBits(uint(z));
    return xx * 4 + yy * 2 + zz;
}

[AutoPyBindCUDA]
[CUDAKernel]
void debug_cb(
    TensorView<float> a,
    TensorView<int> b
    )
{
    b[0] = morton3D(a[0], a[1], a[2]);
}