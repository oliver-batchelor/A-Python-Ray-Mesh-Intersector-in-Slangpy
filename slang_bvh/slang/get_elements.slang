



[AutoPyBindCUDA]
[CUDAKernel]
void triangle_aabb(
    TensorView<float> vertices, TensorView<int> indices, TensorView<float> aabb) {

    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    if (dispatchIdx.x >= indices.size(0)) return;

    int idx = dispatchIdx.x;
    float3 minV = float3(1e9, 1e9, 1e9);
    float3 maxV = float3(-1e9, -1e9, -1e9);
    
    for (int i = 0; i < 3; i++) {
        int v_i = indices[idx, i];
        float3 v = float3(vertices[v_i, 0], vertices[v_i, 1], vertices[v_i, 2]);

        minV = min(minV, v);
        maxV = max(maxV, v);
    }

    for (int i = 0; i < 3; i++) {
      aabb[idx, i] = minV[i];
      aabb[idx, i + 3] = maxV[i];
    }
}

