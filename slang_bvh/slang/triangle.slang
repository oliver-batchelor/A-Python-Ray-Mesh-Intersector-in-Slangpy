
import "geometry.slang";


struct Triangle {
  float3 v0;
  float3 v1;
  float3 v2;

  Optional<float> hit(Ray ray) {
      float epsilon = 1e-9;
      float3 E1 = v1 - v0;
      float3 E2 = v2 - v0;
      float3 P = cross(ray.direction, E2);
      float det = dot(E1, P);
      if (det > -epsilon && det < epsilon)
          return none;

      float invDet = 1 / det;

      float3 T = ray.origin - v0;
      float u = dot(T,P) * invDet;
      if (u < 0 || u > 1) return none;

      float3 Q = cross(T, E1);
      float v = dot(ray.direction, Q) * invDet;
      if (v < 0 || u + v > 1) return none;

      return dot(E2, Q) * invDet;
    }

    static Triangle load(TensorView<float> vert, TensorView<int> indices, int idx) {
        float3 v0 = load_float3(vert, indices[idx, 0]);
        float3 v1 = load_float3(vert, indices[idx, 1]);
        float3 v2 = load_float3(vert, indices[idx, 2]);
        return Triangle(v0, v1, v2);
    }
};


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

