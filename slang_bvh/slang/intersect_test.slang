#define MAX_STACK_SIZE 64


float3 load_float3(TensorView<float> tensor, int idx) {
    return float3(tensor[idx, 0], tensor[idx, 1], tensor[idx, 2]);
}


struct GPUBvhNode {
    int index;
    int left;          // pointer to the left child or INVALID_POINTER in case of leaf
    int right;         // pointer to the right child or INVALID_POINTER in case of leaf
    uint primitiveIdx; // custom value that is copied from the input Element or 0 in case of inner node

  static GPUBvhNode load(TensorView<int> bvh_info, int nodeIdx) {
    return GPUBvhNode(
        nodeIdx,
        bvh_info[nodeIdx, 0], // left
        bvh_info[nodeIdx, 1], // right
        bvh_info[nodeIdx, 2]  // primitiveIdx
    );
  }

};

struct AABB{
  float3 lower;
  float3 upper;

  static AABB load(TensorView<float> aabbs, int nodeIdx) {
    float3 lower = float3(aabbs[nodeIdx, 0], aabbs[nodeIdx, 1], aabbs[nodeIdx, 2]);
    float3 upper = float3(aabbs[nodeIdx, 3], aabbs[nodeIdx, 4], aabbs[nodeIdx, 5]);

    return AABB(lower, upper);
  }

  /*
  intersection code from https://github.com/maxilevi/raytracer/tree/master
  */

  bool hit(Ray ray) {
    float t_min = ray.t_min;
    float t_max = ray.t_max;

    for (int i = 0; i < 3; ++i)
    {
        float ray_d_i = ray.direction[i];
        if (ray_d_i == 0.f) ray_d_i = 0.000001f;
        float inverse_direction = 1.0 / ray_d_i;
        float t0 = (lower[i] - ray.origin[i]) * inverse_direction;
        float t1 = (upper[i] - ray.origin[i]) * inverse_direction;
        if (inverse_direction < 0.0) {
            float tmp = t1;
            t1 = t0;
            t0 = tmp;
        }
        t_min = t0 > t_min ? t0 : t_min;
        t_max = t1 < t_max ? t1 : t_max;
        if (t_max < t_min)
            return false;
    }

    return true;
  }

};

struct Hit {
  float t_hit;
  int idx;
};

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

    static Triangle load(TensorView<float> vert, TensorView<int> v_indx, int idx) {
        float3 v0 = load_float3(vert, v_indx[idx, 0]);
        float3 v1 = load_float3(vert, v_indx[idx, 1]);
        float3 v2 = load_float3(vert, v_indx[idx, 2]);
        return Triangle(v0, v1, v2);
    }
};

struct Ray {
    float3 origin;
    float3 direction;

    float t_min;
    float t_max;

    static Ray load(TensorView<float> origins, TensorView<float> directions, int idx) {
        float3 origin = float3(origins[idx, 0], origins[idx, 1], origins[idx, 2]);
        float3 direction = float3(directions[idx, 0], directions[idx, 1], directions[idx, 2]);

        return Ray(origin, direction, 0.f, 1e9);
    }
};




Optional<Hit> bvh_hit(TensorView<int> bvh_info, TensorView<float> bvh_aabb,
             TensorView<float> vert, TensorView<int> v_indx,
             Ray ray) {
  GPUBvhNode stack[MAX_STACK_SIZE];
  int count = 0;

  /* Add root node */
  stack[count++] = GPUBvhNode.load(bvh_info, 0);
  Optional<Hit> closest_hit = none;

  while (count > 0) {
    GPUBvhNode node = stack[--count];
    let aabb = AABB.load(bvh_aabb, node.index);

    
    if (!aabb.hit(ray))
        continue;

    if (node.primitiveIdx == -1) {
        stack[count++] = GPUBvhNode.load(bvh_info, node.left);
        stack[count++] = GPUBvhNode.load(bvh_info, node.right);

    } else {  
      // Leaf node
      let triangle = Triangle.load(vert, v_indx, node.primitiveIdx);
    
      if (let hit = triangle.hit(ray)) {
        ray.t_max = min(hit, ray.t_max);
        closest_hit = Hit(hit, node.primitiveIdx);
        
      }
    }
  }

  return closest_hit;
}



[AutoPyBindCUDA]
[CUDAKernel]
void intersect(int num_rays, TensorView<float> ray_origins, TensorView<float> ray_directions,
               TensorView<int> bvh_info, TensorView<float> bvh_aabb,
               TensorView<float> vert, TensorView<int> v_indx,
               TensorView<float> hit_t, TensorView<int> hit_idx)
{
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    uint ray_index = dispatchIdx.x;
    if (ray_index >= num_rays) return;

    let ray = Ray.load(ray_origins, ray_directions, ray_index);

    if(let hit = bvh_hit(bvh_info, bvh_aabb, vert, v_indx, ray)) {
        hit_t[ray_index] = hit.t_hit;
        hit_idx[ray_index] = hit.idx;

    } else {
        hit_t[ray_index] = 1e9;
        hit_idx[ray_index] = -1;
        
    }

}