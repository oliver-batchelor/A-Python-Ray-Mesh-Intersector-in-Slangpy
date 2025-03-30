
float3 load_float3(TensorView<float> tensor, int idx) {
    return float3(tensor[idx, 0], tensor[idx, 1], tensor[idx, 2]);
}

struct LBVHConstructionInfo {
    uint parent;         // pointer to the parent
    int visitationCount; // number of threads that arrived
};


struct BVHNode {
    int index;
    int left;          // pointer to the left child (or zero in case of leaf)
    int right;        
    uint primitive_idx; // index of the primitive in the leaf node, -1 if not a leaf

  static BVHNode load(TensorView<int> bvh_info, int node_idx) {
    return BVHNode(
        node_idx,
        bvh_info[node_idx, 0], // left
        bvh_info[node_idx, 1], // right
        bvh_info[node_idx, 2]  // primitive_idx
    );
  }

  void store(TensorView<int> bvh_info) {
    bvh_info[index, 0] = left;
    bvh_info[index, 1] = right;
    bvh_info[index, 2] = int(primitive_idx);
  }

};


struct Hit {
  float t_hit;
  int idx;
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

    static Ray load(TensorView<float> origins, TensorView<float> directions, int2 idx) {
        float3 origin = float3(origins[idx.x, idx.y, 0], origins[idx.x, idx.y, 1], origins[idx.x, idx.y, 2]);
        float3 direction = float3(directions[idx.x, idx.y, 0], directions[idx.x, idx.y, 1], directions[idx.x, idx.y, 2]);

        return Ray(origin, direction, 0.f, 1e9);
    }

  };

struct AABB {
  float3 lower;
  float3 upper;

  static AABB load(TensorView<float> aabbs, int node_idx) {
    float3 lower = float3(aabbs[node_idx, 0], aabbs[node_idx, 1], aabbs[node_idx, 2]);
    float3 upper = float3(aabbs[node_idx, 3], aabbs[node_idx, 4], aabbs[node_idx, 5]);

    return AABB(lower, upper);
  }

  static AABB invalid() {
    return AABB(float3(1e9), float3(-1e9));
  }

  void store(TensorView<float> aabbs, int node_idx) {
    for(int i = 0; i < 3; i++) {
      aabbs[node_idx, i] = lower[i];
      aabbs[node_idx, i + 3] = upper[i];
    }
  }

  AABB expand(float3 point) {
    return AABB(
        min(lower, point),
        max(upper, point)
    );
  }

  AABB union(AABB other) {
    return AABB(
        min(lower, other.lower),
        max(upper, other.upper)
    );
  }

  bool hit(Ray ray) {
    float t_min = ray.t_min;
    float t_max = ray.t_max;

    for (int i = 0; i < 3; ++i) {
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


