#define MAX_STACK_SIZE 64

import "geometry.slang";
import "triangle.slang";

interface IPrimitiveType {
  Optional<float> hit(Ray ray, int primitive_idx);

};


struct Triangles : IPrimitiveType {
  TensorView<float> vertices;
  TensorView<int> indices;

  Optional<float> hit(Ray ray, int primitive_idx) {
    let tri = Triangle.load(vertices, indices, primitive_idx);
    return tri.hit(ray);
  }
};



Hit bvh_hit<Primitives : IPrimitiveType>(
    TensorView<int> bvh_info,
    TensorView<float> bvh_aabb,
    Primitives primitives,
    Ray ray) 
  {
  BVHNode stack[MAX_STACK_SIZE];
  int count = 0;

  /* Add root node */
  stack[count++] = BVHNode.load(bvh_info, 0);
  Hit closest_hit = Hit(1e9, -1);

  while (count > 0) {
    BVHNode node = stack[--count];
    let aabb = AABB.load(bvh_aabb, node.index);

    
    if (!aabb.hit(ray))
        continue;

    if (node.primitive_idx == -1) {
        stack[count++] = BVHNode.load(bvh_info, node.left);
        stack[count++] = BVHNode.load(bvh_info, node.right);

    } else {  
      // Leaf node
    
      if (let t = primitives.hit(ray, node.primitive_idx)) {
        if (closest_hit.t > t) {
          // ray.t_max = t;
          closest_hit = Hit(t, node.primitive_idx);
        }
        
      }
    }
  }

  return closest_hit;
}


[AutoPyBindCUDA]
[CUDAKernel]
void intersect(TensorView<float> ray_origins, TensorView<float> ray_directions,
               TensorView<int> bvh_info, TensorView<float> bvh_aabb,
               TensorView<float> vertices, TensorView<int> indices,
               TensorView<float> hit_t, TensorView<int> hit_idx)
{
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    uint2 pixel = dispatchIdx.xy;
    if (pixel.x >= ray_origins.size(0) || pixel.y >= ray_origins.size(1)) return;

    let ray = Ray.load(ray_origins, ray_directions, pixel);

    let triangles = Triangles(vertices, indices);
    let hit = bvh_hit(bvh_info, bvh_aabb, triangles, ray);

    hit_t[pixel] = hit.t;
    hit_idx[pixel] = hit.idx;
}