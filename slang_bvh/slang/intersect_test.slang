#define MAX_STACK_SIZE 64

import "geometry.slang";
import "triangle.slang";


Optional<Hit> bvh_hit(TensorView<int> bvh_info, TensorView<float> bvh_aabb,
             TensorView<float> vert, TensorView<int> v_indx,
             Ray ray) {
  BVHNode stack[MAX_STACK_SIZE];
  int count = 0;

  /* Add root node */
  stack[count++] = BVHNode.load(bvh_info, 0);
  Optional<Hit> closest_hit = none;

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
      let triangle = Triangle.load(vert, v_indx, node.primitive_idx);
    
      if (let hit = triangle.hit(ray)) {
        ray.t_max = min(hit, ray.t_max);
        closest_hit = Hit(hit, node.primitive_idx);
        
      }
    }
  }

  return closest_hit;
}


[AutoPyBindCUDA]
[CUDAKernel]
void intersect(TensorView<float> ray_origins, TensorView<float> ray_directions,
               TensorView<int> bvh_info, TensorView<float> bvh_aabb,
               TensorView<float> vert, TensorView<int> v_indx,
               TensorView<float> hit_t, TensorView<int> hit_idx)
{
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    uint2 pixel = dispatchIdx.xy;
    if (pixel.x >= ray_origins.size(0) || pixel.y >= ray_origins.size(1)) return;

    let ray = Ray.load(ray_origins, ray_directions, pixel);

    if(let hit = bvh_hit(bvh_info, bvh_aabb, vert, v_indx, ray)) {
        hit_t[pixel] = hit.t_hit;
        hit_idx[pixel] = hit.idx;

    } else {
        hit_t[pixel] = 1e9;
        hit_idx[pixel] = -1;
        
    }

}