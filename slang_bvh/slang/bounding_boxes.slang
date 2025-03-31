import "geometry.slang";



// For bottom-up construction - only uses child bounds
AABB bounds_from_children(BVHNode node, TensorView<float> bvh_aabb) {
    AABB bounds = AABB.invalid();
    
    if (node.left != 0) bounds = bounds.union(AABB.load(bvh_aabb, node.left));
    if (node.right != 0) bounds = bounds.union(AABB.load(bvh_aabb, node.right));
    return bounds;
}


[AutoPyBindCUDA]
[CUDAKernel]
void bounding_boxes(uint num_elements,
                TensorView<int> bvh_info, TensorView<float> bvh_aabb,
                TensorView<int> bvh_construction_infos)
{
    uint idx = cudaThreadIdx().x + cudaBlockIdx().x * cudaBlockDim().x;
    if (idx >= num_elements) return;
    const int leaf_offset = int(num_elements) - 1;

    uint node_idx = bvh_construction_infos[leaf_offset + idx, 0];
    while (true) {
        int visitations;
        bvh_construction_infos.InterlockedAdd(uint2(node_idx, 1), 1, visitations);
        if (visitations < 1) break;

        BVHNode node = BVHNode.load(bvh_info, node_idx);
        bounds_from_children(node, bvh_aabb).store(bvh_aabb, node_idx);
        
        if (node_idx == 0) break;
        node_idx = bvh_construction_infos[node_idx, 0];
    }
}






