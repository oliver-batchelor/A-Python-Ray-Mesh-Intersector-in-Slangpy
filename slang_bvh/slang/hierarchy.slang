
import "geometry.slang";


int clz(uint) {
  __intrinsic_asm R"(__clz($0))";
}

int delta(int i, uint codeI, int j, uint num_elements, TensorView<int> sorted_codes) {
    if (j < 0 || j > num_elements - 1) {
        return -1;
    }
    uint codeJ = sorted_codes[j, 0];

    // handle duplicate morton codes
    if (codeI == codeJ) {
        // add 32 for common prefix of codeI ^ codeJ
        return 32 + clz(i ^ j);
    }
    return clz(codeI ^ codeJ);
}

void determineRange(int idx, out int lower, out int upper, uint num_elements, TensorView<int> sorted_codes) {
    // determine direction of the range (+1 or -1)
    const uint code = sorted_codes[idx];
    const int deltaL = delta(idx, code, idx - 1, num_elements, sorted_codes);
    const int deltaR = delta(idx, code, idx + 1, num_elements, sorted_codes);
    const int d = (deltaR >= deltaL) ? 1 : -1;

    // compute upper bound for the length of the range
    const int deltaMin = min(deltaL, deltaR);// delta(idx, code, idx - d);
    int lMax = 2;
    while (delta(idx, code, idx + lMax * d, num_elements, sorted_codes) > deltaMin) {
        lMax = lMax << 1;
    }

    // find the other end using binary search
    int l = 0;
    for (int t = lMax >> 1; t > 0; t >>= 1) {
        if (delta(idx, code, idx + (l + t) * d, num_elements, sorted_codes) > deltaMin) {
            l += t;
        }
    }
    int jdx = idx + l * d;

    // ensure idx < jdx
    lower = min(idx, jdx);
    upper = max(idx, jdx);
}

int findSplit(int first, int last, uint num_elements, TensorView<int> sorted_codes) {
    uint firstCode = sorted_codes[first, 0];

    // Calculate the number of highest bits that are the same
    // for all objects, using the count-leading-zeros intrinsic.
    int commonPrefix = delta(first, firstCode, last, num_elements, sorted_codes);

    // Use binary search to find where the next bit differs.
    // Specifically, we are looking for the highest object that
    // shares more than commonPrefix bits with the first one.
    int split = first;// initial guess
    int stride = last - first;
    do {
        stride = (stride + 1) >> 1;// exponential decrease
        int newSplit = split + stride;// proposed new position
        if (newSplit < last) {
            int splitPrefix = delta(first, firstCode, newSplit, num_elements, sorted_codes);
            if (splitPrefix > commonPrefix) {
                split = newSplit;// accept proposal
            }
        }
    } while (stride > 1);

    return split;
}

[AutoPyBindCUDA]
[CUDAKernel]
void hierarchy(uint num_elements, 
               TensorView<int> ele_primitive_idx,
               TensorView<float> ele_aabb,
               TensorView<int> sorted_codes, 
               
               // outputs
               TensorView<int> bvh_info, 
               TensorView<float> bvh_aabb,
               TensorView<int> bvh_construction_infos)
{
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    uint bvh_idx = dispatchIdx.x;
    if (bvh_idx >= num_elements) return;
    const int leaf_offset = int(num_elements) - 1;

    // construct leaf nodes
    if (bvh_idx < num_elements) {
        int sorted_idx = sorted_codes[bvh_idx, 1];
        uint primitive_idx = ele_primitive_idx[sorted_idx, 0];

        int node_idx = leaf_offset + bvh_idx;
        let node = BVHNode(node_idx, 0, 0, primitive_idx);

        node.store(bvh_info);
        let aabb = AABB.load(ele_aabb, primitive_idx);
        aabb.store(bvh_aabb, node_idx);
    }

    // construct internal nodes
    if (bvh_idx < num_elements - 1) {
        int first, last;
        determineRange(int(bvh_idx), first, last, num_elements, sorted_codes);
        int split = findSplit(first, last, num_elements, sorted_codes);

        let node = BVHNode(bvh_idx,
          (split == first) ? leaf_offset + split : split,
          (split + 1 == last) ? leaf_offset + split + 1 : split + 1,
          -1
        );

        node.store(bvh_info);

        let aabb = AABB.invalid();
        aabb.store(bvh_aabb, bvh_idx);

        // Store construction info for children
        bvh_construction_infos[node.left, 0] = bvh_idx;
        bvh_construction_infos[node.left, 1] = 0;

        bvh_construction_infos[node.right, 0] = bvh_idx;
        bvh_construction_infos[node.right, 1] = 0;
    }

    // Set root node construction info
    if (bvh_idx == 0) {
        bvh_construction_infos[0, 0] = 0;
        bvh_construction_infos[0, 1] = 0;
    }
}