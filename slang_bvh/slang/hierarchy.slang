#define INVALID_POINTER 0x0

// Utility functions at the top
float3 loadFloat3(TensorView<float> tensor, int idx, int offset) {
    return float3(
        tensor[idx, offset + 0],
        tensor[idx, offset + 1],
        tensor[idx, offset + 2]
    );
}

void storeFloat3(TensorView<float> tensor, int idx, int offset, float3 value) {
    tensor[idx, offset + 0] = value.x;
    tensor[idx, offset + 1] = value.y;
    tensor[idx, offset + 2] = value.z;
}

void storeLBVHNode(int idx, int left, int right, uint primitiveIdx, float3 aabbMin, float3 aabbMax,
                   TensorView<int> info, TensorView<float> aabb) {
    info[idx, 0] = left;
    info[idx, 1] = right;
    info[idx, 2] = int(primitiveIdx);
    storeFloat3(aabb, idx, 0, aabbMin);
    storeFloat3(aabb, idx, 3, aabbMax);
}

struct Element {
    uint primitiveIdx;
    float3 aabbMin;
    float3 aabbMax;
};



struct LBVHConstructionInfo {
    uint parent;         // pointer to the parent
    int visitationCount; // number of threads that arrived
};

int clz(uint) {
  __intrinsic_asm R"(__clz($0))";
}

int delta(int i, uint codeI, int j, uint g_num_elements, TensorView<int> g_sorted_morton_codes) {
    if (j < 0 || j > g_num_elements - 1) {
        return -1;
    }
    uint codeJ = g_sorted_morton_codes[j, 0];
    if (codeI == codeJ) {
        // handle duplicate morton codes
        uint elementIdxI = i;// g_sorted_morton_codes[i].elementIdx;
        uint elementIdxJ = j;// g_sorted_morton_codes[j].elementIdx;
        // add 32 for common prefix of codeI ^ codeJ
        return 32 + clz(elementIdxI ^ elementIdxJ);
    }
    return clz(codeI ^ codeJ);
}

void determineRange(int idx, out int lower, out int upper, uint g_num_elements, TensorView<int> g_sorted_morton_codes) {
    // determine direction of the range (+1 or -1)
    const uint code = g_sorted_morton_codes[idx];
    const int deltaL = delta(idx, code, idx - 1, g_num_elements, g_sorted_morton_codes);
    const int deltaR = delta(idx, code, idx + 1, g_num_elements, g_sorted_morton_codes);
    const int d = (deltaR >= deltaL) ? 1 : -1;

    // compute upper bound for the length of the range
    const int deltaMin = min(deltaL, deltaR);// delta(idx, code, idx - d);
    int lMax = 2;
    while (delta(idx, code, idx + lMax * d, g_num_elements, g_sorted_morton_codes) > deltaMin) {
        lMax = lMax << 1;
    }

    // find the other end using binary search
    int l = 0;
    for (int t = lMax >> 1; t > 0; t >>= 1) {
        if (delta(idx, code, idx + (l + t) * d, g_num_elements, g_sorted_morton_codes) > deltaMin) {
            l += t;
        }
    }
    int jdx = idx + l * d;

    // ensure idx < jdx
    lower = min(idx, jdx);
    upper = max(idx, jdx);
}

int findSplit(int first, int last, uint g_num_elements, TensorView<int> g_sorted_morton_codes) {
    uint firstCode = g_sorted_morton_codes[first, 0];

    // Calculate the number of highest bits that are the same
    // for all objects, using the count-leading-zeros intrinsic.
    int commonPrefix = delta(first, firstCode, last, g_num_elements, g_sorted_morton_codes);

    // Use binary search to find where the next bit differs.
    // Specifically, we are looking for the highest object that
    // shares more than commonPrefix bits with the first one.
    int split = first;// initial guess
    int stride = last - first;
    do {
        stride = (stride + 1) >> 1;// exponential decrease
        int newSplit = split + stride;// proposed new position
        if (newSplit < last) {
            int splitPrefix = delta(first, firstCode, newSplit, g_num_elements, g_sorted_morton_codes);
            if (splitPrefix > commonPrefix) {
                split = newSplit;// accept proposal
            }
        }
    } while (stride > 1);

    return split;
}

[AutoPyBindCUDA]
[CUDAKernel]
void hierarchy(uint g_num_elements, 
  
               TensorView<int> ele_primitiveIdx,
               TensorView<float> ele_aabb,
               TensorView<int> g_sorted_morton_codes, TensorView<int> bvh_info, TensorView<float> bvh_aabb,
               TensorView<int> bvh_construction_infos)
{
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    uint gID = dispatchIdx.x;
    if (gID >= g_num_elements) return;
    const int LEAF_OFFSET = int(g_num_elements) - 1;

    // construct leaf nodes
    if (gID < g_num_elements) {
        int sortedIdx = g_sorted_morton_codes[gID, 1];
        float3 aabbMin = loadFloat3(ele_aabb, sortedIdx, 0);
        float3 aabbMax = loadFloat3(ele_aabb, sortedIdx, 3);
        uint primitiveIdx = ele_primitiveIdx[sortedIdx, 0];

        
        storeLBVHNode(LEAF_OFFSET + gID, INVALID_POINTER, INVALID_POINTER, 
                     primitiveIdx, aabbMin, aabbMax, bvh_info, bvh_aabb);
    }

    // construct internal nodes
    if (gID < g_num_elements - 1) {
        int first, last;
        determineRange(int(gID), first, last, g_num_elements, g_sorted_morton_codes);
        int split = findSplit(first, last, g_num_elements, g_sorted_morton_codes);

        int childA = (split == first) ? LEAF_OFFSET + split : split;
        int childB = (split + 1 == last) ? LEAF_OFFSET + split + 1 : split + 1;

        float3 aabbMin = float3(1e9);
        float3 aabbMax = float3(-1e9);
        
        storeLBVHNode(gID, childA, childB, -1, aabbMin, aabbMax, bvh_info, bvh_aabb);

        // Store construction info for children
        bvh_construction_infos[childA, 0] = gID;
        bvh_construction_infos[childA, 1] = 0;
        bvh_construction_infos[childB, 0] = gID;
        bvh_construction_infos[childB, 1] = 0;
    }

    // Set root node construction info
    if (gID == 0) {
        bvh_construction_infos[0, 0] = 0;
        bvh_construction_infos[0, 1] = 0;
    }
}