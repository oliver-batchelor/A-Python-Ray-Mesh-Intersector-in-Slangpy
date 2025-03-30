#define WORKGROUP_SIZE 256// assert WORKGROUP_SIZE >= RADIX_SORT_BINS
#define RADIX_SORT_BINS 256
#define SUBGROUP_SIZE 32// 32 NVIDIA; 64 AMD

#define BITS 32// sorting uint32_t
#define ITERATIONS 4// 4 iterations, sorting 8 bits per iteration

groupshared uint[RADIX_SORT_BINS] histogram;
groupshared uint[RADIX_SORT_BINS / SUBGROUP_SIZE] sums; // subgroup reductions
groupshared uint[RADIX_SORT_BINS] local_offsets;        // local exclusive scan (prefix sum) (inside subgroups)
groupshared uint[RADIX_SORT_BINS] global_offsets;       // global exclusive scan (prefix sum)

struct BinFlags {
    uint flags[WORKGROUP_SIZE / BITS];
};
groupshared BinFlags[RADIX_SORT_BINS] bin_flags;

// only used on the GPU side during construction; it is necessary to allocate the (empty) buffer
struct KeyValue {
    uint key; // key for sorting
    uint value; // pointer into element buffer

    static KeyValue load(TensorView<int> g_elements_in, int idx) {
      return KeyValue(
          uint(g_elements_in[idx, 0]),
          uint(g_elements_in[idx, 1]));
    }

    void store(TensorView<int> g_elements_out, int idx) {
      g_elements_out[idx, 0] = uint(key);
      g_elements_out[idx, 1] = uint(value);
    }
};


[AutoPyBindCUDA]
[CUDAKernel]
void radix_sort(int num_elements, TensorView<int> g_elements_in, TensorView<int> g_elements_out)
{
    uint lID = cudaThreadIdx().x;
    uint sID = cudaThreadIdx().x / WaveGetLaneCount();
    uint lsID = WaveGetLaneIndex();

    const WaveMask mask = WaveGetConvergedMask();

    for (uint iteration = 0; iteration < ITERATIONS; iteration++) {
        uint shift = 8 * iteration;

        // initialize histogram
        if (lID < RADIX_SORT_BINS) {
            histogram[lID] = 0U;
        }
        GroupMemoryBarrierWithGroupSync();

        for (uint ID = lID; ID < num_elements; ID += WORKGROUP_SIZE) {
            // get current element based on iteration parity
            const uint element = iteration % 2 == 0 ? g_elements_in[ID,0] : g_elements_out[ID,0];
            // determine the bin for current element
            const uint bin = (element >> shift) & (RADIX_SORT_BINS - 1);
            // increment the histogram
            InterlockedAdd(histogram[bin], 1U);
        }

        GroupMemoryBarrierWithGroupSync();

        // subgroup reductions and subgroup prefix sums
        if (lID < RADIX_SORT_BINS) {
            uint histogram_count = histogram[lID];
            uint sum = WaveMaskSum(mask, histogram_count);
            uint prefix_sum = WaveMaskPrefixSum(mask, histogram_count);
            local_offsets[lID] = prefix_sum;
            if (WaveMaskIsFirstLane(mask)) {
                // one thread inside the warp/subgroup enters this section
                sums[sID] = sum;
            }
        }
        GroupMemoryBarrierWithGroupSync();

        // global prefix sums (offsets)
        if (sID == 0) {
            uint offset = 0;
            for (uint i = lsID; i < RADIX_SORT_BINS; i += SUBGROUP_SIZE) {
                global_offsets[i] = offset + local_offsets[i];
                offset += sums[i / SUBGROUP_SIZE];
            }
        }
        GroupMemoryBarrierWithGroupSync();

        //     ==== scatter keys according to global offsets =====
        const uint flags_bin = lID / BITS;
        const uint flags_bit = 1 << (lID % BITS);

        for (uint blockID = 0; blockID < num_elements; blockID += WORKGROUP_SIZE) {
            GroupMemoryBarrierWithGroupSync();

            const uint ID = blockID + lID;

            // initialize bin flags
            if (lID < RADIX_SORT_BINS) {
                for (int i = 0; i < WORKGROUP_SIZE / BITS; i++) {
                    bin_flags[lID].flags[i] = 0U; // init all bin flags to 0
                }
            }
            GroupMemoryBarrierWithGroupSync();

            KeyValue element_in;
            uint binID = 0;
            uint binOffset = 0;
            if (ID < num_elements) {
                if (iteration % 2 == 0) {
                    element_in = KeyValue.load(g_elements_in, ID);
                } else {
                    element_in = KeyValue.load(g_elements_out, ID);
                }
                binID = (element_in.key >> shift) & uint(RADIX_SORT_BINS - 1);
                // offset for group
                binOffset = global_offsets[binID];
                // add bit to flag
                InterlockedAdd(bin_flags[binID].flags[flags_bin], flags_bit);
            }
            GroupMemoryBarrierWithGroupSync();

            if (ID < num_elements) {
                // calculate output index of element
                uint prefix = 0;
                uint count = 0;
                for (uint i = 0; i < WORKGROUP_SIZE / BITS; i++) {
                    const uint bits = bin_flags[binID].flags[i];
                    const uint full_count = countbits(bits);
                    const uint partial_count = countbits(bits & (flags_bit - 1));
                    prefix += (i < flags_bin) ? full_count : 0U;
                    prefix += (i == flags_bin) ? partial_count : 0U;
                    count += full_count;
                }
                if (iteration % 2 == 0) {
                    element_in.store(g_elements_out, binOffset + prefix);
                } else {
                    element_in.store(g_elements_in, binOffset + prefix);
                }

                
                if (prefix == count - 1) {
                    InterlockedAdd(global_offsets[binID], count);
                }
            }
        }
    }
}