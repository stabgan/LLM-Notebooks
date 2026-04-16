#include <metal_stdlib>
using namespace metal;

// Fused softmax kernel with shared memory for max reduction
// Each threadgroup processes one row of the input matrix
kernel void softmax_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& N [[buffer(2)]],       // number of columns
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint gid [[threadgroup_position_in_grid]]
) {
    // Shared memory for reductions
    threadgroup float shared_max[256];
    threadgroup float shared_sum[256];
    
    uint row = gid;
    uint base = row * N;
    
    // Step 1: Find max for numerical stability (parallel reduction)
    float local_max = -INFINITY;
    for (uint i = tid; i < N; i += tg_size) {
        local_max = max(local_max, input[base + i]);
    }
    shared_max[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduce within threadgroup
    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_max[tid] = max(shared_max[tid], shared_max[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = shared_max[0];
    
    // Step 2: Compute exp(x - max) and sum
    float local_sum = 0.0;
    for (uint i = tid; i < N; i += tg_size) {
        float val = exp(input[base + i] - row_max);
        output[base + i] = val;
        local_sum += val;
    }
    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_sum = shared_sum[0];
    
    // Step 3: Normalize
    for (uint i = tid; i < N; i += tg_size) {
        output[base + i] /= row_sum;
    }
}
