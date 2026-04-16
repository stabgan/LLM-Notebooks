#include <metal_stdlib>
using namespace metal;

// RMS Normalization kernel
// output = x / sqrt(mean(x^2) + eps) * weight
kernel void rmsnorm_kernel(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& D [[buffer(3)]],        // dimension
    constant float& eps [[buffer(4)]],     // epsilon
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint gid [[threadgroup_position_in_grid]]
) {
    threadgroup float shared_sum[256];
    
    uint row = gid;
    uint base = row * D;
    
    // Compute sum of squares
    float local_sum = 0.0;
    for (uint i = tid; i < D; i += tg_size) {
        float val = input[base + i];
        local_sum += val * val;
    }
    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduce
    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    float rms = rsqrt(shared_sum[0] / float(D) + eps);
    
    // Normalize and scale
    for (uint i = tid; i < D; i += tg_size) {
        output[base + i] = input[base + i] * rms * weight[i];
    }
}
