#include <metal_stdlib>
using namespace metal;

// Tiled matrix multiplication using threadgroup shared memory
// C = A @ B where A is (M, K) and B is (K, N)
constant uint TILE_SIZE = 16;

kernel void matmul_tiled_kernel(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[threadgroup_position_in_grid]]
) {
    // Shared memory tiles
    threadgroup float tileA[TILE_SIZE][TILE_SIZE];
    threadgroup float tileB[TILE_SIZE][TILE_SIZE];
    
    uint row = gid.y * TILE_SIZE + tid.y;
    uint col = gid.x * TILE_SIZE + tid.x;
    
    float sum = 0.0;
    
    // Iterate over tiles along K dimension
    for (uint t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tile of A into shared memory
        uint a_col = t * TILE_SIZE + tid.x;
        if (row < M && a_col < K) {
            tileA[tid.y][tid.x] = A[row * K + a_col];
        } else {
            tileA[tid.y][tid.x] = 0.0;
        }
        
        // Load tile of B into shared memory
        uint b_row = t * TILE_SIZE + tid.y;
        if (b_row < K && col < N) {
            tileB[tid.y][tid.x] = B[b_row * N + col];
        } else {
            tileB[tid.y][tid.x] = 0.0;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute partial dot product from this tile
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += tileA[tid.y][k] * tileB[k][tid.x];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
