#pragma once

// Phase 2: Optimized TensorCore GEMM Implementation
// Optimizations:
// 1. Larger blocks: 8 warps (256 threads) for 64x64 output tiles
// 2. Software pipelining: Double buffering to overlap load/compute
// 3. Better memory access patterns

#include "utils/check_error.cuh"
#include "utils/tensor.cuh"
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Optimized kernel with 8 warps per block (64x64 output)
template <typename T>
__global__ void op_mm_tensorcore_optimized_kernel(
    const Tensor<__half> A,
    const Tensor<__half> B,
    Tensor<T> C)
{
    const int warpId = threadIdx.y;  // 0-7
    const int laneId = threadIdx.x;  // 0-31
    
    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;
    
    // 8 warps arranged as 2x4: 2 rows x 4 cols = 32x64 output per block
    // Actually, let's do 4x2: 4 rows x 2 cols = 64x32, then transpose grid
    // Or simpler: 2x4 = 32 rows x 64 cols
    const int warpRowInBlock = warpId / 4;  // 0 or 1
    const int warpColInBlock = warpId % 4;  // 0, 1, 2, or 3
    
    const int m = blockRow * 32 + warpRowInBlock * WMMA_M;
    const int n = blockCol * 64 + warpColInBlock * WMMA_N;
    
    // Double buffered shared memory for pipelining
    __shared__ __half smem_a[2][8][WMMA_M * WMMA_K + 8];
    __shared__ __half smem_b[2][8][WMMA_K * WMMA_N + 8];
    
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_a[2];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> frag_b[2];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_c;
    
    wmma::fill_fragment(frag_c, 0.0f);
    
    // Load first tile (pipeline stage 0)
    int k = 0;
    if (k < A.w) {
        #pragma unroll
        for (int load_idx = laneId; load_idx < WMMA_M * WMMA_K; load_idx += 32) {
            int i = load_idx / WMMA_K;
            int j = load_idx % WMMA_K;
            int row = m + i;
            int col = k + j;
            smem_a[0][warpId][load_idx] = (row < A.h && col < A.w) ? Index(A, row, col) : __float2half(0.0f);
        }
        
        #pragma unroll
        for (int load_idx = laneId; load_idx < WMMA_K * WMMA_N; load_idx += 32) {
            int j = load_idx / WMMA_K;
            int i = load_idx % WMMA_K;
            int row = k + i;
            int col = n + j;
            smem_b[0][warpId][j * WMMA_K + i] = (row < B.h && col < B.w) ? Index(B, row, col) : __float2half(0.0f);
        }
    }
    
    __syncthreads();
    
    if (k < A.w) {
        wmma::load_matrix_sync(frag_a[0], smem_a[0][warpId], WMMA_K);
        wmma::load_matrix_sync(frag_b[0], smem_b[0][warpId], WMMA_K);
    }
    
    // Main loop with pipelining
    for (k = WMMA_K; k < A.w; k += WMMA_K) {
        // Pipeline: Load next tile while computing current
        int next_k = k;
        int buf_idx = (k / WMMA_K) % 2;
        int prev_buf = 1 - buf_idx;
        
        // Load next A and B tiles
        #pragma unroll
        for (int load_idx = laneId; load_idx < WMMA_M * WMMA_K; load_idx += 32) {
            int i = load_idx / WMMA_K;
            int j = load_idx % WMMA_K;
            int row = m + i;
            int col = next_k + j;
            smem_a[buf_idx][warpId][load_idx] = (row < A.h && col < A.w) ? Index(A, row, col) : __float2half(0.0f);
        }
        
        #pragma unroll
        for (int load_idx = laneId; load_idx < WMMA_K * WMMA_N; load_idx += 32) {
            int j = load_idx / WMMA_K;
            int i = load_idx % WMMA_K;
            int row = next_k + i;
            int col = n + j;
            smem_b[buf_idx][warpId][j * WMMA_K + i] = (row < B.h && col < B.w) ? Index(B, row, col) : __float2half(0.0f);
        }
        
        // Compute with previous tiles
        wmma::mma_sync(frag_c, frag_a[prev_buf], frag_b[prev_buf], frag_c);
        
        __syncthreads();
        
        // Load next fragments
        wmma::load_matrix_sync(frag_a[buf_idx], smem_a[buf_idx][warpId], WMMA_K);
        wmma::load_matrix_sync(frag_b[buf_idx], smem_b[buf_idx][warpId], WMMA_K);
    }
    
    // Final computation
    if (A.w > 0) {
        int final_buf = ((A.w - WMMA_K) / WMMA_K) % 2;
        wmma::mma_sync(frag_c, frag_a[final_buf], frag_b[final_buf], frag_c);
    }
    
    // Store result
    __shared__ float smem_c[8][WMMA_M * WMMA_N + 8];
    wmma::store_matrix_sync(smem_c[warpId], frag_c, WMMA_N, wmma::mem_row_major);
    __syncthreads();
    
    for (int elem = 0; elem < 8; elem++) {
        int elem_idx = laneId + elem * 32;
        if (elem_idx < WMMA_M * WMMA_N) {
            int i = elem_idx / WMMA_N;
            int j = elem_idx % WMMA_N;
            int row = m + i;
            int col = n + j;
            if (row < C.h && col < C.w) {
                Index(C, row, col) = static_cast<T>(smem_c[warpId][elem_idx]);
            }
        }
    }
}

// For now, keep the original as default since optimized needs testing
// We'll use the working version and document optimization opportunities

