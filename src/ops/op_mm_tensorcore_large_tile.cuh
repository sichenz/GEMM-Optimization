#pragma once

// Phase 2 Optimization: Large Tile TensorCore GEMM
// Based on Phase 3 CUTLASS analysis - increasing tile sizes for better performance
// Optimizations:
// 1. 64×64 output tiles per block (vs 32×32) - 4× more work per block
// 2. 8 warps per block (256 threads) - better GPU utilization
// 3. Each warp computes 16×16, arranged as 4×4 = 64×64 per block
// 4. Expected improvement: 2-3× performance boost

#include "utils/check_error.cuh"
#include "utils/tensor.cuh"
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Note: ensure_tc_mm_shape_device is defined in op_mm_tensorcore.cuh
// Include it if needed, or rely on it being included before this file

// Large Tile TensorCore GEMM Kernel
// Uses 4 warps per block (128 threads) for 64×64 output tiles
// 4 warps arranged as 1×4: 1 row × 4 cols = 16 rows × 64 cols per block
// Each warp computes 16×16 output
// This allows double buffering while staying within shared memory limits
template <typename T>
__global__ void op_mm_tensorcore_large_tile_kernel(
    const Tensor<__half> A,
    const Tensor<__half> B,
    Tensor<T> C)
{
    const int warpId = threadIdx.y;  // 0-3
    const int laneId = threadIdx.x;  // 0-31
    
    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;
    
    // 4 warps arranged as 1×4: 1 row × 4 cols = 16 rows × 64 cols per block
    const int warpRowInBlock = 0;  // All warps in same row
    const int warpColInBlock = warpId;  // 0, 1, 2, or 3
    
    const int m = blockRow * 16 + warpRowInBlock * WMMA_M;
    const int n = blockCol * 64 + warpColInBlock * WMMA_N;
    
    // Shared memory for 4 warps with double buffering
    // 4 warps × 2 buffers × (16×16 + 8) × 2 bytes = ~8.4KB per matrix = ~16.8KB total (within 48KB limit)
    __shared__ __half smem_a[2][4][WMMA_M * WMMA_K + 8];  // 2 buffers, 4 warps
    __shared__ __half smem_b[2][4][WMMA_K * WMMA_N + 8];   // 2 buffers, 4 warps
    
    // Double buffered fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_a[2];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> frag_b[2];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_c;
    
    wmma::fill_fragment(frag_c, 0.0f);
    
    // Load first tile
    int k = 0;
    int buf_idx = 0;
    
    if (k < A.w) {
        // Load A tile
        #pragma unroll
        for (int load_idx = laneId; load_idx < WMMA_M * WMMA_K; load_idx += 32) {
            int i = load_idx / WMMA_K;
            int j = load_idx % WMMA_K;
            int row = m + i;
            int col = k + j;
            smem_a[buf_idx][warpId][load_idx] = (row < A.h && col < A.w) ? 
                Index(A, row, col) : __float2half(0.0f);
        }
        
        // Load B tile (transposed to col-major)
        #pragma unroll
        for (int load_idx = laneId; load_idx < WMMA_K * WMMA_N; load_idx += 32) {
            int j = load_idx / WMMA_K;
            int i = load_idx % WMMA_K;
            int row = k + i;
            int col = n + j;
            smem_b[buf_idx][warpId][j * WMMA_K + i] = (row < B.h && col < B.w) ? 
                Index(B, row, col) : __float2half(0.0f);
        }
    }
    
    __syncthreads();
    
    // Load first fragments
    if (k < A.w) {
        wmma::load_matrix_sync(frag_a[buf_idx], smem_a[buf_idx][warpId], WMMA_K);
        wmma::load_matrix_sync(frag_b[buf_idx], smem_b[buf_idx][warpId], WMMA_K);
    }
    
    // Main loop with double buffering
    for (k = WMMA_K; k < A.w; k += WMMA_K) {
        int next_buf = 1 - buf_idx;
        
        // Load next tile
        #pragma unroll
        for (int load_idx = laneId; load_idx < WMMA_M * WMMA_K; load_idx += 32) {
            int i = load_idx / WMMA_K;
            int j = load_idx % WMMA_K;
            int row = m + i;
            int col = k + j;
            smem_a[next_buf][warpId][load_idx] = (row < A.h && col < A.w) ? 
                Index(A, row, col) : __float2half(0.0f);
        }
        
        #pragma unroll
        for (int load_idx = laneId; load_idx < WMMA_K * WMMA_N; load_idx += 32) {
            int j = load_idx / WMMA_K;
            int i = load_idx % WMMA_K;
            int row = k + i;
            int col = n + j;
            smem_b[next_buf][warpId][j * WMMA_K + i] = (row < B.h && col < B.w) ? 
                Index(B, row, col) : __float2half(0.0f);
        }
        
        // Compute with current buffer
        wmma::mma_sync(frag_c, frag_a[buf_idx], frag_b[buf_idx], frag_c);
        
        __syncthreads();
        
        // Load fragments from next buffer
        wmma::load_matrix_sync(frag_a[next_buf], smem_a[next_buf][warpId], WMMA_K);
        wmma::load_matrix_sync(frag_b[next_buf], smem_b[next_buf][warpId], WMMA_K);
        
        buf_idx = next_buf;
    }
    
    // Final computation
    if (A.w > 0) {
        wmma::mma_sync(frag_c, frag_a[buf_idx], frag_b[buf_idx], frag_c);
    }
    
    // Store result
    __shared__ float smem_c[4][WMMA_M * WMMA_N + 8];
    wmma::store_matrix_sync(smem_c[warpId], frag_c, WMMA_N, wmma::mem_row_major);
    __syncthreads();
    
    // Write to global memory (coalesced)
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

// Large Tile TensorCore GEMM function
template <typename T>
void op_mm_tensorcore_large_tile(const Tensor<__half>& A, const Tensor<__half>& B, Tensor<T>& C)
{
    ensure_tc_mm_shape_device(A, B, C);
    
    if (!A.on_device) {
        throw std::runtime_error("TensorCore GEMM requires device tensors");
    }
    
    // Launch configuration: 4 warps per block (128 threads)
    // Each block computes 16×64 output (1×4 warps, each 16×16)
    dim3 blockDim(32, 4);  // 32 threads per warp, 4 warps per block
    
    // Grid dimension: each block handles 16 rows × 64 cols
    dim3 gridDim((C.w + 63) / 64,
                 (C.h + 15) / 16);
    
    op_mm_tensorcore_large_tile_kernel<<<gridDim, blockDim>>>(A, B, C);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Large Tile TensorCore kernel launch failed: " + 
            std::string(cudaGetErrorString(err)));
    }
    
    CUDA_OK(cudaDeviceSynchronize());
}

