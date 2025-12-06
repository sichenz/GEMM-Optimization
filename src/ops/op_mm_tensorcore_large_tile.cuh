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
// Uses 8 warps per block (256 threads) for 64×64 output tiles
// 8 warps arranged as 2×4: 2 rows × 4 cols = 32 rows × 64 cols per block
// Each warp computes 16×16 output
template <typename T>
__global__ void op_mm_tensorcore_large_tile_kernel(
    const Tensor<__half> A,
    const Tensor<__half> B,
    Tensor<T> C)
{
    const int warpId = threadIdx.y;  // 0-7
    const int laneId = threadIdx.x;  // 0-31
    
    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;
    
    // 8 warps arranged as 2×4: 2 rows × 4 cols = 32 rows × 64 cols per block
    const int warpRowInBlock = warpId / 4;  // 0 or 1
    const int warpColInBlock = warpId % 4;  // 0, 1, 2, or 3
    
    const int m = blockRow * 32 + warpRowInBlock * WMMA_M;
    const int n = blockCol * 64 + warpColInBlock * WMMA_N;
    
    // Shared memory for 8 warps
    // Using single buffer to avoid exceeding 48KB shared memory limit
    // 8 warps × 1 buffer × (16×16 + 8) × 2 bytes = ~8.4KB per matrix = ~16.8KB total (within limit)
    __shared__ __half smem_a[8][WMMA_M * WMMA_K + 8];  // 1 buffer, 8 warps
    __shared__ __half smem_b[8][WMMA_K * WMMA_N + 8];   // 1 buffer, 8 warps
    
    // Single buffer fragments (no double buffering to save shared memory)
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> frag_b;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_c;
    
    wmma::fill_fragment(frag_c, 0.0f);
    
    // Main loop: Load, compute, repeat (no pipelining to save shared memory)
    for (int k = 0; k < A.w; k += WMMA_K) {
        // Load A tile
        #pragma unroll
        for (int load_idx = laneId; load_idx < WMMA_M * WMMA_K; load_idx += 32) {
            int i = load_idx / WMMA_K;
            int j = load_idx % WMMA_K;
            int row = m + i;
            int col = k + j;
            smem_a[warpId][load_idx] = (row < A.h && col < A.w) ? 
                Index(A, row, col) : __float2half(0.0f);
        }
        
        // Load B tile (transposed to col-major)
        #pragma unroll
        for (int load_idx = laneId; load_idx < WMMA_K * WMMA_N; load_idx += 32) {
            int j = load_idx / WMMA_K;
            int i = load_idx % WMMA_K;
            int row = k + i;
            int col = n + j;
            smem_b[warpId][j * WMMA_K + i] = (row < B.h && col < B.w) ? 
                Index(B, row, col) : __float2half(0.0f);
        }
        
        __syncthreads();
        
        // Load fragments
        wmma::load_matrix_sync(frag_a, smem_a[warpId], WMMA_K);
        wmma::load_matrix_sync(frag_b, smem_b[warpId], WMMA_K);
        
        // Compute
        wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
        
        __syncthreads();
    }
    
    // Store result
    __shared__ float smem_c[8][WMMA_M * WMMA_N + 8];
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
    
    // Launch configuration: 8 warps per block (256 threads)
    // Each block computes 32×64 output (2×4 warps, each 16×16)
    dim3 blockDim(32, 8);  // 32 threads per warp, 8 warps per block
    
    // Grid dimension: each block handles 32 rows × 64 cols
    dim3 gridDim((C.w + 63) / 64,
                 (C.h + 31) / 32);
    
    op_mm_tensorcore_large_tile_kernel<<<gridDim, blockDim>>>(A, B, C);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Large Tile TensorCore kernel launch failed: " + 
            std::string(cudaGetErrorString(err)));
    }
    
    CUDA_OK(cudaDeviceSynchronize());
}

