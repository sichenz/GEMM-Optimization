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

// Shape validation for TensorCore GEMM
template <typename AT, typename BT, typename OT>
inline void ensure_tc_mm_shape_device(const Tensor<AT> &a, const Tensor<BT> &b, const Tensor<OT> &out)
{
    if (a.h != out.h || b.w != out.w || a.w != b.h)
        throw std::runtime_error("a,b,out tensor shape mismatch a:" +
            a.repr() + ", b:" + b.repr() + ", out:" + out.repr());
    if (a.on_device != b.on_device || a.on_device != out.on_device)
        throw std::runtime_error("a,b,out tensor device mismatch a:" +
            a.repr() + ", b:" + b.repr() + ", out:" + out.repr());
}

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
    // Using double buffering for pipelining
    __shared__ __half smem_a[2][8][WMMA_M * WMMA_K + 8];  // 2 buffers, 8 warps
    __shared__ __half smem_b[2][8][WMMA_K * WMMA_N + 8];   // 2 buffers, 8 warps
    
    // Double buffered fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_a[2];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> frag_b[2];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_c;
    
    wmma::fill_fragment(frag_c, 0.0f);
    
    // Load first tile (pipeline stage 0)
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
    
    // Main loop with software pipelining (double buffering)
    for (k = WMMA_K; k < A.w; k += WMMA_K) {
        int next_buf = 1 - buf_idx;
        
        // STAGE 1: Load next tile (k+1) into next buffer
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
        
        // STAGE 2: Compute with current buffer (k)
        wmma::mma_sync(frag_c, frag_a[buf_idx], frag_b[buf_idx], frag_c);
        
        // Synchronize: ensure loads complete before switching buffers
        __syncthreads();
        
        // STAGE 3: Load fragments from next buffer
        wmma::load_matrix_sync(frag_a[next_buf], smem_a[next_buf][warpId], WMMA_K);
        wmma::load_matrix_sync(frag_b[next_buf], smem_b[next_buf][warpId], WMMA_K);
        
        buf_idx = next_buf; // Toggle buffer
    }
    
    // Final computation with last buffered tile
    if (A.w > 0) {
        wmma::mma_sync(frag_c, frag_a[buf_idx], frag_b[buf_idx], frag_c);
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

