#pragma once

// Phase 2: TensorCore GEMM Implementation
// This implements matrix multiplication using NVIDIA TensorCores via WMMA API
// TensorCores provide 5-7x speedup over regular FP32 GEMM for large matrices
//
// Key concepts:
// - WMMA (Warp Matrix Multiply Accumulate) API for TensorCore operations
// - FP16 input matrices, FP32 accumulation (mixed precision)
// - Warp-level operations: each warp computes a 16x16x16 matrix multiply
// - Multiple warps per threadblock for better utilization

#include "utils/check_error.cuh"
#include "utils/tensor.cuh"
#include <cuda_fp16.h>
#include <mma.h>
#include <algorithm>

using namespace nvcuda;

// WMMA tile dimensions for TensorCores (Turing/Volta)
// Each warp computes a 16x16x16 matrix multiply
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Threadblock tile dimensions
// We'll use 128x128 output tiles with 8 warps (2x4 warps)
// Each warp handles 16x16, so 2 warps = 32 rows, 4 warps = 64 cols
#define BLOCK_TILE_M 128  // 8 warps * 16 = 128
#define BLOCK_TILE_N 128  // 8 warps * 16 = 128
#define WARP_TILE_M 16
#define WARP_TILE_N 16

// Helper function to validate matrix dimensions for TensorCore GEMM
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

// TensorCore GEMM Kernel
// Algorithm: Warp-level matrix multiply using TensorCores
//
// How it works:
// 1. Each warp loads a 16x16 tile from A and B (FP16)
// 2. Uses wmma::load_matrix_sync to load into fragment registers
// 3. Uses wmma::mma_sync to compute 16x16x16 matrix multiply
// 4. Accumulates results in FP32 fragments
// 5. Stores results back to global memory (FP32)
//
// Thread organization:
// - 32 threads per warp (threadIdx.x = 0-31)
// - 4 warps per threadblock (threadIdx.y = 0-3, 128 threads total)
// - Each warp computes one 16x16 output tile
// - Each block computes 32x32 output (2x2 warps, each 16x16)
template <typename T>
__global__ void op_mm_tensorcore_kernel(
    const Tensor<__half> A,      // FP16 input matrix A (row-major)
    const Tensor<__half> B,      // FP16 input matrix B (row-major, will transpose in load)
    Tensor<T> C)                 // FP32 output matrix C (row-major)
{
    // Warp and lane indices
    const int warpId = threadIdx.y;  // Warp ID within block (0-3)
    const int laneId = threadIdx.x;  // Thread ID within warp (0-31)
    
    // Calculate which 16x16 tile this warp is responsible for
    // Each block handles 64x64 output (4 warps arranged as 2x2)
    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;
    
    // Warp position within block
    // Arrange 4 warps in 2x2 to cover 32x32, then use 2 blocks to cover 64x64
    // Actually, simpler: use 8 warps per block (256 threads) to cover 64x64
    // But we only have 4 warps, so let's use 2x2 arrangement covering 32x32 per block
    // Warp 0: (0,0) -> rows 0-15, cols 0-15
    // Warp 1: (0,1) -> rows 0-15, cols 16-31
    // Warp 2: (1,0) -> rows 16-31, cols 0-15
    // Warp 3: (1,1) -> rows 16-31, cols 16-31
    const int warpRowInBlock = warpId / 2;  // 0 or 1
    const int warpColInBlock = warpId % 2;  // 0 or 1
    
    // Global row/col indices for this warp's output tile
    // Each block handles 32 rows x 32 cols (2x2 warps, each 16x16)
    const int m = blockRow * 32 + warpRowInBlock * WMMA_M;
    const int n = blockCol * 32 + warpColInBlock * WMMA_N;
    
    // Note: We don't early return here because we need all warps to participate
    // in shared memory operations. We'll check bounds when writing output.
    
    // Shared memory for tiles (WMMA requires shared or global memory, not local)
    // Each warp needs its own tile space
    __shared__ __half smem_a[4][WMMA_M * WMMA_K + 8];  // 4 warps, +8 for alignment
    __shared__ __half smem_b[4][WMMA_K * WMMA_N + 8];  // 4 warps, +8 for alignment
    
    // Declare WMMA fragments
    // Fragment A: 16x16 tile from matrix A (FP16, row-major)
    // Fragment B: 16x16 tile from matrix B (FP16, col-major for WMMA)
    // Fragment C: 16x16 accumulator (FP32)
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> frag_b;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_c;
    
    // Initialize accumulator to zero
    wmma::fill_fragment(frag_c, 0.0f);
    
    // Loop over K dimension in chunks of 16
    // For C[m:m+16, n:n+16] = sum over k: A[m:m+16, k:k+16] * B[k:k+16, n:n+16]
    for (int k = 0; k < A.w; k += WMMA_K)
    {
        // Load tile from A into shared memory (row-major)
        // A[m:m+16, k:k+16] - 16 rows, 16 cols
        // Each thread loads multiple elements
        #pragma unroll
        for (int load_idx = laneId; load_idx < WMMA_M * WMMA_K; load_idx += 32) {
            int i = load_idx / WMMA_K;
            int j = load_idx % WMMA_K;
            int row = m + i;
            int col = k + j;
            if (row < A.h && col < A.w) {
                smem_a[warpId][load_idx] = Index(A, row, col);
            } else {
                smem_a[warpId][load_idx] = __float2half(0.0f);
            }
        }
        __syncthreads();  // Ensure all warps have loaded
        
        // Load from shared memory into fragment
        wmma::load_matrix_sync(frag_a, smem_a[warpId], WMMA_K);
        
        // Load tile from B into shared memory (col-major for WMMA)
        // B[k:k+16, n:n+16] - need to transpose to col-major for WMMA
        // WMMA col-major layout: B_col[j*K + i] = B_row[i*N + j]
        #pragma unroll
        for (int load_idx = laneId; load_idx < WMMA_K * WMMA_N; load_idx += 32) {
            // For col-major: j varies fastest, so j = load_idx / K, i = load_idx % K
            int j = load_idx / WMMA_K;  // Column index in output (0-15)
            int i = load_idx % WMMA_K;  // Row index in B (0-15)
            int row = k + i;  // Global row in B
            int col = n + j;  // Global column in B
            // Store in col-major order: smem_b[j*K + i] = B[row][col]
            if (row < B.h && col < B.w) {
                smem_b[warpId][j * WMMA_K + i] = Index(B, row, col);
            } else {
                smem_b[warpId][j * WMMA_K + i] = __float2half(0.0f);
            }
        }
        __syncthreads();  // Ensure all warps have loaded
        
        // Load from shared memory into fragment (col-major)
        wmma::load_matrix_sync(frag_b, smem_b[warpId], WMMA_K);
        
        // Perform matrix multiply-accumulate using TensorCores
        // frag_c += frag_a * frag_b
        wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
    }
    
    // Store result to global memory (FP32 fragment to output)
    // Use shared memory for storing (WMMA doesn't like local memory)
    __shared__ float smem_c[4][WMMA_M * WMMA_N + 8];  // 4 warps
    
    // Store fragment to shared memory first (all warps do this)
    wmma::store_matrix_sync(smem_c[warpId], frag_c, WMMA_N, wmma::mem_row_major);
    __syncthreads();
    
    // Write from shared memory to global memory
    // Each thread in warp writes 8 elements (256 elements / 32 threads = 8 per thread)
    // Thread 0 writes: 0, 32, 64, 96, 128, 160, 192, 224
    // Thread 1 writes: 1, 33, 65, 97, 129, 161, 193, 225
    // ... Thread 31 writes: 31, 63, 95, 127, 159, 191, 223, 255
    for (int elem = 0; elem < 8; elem++) {
        int elem_idx = laneId + elem * 32;  // Each thread handles 8 elements
        if (elem_idx < WMMA_M * WMMA_N) {
            int i = elem_idx / WMMA_N;  // Row within tile (0-15)
            int j = elem_idx % WMMA_N;  // Column within tile (0-15)
            int row = m + i;
            int col = n + j;
            // Only write if within bounds
            if (row < C.h && col < C.w) {
                Index(C, row, col) = static_cast<T>(smem_c[warpId][elem_idx]);
            }
        }
    }
}

// Main TensorCore GEMM function: compute C = A @ B using TensorCores
// Inputs: A and B are FP16, output C is FP32
template <typename T>
void op_mm_tensorcore(const Tensor<__half>& A, const Tensor<__half>& B, Tensor<T>& C)
{
    // Validate matrix dimensions
    ensure_tc_mm_shape_device(A, B, C);
    
    if (!A.on_device) {
        throw std::runtime_error("TensorCore GEMM requires device tensors");
    }
    
    // Launch configuration
    // Each threadblock has 4 warps (128 threads)
    // Block dimension: 32 threads (warp size) x 4 warps
    // Each block computes 64x64 output (4 warps in 1 row covering 64 cols, but we need rows too)
    // Actually, let's use 2 blocks: one for rows 0-63, one for rows 64-127
    // Or better: use 2x2 arrangement but fix the indexing
    dim3 blockDim(32, 4);  // 32 threads per warp, 4 warps per block
    
    // Grid dimension: number of threadblocks needed
    // Each block handles 32x32 output (2x2 warps, each 16x16)
    dim3 gridDim((C.w + 31) / 32,
                 (C.h + 31) / 32);
    
    // Launch TensorCore kernel
    op_mm_tensorcore_kernel<<<gridDim, blockDim>>>(A, B, C);
    
    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("TensorCore kernel launch failed: " + 
            std::string(cudaGetErrorString(err)));
    }
    
    // Synchronize and check for runtime errors
    CUDA_OK(cudaDeviceSynchronize());
}

