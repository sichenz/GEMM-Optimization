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
static void ensure_tc_mm_shape_device(const Tensor<AT> &a, const Tensor<BT> &b, const Tensor<OT> &out)
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
// - Each block computes 64x64 output (4 warps * 16 = 64)
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
    
    // Warp position within block (2x2 arrangement)
    const int warpRowInBlock = warpId / 2;  // 0 or 1
    const int warpColInBlock = warpId % 2;  // 0 or 1
    
    // Global row/col indices for this warp's output tile
    const int m = blockRow * 64 + warpRowInBlock * WMMA_M;
    const int n = blockCol * 64 + warpColInBlock * WMMA_N;
    
    // Bounds check - skip if out of bounds
    if (m >= C.h || n >= C.w) {
        return;
    }
    
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
        // Load tile from A into fragment (row-major)
        // A[m:m+16, k:k+16] - 16 rows, 16 cols
        // Use temporary array - WMMA will handle it
        __half temp_a[WMMA_M * WMMA_K];
        #pragma unroll
        for (int i = 0; i < WMMA_M; i++) {
            #pragma unroll
            for (int j = 0; j < WMMA_K; j++) {
                int row = m + i;
                int col = k + j;
                if (row < A.h && col < A.w) {
                    temp_a[i * WMMA_K + j] = Index(A, row, col);
                } else {
                    temp_a[i * WMMA_K + j] = __float2half(0.0f);
                }
            }
        }
        // Note: WMMA warnings about local memory are expected but code should still work
        wmma::load_matrix_sync(frag_a, temp_a, WMMA_K);
        
        // Load tile from B into fragment (col-major for WMMA)
        // B[k:k+16, n:n+16] - need to transpose to col-major
        __half temp_b[WMMA_K * WMMA_N];
        #pragma unroll
        for (int i = 0; i < WMMA_K; i++) {
            #pragma unroll
            for (int j = 0; j < WMMA_N; j++) {
                int row = k + i;
                int col = n + j;
                // Transpose: col-major means j varies fastest
                if (row < B.h && col < B.w) {
                    temp_b[j * WMMA_K + i] = Index(B, row, col);
                } else {
                    temp_b[j * WMMA_K + i] = __float2half(0.0f);
                }
            }
        }
        wmma::load_matrix_sync(frag_b, temp_b, WMMA_K);
        
        // Perform matrix multiply-accumulate using TensorCores
        // frag_c += frag_a * frag_b
        wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
    }
    
    // Store result to global memory (FP32 fragment to output)
    if (m < C.h && n < C.w) {
        float temp_c[WMMA_M * WMMA_N];
        wmma::store_matrix_sync(temp_c, frag_c, WMMA_N, wmma::mem_row_major);
        
        // Write to output with bounds checking
        int rows = std::min(WMMA_M, C.h - m);
        int cols = std::min(WMMA_N, C.w - n);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                Index(C, m + i, n + j) = static_cast<T>(temp_c[i * WMMA_N + j]);
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
    // Each block computes 64x64 output (4 warps arranged as 2x2, each warp does 16x16)
    dim3 blockDim(32, 4);  // 32 threads per warp, 4 warps per block
    
    // Grid dimension: number of threadblocks needed
    // Each block handles 64x64 output
    dim3 gridDim((C.w + 63) / 64,
                 (C.h + 63) / 64);
    
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

