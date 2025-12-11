#pragma once

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
#include <stdexcept>  // <-- needed for std::runtime_error

using namespace nvcuda;

// WMMA tile dimensions for TensorCores (Turing/Volta/Hopper compatible)
// Each warp computes a 16x16x16 matrix multiply
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Threadblock tile dimensions
// We conceptually talk about 128x128 tiles in the phase design, but this
// simple kernel actually uses 4 warps per block -> 32x32 output per block.
// (The bigger blocking is handled at the grid level.)
#define BLOCK_TILE_M 128
#define BLOCK_TILE_N 128
#define WARP_TILE_M 16
#define WARP_TILE_N 16

// Helper function to validate matrix dimensions for TensorCore GEMM
template <typename AT, typename BT, typename OT>
static void ensure_tc_mm_shape_device(const Tensor<AT> &a,
                                      const Tensor<BT> &b,
                                      const Tensor<OT> &out)
{
    if (a.h != out.h || b.w != out.w || a.w != b.h) {
        throw std::runtime_error(
            "a,b,out tensor shape mismatch a:" + a.repr() +
            ", b:" + b.repr() + ", out:" + out.repr());
    }

    if (a.on_device != b.on_device || a.on_device != out.on_device) {
        throw std::runtime_error(
            "a,b,out tensor device mismatch a:" + a.repr() +
            ", b:" + b.repr() + ", out:" + out.repr());
    }
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
    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;
    
    // Warp position within block (2x2 layout)
    // Warp 0: (0,0), Warp 1: (0,1), Warp 2: (1,0), Warp 3: (1,1)
    const int warpRowInBlock = warpId / 2;  // 0 or 1
    const int warpColInBlock = warpId % 2;  // 0 or 1
    
    // Global row/col indices for this warp's output tile
    const int m = blockRow * 32 + warpRowInBlock * WMMA_M;
    const int n = blockCol * 32 + warpColInBlock * WMMA_N;
    
    // Shared memory for tiles (one tile per warp)
    __shared__ __half smem_a[4][WMMA_M * WMMA_K + 8];  // +8 for alignment padding
    __shared__ __half smem_b[4][WMMA_K * WMMA_N + 8];
    
    // WMMA fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                   __half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                   __half, wmma::col_major> frag_b;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                   float> frag_c;
    
    // Initialize accumulator to zero
    wmma::fill_fragment(frag_c, 0.0f);
    
    // Loop over K dimension in chunks of 16
    for (int k = 0; k < A.w; k += WMMA_K)
    {
        // Load tile from A into shared memory (row-major)
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
        __syncthreads();
        
        // Load from shared memory into fragment (A)
        wmma::load_matrix_sync(frag_a, smem_a[warpId], WMMA_K);
        
        // Load tile from B into shared memory (col-major for WMMA)
        #pragma unroll
        for (int load_idx = laneId; load_idx < WMMA_K * WMMA_N; load_idx += 32) {
            int j = load_idx / WMMA_K;  // column within tile
            int i = load_idx % WMMA_K;  // row within tile
            int row = k + i;
            int col = n + j;
            if (row < B.h && col < B.w) {
                // store in col-major order j*K + i
                smem_b[warpId][j * WMMA_K + i] = Index(B, row, col);
            } else {
                smem_b[warpId][j * WMMA_K + i] = __float2half(0.0f);
            }
        }
        __syncthreads();
        
        // Load from shared memory into fragment (B, col-major)
        wmma::load_matrix_sync(frag_b, smem_b[warpId], WMMA_K);
        
        // TensorCore MMA: frag_c += frag_a * frag_b
        wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
    }
    
    // Store result to global memory via shared memory
    __shared__ float smem_c[4][WMMA_M * WMMA_N + 8];
    
    wmma::store_matrix_sync(smem_c[warpId], frag_c, WMMA_N, wmma::mem_row_major);
    __syncthreads();
    
    // Each thread writes 8 elements of the 16x16 tile
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

// Main TensorCore GEMM function: compute C = A @ B using TensorCores
// Inputs: A and B are FP16, output C is FP32 (or template T)
template <typename T>
void op_mm_tensorcore(const Tensor<__half>& A,
                      const Tensor<__half>& B,
                      Tensor<T>& C)
{
    // Validate matrix dimensions
    ensure_tc_mm_shape_device(A, B, C);
    
    if (!A.on_device) {
        throw std::runtime_error("TensorCore GEMM requires device tensors");
    }
    
    // Launch configuration: 4 warps (128 threads) per block
    dim3 blockDim(32, 4);  // 32 threads per warp, 4 warps per block
    
    // Each block handles 32x32 output
    dim3 gridDim((C.w + 31) / 32,
                 (C.h + 31) / 32);
    
    // Launch TensorCore kernel
    op_mm_tensorcore_kernel<<<gridDim, blockDim>>>(A, B, C);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            "TensorCore kernel launch failed: " +
            std::string(cudaGetErrorString(err)));
    }
    
    CUDA_OK(cudaDeviceSynchronize());
}
