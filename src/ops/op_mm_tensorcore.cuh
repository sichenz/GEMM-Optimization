#pragma once

// TensorCore GEMM implementation using WMMA API
// TensorCores are way faster than regular FP32 cores (5-7x speedup)
// This uses FP16 inputs with FP32 accumulation (mixed precision)

#include "utils/check_error.cuh"
#include "utils/tensor.cuh"
#include <cuda_fp16.h>
#include <mma.h>
#include <algorithm>

using namespace nvcuda;

// WMMA tile size - each warp does 16x16x16
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Check that matrix dimensions match for GEMM
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

// TensorCore GEMM kernel
// Each warp computes a 16x16 output tile using TensorCores
// 4 warps per block = 32x32 output per block
template <typename T>
__global__ void op_mm_tensorcore_kernel(
    const Tensor<__half> A,      // FP16 input matrix A (row-major)
    const Tensor<__half> B,      // FP16 input matrix B (row-major, will transpose in load)
    Tensor<T> C)                 // FP32 output matrix C (row-major)
{
    const int warpId = threadIdx.y;  // 0-3
    const int laneId = threadIdx.x;  // 0-31
    
    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;
    
    // 4 warps arranged as 2x2: each warp does 16x16, so block does 32x32
    const int warpRowInBlock = warpId / 2;  // 0 or 1
    const int warpColInBlock = warpId % 2;  // 0 or 1
    
    const int m = blockRow * 32 + warpRowInBlock * WMMA_M;
    const int n = blockCol * 32 + warpColInBlock * WMMA_N;
    
    // Shared memory for tiles (WMMA needs shared memory, not local)
    __shared__ __half smem_a[4][WMMA_M * WMMA_K + 8];
    __shared__ __half smem_b[4][WMMA_K * WMMA_N + 8];
    
    // WMMA fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> frag_b;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_c;
    
    wmma::fill_fragment(frag_c, 0.0f);
    
    // Loop over K dimension
    for (int k = 0; k < A.w; k += WMMA_K)
    {
        // Load A tile into shared memory
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
        
        wmma::load_matrix_sync(frag_a, smem_a[warpId], WMMA_K);
        
        // Load B tile (transpose to col-major for WMMA)
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
        __syncthreads();
        
        wmma::load_matrix_sync(frag_b, smem_b[warpId], WMMA_K);
        
        // Do the matrix multiply on TensorCores
        wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
    }
    
    // Store result to global memory
    __shared__ float smem_c[4][WMMA_M * WMMA_N + 8];
    
    wmma::store_matrix_sync(smem_c[warpId], frag_c, WMMA_N, wmma::mem_row_major);
    __syncthreads();
    
    // Write to global memory (each thread writes 8 elements)
    for (int elem = 0; elem < 8; elem++) {
        int elem_idx = laneId + elem * 32;  // Each thread handles 8 elements
        if (elem_idx < WMMA_M * WMMA_N) {
            int i = elem_idx / WMMA_N;  // Row within tile (0-15)
            int j = elem_idx % WMMA_N;  // Column within tile (0-15)
            int row = m + i;
            int col = n + j;
            if (row < C.h && col < C.w) {
                Index(C, row, col) = static_cast<T>(smem_c[warpId][elem_idx]);
            }
        }
    }
}

// Main function: C = A @ B using TensorCores
template <typename T>
void op_mm_tensorcore(const Tensor<__half>& A, const Tensor<__half>& B, Tensor<T>& C)
{
    ensure_tc_mm_shape_device(A, B, C);
    
    if (!A.on_device) {
        throw std::runtime_error("TensorCore GEMM requires device tensors");
    }
    
    // Launch config: 4 warps per block, each block does 32x32 output
    dim3 blockDim(32, 4);
    dim3 gridDim((C.w + 31) / 32, (C.h + 31) / 32);
    
    op_mm_tensorcore_kernel<<<gridDim, blockDim>>>(A, B, C);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("TensorCore kernel launch failed: " + 
            std::string(cudaGetErrorString(err)));
    }
    
    CUDA_OK(cudaDeviceSynchronize());
}

