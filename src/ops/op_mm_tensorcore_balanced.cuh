#pragma once

// Balanced TensorCore GEMM - Optimized for Occupancy
// Key insight: Register pressure from 8 warps limits occupancy
// Solution: Use 4 warps with optimized memory access and better pipelining
// This should achieve higher occupancy and better GPU utilization

#include "utils/check_error.cuh"
#include "utils/tensor.cuh"
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// ensure_tc_mm_shape_device is in op_mm_tensorcore.cuh

// Balanced kernel: 4 warps for better occupancy, optimized memory access
template <typename T>
__global__ void op_mm_tensorcore_balanced_kernel(
    const Tensor<__half> A,
    const Tensor<__half> B,
    Tensor<T> C)
{
    const int warpId = threadIdx.y;  // 0-3 (4 warps)
    const int laneId = threadIdx.x;  // 0-31
    
    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;
    
    // 4 warps arranged as 2×2: 2 rows × 2 cols = 32 rows × 32 cols per block
    const int warpRowInBlock = warpId / 2;  // 0-1
    const int warpColInBlock = warpId % 2;  // 0-1
    
    const int m = blockRow * 32 + warpRowInBlock * WMMA_M;
    const int n = blockCol * 32 + warpColInBlock * WMMA_N;
    
    // Double buffered shared memory with padding
    __shared__ __half smem_a[2][4][WMMA_M * WMMA_K + 8];
    __shared__ __half smem_b[2][4][WMMA_K * WMMA_N + 8];
    
    // WMMA fragments - single buffer to reduce register pressure
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> frag_b;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_c;
    
    wmma::fill_fragment(frag_c, 0.0f);
    
    // Precompute addresses to reduce overhead
    const int A_base = A.offset;
    const int A_stride_h = A.stride_h;
    const int A_stride_w = A.stride_w;
    const int B_base = B.offset;
    const int B_stride_h = B.stride_h;
    const int B_stride_w = B.stride_w;
    
    // Load first tile
    int k = 0;
    int buf_idx = 0;
    
    if (k < A.w) {
        // Load A tile - coalesced
        #pragma unroll
        for (int load_idx = laneId; load_idx < WMMA_M * WMMA_K; load_idx += 32) {
            int i = load_idx / WMMA_K;
            int j = load_idx % WMMA_K;
            int row = m + i;
            int col = k + j;
            __half val = __float2half(0.0f);
            if (row < A.h && col < A.w) {
                val = A.rawp[A_base + row * A_stride_h + col * A_stride_w];
            }
            smem_a[buf_idx][warpId][load_idx] = val;
        }
        
        // Load B tile - coalesced (column-major access)
        #pragma unroll
        for (int load_idx = laneId; load_idx < WMMA_K * WMMA_N; load_idx += 32) {
            int j = load_idx % WMMA_N;  // Column varies fastest
            int i = load_idx / WMMA_N;
            int row = k + i;
            int col = n + j;
            __half val = __float2half(0.0f);
            if (row < B.h && col < B.w) {
                val = B.rawp[B_base + row * B_stride_h + col * B_stride_w];
            }
            smem_b[buf_idx][warpId][j * WMMA_K + i] = val;
        }
    }
    
    __syncthreads();
    
    if (k < A.w) {
        wmma::load_matrix_sync(frag_a, smem_a[buf_idx][warpId], WMMA_K);
        wmma::load_matrix_sync(frag_b, smem_b[buf_idx][warpId], WMMA_K);
    }
    
    // Main loop with improved pipelining
    for (k = WMMA_K; k < A.w; k += WMMA_K) {
        int next_buf = 1 - buf_idx;
        
        // Load next tile (overlaps with computation)
        #pragma unroll
        for (int load_idx = laneId; load_idx < WMMA_M * WMMA_K; load_idx += 32) {
            int i = load_idx / WMMA_K;
            int j = load_idx % WMMA_K;
            int row = m + i;
            int col = k + j;
            __half val = __float2half(0.0f);
            if (row < A.h && col < A.w) {
                val = A.rawp[A_base + row * A_stride_h + col * A_stride_w];
            }
            smem_a[next_buf][warpId][load_idx] = val;
        }
        
        #pragma unroll
        for (int load_idx = laneId; load_idx < WMMA_K * WMMA_N; load_idx += 32) {
            int j = load_idx % WMMA_N;
            int i = load_idx / WMMA_N;
            int row = k + i;
            int col = n + j;
            __half val = __float2half(0.0f);
            if (row < B.h && col < B.w) {
                val = B.rawp[B_base + row * B_stride_h + col * B_stride_w];
            }
            smem_b[next_buf][warpId][j * WMMA_K + i] = val;
        }
        
        // Compute with current buffer
        wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
        
        __syncthreads();
        
        // Load fragments from next buffer
        wmma::load_matrix_sync(frag_a, smem_a[next_buf][warpId], WMMA_K);
        wmma::load_matrix_sync(frag_b, smem_b[next_buf][warpId], WMMA_K);
        
        buf_idx = next_buf;
    }
    
    // Final computation
    if (A.w > 0) {
        wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
    }
    
    // Store result
    __shared__ float smem_c[4][WMMA_M * WMMA_N + 8];
    wmma::store_matrix_sync(smem_c[warpId], frag_c, WMMA_N, wmma::mem_row_major);
    __syncthreads();
    
    // Write to global memory - coalesced
    const int C_base = C.offset;
    const int C_stride_h = C.stride_h;
    const int C_stride_w = C.stride_w;
    
    for (int elem = 0; elem < 8; elem++) {
        int elem_idx = laneId + elem * 32;
        if (elem_idx < WMMA_M * WMMA_N) {
            int i = elem_idx / WMMA_N;
            int j = elem_idx % WMMA_N;
            int row = m + i;
            int col = n + j;
            if (row < C.h && col < C.w) {
                C.rawp[C_base + row * C_stride_h + col * C_stride_w] = 
                    static_cast<T>(smem_c[warpId][elem_idx]);
            }
        }
    }
}

template <typename T>
void op_mm_tensorcore_balanced(const Tensor<__half>& A, const Tensor<__half>& B, Tensor<T>& C)
{
    ensure_tc_mm_shape_device(A, B, C);
    
    if (!A.on_device) {
        throw std::runtime_error("TensorCore GEMM requires device tensors");
    }
    
    // Launch config: 4 warps per block (128 threads), 32×32 output per block
    dim3 blockDim(32, 4);
    dim3 gridDim((C.w + 31) / 32, (C.h + 31) / 32);
    
    op_mm_tensorcore_balanced_kernel<<<gridDim, blockDim>>>(A, B, C);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Balanced TensorCore kernel launch failed: " + 
            std::string(cudaGetErrorString(err)));
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("Balanced TensorCore kernel execution failed: " + 
            std::string(cudaGetErrorString(err)));
    }
}

