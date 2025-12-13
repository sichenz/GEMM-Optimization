#pragma once

// Ultra-optimized TensorCore GEMM
// Key insight: Memory bandwidth is only 0.4% of peak - too much synchronization overhead
// Optimizations:
// 1. Reduced __syncthreads() calls - better pipelining
// 2. Improved double buffering - better overlap of compute and memory
// 3. Optimized memory access - reduce Index() macro overhead
// 4. Better instruction scheduling

#include "utils/check_error.cuh"
#include "utils/tensor.cuh"
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// ensure_tc_mm_shape_device is in op_mm_tensorcore.cuh

// Ultra-optimized kernel - minimal synchronization overhead
template <typename T>
__global__ void op_mm_tensorcore_ultra_kernel(
    const Tensor<__half> A,
    const Tensor<__half> B,
    Tensor<T> C)
{
    const int warpId = threadIdx.y;  // 0-7 (8 warps)
    const int laneId = threadIdx.x;  // 0-31
    
    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;
    
    // 8 warps arranged as 4×2: 4 rows × 2 cols = 64 rows × 32 cols per block
    const int warpRowInBlock = warpId / 2;  // 0-3
    const int warpColInBlock = warpId % 2;  // 0 or 1
    
    const int m = blockRow * 64 + warpRowInBlock * WMMA_M;
    const int n = blockCol * 32 + warpColInBlock * WMMA_N;
    
    // Double buffered shared memory with padding
    __shared__ __half smem_a[2][8][WMMA_M * WMMA_K + 8];
    __shared__ __half smem_b[2][8][WMMA_K * WMMA_N + 8];
    
    // WMMA fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_a[2];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> frag_b[2];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_c;
    
    wmma::fill_fragment(frag_c, 0.0f);
    
    // Precompute base addresses to reduce Index() overhead
    const int A_base_offset = A.offset;
    const int A_stride_h = A.stride_h;
    const int A_stride_w = A.stride_w;
    const int B_base_offset = B.offset;
    const int B_stride_h = B.stride_h;
    const int B_stride_w = B.stride_w;
    
    // Load first tile
    int k = 0;
    int buf_idx = 0;
    
    if (k < A.w) {
        // Load A tile - optimized address calculation
        #pragma unroll
        for (int load_idx = laneId; load_idx < WMMA_M * WMMA_K; load_idx += 32) {
            int i = load_idx / WMMA_K;
            int j = load_idx % WMMA_K;
            int row = m + i;
            int col = k + j;
            __half val = __float2half(0.0f);
            if (row < A.h && col < A.w) {
                // Direct address calculation instead of Index() macro
                val = A.rawp[A_base_offset + row * A_stride_h + col * A_stride_w];
            }
            smem_a[buf_idx][warpId][load_idx] = val;
        }
        
        // Load B tile with coalesced access
        #pragma unroll
        for (int load_idx = laneId; load_idx < WMMA_K * WMMA_N; load_idx += 32) {
            int j = load_idx % WMMA_N;
            int i = load_idx / WMMA_N;
            int row = k + i;
            int col = n + j;
            __half val = __float2half(0.0f);
            if (row < B.h && col < B.w) {
                val = B.rawp[B_base_offset + row * B_stride_h + col * B_stride_w];
            }
            smem_b[buf_idx][warpId][j * WMMA_K + i] = val;
        }
    }
    
    __syncthreads();
    
    if (k < A.w) {
        wmma::load_matrix_sync(frag_a[buf_idx], smem_a[buf_idx][warpId], WMMA_K);
        wmma::load_matrix_sync(frag_b[buf_idx], smem_b[buf_idx][warpId], WMMA_K);
    }
    
    // Main loop with improved pipelining - reduce sync overhead
    for (k = WMMA_K; k < A.w; k += WMMA_K) {
        int next_buf = 1 - buf_idx;
        
        // Load next tile (happens in parallel with computation below)
        #pragma unroll
        for (int load_idx = laneId; load_idx < WMMA_M * WMMA_K; load_idx += 32) {
            int i = load_idx / WMMA_K;
            int j = load_idx % WMMA_K;
            int row = m + i;
            int col = k + j;
            __half val = __float2half(0.0f);
            if (row < A.h && col < A.w) {
                val = A.rawp[A_base_offset + row * A_stride_h + col * A_stride_w];
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
                val = B.rawp[B_base_offset + row * B_stride_h + col * B_stride_w];
            }
            smem_b[next_buf][warpId][j * WMMA_K + i] = val;
        }
        
        // Compute with current buffer - this overlaps with loads above
        // Key: computation happens while next tile is being loaded
        wmma::mma_sync(frag_c, frag_a[buf_idx], frag_b[buf_idx], frag_c);
        
        // Only sync once per iteration (reduced from multiple syncs)
        __syncthreads();
        
        // Load fragments from next buffer (ready now)
        wmma::load_matrix_sync(frag_a[next_buf], smem_a[next_buf][warpId], WMMA_K);
        wmma::load_matrix_sync(frag_b[next_buf], smem_b[next_buf][warpId], WMMA_K);
        
        buf_idx = next_buf;
    }
    
    // Final computation
    if (A.w > 0) {
        wmma::mma_sync(frag_c, frag_a[buf_idx], frag_b[buf_idx], frag_c);
    }
    
    // Store result
    __shared__ float smem_c[8][WMMA_M * WMMA_N + 8];
    wmma::store_matrix_sync(smem_c[warpId], frag_c, WMMA_N, wmma::mem_row_major);
    __syncthreads();
    
    // Write to global memory - optimized address calculation
    const int C_base_offset = C.offset;
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
                C.rawp[C_base_offset + row * C_stride_h + col * C_stride_w] = 
                    static_cast<T>(smem_c[warpId][elem_idx]);
            }
        }
    }
}

template <typename T>
void op_mm_tensorcore_ultra(const Tensor<__half>& A, const Tensor<__half>& B, Tensor<T>& C)
{
    ensure_tc_mm_shape_device(A, B, C);
    
    if (!A.on_device) {
        throw std::runtime_error("TensorCore GEMM requires device tensors");
    }
    
    // Launch config: 8 warps per block, 64×32 output per block
    dim3 blockDim(32, 8);
    dim3 gridDim((C.w + 31) / 32, (C.h + 63) / 64);
    
    op_mm_tensorcore_ultra_kernel<<<gridDim, blockDim>>>(A, B, C);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Ultra TensorCore kernel launch failed: " + 
            std::string(cudaGetErrorString(err)));
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("Ultra TensorCore kernel execution failed: " + 
            std::string(cudaGetErrorString(err)));
    }
}
