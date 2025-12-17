#pragma once

// Optimized TensorCore GEMM with double buffering
// Overlaps loading next tile with computing current tile to improve performance

#include "utils/check_error.cuh"
#include "utils/tensor.cuh"
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// ensure_tc_mm_shape_device is in op_mm_tensorcore.cuh

// Double buffered TensorCore kernel
// 4 warps per block, 32x32 output per block
template <typename T>
__global__ void op_mm_tensorcore_optimized_kernel(
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
    
    // Double buffered shared memory with padding to avoid bank conflicts
    __shared__ __half smem_a[2][8][WMMA_M * WMMA_K + 8];
    __shared__ __half smem_b[2][8][WMMA_K * WMMA_N + 8];
    
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
        // Improved: load in coalesced pattern by having threads access consecutive columns
        #pragma unroll
        for (int load_idx = laneId; load_idx < WMMA_K * WMMA_N; load_idx += 32) {
            // Reorder: access B in column-major order for coalescing
            // Threads access consecutive columns: B[k+0, n+0], B[k+0, n+1], ..., B[k+0, n+15], B[k+1, n+0], ...
            int j = load_idx % WMMA_N;  // Column varies fastest for coalescing
            int i = load_idx / WMMA_N;  // Row varies slower
            int row = k + i;
            int col = n + j;
            // Store in col-major order in shared memory: smem_b[j*K + i]
            smem_b[buf_idx][warpId][j * WMMA_K + i] = (row < B.h && col < B.w) ? 
                Index(B, row, col) : __float2half(0.0f);
        }
    }
    
    __syncthreads();
    
    if (k < A.w) {
        wmma::load_matrix_sync(frag_a[buf_idx], smem_a[buf_idx][warpId], WMMA_K);
        wmma::load_matrix_sync(frag_b[buf_idx], smem_b[buf_idx][warpId], WMMA_K);
    }
    
    // Main loop with double buffering
    for (k = WMMA_K; k < A.w; k += WMMA_K) {
        int next_buf = 1 - buf_idx;
        
        // Load next tile - optimized with reduced branching
        #pragma unroll
        for (int load_idx = laneId; load_idx < WMMA_M * WMMA_K; load_idx += 32) {
            int i = load_idx / WMMA_K;
            int j = load_idx % WMMA_K;
            int row = m + i;
            int col = k + j;
            __half val = __float2half(0.0f);
            if (row < A.h && col < A.w) {
                val = Index(A, row, col);
            }
            smem_a[next_buf][warpId][load_idx] = val;
        }
        
        // Load B tile with coalesced access pattern
        #pragma unroll
        for (int load_idx = laneId; load_idx < WMMA_K * WMMA_N; load_idx += 32) {
            int j = load_idx % WMMA_N;  // Column varies fastest for coalescing
            int i = load_idx / WMMA_N;  // Row varies slower
            int row = k + i;
            int col = n + j;
            __half val = __float2half(0.0f);
            if (row < B.h && col < B.w) {
                val = Index(B, row, col);
            }
            smem_b[next_buf][warpId][j * WMMA_K + i] = val;
        }
        
        // Compute with current buffer (overlaps with loads above)
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

template <typename T>
void op_mm_tensorcore_optimized(const Tensor<__half>& A, const Tensor<__half>& B, Tensor<T>& C)
{
    ensure_tc_mm_shape_device(A, B, C);
    
    if (!A.on_device) {
        throw std::runtime_error("TensorCore GEMM requires device tensors");
    }
    
    // Increased to 8 warps per block for better occupancy
    // 8 warps = 256 threads per block (better GPU utilization)
    dim3 blockDim(32, 8);
    // Larger tiles: 64 rows × 32 cols per block (4×2 warps, each 16×16)
    // 4 rows × 2 cols = 64 rows × 32 cols
    dim3 gridDim((C.w + 31) / 32, (C.h + 63) / 64);
    
    op_mm_tensorcore_optimized_kernel<<<gridDim, blockDim>>>(A, B, C);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Optimized TensorCore kernel launch failed: " + 
            std::string(cudaGetErrorString(err)));
    }
    
    CUDA_OK(cudaDeviceSynchronize());
}
