#pragma once

// High-performance TensorCore GEMM implementation
// Key optimizations:
// 1. Larger tile sizes (64x64 output per block with 8 warps)
// 2. Optimized shared memory layout to avoid bank conflicts
// 3. Better double buffering with proper synchronization
// 4. Improved memory access patterns for coalescing

#include "utils/check_error.cuh"
#include "utils/tensor.cuh"
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// ensure_tc_mm_shape_device is in op_mm_tensorcore.cuh

// High-performance TensorCore kernel
// 8 warps per block, 64x64 output per block (4x4 warps, each 16x16)
template <typename T>
__global__ void op_mm_tensorcore_high_perf_kernel(
    const Tensor<__half> A,
    const Tensor<__half> B,
    Tensor<T> C)
{
    const int warpId = threadIdx.y;  // 0-7 (8 warps)
    const int laneId = threadIdx.x;  // 0-31
    
    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;
    
    // 8 warps arranged as 4x2: 4 rows × 2 cols = 64 rows × 32 cols per block
    const int warpRowInBlock = warpId / 2;  // 0-3
    const int warpColInBlock = warpId % 2;  // 0 or 1
    
    const int m = blockRow * 64 + warpRowInBlock * WMMA_M;
    const int n = blockCol * 32 + warpColInBlock * WMMA_N;
    
    // Double buffered shared memory with padding to avoid bank conflicts
    // Padding: +8 elements per row to avoid 32-way bank conflicts
    __shared__ __half smem_a[2][8][WMMA_M * WMMA_K + 8];
    __shared__ __half smem_b[2][8][WMMA_K * WMMA_N + 8];
    
    // WMMA fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_a[2];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> frag_b[2];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_c;
    
    wmma::fill_fragment(frag_c, 0.0f);
    
    // Load first tile
    int k = 0;
    int buf_idx = 0;
    
    if (k < A.w) {
        // Load A tile - ensure coalesced access
        // Each thread loads multiple elements in a coalesced pattern
        #pragma unroll
        for (int load_idx = laneId; load_idx < WMMA_M * WMMA_K; load_idx += 32) {
            int i = load_idx / WMMA_K;
            int j = load_idx % WMMA_K;
            int row = m + i;
            int col = k + j;
            smem_a[buf_idx][warpId][load_idx] = (row < A.h && col < A.w) ? 
                Index(A, row, col) : __float2half(0.0f);
        }
        
        // Load B tile (transposed to col-major) with coalesced access
        #pragma unroll
        for (int load_idx = laneId; load_idx < WMMA_K * WMMA_N; load_idx += 32) {
            // Coalesced: access consecutive columns first
            int j = load_idx % WMMA_N;  // Column varies fastest for coalescing
            int i = load_idx / WMMA_N;  // Row varies slower
            int row = k + i;
            int col = n + j;
            // Store in col-major order in shared memory
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
        
        // Load next tile (overlaps with computation)
        #pragma unroll
        for (int load_idx = laneId; load_idx < WMMA_M * WMMA_K; load_idx += 32) {
            int i = load_idx / WMMA_K;
            int j = load_idx % WMMA_K;
            int row = m + i;
            int col = k + j;
            smem_a[next_buf][warpId][load_idx] = (row < A.h && col < A.w) ? 
                Index(A, row, col) : __float2half(0.0f);
        }
        
        // Load B tile with coalesced access pattern
        #pragma unroll
        for (int load_idx = laneId; load_idx < WMMA_K * WMMA_N; load_idx += 32) {
            // Coalesced: access consecutive columns first
            int j = load_idx % WMMA_N;  // Column varies fastest for coalescing
            int i = load_idx / WMMA_N;  // Row varies slower
            int row = k + i;
            int col = n + j;
            smem_b[next_buf][warpId][j * WMMA_K + i] = (row < B.h && col < B.w) ? 
                Index(B, row, col) : __float2half(0.0f);
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
void op_mm_tensorcore_high_perf(const Tensor<__half>& A, const Tensor<__half>& B, Tensor<T>& C)
{
    ensure_tc_mm_shape_device(A, B, C);
    
    if (!A.on_device) {
        throw std::runtime_error("TensorCore GEMM requires device tensors");
    }
    
    // Launch config: 8 warps per block (256 threads), 64×32 output per block
    // 4 rows × 2 cols warps = 64 rows × 32 cols per block
    dim3 blockDim(32, 8);  // 32 threads per warp, 8 warps per block
    dim3 gridDim((C.w + 31) / 32, (C.h + 63) / 64);
    
    op_mm_tensorcore_high_perf_kernel<<<gridDim, blockDim>>>(A, B, C);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("High-Perf TensorCore kernel launch failed: " + 
            std::string(cudaGetErrorString(err)) + " (grid: " + 
            std::to_string(gridDim.x) + "x" + std::to_string(gridDim.y) + 
            ", block: " + std::to_string(blockDim.x) + "x" + std::to_string(blockDim.y) + ")");
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("High-Perf TensorCore kernel execution failed: " + 
            std::string(cudaGetErrorString(err)));
    }
}

