#pragma once

// Phase 2 Optimization: 3-Stage Pipelined TensorCore GEMM
// Based on Phase 3 CUTLASS analysis findings
// Optimizations:
// 1. 3-stage pipeline (upgrade from 2-stage double buffering)
// 2. Better latency hiding by overlapping load/compute/store
// 3. Expected improvement: 10-20% over 2-stage version

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
static void ensure_tc_mm_shape_device(const Tensor<AT> &a, const Tensor<BT> &b, const Tensor<OT> &out)
{
    if (a.h != out.h || b.w != out.w || a.w != b.h)
        throw std::runtime_error("a,b,out tensor shape mismatch a:" +
            a.repr() + ", b:" + b.repr() + ", out:" + out.repr());
    if (a.on_device != b.on_device || a.on_device != out.on_device)
        throw std::runtime_error("a,b,out tensor device mismatch a:" +
            a.repr() + ", b:" + b.repr() + ", out:" + out.repr());
}

// 3-Stage Pipelined TensorCore GEMM Kernel
// Uses 4 warps per block (128 threads) for 32×32 output tiles
// 3-stage pipeline: Load tile k+1, Compute tile k, Load fragments for k+2
template <typename T>
__global__ void op_mm_tensorcore_3stage_kernel(
    const Tensor<__half> A,
    const Tensor<__half> B,
    Tensor<T> C)
{
    const int warpId = threadIdx.y;  // 0-3
    const int laneId = threadIdx.x;  // 0-31
    
    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;
    
    // 4 warps arranged as 2×2: 2 rows × 2 cols = 32 rows × 32 cols per block
    const int warpRowInBlock = warpId / 2;  // 0 or 1
    const int warpColInBlock = warpId % 2;  // 0 or 1
    
    const int m = blockRow * 32 + warpRowInBlock * WMMA_M;
    const int n = blockCol * 32 + warpColInBlock * WMMA_N;
    
    // 3-stage pipelined shared memory buffers
    // Stage 0, 1, 2: allow 3 tiles in flight simultaneously
    __shared__ __half smem_a[3][4][WMMA_M * WMMA_K + 8];  // 3 buffers, 4 warps
    __shared__ __half smem_b[3][4][WMMA_K * WMMA_N + 8];   // 3 buffers, 4 warps
    
    // 3-stage pipelined fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_a[3];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> frag_b[3];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_c;
    
    wmma::fill_fragment(frag_c, 0.0f);
    
    // Initialize pipeline: Load first two tiles
    // Stage 0: Load tile k=0
    int k = 0;
    int stage = 0;
    
    if (k < A.w) {
        // Load A tile for stage 0
        #pragma unroll
        for (int load_idx = laneId; load_idx < WMMA_M * WMMA_K; load_idx += 32) {
            int i = load_idx / WMMA_K;
            int j = load_idx % WMMA_K;
            int row = m + i;
            int col = k + j;
            smem_a[stage][warpId][load_idx] = (row < A.h && col < A.w) ? 
                Index(A, row, col) : __float2half(0.0f);
        }
        
        // Load B tile for stage 0
        #pragma unroll
        for (int load_idx = laneId; load_idx < WMMA_K * WMMA_N; load_idx += 32) {
            int j = load_idx / WMMA_K;
            int i = load_idx % WMMA_K;
            int row = k + i;
            int col = n + j;
            smem_b[stage][warpId][j * WMMA_K + i] = (row < B.h && col < B.w) ? 
                Index(B, row, col) : __float2half(0.0f);
        }
    }
    
    __syncthreads();
    
    // Load fragments for stage 0
    if (k < A.w) {
        wmma::load_matrix_sync(frag_a[stage], smem_a[stage][warpId], WMMA_K);
        wmma::load_matrix_sync(frag_b[stage], smem_b[stage][warpId], WMMA_K);
    }
    
    // Stage 1: Load tile k=16 (if exists)
    k = WMMA_K;
    stage = 1;
    
    if (k < A.w) {
        // Load A tile for stage 1
        #pragma unroll
        for (int load_idx = laneId; load_idx < WMMA_M * WMMA_K; load_idx += 32) {
            int i = load_idx / WMMA_K;
            int j = load_idx % WMMA_K;
            int row = m + i;
            int col = k + j;
            smem_a[stage][warpId][load_idx] = (row < A.h && col < A.w) ? 
                Index(A, row, col) : __float2half(0.0f);
        }
        
        // Load B tile for stage 1
        #pragma unroll
        for (int load_idx = laneId; load_idx < WMMA_K * WMMA_N; load_idx += 32) {
            int j = load_idx / WMMA_K;
            int i = load_idx % WMMA_K;
            int row = k + i;
            int col = n + j;
            smem_b[stage][warpId][j * WMMA_K + i] = (row < B.h && col < B.w) ? 
                Index(B, row, col) : __float2half(0.0f);
        }
    }
    
    __syncthreads();
    
    // Load fragments for stage 1
    if (k < A.w) {
        wmma::load_matrix_sync(frag_a[stage], smem_a[stage][warpId], WMMA_K);
        wmma::load_matrix_sync(frag_b[stage], smem_b[stage][warpId], WMMA_K);
    }
    
    // Main loop with 3-stage pipelining
    // Pipeline stages: 
    // - Stage 0: Load tile k into smem, load fragments
    // - Stage 1: Compute tile k-16, load tile k+16 into smem
    // - Stage 2: Compute tile k-32, load fragments for tile k
    for (k = 2 * WMMA_K; k < A.w; k += WMMA_K) {
        int stage_idx = (k / WMMA_K) % 3;  // Current stage index (0, 1, 2)
        int compute_stage = (stage_idx + 2) % 3;  // Stage to compute (2 iterations ago)
        int next_stage = (stage_idx + 1) % 3;  // Next stage to load into
        
        // STAGE 1: Load next tile (k) into next buffer
        #pragma unroll
        for (int load_idx = laneId; load_idx < WMMA_M * WMMA_K; load_idx += 32) {
            int i = load_idx / WMMA_K;
            int j = load_idx % WMMA_K;
            int row = m + i;
            int col = k + j;
            smem_a[stage_idx][warpId][load_idx] = (row < A.h && col < A.w) ? 
                Index(A, row, col) : __float2half(0.0f);
        }
        
        #pragma unroll
        for (int load_idx = laneId; load_idx < WMMA_K * WMMA_N; load_idx += 32) {
            int j = load_idx / WMMA_K;
            int i = load_idx % WMMA_K;
            int row = k + i;
            int col = n + j;
            smem_b[stage_idx][warpId][j * WMMA_K + i] = (row < B.h && col < B.w) ? 
                Index(B, row, col) : __float2half(0.0f);
        }
        
        // STAGE 2: Compute with buffer from 2 iterations ago
        wmma::mma_sync(frag_c, frag_a[compute_stage], frag_b[compute_stage], frag_c);
        
        // Synchronize: ensure loads complete before loading fragments
        __syncthreads();
        
        // STAGE 3: Load fragments from current buffer (k-16)
        int load_frag_stage = (stage_idx + 1) % 3;  // Stage loaded in previous iteration
        wmma::load_matrix_sync(frag_a[load_frag_stage], smem_a[load_frag_stage][warpId], WMMA_K);
        wmma::load_matrix_sync(frag_b[load_frag_stage], smem_b[load_frag_stage][warpId], WMMA_K);
    }
    
    // Drain pipeline: Compute remaining tiles
    // After loop: stage 0 and 1 still have tiles to compute
    int remaining_k = A.w - ((A.w / WMMA_K) * WMMA_K);
    int last_stage = ((A.w / WMMA_K) - 1) % 3;
    int second_last_stage = ((A.w / WMMA_K) - 2) % 3;
    
    // Compute second-to-last stage
    if (A.w >= 2 * WMMA_K) {
        wmma::mma_sync(frag_c, frag_a[second_last_stage], frag_b[second_last_stage], frag_c);
    }
    
    // Compute last stage
    if (A.w >= WMMA_K) {
        wmma::mma_sync(frag_c, frag_a[last_stage], frag_b[last_stage], frag_c);
    }
    
    // Store result
    __shared__ float smem_c[4][WMMA_M * WMMA_N + 8];
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

// 3-Stage Pipelined TensorCore GEMM function
template <typename T>
void op_mm_tensorcore_3stage(const Tensor<__half>& A, const Tensor<__half>& B, Tensor<T>& C)
{
    ensure_tc_mm_shape_device(A, B, C);
    
    if (!A.on_device) {
        throw std::runtime_error("TensorCore GEMM requires device tensors");
    }
    
    // Launch configuration: 4 warps per block (128 threads)
    // Each block computes 32×32 output (2×2 warps, each 16×16)
    dim3 blockDim(32, 4);  // 32 threads per warp, 4 warps per block
    
    // Grid dimension: each block handles 32 rows × 32 cols
    dim3 gridDim((C.w + 31) / 32,
                 (C.h + 31) / 32);
    
    op_mm_tensorcore_3stage_kernel<<<gridDim, blockDim>>>(A, B, C);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("3-Stage TensorCore kernel launch failed: " + 
            std::string(cudaGetErrorString(err)));
    }
    
    CUDA_OK(cudaDeviceSynchronize());
}

