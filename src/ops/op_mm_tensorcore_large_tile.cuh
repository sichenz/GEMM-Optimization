#pragma once

// "Large-tile" TensorCore GEMM experiment.
// Inspired by CUTLASS threadblock tiling: we increase the N-dimension tile
// size so each block computes a 16 x 64 region of C instead of 16 x 16.
//
// Block shape: 16 x 64 output tile per threadblock
//   - 4 warps per block (blockDim = (32 threads, 4 warps))
//   - Warps arranged as 1 x 4 within the block
//   - Each warp computes one 16 x 16 tile; together they cover 16 x 64.
//
// This is still a simple 2-buffer pipeline in K, but with a "wider" tile
// in N for better amortization of loads and scheduling.

#include "utils/check_error.cuh"
#include "utils/tensor.cuh"
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

template <typename T>
__global__ void op_mm_tensorcore_large_tile_kernel(
    const Tensor<__half> A,
    const Tensor<__half> B,
    Tensor<T> C)
{
    const int warpId = threadIdx.y;  // 0..3
    const int laneId = threadIdx.x;  // 0..31

    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;

    // 4 warps arranged as 1 x 4: one row, four columns
    const int warpRowInBlock = 0;
    const int warpColInBlock = warpId; // 0..3

    const int m = blockRow * 16 + warpRowInBlock * WMMA_M;
    const int n = blockCol * 64 + warpColInBlock * WMMA_N;

    const int num_tiles_k = (A.w + WMMA_K - 1) / WMMA_K;
    if (num_tiles_k == 0) {
        return;
    }

    // 2-buffer shared memory for A/B tiles
    __shared__ __half smem_a[2][4][WMMA_M * WMMA_K + 8];
    __shared__ __half smem_b[2][4][WMMA_K * WMMA_N + 8];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                   __half, wmma::row_major> frag_a[2];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                   __half, wmma::col_major> frag_b[2];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                   float> frag_c;

    wmma::fill_fragment(frag_c, 0.0f);

    auto load_tile_to_shared = [&] __device__ (int tile_idx, int buf) {
        int k0 = tile_idx * WMMA_K;

        // A tile
        #pragma unroll
        for (int load_idx = laneId; load_idx < WMMA_M * WMMA_K; load_idx += 32) {
            int i = load_idx / WMMA_K;
            int j = load_idx % WMMA_K;
            int row = m + i;
            int col = k0 + j;
            smem_a[buf][warpId][load_idx] =
                (row < A.h && col < A.w) ? Index(A, row, col)
                                         : __float2half(0.0f);
        }

        // B tile (stored col-major in shared)
        #pragma unroll
        for (int load_idx = laneId; load_idx < WMMA_K * WMMA_N; load_idx += 32) {
            int j = load_idx / WMMA_K;
            int i = load_idx % WMMA_K;
            int row = k0 + i;
            int col = n + j;
            smem_b[buf][warpId][j * WMMA_K + i] =
                (row < B.h && col < B.w) ? Index(B, row, col)
                                         : __float2half(0.0f);
        }
    };

    // ---------------------------------------------------------------------
    // 1) Prime: load first tile into buffer 0
    // ---------------------------------------------------------------------
    int buf_idx = 0;
    load_tile_to_shared(0, buf_idx);
    __syncthreads();

    wmma::load_matrix_sync(frag_a[buf_idx], smem_a[buf_idx][warpId], WMMA_K);
    wmma::load_matrix_sync(frag_b[buf_idx], smem_b[buf_idx][warpId], WMMA_K);
    __syncthreads();

    // ---------------------------------------------------------------------
    // 2) Main 2-buffer pipelined loop over remaining tiles
    // ---------------------------------------------------------------------
    for (int tile_idx = 1; tile_idx < num_tiles_k; ++tile_idx) {
        int next_buf = 1 - buf_idx;

        // Load next tile into the alternate buffer
        load_tile_to_shared(tile_idx, next_buf);

        // Compute with current buffer while loads are in flight
        wmma::mma_sync(frag_c, frag_a[buf_idx], frag_b[buf_idx], frag_c);

        __syncthreads();

        // Move fragments for the newly loaded tile
        wmma::load_matrix_sync(frag_a[next_buf], smem_a[next_buf][warpId], WMMA_K);
        wmma::load_matrix_sync(frag_b[next_buf], smem_b[next_buf][warpId], WMMA_K);

        __syncthreads();
        buf_idx = next_buf;
    }

    // Final compute for the last loaded tile
    wmma::mma_sync(frag_c, frag_a[buf_idx], frag_b[buf_idx], frag_c);

    // ---------------------------------------------------------------------
    // 3) Store result
    // ---------------------------------------------------------------------
    __shared__ float smem_c[4][WMMA_M * WMMA_N + 8];

    wmma::store_matrix_sync(
        smem_c[warpId],
        frag_c,
        WMMA_N,
        wmma::mem_row_major
    );

    __syncthreads();

    for (int elem = 0; elem < 8; ++elem) {
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

// Host wrapper
template <typename T>
void op_mm_tensorcore_large_tile(
    const Tensor<__half>& A,
    const Tensor<__half>& B,
    Tensor<T>& C)
{
    ensure_tc_mm_shape_device(A, B, C);

    if (!A.on_device) {
        throw std::runtime_error("TensorCore GEMM requires device tensors");
    }

    // 4 warps per block; each block computes a 16 x 64 tile
    dim3 blockDim(32, 4);
    dim3 gridDim(
        (C.w + 63) / 64,
        (C.h + 15) / 16
    );

    op_mm_tensorcore_large_tile_kernel<<<gridDim, blockDim>>>(A, B, C);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            "Large-tile TensorCore kernel launch failed: " +
            std::string(cudaGetErrorString(err))
        );
    }

    CUDA_OK(cudaDeviceSynchronize());
}
