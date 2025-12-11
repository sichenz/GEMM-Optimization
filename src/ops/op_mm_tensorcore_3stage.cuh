#pragma once

// Idea taken from CUTLASS multistage mainloops, but implemented with
// regular global->shared loads (no cp.async).
//
// Block shape: 32 x 32 output tile per threadblock
//   - 4 warps per block (blockDim = (32 threads, 4 warps))
//   - 2 x 2 warp grid; each warp computes a 16 x 16 sub-tile
//
// NOTE: This is a *software* 3-buffer pipeline. Without cp.async the gain
// over a 2-buffer scheme may be small, but the code mirrors the CUTLASS
// idea of having 3 tiles “in flight”.

#include "utils/check_error.cuh"
#include "utils/tensor.cuh"
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

template <typename T>
__global__ void op_mm_tensorcore_3stage_kernel(
    const Tensor<__half> A,
    const Tensor<__half> B,
    Tensor<T> C)
{
    const int warpId = threadIdx.y;  // 0..3
    const int laneId = threadIdx.x;  // 0..31

    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;

    // 4 warps laid out as 2 x 2 within the block
    const int warpRowInBlock = warpId / 2; // 0 or 1
    const int warpColInBlock = warpId % 2; // 0 or 1

    const int m = blockRow * 32 + warpRowInBlock * WMMA_M;
    const int n = blockCol * 32 + warpColInBlock * WMMA_N;

    // Number of K-tiles for this GEMM
    const int num_tiles_k = (A.w + WMMA_K - 1) / WMMA_K;
    if (num_tiles_k == 0) {
        return;
    }

    // 3-stage shared-memory buffers: [stage][warp][element]
    __shared__ __half smem_a[3][4][WMMA_M * WMMA_K + 8];
    __shared__ __half smem_b[3][4][WMMA_K * WMMA_N + 8];

    // 3 fragment slots, one per stage
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                   __half, wmma::row_major> frag_a[3];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                   __half, wmma::col_major> frag_b[3];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                   float> frag_c;

    wmma::fill_fragment(frag_c, 0.0f);

    // Helper lambda: load one K-tile (by tile index) into a given buffer
    auto load_tile_to_shared = [&] __device__ (int tile_idx, int buf) {
        const int k0 = tile_idx * WMMA_K;

        // Load A sub-tile [m : m+16, k0 : k0+16]
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

        // Load B sub-tile [k0 : k0+16, n : n+16], stored col-major in smem
        #pragma unroll
        for (int load_idx = laneId; load_idx < WMMA_K * WMMA_N; load_idx += 32) {
            int j = load_idx / WMMA_K; // column within 16x16 tile
            int i = load_idx % WMMA_K; // row within 16x16 tile
            int row = k0 + i;
            int col = n + j;
            smem_b[buf][warpId][j * WMMA_K + i] =
                (row < B.h && col < B.w) ? Index(B, row, col)
                                         : __float2half(0.0f);
        }
    };

    // ---------------------------------------------------------------------
    // 1) Prime the pipeline: preload up to 3 tiles into buffers 0,1,2
    // ---------------------------------------------------------------------
    int tiles_to_preload = num_tiles_k < 3 ? num_tiles_k : 3;

    for (int t = 0; t < tiles_to_preload; ++t) {
        int buf = t % 3;
        load_tile_to_shared(t, buf);
        __syncthreads();

        wmma::load_matrix_sync(frag_a[buf], smem_a[buf][warpId], WMMA_K);
        wmma::load_matrix_sync(frag_b[buf], smem_b[buf][warpId], WMMA_K);
        __syncthreads();
    }

    // ---------------------------------------------------------------------
    // 2) Main loop: compute tile t, while keeping tiles t+1, t+2 prefetched
    //    Tile t+3 is loaded into the buffer that just became free.
    // ---------------------------------------------------------------------
    for (int t = 0; t < num_tiles_k; ++t) {
        int curr_buf = t % 3;

        // Compute C += A_t * B_t using current fragments
        wmma::mma_sync(frag_c, frag_a[curr_buf], frag_b[curr_buf], frag_c);

        // Prefetch tile t+3 (if it exists) into the buffer we just consumed
        int next_tile = t + 3;
        if (next_tile < num_tiles_k) {
            int next_buf = curr_buf;

            load_tile_to_shared(next_tile, next_buf);
            __syncthreads();

            wmma::load_matrix_sync(frag_a[next_buf], smem_a[next_buf][warpId], WMMA_K);
            wmma::load_matrix_sync(frag_b[next_buf], smem_b[next_buf][warpId], WMMA_K);
            __syncthreads();
        }
    }

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

    // Coalesced write-back
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
void op_mm_tensorcore_3stage(
    const Tensor<__half>& A,
    const Tensor<__half>& B,
    Tensor<T>& C)
{
    ensure_tc_mm_shape_device(A, B, C);

    if (!A.on_device) {
        throw std::runtime_error("TensorCore GEMM requires device tensors");
    }

    dim3 blockDim(32, 4); // 4 warps per block
    dim3 gridDim(
        (C.w + 31) / 32,
        (C.h + 31) / 32
    );

    op_mm_tensorcore_3stage_kernel<<<gridDim, blockDim>>>(A, B, C);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            "3-stage TensorCore kernel launch failed: " +
            std::string(cudaGetErrorString(err))
        );
    }

    CUDA_OK(cudaDeviceSynchronize());
}
