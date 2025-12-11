#pragma once

// Phase 1.1.3: Lab-1 Tiled GEMM Implementation
// This is our baseline matrix multiplication kernel using tiling/blocking optimization
// The key idea: load small tiles into shared memory to reduce global memory accesses

#include "utils/check_error.cuh"
#include "utils/tensor.cuh"

// Helper function to validate matrix dimensions for matrix multiply
// For C = A @ B: A must be (M x K), B must be (K x N), C must be (M x N)
template <typename AT, typename BT, typename OT>
static void ensure_mm_shape_device(const Tensor<AT> &a, const Tensor<BT> &b, const Tensor<OT> &out)
{
    if (a.h != out.h || b.w != out.w || a.w != b.h)
        throw std::runtime_error("a,b,out tensor shape mismatch a:" +
            a.repr() + ", b:" + b.repr() + ", out:" + out.repr());

    if (a.on_device != b.on_device || a.on_device != out.on_device)
        throw std::runtime_error("a,b,out tensor device mismatch a:" +
            a.repr() + ", b:" + b.repr() + ", out:" + out.repr());
}

// Tile dimension: 32x32 tiles fit nicely in shared memory
// Each threadblock processes one 32x32 output tile
// This is a common choice because:
// - 32x32 = 1024 threads per block (max 1024 threads per block on most GPUs)
// - 32x32x4 bytes = 4KB per tile (fits in shared memory)
#define TILE_DIM 32

// Lab-1 Tiled GEMM Kernel
// Algorithm: Blocked matrix multiplication using shared memory
// 
// Key optimization: Instead of each thread reading directly from global memory,
// we load tiles into shared memory first. This reduces global memory traffic because:
// 1. Each element in a tile is reused multiple times (once per row/column)
// 2. Shared memory is much faster than global memory (~100x faster)
//
// How it works:
// - Each threadblock computes one 32x32 tile of the output matrix C
// - For each tile, we iterate over K dimension in chunks of 32
// - Load 32x32 tile from A and 32x32 tile from B into shared memory
// - All threads in block compute partial dot products
// - Accumulate results in registers, then write to global memory
template <typename T>
__global__ void op_mm_kernel(const Tensor<T> A, const Tensor<T> B, Tensor<T> C)
{
    // Shared memory tiles - visible to all threads in this threadblock
    // This is the key optimization: shared memory is much faster than global memory
    __shared__ T tileA[TILE_DIM][TILE_DIM];
    __shared__ T tileB[TILE_DIM][TILE_DIM];

    // Thread indices within the block (0-31 for both x and y)
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Global position in output matrix C
    // Each block handles one 32x32 tile, so we offset by block index
    int col = blockIdx.x * TILE_DIM + tx;
    int row = blockIdx.y * TILE_DIM + ty;

    // Accumulator for this thread's output element
    // Stored in register (fastest memory)
    T C_val = 0.0f;

    // Loop over K dimension in tiles
    // For C[i][j] = sum over k: A[i][k] * B[k][j]
    // We process K in chunks of TILE_DIM
    for (int t = 0; t < (A.w + TILE_DIM - 1) / TILE_DIM; ++t)
    {
        // Load tile from matrix A into shared memory
        // Each thread loads one element
        int kA = t * TILE_DIM + tx;  // Column index in A
        if (row < A.h && kA < A.w) {
            tileA[ty][tx] = Index(A, row, kA);  // Load from global to shared
        } else {
            tileA[ty][tx] = 0.0f;  // Padding for out-of-bounds
        }

        // Load tile from matrix B into shared memory
        int rB = t * TILE_DIM + ty;  // Row index in B
        if (rB < B.h && col < B.w) {
            tileB[ty][tx] = Index(B, rB, col);  // Load from global to shared
        } else {
            tileB[ty][tx] = 0.0f;  // Padding for out-of-bounds
        }
        
        // Synchronize: wait for all threads to finish loading
        // Critical: can't compute until both tiles are loaded
        __syncthreads();

        // Compute partial dot product using shared memory
        // This is the compute-intensive part
        // Each thread computes one element: sum over k of tileA[ty][k] * tileB[k][tx]
        for (int k = 0; k < TILE_DIM; ++k) {
            C_val += tileA[ty][k] * tileB[k][tx];
        }
        
        // Synchronize again before loading next tiles
        // Prevents race conditions when overwriting shared memory
        __syncthreads();
    }

    // Write result to global memory
    // Only write if within bounds (handles non-multiples of 32)
    if (row < C.h && col < C.w) {
        Index(C, row, col) = C_val;
    }
}


// Main GEMM function: compute C = A @ B
// This is the interface that other code calls
template <typename T>
void op_mm(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C)
{
    // Validate matrix dimensions
    ensure_mm_shape_device(A,B,C);
    
    if (A.on_device) {
        // Launch GPU kernel
        // Block size: 32x32 = 1024 threads per block
        dim3 blockDim(TILE_DIM, TILE_DIM);
        
        // Grid size: number of 32x32 tiles needed to cover output matrix
        // Ceiling division: (size + TILE_DIM - 1) / TILE_DIM
        dim3 gridDim((C.w + TILE_DIM - 1) / TILE_DIM, (C.h + TILE_DIM - 1) / TILE_DIM);
        
        // Launch kernel: <<<gridDim, blockDim>>>
        // Each threadblock computes one 32x32 output tile
        op_mm_kernel<<<gridDim, blockDim>>>(A, B, C);
    } else {
        // Fallback to CPU for non-device tensors (for testing/debugging)
        // Naive triple-loop implementation - very slow but correct
        for (int i = 0; i < A.h; i++) {
            for (int j = 0; j < B.w; j++) {
                T sum = 0;
                for (int k = 0; k < A.w; k++) {
                    sum += Index(A, i, k) * Index(B, k, j);
                }
                Index(C, i, j) = sum;
            }
        }
    }
}