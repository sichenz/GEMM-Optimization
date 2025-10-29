#pragma once

#include "utils/check_error.cuh"
#include "utils/tensor.cuh"
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

#define TILE_DIM 32

template <typename T>
__global__ void op_mm_kernel(const Tensor<T> A, const Tensor<T> B, Tensor<T> C)
{
    __shared__ T tileA[TILE_DIM][TILE_DIM];
    __shared__ T tileB[TILE_DIM][TILE_DIM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * TILE_DIM + tx;
    int row = blockIdx.y * TILE_DIM + ty;

    T C_val = 0.0f;

    for (int t = 0; t < (A.w + TILE_DIM - 1) / TILE_DIM; ++t)
    {
        int kA = t * TILE_DIM + tx;
        if (row < A.h && kA < A.w) {
            tileA[ty][tx] = Index(A, row, kA);
        } else {
            tileA[ty][tx] = 0.0f;
        }

        int rB = t * TILE_DIM + ty;
        if (rB < B.h && col < B.w) {
            tileB[ty][tx] = Index(B, rB, col);
        } else {
            tileB[ty][tx] = 0.0f;
        }
        __syncthreads();

        for (int k = 0; k < TILE_DIM; ++k) {
            C_val += tileA[ty][k] * tileB[k][tx];
        }
        __syncthreads();
    }

    if (row < C.h && col < C.w) {
        Index(C, row, col) = C_val;
    }
}


//compute C = A@B
template <typename T>
void op_mm(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C)
{
    ensure_mm_shape_device(A,B,C);
    //Lab-1: please complete this
    //You need to define separate kernel function(s) and launch them here
    if (A.on_device) {
        dim3 blockDim(TILE_DIM, TILE_DIM);
        dim3 gridDim((C.w + TILE_DIM - 1) / TILE_DIM, (C.h + TILE_DIM - 1) / TILE_DIM);
        op_mm_kernel<<<gridDim, blockDim>>>(A, B, C);
    } else {
        // Fallback to CPU for non-device tensors
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