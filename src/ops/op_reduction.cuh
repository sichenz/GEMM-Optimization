#pragma once

#include "utils/tensor.cuh"

template <typename T, typename IT>
class MaxAccumFunc
{
public:
    //This function compares input x with the current accumulated maximum value stored in accum
    //If x is bigger than accum, stores x in accum and stores x's index (ind_x) to ind_accum
    __host__ __device__ void operator()(const T &x, const IT &ind_x, T &accum, IT &ind_accum)
    {
      //Lab-1: add your code here
      if (x > accum) {
        accum = x;
        ind_accum = ind_x;
      }
    }
};

template <typename T, typename IT>
class SumAccumFunc
{
public:
    //This function adds input x to the current accumulated sum value stored in accum
    //The accumu's value is updated (to add x).  The ind_x and ind_accum arguments are not used.
    __host__ __device__ void operator()(const T &x, const IT &ind_x, T &accum, IT &ind_accum)
    {
        //Lab-1: add your code here
        accum += x;
    }
};

#define REDUCTION_BLOCK_SIZE 256

//This kernel function performs column-wise reduction of the "in" tensor and stores the result in "out" tensor.
//If "get_index" is true, then the index accumulator values are stored in "out_index" and "out" is not touched.
template <typename OpFunc, typename T, typename IT>
__global__ void op_reduction_kernel_colwise(OpFunc f, Tensor<T> in, Tensor<T> out, Tensor<IT> out_index, bool get_index)
{
   //Lab-1: add your code here
   __shared__ T s_val[REDUCTION_BLOCK_SIZE];
   __shared__ IT s_ind[REDUCTION_BLOCK_SIZE];

   int i = blockIdx.x; // Row index
   if (i >= in.h) return;
   int tid = threadIdx.x;

   T my_val;
   IT my_ind;

   if (tid < in.w) {
       my_val = Index(in, i, tid);
       my_ind = tid;
       for (int j = tid + blockDim.x; j < in.w; j += blockDim.x) {
           f(Index(in, i, j), j, my_val, my_ind);
       }
       s_val[tid] = my_val;
       s_ind[tid] = my_ind;
   }
   __syncthreads();

   for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
       if (tid < s && (tid + s) < in.w) {
           f(s_val[tid + s], s_ind[tid + s], s_val[tid], s_ind[tid]);
       }
       __syncthreads();
   }

   if (tid == 0 && in.w > 0) {
       if (get_index) {
           Index(out_index, i, 0) = s_ind[0];
       } else {
           Index(out, i, 0) = s_val[0];
       }
   }
}

//This kernel function performs row-wise reduction of the "in" tensor and stores the result in "out" tensor.
//If "get_index" is true, then the index accumulator values are stored in "out_index" and "out" is not touched.
template <typename OpFunc, typename T, typename IT>
__global__ void op_reduction_kernel_rowwise(OpFunc f, Tensor<T> in, Tensor<T> out, Tensor<IT> out_index, bool get_index)
{
   //Lab-1: add your code here
   __shared__ T s_val[REDUCTION_BLOCK_SIZE];
   __shared__ IT s_ind[REDUCTION_BLOCK_SIZE];

   int j = blockIdx.x; // Column index
   if (j >= in.w) return;
   int tid = threadIdx.x;

   T my_val;
   IT my_ind;

   if (tid < in.h) {
       my_val = Index(in, tid, j);
       my_ind = tid;
       for (int i = tid + blockDim.x; i < in.h; i += blockDim.x) {
           f(Index(in, i, j), i, my_val, my_ind);
       }
       s_val[tid] = my_val;
       s_ind[tid] = my_ind;
   }
   __syncthreads();

   for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
       if (tid < s && (tid + s) < in.h) {
           f(s_val[tid + s], s_ind[tid + s], s_val[tid], s_ind[tid]);
       }
       __syncthreads();
   }

   if (tid == 0 && in.h > 0) {
       if (get_index) {
           Index(out_index, 0, j) = s_ind[0];
       } else {
           Index(out, 0, j) = s_val[0];
       }
   }
}

template <typename OpFunc, typename T, typename IT>
void op_reduction_gpu(OpFunc f, const Tensor<T> &in, Tensor<T> &out, Tensor<IT> &out_index, bool get_index = false)
{
  //Lab-1: add your code here. You need to launch either op_reduction_kernel_colwise or op_reduction_kernel_rowwise
  //depending on the output shape
  int out_h = get_index ? out_index.h : out.h;

  dim3 blockDim(REDUCTION_BLOCK_SIZE);

  if (in.h > out_h) { // Row-wise reduction (reducing columns) -> one block per column
      dim3 gridDim(in.w);
      op_reduction_kernel_rowwise<<<gridDim, blockDim>>>(f, in, out, out_index, get_index);
  } else { // Column-wise reduction (reducing rows) -> one block per row
      dim3 gridDim(in.h);
      op_reduction_kernel_colwise<<<gridDim, blockDim>>>(f, in, out, out_index, get_index);
  }
}

template <typename OpFunc, typename T, typename IT>
void op_reduction_cpu_rowwise(OpFunc f, const Tensor<T> &in, Tensor<T> &out, Tensor<IT> &out_index, bool get_index = false)
{
    for (int j = 0; j < in.w; j++)
    {
        IT accum_ind = 0;
        T accum = Index(in, 0, j);
        for (int i = 1; i < in.h; i++)
        {
            f(Index(in, i, j), i, accum, accum_ind);
        }
        if (get_index)
            Index(out_index, 0, j) = accum_ind;
        else
            Index(out, 0, j) = accum;
    }
}

template <typename OpFunc, typename T, typename IT>
void op_reduction_cpu_colwise(OpFunc f, const Tensor<T> &in, Tensor<T> &out, Tensor<IT> &out_index, bool get_index = false)
{

    for (int i = 0; i < in.h; i++)
    {
        IT accum_ind = 0;
        T accum = Index(in, i, 0);
        for (int j = 1; j < in.w; j++)
        {
            f(Index(in, i, j), j, accum, accum_ind);
        }
        if (get_index)
            Index(out_index, i, 0) = accum_ind;
        else
            Index(out, i, 0) = accum;
    }
}

template <typename OpFunc, typename T, typename IT>
void op_reduction_cpu(OpFunc f, const Tensor<T> &in, Tensor<T> &out, Tensor<IT> &out_index, bool get_index = false)
{
    int out_h = get_index?out_index.h:out.h;
    if (in.h > out_h)
        op_reduction_cpu_rowwise(f, in, out, out_index, get_index);
    else
        op_reduction_cpu_colwise(f, in, out, out_index, get_index);
}

/*-----------------------------------------------------------*/
template <typename AT, typename OT>
static void ensure_reduction_shape_device(const Tensor<AT> &a, const Tensor<OT> &out)
{
    if (a.on_device != out.on_device)
    {
        throw std::runtime_error("ensure_reduction_shape_device2: device mismatch");
    }

    if (a.w == out.w && out.h == 1)
    {
    }
    else if (a.h == out.h && out.w == 1)
    {
    }
    else
    {
        throw std::runtime_error("ensure_reduction_shape_device2: output shape mismatch");
    }
}


template <typename T>
void op_sum(const Tensor<T> &in, Tensor<T> &out)
{
    Tensor<int> out_index;
    SumAccumFunc<T, int> f;
    ensure_reduction_shape_device(in, out);
    if (in.on_device)
    {
        op_reduction_gpu(f, in, out, out_index, false);
    }
    else
    {
        op_reduction_cpu(f, in, out, out_index, false);
    }
}

template <typename T, typename IT>
void op_argmax(const Tensor<T> &in, Tensor<IT> &out_index)
{
    Tensor<T> out;
    MaxAccumFunc<T, IT> f;
    ensure_reduction_shape_device(in, out_index);
    if (in.on_device)
    {
        op_reduction_gpu(f, in, out, out_index, true);
    }
    else
    {
        op_reduction_cpu(f, in, out, out_index, true);
    }
}
