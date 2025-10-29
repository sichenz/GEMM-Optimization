#pragma once
#include "utils/tensor.cuh"
#include <vector>

#define CE_BLOCK_SIZE 256

template <typename T, typename S>
__global__ void cross_entropy_kernel(const Tensor<T> logits, const Tensor<S> targets, Tensor<T> d_logits, T* losses) {
    int i = blockIdx.x; // Batch index
    if (i >= logits.h) return;

    int tid = threadIdx.x;
    const int num_classes = logits.w;
    const int batch_size = logits.h;

    __shared__ T s_data[CE_BLOCK_SIZE];

    // --- 1. Find max logit for numerical stability ---
    T my_max = -1.0e20f;
    if (tid < num_classes) {
        my_max = Index(logits, i, tid);
        for (int j = tid + blockDim.x; j < num_classes; j += blockDim.x) {
            my_max = max(my_max, Index(logits, i, j));
        }
    }
    s_data[tid] = my_max;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && (tid + s) < num_classes) {
            s_data[tid] = max(s_data[tid], s_data[tid + s]);
        }
        __syncthreads();
    }
    T max_logit = (num_classes > 0) ? s_data[0] : 0.0f;
    __syncthreads();

    // --- 2. Calculate sum of exps ---
    T my_sum = 0.0f;
    if (tid < num_classes) {
        for (int j = tid; j < num_classes; j += blockDim.x) {
            T prob = exp(Index(logits, i, j) - max_logit);
            Index(d_logits, i, j) = prob; // Store un-normalized prob
            my_sum += prob;
        }
    }
    s_data[tid] = my_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && (tid + s) < num_classes) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }
    T sum_exp = (num_classes > 0) ? s_data[0] : 1.0f;
    __syncthreads();

    // --- 3. Normalize to get softmax probs, calculate loss, and finalize d_logits ---
    S target_idx = Index(targets, i, 0);
    if (tid < num_classes) {
        for (int j = tid; j < num_classes; j += blockDim.x) {
            T prob = Index(d_logits, i, j) / sum_exp;
            if (j == target_idx) {
                if (tid == j) { // Only one thread writes the loss for the row
                    losses[i] = -log(prob);
                }
                Index(d_logits, i, j) = (prob - 1.0f) / batch_size;
            } else {
                Index(d_logits, i, j) = prob / batch_size;
            }
        }
    }
}

//This function calculates the cross_entropy loss from the "logits" tensor for a batch of training innput
//and the batch's corresponding "target" label tensor and returns the average loss of the batch.
//It also returns the gradient of the logits tensor.
template <typename T, typename S>
T op_cross_entropy_loss(const Tensor<T> &logits, const Tensor<S> &targets,
                               Tensor<T> &d_logits)
{
    if (logits.h != d_logits.h || logits.w != d_logits.w)
    {
        throw std::runtime_error("op_cross_entropy_loss: d_logits shape mismatch");
    }

    if (targets.h != logits.h || targets.w != 1)
    {
        throw std::runtime_error("op_cross_entropy_loss: targets shape mismatch");
    }
    if (logits.on_device != d_logits.on_device || logits.on_device != targets.on_device)
    {
        throw std::runtime_error("op_cross_entropy_loss: device mismatch");
    }

    //Lab-1: please add your code here
    //You need to define separate GPU kernel function(s) and launch them here
    //In order to calculate d_logits, you should derive what its values should be
    //symbolically.
    int batch_size = logits.h;
    if (batch_size == 0) return 0.0f;

    if (logits.on_device) {
        Tensor<T> losses_gpu(batch_size, 1, true);
        dim3 gridDim(batch_size);
        dim3 blockDim(CE_BLOCK_SIZE);
        cross_entropy_kernel<<<gridDim, blockDim>>>(logits, targets, d_logits, losses_gpu.rawp);
        CUDA_OK(cudaGetLastError());

        Tensor<T> losses_cpu(batch_size, 1, false);
        CUDA_OK(cudaMemcpy(losses_cpu.rawp, losses_gpu.rawp, batch_size * sizeof(T), cudaMemcpyDeviceToHost));

        T total_loss = 0.0f;
        for (int i = 0; i < batch_size; i++) {
            total_loss += Index(losses_cpu, i, 0);
        }
        return total_loss / batch_size;
    } else {
        // CPU implementation
        T total_loss = 0.0f;
        for (int i = 0; i < logits.h; i++) {
            // Find max for numerical stability
            T max_l = Index(logits, i, 0);
            for (int j = 1; j < logits.w; j++) {
                if (Index(logits, i, j) > max_l) {
                    max_l = Index(logits, i, j);
                }
            }
            // Compute softmax
            std::vector<T> exp_vals(logits.w);
            T sum_exp = 0.0f;
            for (int j = 0; j < logits.w; j++) {
                exp_vals[j] = exp(Index(logits, i, j) - max_l);
                sum_exp += exp_vals[j];
            }

            S target_idx = Index(targets, i, 0);
            for (int j = 0; j < logits.w; j++) {
                T prob = exp_vals[j] / sum_exp;
                if (j == target_idx) {
                    total_loss += -log(prob);
                    Index(d_logits, i, j) = (prob - 1.0f) / batch_size;
                } else {
                    Index(d_logits, i, j) = prob / batch_size;
                }
            }
        }
        return total_loss / batch_size;
    }
}