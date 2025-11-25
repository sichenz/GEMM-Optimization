// Minimal program to profile TensorCore kernel
// Only runs optimized TensorCore kernel for 4096x4096x4096

#include <cuda_runtime.h>
#include <curand.h>
#include <iostream>
#include "utils/tensor.cuh"
#include "utils/check_error.cuh"
#include "ops/op_mm_tensorcore_optimized.cuh"

int main() {
    int M = 4096, N = 4096, K = 4096;
    
    // Allocate matrices
    Tensor<__half> A{M, K, true};
    Tensor<__half> B{K, N, true};
    Tensor<float> C{M, N, true};
    
    // Initialize with random data
    Tensor<float> A_fp32{M, K, true};
    Tensor<float> B_fp32{K, N, true};
    
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 12345);
    curandGenerateUniform(gen, A_fp32.rawp, M * K);
    curandGenerateUniform(gen, B_fp32.rawp, K * N);
    
    // Convert to FP16
    int threads = 256;
    int blocks_A = (M * K + threads - 1) / threads;
    int blocks_B = (K * N + threads - 1) / threads;
    
    auto convert_fp32_to_fp16 = [](float* src, __half* dst, int n) {
        int blocks = (n + 255) / 256;
        auto kernel = [] __device__ (float* src, __half* dst, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) dst[idx] = __float2half(src[idx]);
        };
        kernel<<<blocks, 256>>>(src, dst, n);
    };
    
    convert_fp32_to_fp16(A_fp32.rawp, A.rawp, M * K);
    convert_fp32_to_fp16(B_fp32.rawp, B.rawp, K * N);
    CUDA_OK(cudaDeviceSynchronize());
    
    // Initialize C to zero
    CUDA_OK(cudaMemset(C.rawp, 0, M * N * sizeof(float)));
    
    // Warmup
    for (int i = 0; i < 3; i++) {
        op_mm_tensorcore_optimized(A, B, C);
    }
    CUDA_OK(cudaDeviceSynchronize());
    
    std::cout << "Running optimized TensorCore kernel for " 
              << M << "x" << N << "x" << K << std::endl;
    
    // This is the kernel we want to profile
    op_mm_tensorcore_optimized(A, B, C);
    CUDA_OK(cudaDeviceSynchronize());
    
    std::cout << "Kernel execution complete" << std::endl;
    
    curandDestroyGenerator(gen);
    return 0;
}

