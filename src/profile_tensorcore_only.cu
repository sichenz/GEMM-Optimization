#include <cuda_runtime.h>
#include <curand.h>
#include <iostream>
#include "utils/tensor.cuh"
#include "utils/check_error.cuh"
#include "ops/op_mm_tensorcore_optimized.cuh"
#include "ops/op_elemwise.cuh"

// Minimal program to profile only the optimized TensorCore kernel
// This makes profiling easier by running only one kernel launch

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    if (prop.major < 7) {
        std::cerr << "TensorCores require Compute Capability >= 7.0" << std::endl;
        return 1;
    }
    
    std::cout << "Profiling TensorCore Optimized Kernel (4096x4096x4096)" << std::endl;
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << std::endl;
    
    const int M = 4096;
    const int N = 4096;
    const int K = 4096;
    
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
    
    // Simple conversion kernel
    auto convert = [](float* src, __half* dst, int n) {
        for (int i = 0; i < n; i++) {
            dst[i] = __float2half(src[i]);
        }
    };
    
    // Convert on device (simplified - using elementwise)
    op_uniform_fill(A, 0.0f, 1.0f);
    op_uniform_fill(B, 0.0f, 1.0f);
    
    CUDA_OK(cudaMemset(C.rawp, 0, M * N * sizeof(float)));
    CUDA_OK(cudaDeviceSynchronize());
    
    std::cout << "Running optimized TensorCore kernel..." << std::endl;
    
    // Run the kernel (this is what we'll profile)
    op_mm_tensorcore_optimized(A, B, C);
    
    std::cout << "Kernel execution complete!" << std::endl;
    
    curandDestroyGenerator(gen);
    
    return 0;
}

