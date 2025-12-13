#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cmath>

#include "utils/tensor.cuh"
#include "utils/check_error.cuh"
#include "ops/op_mm.cuh"
#include "ops/op_mm_tensorcore.cuh"  // Must be included first (defines ensure_tc_mm_shape_device)
#include "ops/op_mm_tensorcore_optimized.cuh"
#include "ops/op_mm_tensorcore_large_tile.cuh"
#include "ops/op_mm_tensorcore_high_perf.cuh"
#include "ops/op_mm_tensorcore_aggressive.cuh"
#include "ops/op_mm_tensorcore_ultra.cuh"
#include "ops/op_mm_tensorcore_64x64.cuh"
// Note: op_mm_tensorcore_3stage.cuh disabled due to performance issues
#include "ops/op_elemwise.cuh"

// Benchmarking framework for measuring GEMM performance
// This file handles timing, running benchmarks, and validating correctness
// I'm using CUDA events for timing since they're more accurate than CPU timers

unsigned long long randgen_seed = 12345;

// CUDA event timer - more accurate than CPU timers
// Events measure GPU time directly, which is important since GPU execution is async
class CudaTimer {
public:
    CudaTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }
    
    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    
    void start() {
        cudaEventRecord(start_);
    }
    
    float stop() {
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start_, stop_);
        return milliseconds;
    }
    
private:
    cudaEvent_t start_, stop_;
};

// Structure to store benchmark results
struct BenchmarkResult {
    int M, N, K;              // Matrix dimensions
    float time_ms;            // Execution time in milliseconds
    double gflops;            // Giga-FLOPS achieved
    double bandwidth_gb_s;    // Memory bandwidth in GB/s
    double efficiency_percent; // Efficiency vs peak (calculated later)
    std::string kernel_name;  // Which kernel was tested
    std::string dtype;        // Data type (FP32 or FP16)
};

// Calculate GFLOPS - total operations is 2*M*N*K (multiply-add counts as 2 ops)
double calculateGFLOPS(int M, int N, int K, float time_ms) {
    double operations = 2.0 * M * N * K;  // Total FLOPS
    double gflops = (operations / 1e9) / (time_ms / 1000.0);  // Convert to GFLOPS
    return gflops;
}

// Calculate memory bandwidth - total bytes = A + B + C (read A, read B, write C)
double calculateBandwidth(int M, int N, int K, float time_ms, int bytes_per_element) {
    double bytes = (double)(M * K + K * N + M * N) * bytes_per_element;
    double bandwidth = (bytes / 1e9) / (time_ms / 1000.0);  // GB/s
    return bandwidth;
}

// Benchmark the Lab-1 tiled GEMM kernel
BenchmarkResult benchmarkLab1GEMM(int M, int N, int K, int warmup_iters, int bench_iters) {
    // Allocate matrices on GPU
    Tensor<float> A{M, K, true};
    Tensor<float> B{K, N, true};
    Tensor<float> C{M, N, true};
    
    // Initialize with random data
    op_uniform_fill(A, 0.0f, 1.0f);
    op_uniform_fill(B, 0.0f, 1.0f);
    
    CUDA_OK(cudaDeviceSynchronize());
    
    CudaTimer timer;
    
    // Warmup runs - GPU needs to "warm up" first
    for (int i = 0; i < warmup_iters; i++) {
        op_mm(A, B, C);
    }
    CUDA_OK(cudaDeviceSynchronize());
    
    // Actual benchmark - run multiple times and average
    timer.start();
    for (int i = 0; i < bench_iters; i++) {
        op_mm(A, B, C);
    }
    CUDA_OK(cudaDeviceSynchronize());
    float time_ms = timer.stop() / bench_iters;
    
    BenchmarkResult result;
    result.M = M;
    result.N = N;
    result.K = K;
    result.time_ms = time_ms;
    result.gflops = calculateGFLOPS(M, N, K, time_ms);
    result.bandwidth_gb_s = calculateBandwidth(M, N, K, time_ms, sizeof(float));
    result.kernel_name = "Lab1_Tiled";
    result.dtype = "FP32";
    
    return result;
}

// Benchmark cuBLAS SGEMM (FP32) - this is our performance target
// cuBLAS is NVIDIA's optimized library, so this is what we're trying to match
BenchmarkResult benchmarkCublasSGEMM(int M, int N, int K, int warmup_iters, int bench_iters) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // cuBLAS uses raw pointers, not our Tensor class
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));
    
    // Generate random data
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 12345);
    curandGenerateUniform(gen, d_A, M * K);
    curandGenerateUniform(gen, d_B, K * N);
    
    // cuBLAS GEMM: C = alpha * A * B + beta * C
    // alpha=1.0, beta=0.0 means C = A * B
    float alpha = 1.0f, beta = 0.0f;
    CudaTimer timer;
    
    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
        // Note: cuBLAS uses column-major, so we swap M and N
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &alpha,
                    d_B, N, d_A, K,
                    &beta, d_C, N);
    }
    cudaDeviceSynchronize();
    
    // Benchmark
    timer.start();
    for (int i = 0; i < bench_iters; i++) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &alpha,
                    d_B, N, d_A, K,
                    &beta, d_C, N);
    }
    cudaDeviceSynchronize();
    float time_ms = timer.stop() / bench_iters;
    
    BenchmarkResult result;
    result.M = M;
    result.N = N;
    result.K = K;
    result.time_ms = time_ms;
    result.gflops = calculateGFLOPS(M, N, K, time_ms);
    result.bandwidth_gb_s = calculateBandwidth(M, N, K, time_ms, sizeof(float));
    result.kernel_name = "cuBLAS_SGEMM";
    result.dtype = "FP32";
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    curandDestroyGenerator(gen);
    cublasDestroy(handle);
    
    return result;
}

// Convert FP32 to FP16 for TensorCore (TensorCores need FP16 input)
__global__ void convert_fp32_to_fp16_kernel(float* in, __half* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half(in[idx]);  // CUDA's built-in conversion
    }
}

// Helper kernel to convert FP16 to FP32
__global__ void convert_fp16_to_fp32_kernel(__half* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __half2float(in[idx]);  // CUDA's built-in conversion
    }
}

// Benchmark cuBLAS HGEMM (FP16 TensorCore path)
// TensorCores are way faster than regular FP32 cores
// Uses FP16 input, FP32 accumulation (mixed precision)
BenchmarkResult benchmarkCublasHGEMM(int M, int N, int K, int warmup_iters, int bench_iters) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    // Enable TensorCore mode
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    
    __half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(__half));
    cudaMalloc(&d_B, K * N * sizeof(__half));
    cudaMalloc(&d_C, M * N * sizeof(__half));
    
    float *d_A_fp32, *d_B_fp32;
    cudaMalloc(&d_A_fp32, M * K * sizeof(float));
    cudaMalloc(&d_B_fp32, K * N * sizeof(float));
    
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 12345);
    curandGenerateUniform(gen, d_A_fp32, M * K);
    curandGenerateUniform(gen, d_B_fp32, K * N);
    
    int threads = 256;
    int blocks_A = (M * K + threads - 1) / threads;
    int blocks_B = (K * N + threads - 1) / threads;
    
    convert_fp32_to_fp16_kernel<<<blocks_A, threads>>>(d_A_fp32, d_A, M * K);
    convert_fp32_to_fp16_kernel<<<blocks_B, threads>>>(d_B_fp32, d_B, K * N);
    cudaDeviceSynchronize();
    
    __half alpha = __float2half(1.0f), beta = __float2half(0.0f);
    CudaTimer timer;
    
    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
        cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &alpha,
                    d_B, N, d_A, K,
                    &beta, d_C, N);
    }
    cudaDeviceSynchronize();
    
    // Benchmark
    timer.start();
    for (int i = 0; i < bench_iters; i++) {
        cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &alpha,
                    d_B, N, d_A, K,
                    &beta, d_C, N);
    }
    cudaDeviceSynchronize();
    float time_ms = timer.stop() / bench_iters;
    
    BenchmarkResult result;
    result.M = M;
    result.N = N;
    result.K = K;
    result.time_ms = time_ms;
    result.gflops = calculateGFLOPS(M, N, K, time_ms);
    result.bandwidth_gb_s = calculateBandwidth(M, N, K, time_ms, sizeof(__half));
    result.kernel_name = "cuBLAS_HGEMM_TensorCore";
    result.dtype = "FP16";
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_A_fp32);
    cudaFree(d_B_fp32);
    curandDestroyGenerator(gen);
    cublasDestroy(handle);
    
    return result;
}

// Benchmark our TensorCore GEMM implementation
BenchmarkResult benchmarkTensorCoreGEMM(int M, int N, int K, int warmup_iters, int bench_iters) {
    // Allocate FP16 inputs, FP32 output
    Tensor<__half> A{M, K, true};
    Tensor<__half> B{K, N, true};
    Tensor<float> C{M, N, true};
    
    // Initialize C to zero (needed for correctness)
    CUDA_OK(cudaMemset(C.rawp, 0, M * N * sizeof(float)));
    
    // Generate random data in FP32, then convert to FP16
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
    convert_fp32_to_fp16_kernel<<<blocks_A, threads>>>(A_fp32.rawp, A.rawp, M * K);
    convert_fp32_to_fp16_kernel<<<blocks_B, threads>>>(B_fp32.rawp, B.rawp, K * N);
    CUDA_OK(cudaDeviceSynchronize());
    
    CudaTimer timer;
    
    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
        op_mm_tensorcore(A, B, C);
    }
    CUDA_OK(cudaDeviceSynchronize());
    
    // Benchmark
    timer.start();
    for (int i = 0; i < bench_iters; i++) {
        op_mm_tensorcore(A, B, C);
    }
    CUDA_OK(cudaDeviceSynchronize());
    float time_ms = timer.stop() / bench_iters;
    
    BenchmarkResult result;
    result.M = M;
    result.N = N;
    result.K = K;
    result.time_ms = time_ms;
    result.gflops = calculateGFLOPS(M, N, K, time_ms);
    result.bandwidth_gb_s = calculateBandwidth(M, N, K, time_ms, sizeof(__half));
    result.kernel_name = "Lab2_TensorCore";
    result.dtype = "FP16";
    
    curandDestroyGenerator(gen);
    
    return result;
}

// Benchmark optimized TensorCore GEMM (with double buffering)
BenchmarkResult benchmarkTensorCoreOptimizedGEMM(int M, int N, int K, int warmup_iters, int bench_iters) {
    Tensor<__half> A{M, K, true};
    Tensor<__half> B{K, N, true};
    Tensor<float> C{M, N, true};
    
    CUDA_OK(cudaMemset(C.rawp, 0, M * N * sizeof(float)));
    
    Tensor<float> A_fp32{M, K, true};
    Tensor<float> B_fp32{K, N, true};
    
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 12345);
    curandGenerateUniform(gen, A_fp32.rawp, M * K);
    curandGenerateUniform(gen, B_fp32.rawp, K * N);
    
    int threads = 256;
    int blocks_A = (M * K + threads - 1) / threads;
    int blocks_B = (K * N + threads - 1) / threads;
    convert_fp32_to_fp16_kernel<<<blocks_A, threads>>>(A_fp32.rawp, A.rawp, M * K);
    convert_fp32_to_fp16_kernel<<<blocks_B, threads>>>(B_fp32.rawp, B.rawp, K * N);
    CUDA_OK(cudaDeviceSynchronize());
    
    CudaTimer timer;
    
    for (int i = 0; i < warmup_iters; i++) {
        op_mm_tensorcore_optimized(A, B, C);
    }
    CUDA_OK(cudaDeviceSynchronize());
    
    timer.start();
    for (int i = 0; i < bench_iters; i++) {
        op_mm_tensorcore_optimized(A, B, C);
    }
    CUDA_OK(cudaDeviceSynchronize());
    float time_ms = timer.stop() / bench_iters;
    
    BenchmarkResult result;
    result.M = M;
    result.N = N;
    result.K = K;
    result.time_ms = time_ms;
    result.gflops = calculateGFLOPS(M, N, K, time_ms);
    result.bandwidth_gb_s = calculateBandwidth(M, N, K, time_ms, sizeof(__half));
    result.kernel_name = "Lab2_TensorCore_Optimized";
    result.dtype = "FP16";
    
    curandDestroyGenerator(gen);
    
    return result;
}

// 3-stage pipelined TensorCore GEMM - DISABLED
// This had a bug that made it 54% slower, so I disabled it for now
/*
BenchmarkResult benchmarkTensorCore3StageGEMM(int M, int N, int K, int warmup_iters, int bench_iters) {
    // Allocate FP16 input matrices and FP32 output
    Tensor<__half> A{M, K, true};
    Tensor<__half> B{K, N, true};
    Tensor<float> C{M, N, true};
    
    // Initialize C to zero (important for correctness)
    CUDA_OK(cudaMemset(C.rawp, 0, M * N * sizeof(float)));
    
    // Initialize with random data (convert FP32 to FP16)
    Tensor<float> A_fp32{M, K, true};
    Tensor<float> B_fp32{K, N, true};
    
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 12345);
    curandGenerateUniform(gen, A_fp32.rawp, M * K);
    curandGenerateUniform(gen, B_fp32.rawp, K * N);
    
    // Convert FP32 to FP16
    int threads = 256;
    int blocks_A = (M * K + threads - 1) / threads;
    int blocks_B = (K * N + threads - 1) / threads;
    convert_fp32_to_fp16_kernel<<<blocks_A, threads>>>(A_fp32.rawp, A.rawp, M * K);
    convert_fp32_to_fp16_kernel<<<blocks_B, threads>>>(B_fp32.rawp, B.rawp, K * N);
    CUDA_OK(cudaDeviceSynchronize());
    
    CudaTimer timer;
    
    // Warmup runs
    for (int i = 0; i < warmup_iters; i++) {
        op_mm_tensorcore_3stage(A, B, C);
    }
    CUDA_OK(cudaDeviceSynchronize());
    
    // Benchmark runs
    timer.start();
    for (int i = 0; i < bench_iters; i++) {
        op_mm_tensorcore_3stage(A, B, C);
    }
    CUDA_OK(cudaDeviceSynchronize());
    float time_ms = timer.stop() / bench_iters;
    
    BenchmarkResult result;
    result.M = M;
    result.N = N;
    result.K = K;
    result.time_ms = time_ms;
    result.gflops = calculateGFLOPS(M, N, K, time_ms);
    result.bandwidth_gb_s = calculateBandwidth(M, N, K, time_ms, sizeof(__half));
    result.kernel_name = "Lab2_TensorCore_3Stage";
    result.dtype = "FP16";
    
    curandDestroyGenerator(gen);
    
    return result;
}
*/

// Benchmark high-performance TensorCore GEMM (8 warps, larger tiles)
BenchmarkResult benchmarkTensorCoreHighPerfGEMM(int M, int N, int K, int warmup_iters, int bench_iters) {
    std::cerr << "[DEBUG] benchmarkTensorCoreHighPerfGEMM called for M=" << M << " N=" << N << " K=" << K << std::endl;
    Tensor<__half> A{M, K, true};
    Tensor<__half> B{K, N, true};
    Tensor<float> C{M, N, true};
    
    CUDA_OK(cudaMemset(C.rawp, 0, M * N * sizeof(float)));
    
    Tensor<float> A_fp32{M, K, true};
    Tensor<float> B_fp32{K, N, true};
    
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 12345);
    curandGenerateUniform(gen, A_fp32.rawp, M * K);
    curandGenerateUniform(gen, B_fp32.rawp, K * N);
    
    int threads = 256;
    int blocks_A = (M * K + threads - 1) / threads;
    int blocks_B = (K * N + threads - 1) / threads;
    convert_fp32_to_fp16_kernel<<<blocks_A, threads>>>(A_fp32.rawp, A.rawp, M * K);
    convert_fp32_to_fp16_kernel<<<blocks_B, threads>>>(B_fp32.rawp, B.rawp, K * N);
    CUDA_OK(cudaDeviceSynchronize());
    
    CudaTimer timer;
    
    for (int i = 0; i < warmup_iters; i++) {
        op_mm_tensorcore_high_perf(A, B, C);
    }
    CUDA_OK(cudaDeviceSynchronize());
    
    timer.start();
    for (int i = 0; i < bench_iters; i++) {
        op_mm_tensorcore_high_perf(A, B, C);
    }
    CUDA_OK(cudaDeviceSynchronize());
    float time_ms = timer.stop() / bench_iters;
    
    BenchmarkResult result;
    result.M = M;
    result.N = N;
    result.K = K;
    result.time_ms = time_ms;
    result.gflops = calculateGFLOPS(M, N, K, time_ms);
    result.bandwidth_gb_s = calculateBandwidth(M, N, K, time_ms, sizeof(__half));
    result.kernel_name = "Lab2_TensorCore_HighPerf";
    result.dtype = "FP16";
    
    curandDestroyGenerator(gen);
    
    return result;
}

// Benchmark aggressively optimized TensorCore GEMM (vectorized loads)
BenchmarkResult benchmarkTensorCoreAggressiveGEMM(int M, int N, int K, int warmup_iters, int bench_iters) {
    Tensor<__half> A{M, K, true};
    Tensor<__half> B{K, N, true};
    Tensor<float> C{M, N, true};
    
    CUDA_OK(cudaMemset(C.rawp, 0, M * N * sizeof(float)));
    
    Tensor<float> A_fp32{M, K, true};
    Tensor<float> B_fp32{K, N, true};
    
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 12345);
    curandGenerateUniform(gen, A_fp32.rawp, M * K);
    curandGenerateUniform(gen, B_fp32.rawp, K * N);
    
    int threads = 256;
    int blocks_A = (M * K + threads - 1) / threads;
    int blocks_B = (K * N + threads - 1) / threads;
    convert_fp32_to_fp16_kernel<<<blocks_A, threads>>>(A_fp32.rawp, A.rawp, M * K);
    convert_fp32_to_fp16_kernel<<<blocks_B, threads>>>(B_fp32.rawp, B.rawp, K * N);
    CUDA_OK(cudaDeviceSynchronize());
    
    CudaTimer timer;
    
    for (int i = 0; i < warmup_iters; i++) {
        op_mm_tensorcore_aggressive(A, B, C);
    }
    CUDA_OK(cudaDeviceSynchronize());
    
    timer.start();
    for (int i = 0; i < bench_iters; i++) {
        op_mm_tensorcore_aggressive(A, B, C);
    }
    CUDA_OK(cudaDeviceSynchronize());
    float time_ms = timer.stop() / bench_iters;
    
    BenchmarkResult result;
    result.M = M;
    result.N = N;
    result.K = K;
    result.time_ms = time_ms;
    result.gflops = calculateGFLOPS(M, N, K, time_ms);
    result.bandwidth_gb_s = calculateBandwidth(M, N, K, time_ms, sizeof(__half));
    result.kernel_name = "Lab2_TensorCore_Aggressive";
    result.dtype = "FP16";
    
    curandDestroyGenerator(gen);
    
    return result;
}

// Benchmark large tile TensorCore GEMM (64x64 tiles)
BenchmarkResult benchmarkTensorCoreLargeTileGEMM(int M, int N, int K, int warmup_iters, int bench_iters) {
    Tensor<__half> A{M, K, true};
    Tensor<__half> B{K, N, true};
    Tensor<float> C{M, N, true};
    
    CUDA_OK(cudaMemset(C.rawp, 0, M * N * sizeof(float)));
    
    Tensor<float> A_fp32{M, K, true};
    Tensor<float> B_fp32{K, N, true};
    
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 12345);
    curandGenerateUniform(gen, A_fp32.rawp, M * K);
    curandGenerateUniform(gen, B_fp32.rawp, K * N);
    
    int threads = 256;
    int blocks_A = (M * K + threads - 1) / threads;
    int blocks_B = (K * N + threads - 1) / threads;
    convert_fp32_to_fp16_kernel<<<blocks_A, threads>>>(A_fp32.rawp, A.rawp, M * K);
    convert_fp32_to_fp16_kernel<<<blocks_B, threads>>>(B_fp32.rawp, B.rawp, K * N);
    CUDA_OK(cudaDeviceSynchronize());
    
    CudaTimer timer;
    
    for (int i = 0; i < warmup_iters; i++) {
        op_mm_tensorcore_large_tile(A, B, C);
    }
    CUDA_OK(cudaDeviceSynchronize());
    
    timer.start();
    for (int i = 0; i < bench_iters; i++) {
        op_mm_tensorcore_large_tile(A, B, C);
    }
    CUDA_OK(cudaDeviceSynchronize());
    float time_ms = timer.stop() / bench_iters;
    
    BenchmarkResult result;
    result.M = M;
    result.N = N;
    result.K = K;
    result.time_ms = time_ms;
    result.gflops = calculateGFLOPS(M, N, K, time_ms);
    result.bandwidth_gb_s = calculateBandwidth(M, N, K, time_ms, sizeof(__half));
    result.kernel_name = "Lab2_TensorCore_LargeTile";
    result.dtype = "FP16";
    
    curandDestroyGenerator(gen);
    
    return result;
}

// Benchmark ultra-optimized TensorCore GEMM (minimal overhead)
BenchmarkResult benchmarkTensorCoreUltraGEMM(int M, int N, int K, int warmup_iters, int bench_iters) {
    Tensor<__half> A{M, K, true};
    Tensor<__half> B{K, N, true};
    Tensor<float> C{M, N, true};
    
    CUDA_OK(cudaMemset(C.rawp, 0, M * N * sizeof(float)));
    
    Tensor<float> A_fp32{M, K, true};
    Tensor<float> B_fp32{K, N, true};
    
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 12345);
    curandGenerateUniform(gen, A_fp32.rawp, M * K);
    curandGenerateUniform(gen, B_fp32.rawp, K * N);
    
    int threads = 256;
    int blocks_A = (M * K + threads - 1) / threads;
    int blocks_B = (K * N + threads - 1) / threads;
    convert_fp32_to_fp16_kernel<<<blocks_A, threads>>>(A_fp32.rawp, A.rawp, M * K);
    convert_fp32_to_fp16_kernel<<<blocks_B, threads>>>(B_fp32.rawp, B.rawp, K * N);
    CUDA_OK(cudaDeviceSynchronize());
    
    CudaTimer timer;
    
    for (int i = 0; i < warmup_iters; i++) {
        op_mm_tensorcore_ultra(A, B, C);
    }
    CUDA_OK(cudaDeviceSynchronize());
    
    timer.start();
    for (int i = 0; i < bench_iters; i++) {
        op_mm_tensorcore_ultra(A, B, C);
    }
    CUDA_OK(cudaDeviceSynchronize());
    float time_ms = timer.stop() / bench_iters;
    
    BenchmarkResult result;
    result.M = M;
    result.N = N;
    result.K = K;
    result.time_ms = time_ms;
    result.gflops = calculateGFLOPS(M, N, K, time_ms);
    result.bandwidth_gb_s = calculateBandwidth(M, N, K, time_ms, sizeof(__half));
    result.kernel_name = "Lab2_TensorCore_Ultra";
    result.dtype = "FP16";
    
    curandDestroyGenerator(gen);
    
    return result;
}

// Validate our TensorCore kernel against cuBLAS
bool validateTensorCoreCorrectness(int M, int N, int K, float tolerance = 1e-3f) {
    std::cout << "\nValidating TensorCore Correctness" << std::endl;
    std::cout << "Matrix size: " << M << "x" << N << "x" << K << std::endl;
    
    // Allocate matrices
    Tensor<__half> A{M, K, true};
    Tensor<__half> B{K, N, true};
    Tensor<float> C_tensorcore{M, N, true};
    Tensor<float> C_cublas{M, N, true};
    
    // Initialize output matrices to zero
    CUDA_OK(cudaMemset(C_tensorcore.rawp, 0, M * N * sizeof(float)));
    CUDA_OK(cudaMemset(C_cublas.rawp, 0, M * N * sizeof(float)));
    
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
    convert_fp32_to_fp16_kernel<<<blocks_A, threads>>>(A_fp32.rawp, A.rawp, M * K);
    convert_fp32_to_fp16_kernel<<<blocks_B, threads>>>(B_fp32.rawp, B.rawp, K * N);
    CUDA_OK(cudaDeviceSynchronize());
    
    // Run TensorCore GEMM
    op_mm_tensorcore(A, B, C_tensorcore);
    
    // Run cuBLAS TensorCore GEMM for comparison
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    
    // cuBLAS HGEMM outputs FP16, so we need FP16 output tensor
    Tensor<__half> C_cublas_fp16{M, N, true};
    
    __half alpha = __float2half(1.0f), beta = __float2half(0.0f);
    cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K, &alpha,
                B.rawp, N, A.rawp, K,
                &beta, C_cublas_fp16.rawp, N);
    CUDA_OK(cudaDeviceSynchronize());
    
    // Convert cuBLAS FP16 output to FP32 for comparison
    int blocks_C = (M * N + threads - 1) / threads;
    convert_fp16_to_fp32_kernel<<<blocks_C, threads>>>(C_cublas_fp16.rawp, C_cublas.rawp, M * N);
    CUDA_OK(cudaDeviceSynchronize());
    
    // Compare results
    Tensor<float> C_tensorcore_host{M, N, false};
    Tensor<float> C_cublas_host{M, N, false};
    C_tensorcore.toHost(C_tensorcore_host);
    C_cublas.toHost(C_cublas_host);
    
    int errors = 0;
    float max_diff = 0.0f;
    float max_rel_diff = 0.0f;
    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float tc_val = Index(C_tensorcore_host, i, j);
            float cublas_val = Index(C_cublas_host, i, j);
            float diff = std::abs(tc_val - cublas_val);
            float rel_diff = (cublas_val != 0.0f) ? diff / std::abs(cublas_val) : diff;
            
            max_diff = std::max(max_diff, diff);
            max_rel_diff = std::max(max_rel_diff, rel_diff);
            
            if (diff > tolerance * std::max(std::abs(tc_val), std::abs(cublas_val))) {
                errors++;
                if (errors <= 5) {  // Print first 5 errors
                    std::cout << "  Mismatch at (" << i << "," << j << "): "
                              << "TensorCore=" << tc_val << ", cuBLAS=" << cublas_val
                              << ", diff=" << diff << std::endl;
                }
            }
        }
    }
    
    std::cout << "Max absolute difference: " << max_diff << std::endl;
    std::cout << "Max relative difference: " << max_rel_diff << std::endl;
    std::cout << "Total mismatches: " << errors << " / " << (M * N) << std::endl;
    
    bool passed = (errors == 0);
    if (passed) {
        std::cout << "Validation PASSED (tolerance: " << tolerance << ")" << std::endl;
    } else {
        std::cout << "Validation FAILED (tolerance: " << tolerance << ")" << std::endl;
    }
    
    cublasDestroy(handle);
    curandDestroyGenerator(gen);
    
    return passed;
}

void printResult(const BenchmarkResult& r, std::ostream& out) {
    out << std::setw(15) << r.kernel_name
        << std::setw(8) << r.dtype
        << std::setw(6) << r.M
        << std::setw(6) << r.N
        << std::setw(6) << r.K
        << std::setw(12) << std::fixed << std::setprecision(3) << r.time_ms
        << std::setw(12) << std::fixed << std::setprecision(2) << r.gflops
        << std::setw(15) << std::fixed << std::setprecision(2) << r.bandwidth_gb_s
        << std::endl;
}

void saveResultsCSV(const std::vector<BenchmarkResult>& results, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << filename << std::endl;
        return;
    }
    
    file << "Kernel,DType,M,N,K,Time_ms,GFLOPS,Bandwidth_GB_s" << std::endl;
    for (const auto& r : results) {
        file << r.kernel_name << ","
             << r.dtype << ","
             << r.M << ","
             << r.N << ","
             << r.K << ","
             << std::fixed << std::setprecision(4) << r.time_ms << ","
             << std::fixed << std::setprecision(2) << r.gflops << ","
             << std::fixed << std::setprecision(2) << r.bandwidth_gb_s
             << std::endl;
    }
    file.close();
    std::cout << "Results saved to " << filename << std::endl;
}

int main() {
    // Main benchmarking loop - tests different matrix sizes
    std::cout << "GEMM Benchmark Suite" << std::endl;
    std::cout << std::endl;
    
    // Print GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << std::endl;
    
    // Test configurations: square matrices and some rectangular ones
    std::vector<std::tuple<int, int, int>> configs = {
        {128, 128, 128},      // Small square
        {256, 256, 256},
        {512, 512, 512},
        {1024, 1024, 1024},   // Medium square
        {2048, 2048, 2048},
        {4096, 4096, 4096},   // Large square
        {8192, 8192, 8192},   // Very large square
        {4096, 256, 1024},    // Rectangular (common in transformers)
        {1024, 4096, 512},
        {2048, 512, 2048},
        {512, 2048, 512}
    };
    
    // Warmup and benchmark iteration counts
    int warmup_iters = 5;
    int bench_iters = 20;
    
    std::vector<BenchmarkResult> all_results;
    
    std::cout << std::setw(15) << "Kernel"
              << std::setw(8) << "DType"
              << std::setw(6) << "M"
              << std::setw(6) << "N"
              << std::setw(6) << "K"
              << std::setw(12) << "Time(ms)"
              << std::setw(12) << "GFLOPS"
              << std::setw(15) << "Bandwidth(GB/s)"
              << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    for (const auto& config : configs) {
        int M, N, K;
        std::tie(M, N, K) = config;
        
        std::cout << "\nTesting M=" << M << ", N=" << N << ", K=" << K << std::endl;
        
        try {
            auto result = benchmarkLab1GEMM(M, N, K, warmup_iters, bench_iters);
            printResult(result, std::cout);
            all_results.push_back(result);
        } catch (const std::exception& e) {
            std::cerr << "Lab-1 GEMM failed: " << e.what() << std::endl;
        }
        
        try {
            auto result = benchmarkCublasSGEMM(M, N, K, warmup_iters, bench_iters);
            printResult(result, std::cout);
            all_results.push_back(result);
        } catch (const std::exception& e) {
            std::cerr << "cuBLAS SGEMM failed: " << e.what() << std::endl;
        }
        
        if (prop.major >= 7) {
            try {
                auto result = benchmarkCublasHGEMM(M, N, K, warmup_iters, bench_iters);
                printResult(result, std::cout);
                all_results.push_back(result);
            } catch (const std::exception& e) {
                std::cerr << "cuBLAS HGEMM failed: " << e.what() << std::endl;
            }
            
            // Our TensorCore implementation
            try {
                auto result = benchmarkTensorCoreGEMM(M, N, K, warmup_iters, bench_iters);
                printResult(result, std::cout);
                all_results.push_back(result);
            } catch (const std::exception& e) {
                std::cerr << "TensorCore GEMM failed: " << e.what() << std::endl;
            }
            
            // Optimized version (double buffering)
            try {
                auto result = benchmarkTensorCoreOptimizedGEMM(M, N, K, warmup_iters, bench_iters);
                printResult(result, std::cout);
                all_results.push_back(result);
            } catch (const std::exception& e) {
                std::cerr << "Optimized TensorCore GEMM failed: " << e.what() << std::endl;
            }
            
            // Large tile version
            try {
                auto result = benchmarkTensorCoreLargeTileGEMM(M, N, K, warmup_iters, bench_iters);
                printResult(result, std::cout);
                all_results.push_back(result);
            } catch (const std::exception& e) {
                std::cerr << "Large Tile TensorCore GEMM failed: " << e.what() << std::endl;
            }
            
            // High-performance version (8 warps, larger tiles, optimized memory access)
            std::cout << "[DEBUG] About to call HighPerf for M=" << M << " N=" << N << " K=" << K << std::endl;
            try {
                std::cout << "[DEBUG] Starting HighPerf benchmark for M=" << M << " N=" << N << " K=" << K << std::endl;
                auto result = benchmarkTensorCoreHighPerfGEMM(M, N, K, warmup_iters, bench_iters);
                std::cout << "[DEBUG] HighPerf benchmark completed, GFLOPS=" << result.gflops << std::endl;
                printResult(result, std::cout);
                all_results.push_back(result);
            } catch (const std::exception& e) {
                std::cerr << "High-Perf TensorCore GEMM failed for M=" << M << " N=" << N << " K=" << K 
                          << ": " << e.what() << std::endl;
            } catch (...) {
                std::cerr << "High-Perf TensorCore GEMM failed for M=" << M << " N=" << N << " K=" << K 
                          << ": Unknown error" << std::endl;
            }
            
            // Aggressively optimized version (vectorized loads)
            try {
                auto result = benchmarkTensorCoreAggressiveGEMM(M, N, K, warmup_iters, bench_iters);
                printResult(result, std::cout);
                all_results.push_back(result);
            } catch (const std::exception& e) {
                std::cerr << "Aggressive TensorCore GEMM failed for M=" << M << " N=" << N << " K=" << K 
                          << ": " << e.what() << std::endl;
            } catch (...) {
                std::cerr << "Aggressive TensorCore GEMM failed for M=" << M << " N=" << N << " K=" << K 
                          << ": Unknown error" << std::endl;
            }
            
            // Ultra-optimized version (minimal overhead)
            try {
                auto result = benchmarkTensorCoreUltraGEMM(M, N, K, warmup_iters, bench_iters);
                printResult(result, std::cout);
                all_results.push_back(result);
            } catch (const std::exception& e) {
                std::cerr << "Ultra TensorCore GEMM failed for M=" << M << " N=" << N << " K=" << K 
                          << ": " << e.what() << std::endl;
            } catch (...) {
                std::cerr << "Ultra TensorCore GEMM failed for M=" << M << " N=" << N << " K=" << K 
                          << ": Unknown error" << std::endl;
            }
            
            // 64×64 tile version (16 warps, larger tiles)
            try {
                auto result = benchmarkTensorCore64x64GEMM(M, N, K, warmup_iters, bench_iters);
                printResult(result, std::cout);
                all_results.push_back(result);
            } catch (const std::exception& e) {
                std::cerr << "64×64 TensorCore GEMM failed for M=" << M << " N=" << N << " K=" << K 
                          << ": " << e.what() << std::endl;
            } catch (...) {
                std::cerr << "64×64 TensorCore GEMM failed for M=" << M << " N=" << N << " K=" << K 
                          << ": Unknown error" << std::endl;
            }
        }
    }
    
    saveResultsCSV(all_results, "results/benchmark_results.csv");
    
    std::cout << "\nBenchmark completed!" << std::endl;
    
    // Validate correctness
    if (prop.major >= 7) {
        std::cout << "\nValidating TensorCore Correctness" << std::endl;
        
        std::vector<std::tuple<int, int, int>> test_sizes = {
            {128, 128, 128},
            {512, 512, 512},
            {1024, 1024, 1024}
        };
        
        int passed = 0;
        int total = test_sizes.size();
        
        for (const auto& size : test_sizes) {
            int M, N, K;
            std::tie(M, N, K) = size;
            if (validateTensorCoreCorrectness(M, N, K, 1e-2f)) {
                passed++;
            }
        }
        
        std::cout << "\nValidation Summary: " << passed << " / " << total << " tests passed" << std::endl;
    }
    return 0;
}