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
#include "ops/op_elemwise.cuh"

unsigned long long randgen_seed = 12345;

// Timing utility
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

// Performance metrics structure
struct BenchmarkResult {
    int M, N, K;
    float time_ms;
    double gflops;
    double bandwidth_gb_s;
    double efficiency_percent;
    std::string kernel_name;
    std::string dtype;
};

// Calculate GFLOPS
double calculateGFLOPS(int M, int N, int K, float time_ms) {
    // GEMM: 2*M*N*K operations (multiply-add)
    double operations = 2.0 * M * N * K;
    double gflops = (operations / 1e9) / (time_ms / 1000.0);
    return gflops;
}

// Calculate memory bandwidth
double calculateBandwidth(int M, int N, int K, float time_ms, int bytes_per_element) {
    // Memory accesses: read A (M*K), read B (K*N), write C (M*N)
    double bytes = (double)(M * K + K * N + M * N) * bytes_per_element;
    double bandwidth = (bytes / 1e9) / (time_ms / 1000.0);
    return bandwidth;
}

// Benchmark Lab-1 GEMM (FP32)
BenchmarkResult benchmarkLab1GEMM(int M, int N, int K, int warmup_iters, int bench_iters) {
    Tensor<float> A{M, K, true};
    Tensor<float> B{K, N, true};
    Tensor<float> C{M, N, true};
    
    op_uniform_fill(A, 0.0f, 1.0f);
    op_uniform_fill(B, 0.0f, 1.0f);
    
    CudaTimer timer;
    
    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
        op_mm(A, B, C);
    }
    CUDA_OK(cudaDeviceSynchronize());
    
    // Benchmark
    timer.start();
    for (int i = 0; i < bench_iters; i++) {
        op_mm(A, B, C);
    }
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

// Benchmark cuBLAS SGEMM (FP32)
BenchmarkResult benchmarkCublasSGEMM(int M, int N, int K, int warmup_iters, int bench_iters) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));
    
    // Initialize with random data
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 12345);
    curandGenerateUniform(gen, d_A, M * K);
    curandGenerateUniform(gen, d_B, K * N);
    
    float alpha = 1.0f, beta = 0.0f;
    CudaTimer timer;
    
    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
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

// Benchmark cuBLAS with mixed precision (FP16 TensorCore)
__global__ void convert_fp32_to_fp16_kernel(float* in, __half* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half(in[idx]);
    }
}

BenchmarkResult benchmarkCublasHGEMM(int M, int N, int K, int warmup_iters, int bench_iters) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH); // Enable TensorCores
    
    __half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(__half));
    cudaMalloc(&d_B, K * N * sizeof(__half));
    cudaMalloc(&d_C, M * N * sizeof(__half));
    
    // Initialize with random FP32 data, then convert to FP16
    float *d_A_fp32, *d_B_fp32;
    cudaMalloc(&d_A_fp32, M * K * sizeof(float));
    cudaMalloc(&d_B_fp32, K * N * sizeof(float));
    
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 12345);
    curandGenerateUniform(gen, d_A_fp32, M * K);
    curandGenerateUniform(gen, d_B_fp32, K * N);
    
    // Convert FP32 to FP16
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
    std::cout << "GEMM Benchmark Suite" << std::endl;
    std::cout << "====================" << std::endl << std::endl;
    
    // Get GPU specs for calculating efficiency
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << std::endl;
    
    // Test configurations
    std::vector<std::tuple<int, int, int>> configs = {
        // Square matrices
        {128, 128, 128},
        {256, 256, 256},
        {512, 512, 512},
        {1024, 1024, 1024},
        {2048, 2048, 2048},
        {4096, 4096, 4096},
        {8192, 8192, 8192},
        
        // Rectangular matrices
        {4096, 256, 1024},
        {1024, 4096, 512},
        {2048, 512, 2048},
        {512, 2048, 512}
    };
    
    int warmup_iters = 5;
    int bench_iters = 20;
    
    std::vector<BenchmarkResult> all_results;
    
    // Print header
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
        
        // Lab-1 GEMM
        try {
            auto result = benchmarkLab1GEMM(M, N, K, warmup_iters, bench_iters);
            printResult(result, std::cout);
            all_results.push_back(result);
        } catch (const std::exception& e) {
            std::cerr << "Lab-1 GEMM failed: " << e.what() << std::endl;
        }
        
        // cuBLAS SGEMM (FP32)
        try {
            auto result = benchmarkCublasSGEMM(M, N, K, warmup_iters, bench_iters);
            printResult(result, std::cout);
            all_results.push_back(result);
        } catch (const std::exception& e) {
            std::cerr << "cuBLAS SGEMM failed: " << e.what() << std::endl;
        }
        
        // cuBLAS HGEMM (FP16 TensorCore) - only for compute >= 7.0
        if (prop.major >= 7) {
            try {
                auto result = benchmarkCublasHGEMM(M, N, K, warmup_iters, bench_iters);
                printResult(result, std::cout);
                all_results.push_back(result);
            } catch (const std::exception& e) {
                std::cerr << "cuBLAS HGEMM failed: " << e.what() << std::endl;
            }
        }
    }
    
    // Save results
    saveResultsCSV(all_results, "results/benchmark_results.csv");
    
    std::cout << "\nBenchmark completed!" << std::endl;
    return 0;
}