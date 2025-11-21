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
#include "ops/op_mm_tensorcore.cuh"
#include "ops/op_elemwise.cuh"

// Phase 1.1.3: Set up benchmarking framework
// This file implements the timing harness for measuring FLOPS, memory bandwidth, and latency
// We benchmark both our Lab-1 kernel and cuBLAS baselines

unsigned long long randgen_seed = 12345;

// Timing utility - uses CUDA events for accurate GPU timing
// CUDA events are more accurate than CPU timers because they measure GPU execution time
// directly, accounting for async execution and avoiding CPU-GPU synchronization overhead
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

// Performance metrics structure - stores results for each benchmark run
struct BenchmarkResult {
    int M, N, K;              // Matrix dimensions
    float time_ms;            // Execution time in milliseconds
    double gflops;            // Giga-FLOPS achieved
    double bandwidth_gb_s;    // Memory bandwidth in GB/s
    double efficiency_percent; // Efficiency vs peak (calculated later)
    std::string kernel_name;  // Which kernel was tested
    std::string dtype;        // Data type (FP32 or FP16)
};

// Calculate GFLOPS (Giga Floating Point Operations Per Second)
// GEMM operation count: 2*M*N*K (each element of C requires K multiply-adds = 2 ops)
double calculateGFLOPS(int M, int N, int K, float time_ms) {
    double operations = 2.0 * M * N * K;  // Total FLOPS
    double gflops = (operations / 1e9) / (time_ms / 1000.0);  // Convert to GFLOPS
    return gflops;
}

// Calculate memory bandwidth utilization
// Total bytes transferred = A matrix + B matrix + C matrix (read A, read B, write C)
double calculateBandwidth(int M, int N, int K, float time_ms, int bytes_per_element) {
    double bytes = (double)(M * K + K * N + M * N) * bytes_per_element;
    double bandwidth = (bytes / 1e9) / (time_ms / 1000.0);  // GB/s
    return bandwidth;
}

// Phase 1.1.3: Benchmark Lab-1 tiled GEMM implementation
// This measures our baseline kernel performance across different matrix sizes
BenchmarkResult benchmarkLab1GEMM(int M, int N, int K, int warmup_iters, int bench_iters) {
    // Allocate matrices on GPU (on_device=true)
    Tensor<float> A{M, K, true};
    Tensor<float> B{K, N, true};
    Tensor<float> C{M, N, true};
    
    // Initialize with random data (uniform distribution 0-1)
    op_uniform_fill(A, 0.0f, 1.0f);
    op_uniform_fill(B, 0.0f, 1.0f);
    
    // Synchronize after initialization to ensure data is ready
    CUDA_OK(cudaDeviceSynchronize());
    
    CudaTimer timer;
    
    // Warmup runs - GPU needs a few iterations to "warm up" (cache, clock speeds, etc.)
    // This ensures consistent timing for the actual benchmark
    for (int i = 0; i < warmup_iters; i++) {
        op_mm(A, B, C);
    }
    CUDA_OK(cudaDeviceSynchronize());
    
    // Benchmark - measure total time including synchronization
    // We run multiple iterations and average to get more stable results
    timer.start();
    for (int i = 0; i < bench_iters; i++) {
        op_mm(A, B, C);
    }
    CUDA_OK(cudaDeviceSynchronize()); // Critical: sync before stopping timer
    float time_ms = timer.stop() / bench_iters;  // Average time per iteration
    
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

// Phase 1.2.1: Benchmark cuBLAS SGEMM (FP32) - our performance target
// cuBLAS is NVIDIA's highly optimized library, so this represents the "best case" performance
// We use this as a baseline to compare against our implementations
BenchmarkResult benchmarkCublasSGEMM(int M, int N, int K, int warmup_iters, int bench_iters) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // Allocate device memory directly (cuBLAS uses raw pointers, not our Tensor class)
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));
    
    // Generate random data using cuRAND (CUDA's random number generator)
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 12345);
    curandGenerateUniform(gen, d_A, M * K);
    curandGenerateUniform(gen, d_B, K * N);
    
    // cuBLAS GEMM parameters: C = alpha * A * B + beta * C
    // alpha=1.0, beta=0.0 means C = A * B
    float alpha = 1.0f, beta = 0.0f;
    CudaTimer timer;
    
    // Warmup runs
    for (int i = 0; i < warmup_iters; i++) {
        // Note: cuBLAS uses column-major format, so we swap M and N
        // CUBLAS_OP_N means no transpose
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &alpha,
                    d_B, N, d_A, K,
                    &beta, d_C, N);
    }
    cudaDeviceSynchronize();
    
    // Benchmark - same pattern as Lab-1
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

// Helper kernel to convert FP32 to FP16 for TensorCore operations
// TensorCores require FP16 input but can accumulate in FP32
__global__ void convert_fp32_to_fp16_kernel(float* in, __half* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half(in[idx]);  // CUDA's built-in conversion
    }
}

// Phase 1.2.1: Benchmark cuBLAS HGEMM (FP16 TensorCore path)
// This uses TensorCores which are much faster than regular FP32 cores
// Input: FP16, Accumulation: FP32, Output: FP32 (mixed precision)
BenchmarkResult benchmarkCublasHGEMM(int M, int N, int K, int warmup_iters, int bench_iters) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    // Enable TensorCore math mode - this tells cuBLAS to use TensorCores when possible
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

// Phase 2: Benchmark our TensorCore GEMM implementation
BenchmarkResult benchmarkTensorCoreGEMM(int M, int N, int K, int warmup_iters, int bench_iters) {
    // Allocate FP16 input matrices and FP32 output
    Tensor<__half> A{M, K, true};
    Tensor<__half> B{K, N, true};
    Tensor<float> C{M, N, true};
    
    // Initialize with random data (convert FP32 to FP16)
    Tensor<float> A_fp32{M, K, true};
    Tensor<float> B_fp32{K, N, true};
    
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 12345);
    curandGenerateUniform(gen, A_fp32.data, M * K);
    curandGenerateUniform(gen, B_fp32.data, K * N);
    
    // Convert FP32 to FP16
    int threads = 256;
    int blocks_A = (M * K + threads - 1) / threads;
    int blocks_B = (K * N + threads - 1) / threads;
    convert_fp32_to_fp16_kernel<<<blocks_A, threads>>>(A_fp32.data, A.data, M * K);
    convert_fp32_to_fp16_kernel<<<blocks_B, threads>>>(B_fp32.data, B.data, K * N);
    CUDA_OK(cudaDeviceSynchronize());
    
    CudaTimer timer;
    
    // Warmup runs: Essential to ensure the GPU is fully initialized and clocks are boosted
    for (int i = 0; i < warmup_iters; i++) {
        op_mm_tensorcore(A, B, C);
    }
    CUDA_OK(cudaDeviceSynchronize());
    
    // Benchmark runs: Measure the actual performance
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
    // Phase 1.1.3: Main benchmarking harness
    // Tests both square and rectangular matrices across different sizes
    std::cout << "GEMM Benchmark Suite" << std::endl;
    std::cout << "====================" << std::endl << std::endl;
    
    // Print GPU info for reference
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << std::endl;
    
    // Test matrix configurations:
    // - Square matrices: powers of 2 from 128 to 8192
    // - Rectangular matrices: common shapes in ML workloads (e.g., attention mechanisms)
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
    
    // Iteration counts: warmup to stabilize GPU, then multiple benchmark runs for averaging
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
            
            // Phase 2: Benchmark our TensorCore implementation
            try {
                auto result = benchmarkTensorCoreGEMM(M, N, K, warmup_iters, bench_iters);
                printResult(result, std::cout);
                all_results.push_back(result);
            } catch (const std::exception& e) {
                std::cerr << "TensorCore GEMM failed: " << e.what() << std::endl;
            }
        }
    }
    
    saveResultsCSV(all_results, "results/benchmark_results.csv");
    
    std::cout << "\nBenchmark completed!" << std::endl;
    return 0;
}