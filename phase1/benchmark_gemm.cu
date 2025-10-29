#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "utils/tensor.cuh"
#include "ops/op_mm.cuh"
#include "ops/op_elemwise.cuh"

unsigned long long randgen_seed = 42;

// ============================================================================
// GPU Hardware Analysis
// ============================================================================
struct GPUSpecs {
    std::string name;
    int compute_capability_major;
    int compute_capability_minor;
    size_t total_global_mem;
    int multiprocessor_count;
    int max_threads_per_block;
    int max_threads_per_multiprocessor;
    size_t shared_mem_per_block;
    size_t shared_mem_per_multiprocessor;
    int warp_size;
    int max_registers_per_block;
    size_t l2_cache_size;
    int memory_clock_rate; // kHz
    int memory_bus_width; // bits
    
    // Derived specs
    float peak_fp32_tflops;
    float peak_fp16_tflops;
    float memory_bandwidth_gbps;
    
    void print() const {
        std::cout << "\n=== GPU Hardware Specifications ===" << std::endl;
        std::cout << "Device Name: " << name << std::endl;
        std::cout << "Compute Capability: " << compute_capability_major << "." 
                  << compute_capability_minor << std::endl;
        std::cout << "Global Memory: " << total_global_mem / (1024.0 * 1024.0 * 1024.0) 
                  << " GB" << std::endl;
        std::cout << "Number of SMs: " << multiprocessor_count << std::endl;
        std::cout << "Max Threads/Block: " << max_threads_per_block << std::endl;
        std::cout << "Max Threads/SM: " << max_threads_per_multiprocessor << std::endl;
        std::cout << "Shared Memory/Block: " << shared_mem_per_block / 1024.0 
                  << " KB" << std::endl;
        std::cout << "Shared Memory/SM: " << shared_mem_per_multiprocessor / 1024.0 
                  << " KB" << std::endl;
        std::cout << "L2 Cache Size: " << l2_cache_size / (1024.0 * 1024.0) 
                  << " MB" << std::endl;
        std::cout << "Memory Clock Rate: " << memory_clock_rate / 1e6 << " GHz" << std::endl;
        std::cout << "Memory Bus Width: " << memory_bus_width << " bits" << std::endl;
        std::cout << "\n=== Theoretical Performance ===" << std::endl;
        std::cout << "Peak FP32 Performance: " << peak_fp32_tflops << " TFLOPS" << std::endl;
        std::cout << "Peak FP16 Performance: " << peak_fp16_tflops << " TFLOPS" << std::endl;
        std::cout << "Memory Bandwidth: " << memory_bandwidth_gbps << " GB/s" << std::endl;
        std::cout << "===================================\n" << std::endl;
    }
};

GPUSpecs get_gpu_specs() {
    GPUSpecs specs;
    cudaDeviceProp prop;
    CUDA_OK(cudaGetDeviceProperties(&prop, 0));
    
    specs.name = prop.name;
    specs.compute_capability_major = prop.major;
    specs.compute_capability_minor = prop.minor;
    specs.total_global_mem = prop.totalGlobalMem;
    specs.multiprocessor_count = prop.multiProcessorCount;
    specs.max_threads_per_block = prop.maxThreadsPerBlock;
    specs.max_threads_per_multiprocessor = prop.maxThreadsPerMultiProcessor;
    specs.shared_mem_per_block = prop.sharedMemPerBlock;
    specs.shared_mem_per_multiprocessor = prop.sharedMemPerMultiProcessor;
    specs.warp_size = prop.warpSize;
    specs.max_registers_per_block = prop.regsPerBlock;
    specs.l2_cache_size = prop.l2CacheSize;
    specs.memory_clock_rate = prop.memoryClockRate;
    specs.memory_bus_width = prop.memoryBusWidth;
    
    // Calculate theoretical performance
    // FP32 cores per SM varies by architecture
    int fp32_cores_per_sm;
    if (specs.compute_capability_major == 7 && specs.compute_capability_minor == 5) {
        // T4 (Turing): 64 FP32 cores per SM
        fp32_cores_per_sm = 64;
        // T4 has 320 Tensor Cores (INT8/FP16)
        specs.peak_fp16_tflops = 65.0f; // With Tensor Cores
    } else if (specs.compute_capability_major == 8) {
        // A100: 64 FP32 cores per SM
        fp32_cores_per_sm = 64;
        specs.peak_fp16_tflops = 312.0f; // A100 with Tensor Cores
    } else {
        fp32_cores_per_sm = 64; // Default estimate
        specs.peak_fp16_tflops = 0.0f;
    }
    
    // Clock rate in kHz, convert to GHz
    float clock_ghz = prop.clockRate / 1e6;
    // Peak FLOPS = cores × SMs × clock × 2 (FMA = 2 ops)
    specs.peak_fp32_tflops = (fp32_cores_per_sm * specs.multiprocessor_count * 
                               clock_ghz * 2.0f) / 1000.0f;
    
    // Memory bandwidth = (memory_clock × bus_width × 2) / 8
    // Factor of 2 for DDR, divide by 8 to convert bits to bytes
    specs.memory_bandwidth_gbps = (2.0f * specs.memory_clock_rate * 
                                    specs.memory_bus_width / 8.0f) / 1e6;
    
    return specs;
}

// ============================================================================
// Performance Metrics
// ============================================================================
struct BenchmarkResult {
    std::string kernel_name;
    int m, n, k;
    float time_ms;
    float gflops;
    float memory_bandwidth_gbps;
    float arithmetic_intensity;
    float compute_efficiency; // % of peak
    float memory_efficiency;  // % of peak
    
    void print() const {
        std::cout << std::setw(20) << kernel_name 
                  << " | " << std::setw(5) << m << "×" << std::setw(5) << n << "×" << std::setw(5) << k
                  << " | " << std::setw(8) << std::fixed << std::setprecision(3) << time_ms << " ms"
                  << " | " << std::setw(8) << gflops << " GFLOPS"
                  << " | " << std::setw(6) << std::setprecision(1) << compute_efficiency << "%"
                  << " | AI=" << std::setw(5) << std::setprecision(2) << arithmetic_intensity
                  << std::endl;
    }
    
    std::string csv_header() const {
        return "kernel,m,n,k,time_ms,gflops,bandwidth_gbps,arithmetic_intensity,compute_eff,memory_eff";
    }
    
    std::string to_csv() const {
        std::ostringstream oss;
        oss << kernel_name << "," << m << "," << n << "," << k << ","
            << time_ms << "," << gflops << "," << memory_bandwidth_gbps << ","
            << arithmetic_intensity << "," << compute_efficiency << "," << memory_efficiency;
        return oss.str();
    }
};

// ============================================================================
// Timing Utilities
// ============================================================================
class CudaTimer {
    cudaEvent_t start_, stop_;
public:
    CudaTimer() {
        CUDA_OK(cudaEventCreate(&start_));
        CUDA_OK(cudaEventCreate(&stop_));
    }
    
    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    
    void start() {
        CUDA_OK(cudaEventRecord(start_));
    }
    
    float stop() {
        CUDA_OK(cudaEventRecord(stop_));
        CUDA_OK(cudaEventSynchronize(stop_));
        float ms;
        CUDA_OK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
};

// ============================================================================
// GEMM Benchmarking
// ============================================================================
BenchmarkResult benchmark_gemm(
    const std::string& name,
    std::function<void(const Tensor<float>&, const Tensor<float>&, Tensor<float>&)> kernel,
    int m, int n, int k,
    const GPUSpecs& specs,
    int warmup_iters = 5,
    int bench_iters = 20)
{
    // Create tensors
    Tensor<float> A{m, k, true};
    Tensor<float> B{k, n, true};
    Tensor<float> C{m, n, true};
    
    op_uniform_fill(A, 0.0f, 1.0f);
    op_uniform_fill(B, 0.0f, 1.0f);
    
    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
        kernel(A, B, C);
    }
    CUDA_OK(cudaDeviceSynchronize());
    
    // Benchmark
    CudaTimer timer;
    timer.start();
    for (int i = 0; i < bench_iters; i++) {
        kernel(A, B, C);
    }
    float total_ms = timer.stop();
    float avg_ms = total_ms / bench_iters;
    
    // Calculate metrics
    BenchmarkResult result;
    result.kernel_name = name;
    result.m = m;
    result.n = n;
    result.k = k;
    result.time_ms = avg_ms;
    
    // FLOPS: 2*m*n*k operations (multiply-add)
    double flops = 2.0 * m * n * k;
    result.gflops = (flops / (avg_ms / 1000.0)) / 1e9;
    
    // Memory traffic: read A (m*k), read B (k*n), write C (m*n)
    double bytes = sizeof(float) * (m*k + k*n + m*n);
    result.memory_bandwidth_gbps = (bytes / (avg_ms / 1000.0)) / 1e9;
    
    // Arithmetic intensity: FLOPS / bytes
    result.arithmetic_intensity = flops / bytes;
    
    // Efficiency
    result.compute_efficiency = (result.gflops / specs.peak_fp32_tflops) / 10.0;
    result.memory_efficiency = (result.memory_bandwidth_gbps / specs.memory_bandwidth_gbps) * 100.0;
    
    return result;
}

// ============================================================================
// cuBLAS Wrapper
// ============================================================================
void cublas_gemm(const Tensor<float>& A, const Tensor<float>& B, Tensor<float>& C) {
    static cublasHandle_t handle = nullptr;
    if (handle == nullptr) {
        cublasCreate(&handle);
    }
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // cuBLAS uses column-major, so we compute C^T = B^T * A^T
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                C.w, C.h, A.w,
                &alpha,
                B.rawp, B.w,
                A.rawp, A.w,
                &beta,
                C.rawp, C.w);
}

// ============================================================================
// Test Suite
// ============================================================================
void run_benchmark_suite(const GPUSpecs& specs) {
    std::vector<BenchmarkResult> results;
    
    // Define test cases
    std::vector<std::tuple<int,int,int>> test_cases = {
        // Square matrices
        {128, 128, 128},
        {256, 256, 256},
        {512, 512, 512},
        {1024, 1024, 1024},
        {2048, 2048, 2048},
        {4096, 4096, 4096},
        {8192, 8192, 8192},
        
        // Rectangular matrices (common in ML)
        {4096, 256, 1024},
        {1024, 4096, 512},
        {2048, 1024, 2048},
        {8192, 128, 1024},
        
        // Small matrices
        {64, 64, 64},
        {32, 32, 32},
    };
    
    std::cout << "\n=== Running Benchmark Suite ===" << std::endl;
    std::cout << std::setw(20) << "Kernel" 
              << " | " << std::setw(17) << "Size (M×N×K)"
              << " | " << std::setw(8) << "Time"
              << " | " << std::setw(8) << "GFLOPS"
              << " | " << std::setw(6) << "Eff%"
              << " | " << "AI"
              << std::endl;
    std::cout << std::string(100, '-') << std::endl;
    
    for (const auto& [m, n, k] : test_cases) {
        // Skip very large matrices if insufficient memory
        size_t required_mem = sizeof(float) * (m*k + k*n + m*n);
        if (required_mem > specs.total_global_mem * 0.8) {
            std::cout << "Skipping " << m << "×" << n << "×" << k << " (insufficient memory)" << std::endl;
            continue;
        }
        
        try {
            // Lab-1 kernel
            auto lab1_result = benchmark_gemm("Lab-1 Tiled", op_mm<float>, m, n, k, specs);
            lab1_result.print();
            results.push_back(lab1_result);
            
            // cuBLAS
            auto cublas_result = benchmark_gemm("cuBLAS", cublas_gemm, m, n, k, specs);
            cublas_result.print();
            results.push_back(cublas_result);
            
            std::cout << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error benchmarking " << m << "×" << n << "×" << k << ": " 
                      << e.what() << std::endl;
        }
    }
    
    // Save results to CSV
    std::ofstream csv("benchmark_results.csv");
    csv << results[0].csv_header() << std::endl;
    for (const auto& r : results) {
        csv << r.to_csv() << std::endl;
    }
    csv.close();
    
    std::cout << "\nResults saved to benchmark_results.csv" << std::endl;
}

// ============================================================================
// Roofline Analysis
// ============================================================================
void generate_roofline_data(const GPUSpecs& specs) {
    std::ofstream roofline("roofline_data.csv");
    roofline << "arithmetic_intensity,peak_gflops,memory_bound_gflops" << std::endl;
    
    float peak_gflops = specs.peak_fp32_tflops * 1000.0f;
    float bandwidth_gbps = specs.memory_bandwidth_gbps;
    
    for (float ai = 0.1; ai <= 1000.0; ai *= 1.1) {
        float memory_bound = ai * bandwidth_gbps;
        float actual_gflops = std::min(memory_bound, peak_gflops);
        roofline << ai << "," << peak_gflops << "," << actual_gflops << std::endl;
    }
    
    roofline.close();
    std::cout << "Roofline data saved to roofline_data.csv" << std::endl;
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
    std::cout << "=== Phase 1: GPU GEMM Benchmarking Framework ===" << std::endl;
    
    // Get GPU specifications
    GPUSpecs specs = get_gpu_specs();
    specs.print();
    
    // Generate roofline data
    generate_roofline_data(specs);
    
    // Run benchmark suite
    run_benchmark_suite(specs);
    
    std::cout << "\n=== Benchmark Complete ===" << std::endl;
    std::cout << "Review benchmark_results.csv for detailed performance data" << std::endl;
    std::cout << "Use roofline_data.csv to plot roofline model" << std::endl;
    
    return 0;
}