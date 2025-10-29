// benchmark_cutlass.cu - CUTLASS integration for comparison
// Requires CUTLASS to be cloned and built
#include <iostream>
#include <vector>
#include "src/utils/tensor.cuh"
#include "src/ops/op_elemwise.cuh"

// Include CUTLASS headers (if available)
#ifdef USE_CUTLASS
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm.h"
#endif

unsigned long long randgen_seed = 42;

struct CutlassResult {
    std::string config;
    int m, n, k;
    float time_ms;
    float gflops;
    
    void print() const {
        std::cout << config << " | " 
                  << m << "×" << n << "×" << k << " | "
                  << time_ms << " ms | "
                  << gflops << " GFLOPS" << std::endl;
    }
};

#ifdef USE_CUTLASS

// CUTLASS GEMM configuration
// Using standard FP32 GEMM for T4/Turing
using CutlassGemm = cutlass::gemm::device::Gemm<
    float,                           // ElementA
    cutlass::layout::RowMajor,       // LayoutA
    float,                           // ElementB
    cutlass::layout::RowMajor,       // LayoutB
    float,                           // ElementC
    cutlass::layout::RowMajor,       // LayoutC
    float,                           // ElementAccumulator
    cutlass::arch::OpClassSimt,      // Operator class (SIMT for non-TensorCore)
    cutlass::arch::Sm75              // Architecture (Turing/T4)
>;

CutlassResult benchmark_cutlass_gemm(int m, int n, int k, int warmup = 5, int iters = 20) {
    CutlassResult result;
    result.config = "CUTLASS FP32";
    result.m = m;
    result.n = n;
    result.k = k;
    
    // Allocate device memory
    cutlass::HostTensor<float, cutlass::layout::RowMajor> A({m, k});
    cutlass::HostTensor<float, cutlass::layout::RowMajor> B({k, n});
    cutlass::HostTensor<float, cutlass::layout::RowMajor> C({m, n});
    
    // Initialize with random data
    for (int i = 0; i < m * k; i++) A.host_data()[i] = float(rand()) / RAND_MAX;
    for (int i = 0; i < k * n; i++) B.host_data()[i] = float(rand()) / RAND_MAX;
    
    A.sync_device();
    B.sync_device();
    
    // Setup GEMM arguments
    float alpha = 1.0f;
    float beta = 0.0f;
    
    typename CutlassGemm::Arguments args(
        {m, n, k},
        {A.device_data(), k},
        {B.device_data(), n},
        {C.device_data(), n},
        {C.device_data(), n},
        {alpha, beta}
    );
    
    CutlassGemm gemm_op;
    
    // Warmup
    for (int i = 0; i < warmup; i++) {
        cutlass::Status status = gemm_op(args);
        if (status != cutlass::Status::kSuccess) {
            std::cerr << "CUTLASS error: " << cutlassGetStatusString(status) << std::endl;
            result.time_ms = -1.0f;
            result.gflops = 0.0f;
            return result;
        }
    }
    cudaDeviceSynchronize();
    
    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        gemm_op(args);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float total_ms;
    cudaEventElapsedTime(&total_ms, start, stop);
    result.time_ms = total_ms / iters;
    
    // Calculate GFLOPS
    double flops = 2.0 * m * n * k;
    result.gflops = (flops / (result.time_ms / 1000.0)) / 1e9;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return result;
}

void run_cutlass_benchmarks() {
    std::cout << "\n=== CUTLASS Benchmarks ===" << std::endl;
    
    std::vector<std::tuple<int,int,int>> test_cases = {
        {128, 128, 128},
        {256, 256, 256},
        {512, 512, 512},
        {1024, 1024, 1024},
        {2048, 2048, 2048},
        {4096, 4096, 4096},
        {4096, 256, 1024},
        {1024, 4096, 512},
    };
    
    std::vector<CutlassResult> results;
    
    for (const auto& [m, n, k] : test_cases) {
        std::cout << "Testing " << m << "×" << n << "×" << k << "..." << std::endl;
        auto result = benchmark_cutlass_gemm(m, n, k);
        result.print();
        results.push_back(result);
    }
    
    // Save results
    std::ofstream csv("cutlass_results.csv");
    csv << "config,m,n,k,time_ms,gflops" << std::endl;
    for (const auto& r : results) {
        csv << r.config << "," << r.m << "," << r.n << "," << r.k << ","
            << r.time_ms << "," << r.gflops << std::endl;
    }
    csv.close();
    
    std::cout << "Results saved to cutlass_results.csv" << std::endl;
}

#else

void run_cutlass_benchmarks() {
    std::cout << "\n=== CUTLASS Benchmarks ===" << std::endl;
    std::cout << "CUTLASS not available. To enable:" << std::endl;
    std::cout << "1. Clone CUTLASS: git clone https://github.com/NVIDIA/cutlass.git" << std::endl;
    std::cout << "2. Set CUTLASS_DIR environment variable" << std::endl;
    std::cout << "3. Rebuild with -DUSE_CUTLASS=ON" << std::endl;
    std::cout << "\nAlternatively, use cutlass_profiler from CUTLASS:" << std::endl;
    std::cout << "  cd cutlass/build" << std::endl;
    std::cout << "  ./tools/profiler/cutlass_profiler --operation=Gemm \\" << std::endl;
    std::cout << "    --m=1024 --n=1024 --k=1024 --warmup-iterations=5 \\" << std::endl;
    std::cout << "    --profiling-iterations=20" << std::endl;
}

#endif

int main() {
    run_cutlass_benchmarks();
    return 0;
}