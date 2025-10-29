#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <fstream>
#include <iomanip>

void checkCudaError(cudaError_t error, const char* msg) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " 
                  << cudaGetErrorString(error) << std::endl;
        exit(1);
    }
}

void printDeviceSpecs(std::ostream& out) {
    int deviceCount = 0;
    checkCudaError(cudaGetDeviceCount(&deviceCount), "Get Device Count");
    
    if (deviceCount == 0) {
        out << "No CUDA-capable devices found!" << std::endl;
        return;
    }
    
    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp prop;
        checkCudaError(cudaGetDeviceProperties(&prop, dev), "Get Device Properties");
        
        out << "========================================" << std::endl;
        out << "Device " << dev << ": " << prop.name << std::endl;
        out << "========================================" << std::endl;
        out << std::endl;
        
        // Basic Info
        out << "=== Basic Information ===" << std::endl;
        out << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        out << "Total Global Memory: " 
            << std::fixed << std::setprecision(2)
            << (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0) 
            << " GB" << std::endl;
        out << "Memory Clock Rate: " 
            << prop.memoryClockRate / 1000.0 << " MHz" << std::endl;
        out << "Memory Bus Width: " << prop.memoryBusWidth << " bits" << std::endl;
        out << std::endl;
        
        // Compute Resources
        out << "=== Compute Resources ===" << std::endl;
        out << "Number of SMs: " << prop.multiProcessorCount << std::endl;
        out << "Max Threads per SM: " << prop.maxThreadsPerMultiProcessor << std::endl;
        out << "Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
        out << "Max Block Dimensions: (" 
            << prop.maxThreadsDim[0] << ", "
            << prop.maxThreadsDim[1] << ", "
            << prop.maxThreadsDim[2] << ")" << std::endl;
        out << "Max Grid Dimensions: (" 
            << prop.maxGridSize[0] << ", "
            << prop.maxGridSize[1] << ", "
            << prop.maxGridSize[2] << ")" << std::endl;
        out << "Warp Size: " << prop.warpSize << std::endl;
        out << std::endl;
        
        // Memory Hierarchy
        out << "=== Memory Hierarchy ===" << std::endl;
        out << "L2 Cache Size: " 
            << (double)prop.l2CacheSize / (1024.0 * 1024.0) 
            << " MB" << std::endl;
        out << "Shared Memory per Block: " 
            << (double)prop.sharedMemPerBlock / 1024.0 
            << " KB" << std::endl;
        out << "Shared Memory per SM: " 
            << (double)prop.sharedMemPerMultiprocessor / 1024.0 
            << " KB" << std::endl;
        out << "Registers per Block: " << prop.regsPerBlock << std::endl;
        out << "Registers per SM: " << prop.regsPerMultiprocessor << std::endl;
        out << "Constant Memory: " 
            << (double)prop.totalConstMem / 1024.0 
            << " KB" << std::endl;
        out << std::endl;
        
        // Performance Characteristics
        out << "=== Performance Characteristics ===" << std::endl;
        
        // Memory Bandwidth (theoretical peak)
        double memBandwidthGB = 2.0 * prop.memoryClockRate * 
                                (prop.memoryBusWidth / 8.0) / 1.0e6;
        out << "Peak Memory Bandwidth: " 
            << std::fixed << std::setprecision(1)
            << memBandwidthGB << " GB/s" << std::endl;
        
        // FP32 TFLOPS (base clock, 2 FMA ops per cycle per CUDA core)
        // For T4: 2560 CUDA cores, ~1.5 GHz base
        int cudaCores = 0;
        // Rough estimate based on compute capability
        if (prop.major == 7 && prop.minor == 5) { // Turing (T4)
            cudaCores = prop.multiProcessorCount * 64; // 64 cores per SM
        } else if (prop.major == 8 && prop.minor == 0) { // Ampere (A100)
            cudaCores = prop.multiProcessorCount * 64;
        } else if (prop.major == 8 && prop.minor == 6) { // Ampere (A40)
            cudaCores = prop.multiProcessorCount * 128;
        } else {
            cudaCores = prop.multiProcessorCount * 64; // rough estimate
        }
        
        double fp32TFlops = (cudaCores * prop.clockRate * 2.0) / 1.0e9;
        out << "Estimated FP32 Peak TFLOPS: " 
            << std::fixed << std::setprecision(2)
            << fp32TFlops << " TFLOPS" << std::endl;
        
        // TensorCore capabilities
        if (prop.major >= 7) { // Volta and later
            double fp16TFlops = 0;
            if (prop.major == 7 && prop.minor == 0) { // V100
                fp16TFlops = 112.0; // V100 spec
            } else if (prop.major == 7 && prop.minor == 5) { // T4
                fp16TFlops = 65.0; // T4 spec
            } else if (prop.major == 8 && prop.minor == 0) { // A100
                fp16TFlops = 312.0; // A100 spec
            } else if (prop.major == 8 && prop.minor == 6) { // A40
                fp16TFlops = 149.0; // A40 spec
            }
            
            if (fp16TFlops > 0) {
                out << "TensorCore FP16 Peak TFLOPS: " 
                    << fp16TFlops << " TFLOPS" << std::endl;
                out << "TensorCore Speedup vs FP32: " 
                    << std::fixed << std::setprecision(1)
                    << fp16TFlops / fp32TFlops << "x" << std::endl;
            }
        }
        
        out << "Clock Rate: " 
            << prop.clockRate / 1000.0 << " MHz" << std::endl;
        out << std::endl;
        
        // Feature Support
        out << "=== Feature Support ===" << std::endl;
        out << "Concurrent Kernels: " 
            << (prop.concurrentKernels ? "Yes" : "No") << std::endl;
        out << "ECC Enabled: " 
            << (prop.ECCEnabled ? "Yes" : "No") << std::endl;
        out << "Unified Addressing: " 
            << (prop.unifiedAddressing ? "Yes" : "No") << std::endl;
        out << "Managed Memory: " 
            << (prop.managedMemory ? "Yes" : "No") << std::endl;
        out << "Cooperative Launch: " 
            << (prop.cooperativeLaunch ? "Yes" : "No") << std::endl;
        
        if (prop.major >= 7) {
            out << "TensorCore Support: Yes (Compute " 
                << prop.major << "." << prop.minor << ")" << std::endl;
        } else {
            out << "TensorCore Support: No" << std::endl;
        }
        
        out << std::endl;
    }
}

int main() {
    std::cout << "GPU Specifications Report" << std::endl;
    std::cout << "Generated: " << __DATE__ << " " << __TIME__ << std::endl;
    std::cout << std::endl;
    
    // Print to console
    printDeviceSpecs(std::cout);
    
    // Save to file
    std::ofstream outFile("results/gpu_specs.txt");
    if (outFile.is_open()) {
        outFile << "GPU Specifications Report" << std::endl;
        outFile << "Generated: " << __DATE__ << " " << __TIME__ << std::endl;
        outFile << std::endl;
        printDeviceSpecs(outFile);
        outFile.close();
        std::cout << "Results saved to results/gpu_specs.txt" << std::endl;
    } else {
        std::cerr << "Failed to open output file!" << std::endl;
    }
    
    return 0;
}