#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <ctime>

// Phase 1.1: Hardware Analysis & Baseline Setup
// This file collects GPU specifications needed to understand our target hardware
// and calculate theoretical performance limits for GEMM operations.

// Simple error checking helper - exits if CUDA call fails
void checkCudaError(cudaError_t error, const char* msg) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " 
                  << cudaGetErrorString(error) << std::endl;
        exit(1);
    }
}

// Different GPU architectures have different numbers of CUDA cores per SM
// This function maps compute capability (major.minor) to the correct core count
// We need this to calculate peak FP32 TFLOPS accurately
int getSPcores(cudaDeviceProp devProp) {
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    
    switch (devProp.major) {
        case 2: // Fermi
            cores = (devProp.minor == 1) ? mp * 48 : mp * 32;
            break;
        case 3: // Kepler
            cores = mp * 192;
            break;
        case 5: // Maxwell
            cores = mp * 128;
            break;
        case 6: // Pascal
            cores = (devProp.minor == 1) ? mp * 128 : mp * 64;
            break;
        case 7: // Volta, Turing
            cores = mp * 64;
            break;
        case 8: // Ampere
            cores = (devProp.minor == 0) ? mp * 64 : mp * 128;
            break;
        case 9: // Hopper
            cores = mp * 128;
            break;
        default:
            cores = mp * 64; // Fallback
            break;
    }
    return cores;
}

// Main function to collect and print all GPU specifications
// This is Phase 1.1.1: Document GPU specifications
void printDeviceSpecs(std::ostream& out) {
    int deviceCount = 0;
    checkCudaError(cudaGetDeviceCount(&deviceCount), "Get Device Count");
    
    if (deviceCount == 0) {
        out << "No CUDA-capable devices found!" << std::endl;
        return;
    }
    
    // Loop through all available GPUs (usually just one in our case)
    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp prop;
        checkCudaError(cudaGetDeviceProperties(&prop, dev), "Get Device Properties");
        
        out << "========================================" << std::endl;
        out << "Device " << dev << ": " << prop.name << std::endl;
        out << "========================================" << std::endl;
        out << std::endl;
        
        // Basic Info - GPU name, memory, clock rates
        out << "=== Basic Information ===" << std::endl;
        out << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        out << "Total Global Memory: " 
            << std::fixed << std::setprecision(2)
            << (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0) 
            << " GB" << std::endl;
        out << "Memory Clock Rate: " 
            << std::fixed << std::setprecision(2)
            << prop.memoryClockRate / 1000.0 << " MHz" << std::endl;
        out << "Memory Bus Width: " << prop.memoryBusWidth << " bits" << std::endl;
        out << std::endl;
        
        // Compute Resources
        out << "=== Compute Resources ===" << std::endl;
        out << "Number of SMs: " << prop.multiProcessorCount << std::endl;
        
        int cudaCores = getSPcores(prop);
        out << "CUDA Cores: " << cudaCores << " (total)" << std::endl;
        out << "CUDA Cores per SM: " << cudaCores / prop.multiProcessorCount << std::endl;
        
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
            << std::fixed << std::setprecision(2)
            << (double)prop.l2CacheSize / (1024.0 * 1024.0) 
            << " MB" << std::endl;
        out << "Shared Memory per Block: " 
            << std::fixed << std::setprecision(2)
            << (double)prop.sharedMemPerBlock / 1024.0 
            << " KB" << std::endl;
        out << "Shared Memory per SM: " 
            << std::fixed << std::setprecision(2)
            << (double)prop.sharedMemPerMultiprocessor / 1024.0 
            << " KB" << std::endl;
        out << "Registers per Block: " << prop.regsPerBlock << std::endl;
        out << "Registers per SM: " << prop.regsPerMultiprocessor << std::endl;
        out << "Constant Memory: " 
            << std::fixed << std::setprecision(2)
            << (double)prop.totalConstMem / 1024.0 
            << " KB" << std::endl;
        out << std::endl;
        
        // Performance Characteristics - Phase 1.1.2: Calculate performance ceilings
        out << "=== Performance Characteristics ===" << std::endl;
        
        // Memory Bandwidth (theoretical peak)
        // Formula: 2 * MemClockRate(MHz) * BusWidth(bits) / 8 / 1000
        // The "2" comes from DDR (Double Data Rate) - data transferred on both clock edges
        // This is the maximum rate we can read/write data from global memory
        double memBandwidthGB = 2.0 * prop.memoryClockRate * 
                                (prop.memoryBusWidth / 8.0) / 1.0e6;
        out << "Peak Memory Bandwidth: " 
            << std::fixed << std::setprecision(1)
            << memBandwidthGB << " GB/s" << std::endl;
        
        // FP32 TFLOPS - Peak floating point performance
        // Formula: CUDA_Cores * Clock_Rate(GHz) * 2 (FMA = 2 ops) / 1000
        // FMA (Fused Multiply-Add) counts as 2 operations: multiply + add
        // This tells us the maximum compute throughput for FP32 operations
        double clockRateGHz = prop.clockRate / 1.0e6; // Convert kHz to GHz
        double fp32TFlops = (cudaCores * clockRateGHz * 2.0) / 1000.0;
        
        out << "Estimated FP32 Peak TFLOPS: " 
            << std::fixed << std::setprecision(2)
            << fp32TFlops << " TFLOPS" << std::endl;
        out << "Base Clock Rate: " 
            << std::fixed << std::setprecision(2)
            << clockRateGHz << " GHz" << std::endl;
        
        // TensorCore capabilities
        if (prop.major >= 7) { // Volta and later
            double fp16TFlops = 0;
            std::string tcArch = "";
            
            if (prop.major == 7 && prop.minor == 0) { // V100
                fp16TFlops = 112.0;
                tcArch = "Volta";
            } else if (prop.major == 7 && prop.minor == 5) { // T4, RTX 2000, Quadro RTX
                fp16TFlops = 65.0; // T4 spec (Turing)
                // For Quadro RTX 8000: 72 SMs * 8 TC/SM * 64 ops/TC/cycle * 1.77 GHz
                // More accurate: 72 * 8 * 64 * 1.77 / 1000 = 65.2 TFLOPS
                tcArch = "Turing";
            } else if (prop.major == 8 && prop.minor == 0) { // A100
                fp16TFlops = 312.0;
                tcArch = "Ampere";
            } else if (prop.major == 8 && prop.minor == 6) { // A40, A10
                fp16TFlops = 149.0;
                tcArch = "Ampere";
            } else if (prop.major == 9 && prop.minor == 0) { // H100
                fp16TFlops = 989.0;
                tcArch = "Hopper";
            }
            
            if (fp16TFlops > 0) {
                out << "TensorCore Architecture: " << tcArch << std::endl;
                out << "TensorCore FP16 Peak TFLOPS: " 
                    << std::fixed << std::setprecision(1)
                    << fp16TFlops << " TFLOPS" << std::endl;
                out << "TensorCore Speedup vs FP32: " 
                    << std::fixed << std::setprecision(1)
                    << fp16TFlops / fp32TFlops << "x" << std::endl;
            }
        }
        
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
        
        // Arithmetic Intensity Analysis - Phase 1.1.2: Calculate arithmetic intensity
        // Arithmetic Intensity (AI) = FLOPS / Bytes transferred
        // This tells us if a workload is memory-bound or compute-bound
        // Ridge point: AI where memory bandwidth limit equals compute limit
        out << std::endl;
        out << "=== Arithmetic Intensity Analysis ===" << std::endl;
        double ridgePointFP32 = (fp32TFlops * 1000.0) / memBandwidthGB; // GFLOPS / GB/s
        out << "FP32 Ridge Point: " 
            << std::fixed << std::setprecision(2)
            << ridgePointFP32 << " FLOPS/Byte" << std::endl;
        out << "  (Workloads with AI > " << ridgePointFP32 
            << " are compute-bound)" << std::endl;
        out << "  (Workloads with AI < " << ridgePointFP32 
            << " are memory-bound)" << std::endl;
        
        if (prop.major >= 7) {
            double fp16TFlopsValue = (prop.major == 7 && prop.minor == 5) ? 65.0 : 0;
            if (fp16TFlopsValue > 0) {
                double ridgePointFP16 = (fp16TFlopsValue * 1000.0) / memBandwidthGB;
                out << "FP16 TensorCore Ridge Point: " 
                    << std::fixed << std::setprecision(2)
                    << ridgePointFP16 << " FLOPS/Byte" << std::endl;
            }
        }
        
        out << std::endl;
    }
}

int main() {
    // Phase 1.1.1: Generate GPU specifications report
    // This will be used later to calculate efficiency and understand performance limits
    
    // Get current time for report timestamp
    std::time_t now = std::time(nullptr);
    char timestamp[100];
    std::strftime(timestamp, sizeof(timestamp), "%b %d %Y %H:%M:%S", std::localtime(&now));
    
    std::cout << "========================================" << std::endl;
    std::cout << "GPU Specifications Report" << std::endl;
    std::cout << "Generated: " << timestamp << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    // Print to console for immediate viewing
    printDeviceSpecs(std::cout);
    
    // Save to file for later analysis (used by roofline_analysis.py)
    std::ofstream outFile("results/gpu_specs.txt");
    if (outFile.is_open()) {
        outFile << "========================================" << std::endl;
        outFile << "GPU Specifications Report" << std::endl;
        outFile << "Generated: " << timestamp << std::endl;
        outFile << "========================================" << std::endl;
        outFile << std::endl;
        printDeviceSpecs(outFile);
        outFile.close();
        std::cout << "✓ Results saved to results/gpu_specs.txt" << std::endl;
    } else {
        std::cerr << "✗ Failed to open output file!" << std::endl;
    }
    
    return 0;
}