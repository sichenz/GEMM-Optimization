#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

// Phase 1.2.1: cuBLAS Reference Implementation Benchmarks
// This is a more detailed cuBLAS benchmark tool that tests both FP32 and mixed precision
// Used to establish performance ceiling targets for our implementations

#define CHECK_CUDA(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1);} } while(0)
#define CHECK_CUBLAS(x) do { cublasStatus_t s = (x); if (s != CUBLAS_STATUS_SUCCESS) { \
  fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, int(s)); exit(1);} } while(0)

// Initialize random data for testing
static void init_random(float* p, size_t n) {
  for (size_t i = 0; i < n; ++i) { p[i] = float(rand()) / RAND_MAX - 0.5f; }
}

// Helper kernel to convert FP32 to FP16 for TensorCore operations
__global__ void f32_to_f16(const float* __restrict__ in, __half* __restrict__ out, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __float2half_rn(in[i]);  // Round-to-nearest conversion
}

struct Result {
  int M, N, K;
  const char* dtype;     // "fp32" or "f16f32"
  const char* api;       // "sgemm" or "gemmex"
  float ms_avg;
  double tflops;
  double gbytes_per_s;
};

static double compute_tflops(int M, int N, int K, float ms) {
  double ops = 2.0 * double(M) * double(N) * double(K);   // GEMM flops
  return ops / (ms * 1e-3) / 1e12;
}

static double approx_bytes_gemm_fp32(int M, int N, int K) {
  return (double(M)*K + double(K)*N + double(M)*N) * sizeof(float);
}
static double approx_bytes_gemm_mixed(int M, int N, int K) {
  return (double(M)*K + double(K)*N) * sizeof(__half) + (double(M)*N) * sizeof(float);
}

// Benchmark cuBLAS SGEMM (FP32 single precision)
// This is the standard FP32 GEMM - our performance target for Phase 2
static Result bench_sgemm(cublasHandle_t handle, int M, int N, int K, int iters=50, int warmup=5) {
  float alpha = 1.f, beta = 0.f;  // C = alpha*A*B + beta*C
  size_t asz = size_t(M)*K, bsz = size_t(K)*N, csz = size_t(M)*N;

  // Allocate device memory
  float *A_d, *B_d, *C_d;
  CHECK_CUDA(cudaMalloc(&A_d, asz*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&B_d, bsz*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&C_d, csz*sizeof(float)));

  // Initialize with random data on host, then copy to device
  std::vector<float> hA(asz), hB(bsz), hC(csz, 0.f);
  init_random(hA.data(), asz);
  init_random(hB.data(), bsz);
  CHECK_CUDA(cudaMemcpy(A_d, hA.data(), asz*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(B_d, hB.data(), bsz*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(C_d, hC.data(), csz*sizeof(float), cudaMemcpyHostToDevice));

  // Warmup runs to stabilize GPU performance
  // Note: cuBLAS uses column-major format (Fortran-style)
  // lda=M means leading dimension of A is M (number of rows)
  for (int i=0;i<warmup;i++) {
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M,N,K,
                             &alpha, A_d,M, B_d,K, &beta, C_d,M));
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  // Time
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaEventRecord(start));
  for (int i=0;i<iters;i++) {
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M,N,K,
                             &alpha, A_d,M, B_d,K, &beta, C_d,M));
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));
  float ms_total=0;
  CHECK_CUDA(cudaEventElapsedTime(&ms_total, start, stop));
  float ms_avg = ms_total / iters;

  CHECK_CUDA(cudaFree(A_d));
  CHECK_CUDA(cudaFree(B_d));
  CHECK_CUDA(cudaFree(C_d));

  Result r{M,N,K,"fp32","sgemm", ms_avg, 0,0};
  r.tflops = compute_tflops(M,N,K,ms_avg);
  r.gbytes_per_s = approx_bytes_gemm_fp32(M,N,K) / (ms_avg*1e-3) / 1e9;
  return r;
}

#ifndef CUBLAS_GEMM_DEFAULT_TENSOR_OP
#define CUBLAS_GEMM_DEFAULT_TENSOR_OP CUBLAS_GEMM_DEFAULT
#endif

// Benchmark cuBLAS GemmEx with FP16 input and FP32 accumulation (TensorCore path)
// Phase 1.2.1: This uses TensorCores which are much faster than regular FP32 cores
// Input: FP16, Accumulation: FP32, Output: FP32 (mixed precision)
// This is the performance target for Phase 2 TensorCore implementation
static Result bench_gemmex_f16f32(cublasHandle_t handle, int M, int N, int K, int iters=50, int warmup=5) {
  float alpha = 1.f, beta = 0.f;
  size_t asz = size_t(M)*K, bsz = size_t(K)*N, csz = size_t(M)*N;

  // Device buffers: need both FP32 (for conversion) and FP16 (for computation)
  float *A32_d, *B32_d; __half *A16_d, *B16_d; float *C32_d;
  CHECK_CUDA(cudaMalloc(&A32_d, asz*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&B32_d, bsz*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&A16_d, asz*sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&B16_d, bsz*sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&C32_d, csz*sizeof(float)));

  std::vector<float> hA(asz), hB(bsz), hC(csz, 0.f);
  init_random(hA.data(), asz);
  init_random(hB.data(), bsz);
  CHECK_CUDA(cudaMemcpy(A32_d, hA.data(), asz*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(B32_d, hB.data(), bsz*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(C32_d, 0, csz*sizeof(float)));

  // convert f32 -> f16 on device
  int threads = 256;
  int blocksA = int((asz + threads - 1) / threads);
  int blocksB = int((bsz + threads - 1) / threads);
  f32_to_f16<<<blocksA, threads>>>(A32_d, A16_d, asz);
  f32_to_f16<<<blocksB, threads>>>(B32_d, B16_d, bsz);
  CHECK_CUDA(cudaDeviceSynchronize());

  // warmup
  for (int i=0;i<warmup;i++) {
    CHECK_CUBLAS(cublasGemmEx(handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      M,N,K,
      &alpha,
      A16_d, CUDA_R_16F, M,
      B16_d, CUDA_R_16F, K,
      &beta,
      C32_d, CUDA_R_32F, M,
      CUDA_R_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  // time
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaEventRecord(start));
  for (int i=0;i<iters;i++) {
    CHECK_CUBLAS(cublasGemmEx(handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      M,N,K,
      &alpha,
      A16_d, CUDA_R_16F, M,
      B16_d, CUDA_R_16F, K,
      &beta,
      C32_d, CUDA_R_32F, M,
      CUBLAS_COMPUTE_32F_FAST_16F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));
  float ms_total=0;
  CHECK_CUDA(cudaEventElapsedTime(&ms_total, start, stop));
  float ms_avg = ms_total / iters;

  CHECK_CUDA(cudaFree(A32_d));
  CHECK_CUDA(cudaFree(B32_d));
  CHECK_CUDA(cudaFree(A16_d));
  CHECK_CUDA(cudaFree(B16_d));
  CHECK_CUDA(cudaFree(C32_d));

  Result r{M,N,K,"f16f32","gemmex", ms_avg, 0,0};
  r.tflops = compute_tflops(M,N,K,ms_avg);
  r.gbytes_per_s = approx_bytes_gemm_mixed(M,N,K) / (ms_avg*1e-3) / 1e9;
  return r;
}

static void print_header(bool csv){
  if (csv) {
    std::cout << "api,dtype,M,N,K,ms,TFLOPS,GBps\n";
  } else {
    std::cout << std::left << std::setw(8) << "api"
              << std::setw(8) << "dtype"
              << std::setw(8) << "M"
              << std::setw(8) << "N"
              << std::setw(8) << "K"
              << std::setw(12) << "ms"
              << std::setw(12) << "TFLOPS"
              << std::setw(12) << "GB/s" << "\n";
  }
}

static void print_row(const Result& r, bool csv){
  if (csv) {
    std::cout << r.api << "," << r.dtype << ","
              << r.M << "," << r.N << "," << r.K << ","
              << std::fixed << std::setprecision(4) << r.ms_avg << ","
              << std::setprecision(3) << r.tflops << ","
              << std::setprecision(3) << r.gbytes_per_s << "\n";
  } else {
    std::cout << std::left << std::setw(8) << r.api
              << std::setw(8) << r.dtype
              << std::setw(8) << r.M
              << std::setw(8) << r.N
              << std::setw(8) << r.K
              << std::setw(12) << std::fixed << std::setprecision(4) << r.ms_avg
              << std::setw(12) << std::setprecision(3) << r.tflops
              << std::setw(12) << std::setprecision(3) << r.gbytes_per_s
              << "\n";
  }
}

int main(int argc, char** argv) {
  bool csv = false;
  bool fp32 = true, mixed = true;
  int iters = 50, warmup = 5;

  for (int i=1;i<argc;i++){
    if (!strcmp(argv[i], "--csv")) csv = true;
    else if (!strcmp(argv[i], "--fp32-only")) { fp32=true; mixed=false; }
    else if (!strcmp(argv[i], "--mixed-only")) { fp32=false; mixed=true; }
    else if (!strncmp(argv[i], "--iters=", 8)) iters = atoi(argv[i]+8);
    else if (!strncmp(argv[i], "--warmup=", 9)) warmup = atoi(argv[i]+9);
  }

  // square sizes + a few rectangulars
  std::vector<int> sizes = {128,256,512,1024,1536,2048,4096,8192};
  std::vector<std::tuple<int,int,int>> rect = {
      {4096,256,1024}, {1024,8192,512}, {3000,3000,3000}
  };

  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));

  print_header(csv);

  for (int s : sizes) {
    int M=s, N=s, K=s;
    if (fp32)   { auto r = bench_sgemm(handle,M,N,K,iters,warmup);         print_row(r,csv); }
    if (mixed)  { auto r = bench_gemmex_f16f32(handle,M,N,K,iters,warmup); print_row(r,csv); }
  }
  for (auto [M,N,K] : rect) {
    if (fp32)   { auto r = bench_sgemm(handle,M,N,K,iters,warmup);         print_row(r,csv); }
    if (mixed)  { auto r = bench_gemmex_f16f32(handle,M,N,K,iters,warmup); print_row(r,csv); }
  }

  CHECK_CUBLAS(cublasDestroy(handle));
  return 0;
}