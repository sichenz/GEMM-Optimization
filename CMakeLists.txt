cmake_minimum_required(VERSION 3.20)

project(GEMM_Optimization LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -use_fast_math")

# CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Include paths
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/utils)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/ops)

# GPU Specifications Tool
add_executable(gpu_specs src/gpu_specs.cu)
# FIXED: Support V100 (compute 7.0) instead of only 7.5
set_target_properties(gpu_specs PROPERTIES 
    CUDA_ARCHITECTURES "70;75")
target_compile_options(gpu_specs PRIVATE
    -Wno-unused-function
    --expt-relaxed-constexpr
)
target_link_libraries(gpu_specs PRIVATE 
    CUDA::cudart
    CUDA::cublas
)

# GEMM Benchmark
add_executable(benchmark_gemm src/benchmark_gemm.cu)
# FIXED: Support V100 (compute 7.0) instead of only 7.5
set_target_properties(benchmark_gemm PROPERTIES 
    CUDA_ARCHITECTURES "70;75")
target_compile_options(benchmark_gemm PRIVATE
    -Wno-unused-function
    --expt-relaxed-constexpr
    --extended-lambda
)
target_link_libraries(benchmark_gemm PRIVATE 
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
)

# cuBLAS reference benchmark
add_executable(cublas_bench src/baselines/cublas_bench.cu)
set_target_properties(cublas_bench PROPERTIES CUDA_ARCHITECTURES "70;75;80")
target_link_libraries(cublas_bench PRIVATE CUDA::cublas CUDA::cudart)
target_compile_options(cublas_bench PRIVATE -O3 --expt-relaxed-constexpr)
target_compile_definitions(cublas_bench PRIVATE -D_FORCE_INLINES)

# Profiling test program
add_executable(profile_tensorcore src/profile_tensorcore.cu)
set_target_properties(profile_tensorcore PROPERTIES CUDA_ARCHITECTURES "70;75")
target_compile_options(profile_tensorcore PRIVATE
    -Wno-unused-function
    --expt-relaxed-constexpr
    --extended-lambda
)
target_link_libraries(profile_tensorcore PRIVATE 
    CUDA::cudart
    CUDA::curand
)

# Profile-only executable for TensorCore kernel
add_executable(profile_tensorcore src/profile_tensorcore.cu)
set_target_properties(profile_tensorcore PROPERTIES CUDA_ARCHITECTURES "70;75")
target_compile_options(profile_tensorcore PRIVATE
    -Wno-unused-function
    --expt-relaxed-constexpr
    --extended-lambda
)
target_link_libraries(profile_tensorcore PRIVATE 
    CUDA::cudart
    CUDA::curand
)