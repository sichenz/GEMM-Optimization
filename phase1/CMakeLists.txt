cmake_minimum_required(VERSION 3.20)

project(BareNet LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# ---- CUDA Toolkit ----
find_package(CUDAToolkit REQUIRED)

# ---- cuBLAS (required for benchmarking) ----
find_package(CUDAToolkit REQUIRED COMPONENTS cublas)

# ---- CUTLASS (optional) ----
option(USE_CUTLASS "Build with CUTLASS support" OFF)
if(USE_CUTLASS)
    if(DEFINED ENV{CUTLASS_DIR})
        set(CUTLASS_DIR $ENV{CUTLASS_DIR})
    else()
        set(CUTLASS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../cutlass")
    endif()
    
    if(EXISTS ${CUTLASS_DIR})
        message(STATUS "Found CUTLASS at ${CUTLASS_DIR}")
        include_directories(${CUTLASS_DIR}/include)
        include_directories(${CUTLASS_DIR}/tools/util/include)
        add_definitions(-DUSE_CUTLASS)
    else()
        message(WARNING "CUTLASS not found at ${CUTLASS_DIR}")
        message(WARNING "Clone with: git clone https://github.com/NVIDIA/cutlass.git")
        set(USE_CUTLASS OFF)
    endif()
endif()

# ---- Include paths ----
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/utils)

# ---- Enable language ----
enable_language(CUDA)

# ---- Original executables ----
file(GLOB executables
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

list(FILTER executables EXCLUDE REGEX ".*/bindings\\.(cu|cpp)$")

foreach(exe ${executables})
  get_filename_component(filename ${exe} NAME_WE)
  add_executable(${filename} ${exe})
  set_target_properties(${filename} PROPERTIES CUDA_ARCHITECTURES "75;80;86;89")
  target_compile_options(${filename} PRIVATE
    -Wno-unused-function
    --expt-relaxed-constexpr
  )
  target_include_directories(${filename} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
  )
  target_link_libraries(${filename} PRIVATE 
    CUDA::cudart
    CUDA::curand
  )
endforeach()

# ---- Phase 1 Benchmarking Tool ----
add_executable(benchmark_gemm 
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_gemm.cu
)
set_target_properties(benchmark_gemm PROPERTIES 
    CUDA_ARCHITECTURES "75;80;86;89"
)
target_compile_options(benchmark_gemm PRIVATE
    -Wno-unused-function
    --expt-relaxed-constexpr
)
target_include_directories(benchmark_gemm PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
)
target_link_libraries(benchmark_gemm PRIVATE 
    CUDA::cudart
    CUDA::curand
    CUDA::cublas
)

# ---- CUTLASS Benchmark (if enabled) ----
if(USE_CUTLASS)
    add_executable(cutlass_benchmark 
        ${CMAKE_CURRENT_SOURCE_DIR}/cutlass_benchmark.cu
    )
    set_target_properties(cutlass_benchmark PROPERTIES 
        CUDA_ARCHITECTURES "75;80;86;89"
    )
    target_compile_options(cutlass_benchmark PRIVATE
        -Wno-unused-function
        --expt-relaxed-constexpr
    )
    target_include_directories(cutlass_benchmark PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/utils
        ${CUTLASS_DIR}/include
        ${CUTLASS_DIR}/tools/util/include
    )
    target_link_libraries(cutlass_benchmark PRIVATE 
        CUDA::cudart
        CUDA::curand
    )
endif()

# ---- Installation ----
install(TARGETS benchmark_gemm DESTINATION bin)
if(USE_CUTLASS)
    install(TARGETS cutlass_benchmark DESTINATION bin)
endif()

message(STATUS "=== Build Configuration ===")
message(STATUS "CUDA Architectures: 75;80;86;89")
message(STATUS "cuBLAS: ENABLED")
message(STATUS "CUTLASS: ${USE_CUTLASS}")
message(STATUS "========================")