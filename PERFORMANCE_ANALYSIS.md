# Performance Analysis and Optimizations

## Identified Performance Issues

After analyzing the codebase, we identified several critical performance bottlenecks that explain why our kernels achieve only ~3% of cuBLAS performance:

### 1. **Small Tile Sizes (32x32 output per block)**
- **Problem**: For a 4096x4096 matrix, this creates 128×128 = 16,384 blocks
- **Impact**: High kernel launch overhead, poor GPU utilization
- **Fix**: Increased to 64×64 output tiles (4× reduction in number of blocks)

### 2. **Low Occupancy (4 warps per block)**
- **Problem**: Only 128 threads per block (4 warps × 32 threads)
- **Impact**: Modern GPUs can handle 8 warps per block, limiting parallelism
- **Fix**: Increased to 8 warps per block (256 threads) for better occupancy

### 3. **Non-Coalesced Memory Access for Matrix B**
- **Problem**: B matrix loads accessed different rows first, then different columns
  - Original pattern: `B[k+0, n+0]`, `B[k+1, n+0]`, ..., `B[k+15, n+0]`, `B[k+0, n+1]`, ...
  - This is NOT coalesced because threads access non-consecutive memory locations
- **Impact**: Poor memory bandwidth utilization (~10-20% of peak)
- **Fix**: Reordered access pattern to access consecutive columns:
  - New pattern: `B[k+0, n+0]`, `B[k+0, n+1]`, ..., `B[k+0, n+15]`, `B[k+1, n+0]`, ...
  - This ensures threads access consecutive memory locations (coalesced)

### 4. **Inefficient Shared Memory Layout**
- **Problem**: Potential bank conflicts in shared memory access
- **Fix**: Added padding (+8 elements) to avoid 32-way bank conflicts

### 5. **Suboptimal Double Buffering**
- **Problem**: Double buffering implementation didn't properly overlap computation and memory
- **Fix**: Improved synchronization and load order to better overlap computation with memory loads

## Optimizations Implemented

### Optimized Kernel (`op_mm_tensorcore_optimized.cuh`)
- **Tile Size**: 64×64 output per block (was 32×32)
- **Warps per Block**: 8 warps (was 4 warps)
- **Memory Access**: Coalesced B matrix loads
- **Shared Memory**: Padded to avoid bank conflicts

### High-Performance Kernel (`op_mm_tensorcore_high_perf.cuh`)
- Same optimizations as optimized kernel
- Additional focus on memory access patterns

## Expected Performance Improvements

With these optimizations, we expect:
- **2-4× improvement** from larger tiles and better occupancy
- **1.5-2× improvement** from coalesced memory access
- **Overall: 3-8× improvement** (from ~3% to ~10-25% of cuBLAS)

## Matrix Sizes Tested

The benchmark tests the following matrix sizes:
- Square: 128, 256, 512, 1024, 2048, 4096, 8192
- Rectangular: (4096, 256, 1024), (1024, 4096, 512), (2048, 512, 2048), (512, 2048, 512)

## Key Metrics to Report

When presenting results, be prepared to answer:
1. **Matrix sizes**: All sizes listed above
2. **Blocks launched**: 
   - Old: (4096/32)² = 128² = 16,384 blocks for 4096×4096
   - New: (4096/64)² = 64² = 4,096 blocks for 4096×4096
3. **Warps per block**: 8 warps (256 threads)
4. **Occupancy**: Higher occupancy due to more warps per block
5. **Memory access**: Coalesced for both A and B matrices

## Next Steps

1. Run benchmarks to verify performance improvements
2. Profile with Nsight Compute to identify remaining bottlenecks
3. Consider additional optimizations:
   - Vectorized memory loads (uint4, uint2)
   - 3-5 stage pipelining (like CUTLASS)
   - Larger tile sizes (128×128) if shared memory allows

