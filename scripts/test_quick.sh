#!/bin/bash
# Quick test script for local or interactive testing
# Usage: bash scripts/test_quick.sh

set -e

echo "Quick GEMM Benchmark Test"
echo "========================"
echo ""

# Check if we're in the right directory
if [ ! -f "CMakeLists.txt" ]; then
    echo "Error: Please run from project root directory"
    exit 1
fi

# Build
echo "Building project..."
rm -rf build
mkdir -p build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8 benchmark_gemm

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

echo "Build successful!"
echo ""

# Run benchmark
cd ..
echo "Running benchmark..."
echo ""

./build/benchmark_gemm 2>&1 | tee results/quick_test_output.txt

echo ""
echo "Test complete! Results saved to results/quick_test_output.txt"
echo ""
echo "Quick summary:"
grep -E "(Lab2_TensorCore|cuBLAS)" results/benchmark_results.csv | \
    awk -F',' '{printf "%-35s: %8.2f GFLOPS\n", $1, $6}' | head -10

