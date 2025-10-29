#!/bin/bash
# setup_phase1.sh - Complete setup for Phase 1 benchmarking

set -e  # Exit on error

echo "=== Phase 1: GPU GEMM Benchmarking Setup ==="
echo ""

# Check CUDA availability
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found. Please install CUDA toolkit."
    exit 1
fi

echo "✓ CUDA found: $(nvcc --version | grep release | awk '{print $5}' | sed 's/,//')"

# Check for lab1 directory
if [ ! -d "lab1" ]; then
    echo "ERROR: lab1 directory not found. Please run from project root."
    exit 1
fi

cd lab1

# Copy new files
echo ""
echo "Step 1: Setting up benchmark files..."
cp ../benchmark_gemm.cu .
cp ../cutlass_benchmark.cu .
cp ../CMakeLists_phase1.txt CMakeLists.txt

# Setup Python environment
echo ""
echo "Step 2: Setting up Python environment..."
if ! command -v python3 &> /dev/null; then
    echo "WARNING: python3 not found. Analysis scripts will not work."
else
    python3 -m pip install --user pandas matplotlib numpy seaborn || {
        echo "WARNING: Failed to install Python packages. You may need to install manually:"
        echo "  pip install pandas matplotlib numpy seaborn"
    }
    echo "✓ Python dependencies installed"
fi

# Optional: Setup CUTLASS
echo ""
echo "Step 3: CUTLASS setup (optional)..."
read -p "Do you want to set up CUTLASS for comparison? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ ! -d "../cutlass" ]; then
        echo "Cloning CUTLASS (this may take a few minutes)..."
        cd ..
        git clone https://github.com/NVIDIA/cutlass.git
        cd cutlass
        mkdir -p build && cd build
        echo "Building CUTLASS..."
        cmake .. -DCUTLASS_NVCC_ARCHS=75 -DCUTLASS_ENABLE_EXAMPLES=OFF
        make -j$(nproc) cutlass_profiler
        cd ../../lab1
        echo "✓ CUTLASS installed and built"
    else
        echo "✓ CUTLASS already exists"
    fi
    USE_CUTLASS=ON
else
    echo "Skipping CUTLASS setup. You can install it later."
    USE_CUTLASS=OFF
fi

# Build project
echo ""
echo "Step 4: Building project..."
mkdir -p build
cd build

if [ "$USE_CUTLASS" = "ON" ]; then
    cmake .. -DUSE_CUTLASS=ON -DCUTLASS_DIR=../../cutlass
else
    cmake ..
fi

make -j$(nproc)

if [ $? -eq 0 ]; then
    echo "✓ Build successful"
else
    echo "ERROR: Build failed"
    exit 1
fi

cd ..

# Create run script
cat > run_phase1.sh << 'EOF'
#!/bin/bash
# run_phase1.sh - Execute Phase 1 benchmarks and analysis

set -e

echo "=== Phase 1: Running Benchmarks ==="
echo ""

cd build

# Run benchmarks
echo "Running GEMM benchmarks (this may take several minutes)..."
./benchmark_gemm

# Run CUTLASS if available
if [ -f "./cutlass_benchmark" ]; then
    echo ""
    echo "Running CUTLASS benchmarks..."
    ./cutlass_benchmark
fi

cd ..

# Run analysis
echo ""
echo "=== Generating Analysis and Visualizations ==="
if command -v python3 &> /dev/null; then
    python3 ../analyze_phase1.py
    echo ""
    echo "✓ Analysis complete!"
    echo ""
    echo "Generated files:"
    echo "  - benchmark_results.csv"
    echo "  - performance_comparison.png"
    echo "  - roofline_plot.png"
    echo "  - bottleneck_analysis.png"
    echo "  - performance_report.txt"
    if [ -f "cutlass_results.csv" ]; then
        echo "  - cutlass_results.csv"
    fi
else
    echo "WARNING: Python3 not found. Skipping analysis."
    echo "You can run manually: python3 analyze_phase1.py"
fi

echo ""
echo "=== Phase 1 Complete ==="
echo "Review performance_report.txt for detailed analysis"
EOF

chmod +x run_phase1.sh

# Copy analysis script
cp ../analyze_phase1.py .

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To run Phase 1 benchmarks:"
echo "  cd lab1"
echo "  ./run_phase1.sh"
echo ""
echo "This will:"
echo "  1. Run Lab-1 tiled GEMM benchmarks"
echo "  2. Run cuBLAS benchmarks"
if [ "$USE_CUTLASS" = "ON" ]; then
    echo "  3. Run CUTLASS benchmarks"
    echo "  4. Generate performance analysis and visualizations"
else
    echo "  3. Generate performance analysis and visualizations"
fi
echo ""
echo "Results will be saved to:"
echo "  - CSV files for raw data"
echo "  - PNG files for visualizations"
echo "  - performance_report.txt for detailed analysis"