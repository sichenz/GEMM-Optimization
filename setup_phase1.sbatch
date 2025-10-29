#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=setup_phase1
#SBATCH --mail-user=sz4972@nyu.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=setup_phase1.out
#SBATCH --gres=gpu:1

module purge

cd /scratch/$USER/env

singularity exec --nv \
    --overlay /scratch/$USER/env/overlay-50G-10M.ext3:rw \
    /scratch/work/public/singularity/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "
source /ext3/env.sh
conda activate test
cd /scratch/$USER/GEMM-Optimization

set -e  # Exit on error

echo '=== Phase 1: GPU GEMM Benchmarking Setup ==='
echo ''

# Check CUDA availability
if ! command -v nvcc &> /dev/null; then
    echo 'ERROR: nvcc not found. Please install CUDA toolkit.'
    exit 1
fi

echo \"✓ CUDA found: \$(nvcc --version | grep release | awk '{print \$5}' | sed 's/,//')\"

# Check for lab1 directory
if [ ! -d 'phase1' ]; then
    echo 'ERROR: phase1 directory not found. Please run from project root.'
    exit 1
fi

cd phase1

# Check for required files
echo ''
echo 'Step 1: Checking benchmark files...'
if [ ! -f 'benchmark_gemm.cu' ]; then
    echo 'ERROR: benchmark_gemm.cu not found in phase1 directory'
    exit 1
fi
if [ ! -f 'benchmark_cutlass.cu' ]; then
    echo 'ERROR: benchmark_cutlass.cu not found in phase1 directory'
    exit 1
fi
if [ ! -f 'CMakeLists_phase1.txt' ]; then
    echo 'ERROR: CMakeLists_phase1.txt not found in phase1 directory'
    exit 1
fi
# Rename CMakeLists if needed
if [ -f 'CMakeLists_phase1.txt' ] && [ ! -f 'CMakeLists.txt' ]; then
    cp CMakeLists_phase1.txt CMakeLists.txt
fi
echo '✓ All required files present'

# Setup Python environment
echo ''
echo 'Step 2: Setting up Python environment...'
if ! command -v python3 &> /dev/null; then
    echo 'WARNING: python3 not found. Analysis scripts will not work.'
else
    python3 -m pip install --user pandas matplotlib numpy seaborn || {
        echo 'WARNING: Failed to install Python packages. You may need to install manually:'
        echo '  pip install pandas matplotlib numpy seaborn'
    }
    echo '✓ Python dependencies installed'
fi

# Optional: Setup CUTLASS
echo ''
echo 'Step 3: CUTLASS setup (non-interactive mode - skipping)...'
# Non-interactive mode: Skip CUTLASS setup by default
# To enable CUTLASS, set USE_CUTLASS=ON manually before running
USE_CUTLASS=OFF
echo 'Skipping CUTLASS setup. You can install it later.'

# Build project
echo ''
echo 'Step 4: Building project...'
mkdir -p build
cd build

if [ \"\$USE_CUTLASS\" = 'ON' ]; then
    cmake .. -DUSE_CUTLASS=ON -DCUTLASS_DIR=../../cutlass
else
    cmake ..
fi

make -j\$(nproc)

if [ \$? -eq 0 ]; then
    echo '✓ Build successful'
else
    echo 'ERROR: Build failed'
    exit 1
fi

cd ..

# Create run script
cat > run_phase1.sh << 'EOFINNER'
#!/bin/bash
# run_phase1.sh - Execute Phase 1 benchmarks and analysis

set -e

echo \"=== Phase 1: Running Benchmarks ===\"
echo \"\"

cd build

# Run benchmarks
echo \"Running GEMM benchmarks (this may take several minutes)...\"
./benchmark_gemm

# Run CUTLASS if available
if [ -f \"./benchmark_cutlass\" ]; then
    echo \"\"
    echo \"Running CUTLASS benchmarks...\"
    ./benchmark_cutlass
fi

cd ..

# Run analysis
echo \"\"
echo \"=== Generating Analysis and Visualizations ===\"
if command -v python3 &> /dev/null; then
    python3 ../analyze_phase1.py
    echo \"\"
    echo \"✓ Analysis complete!\"
    echo \"\"
    echo \"Generated files:\"
    echo \"  - benchmark_results.csv\"
    echo \"  - performance_comparison.png\"
    echo \"  - roofline_plot.png\"
    echo \"  - bottleneck_analysis.png\"
    echo \"  - performance_report.txt\"
    if [ -f \"cutlass_results.csv\" ]; then
        echo \"  - cutlass_results.csv\"
    fi
else
    echo \"WARNING: Python3 not found. Skipping analysis.\"
    echo \"You can run manually: python3 analyze_phase1.py\"
fi

echo \"\"
echo \"=== Phase 1 Complete ===\"
echo \"Review performance_report.txt for detailed analysis\"
EOFINNER

chmod +x run_phase1.sh

# Check for analysis script
if [ ! -f 'analyze_phase1.py' ]; then
    if [ -f '../analyze_phase1.py' ]; then
        cp ../analyze_phase1.py .
    else
        echo 'WARNING: analyze_phase1.py not found'
    fi
fi

echo ''
echo '=== Setup Complete ==='
echo ''
echo 'To run Phase 1 benchmarks:'
echo '  cd phase1'
echo '  ./run_phase1.sh'
echo ''
echo 'This will:'
echo '  1. Run Lab-1 tiled GEMM benchmarks'
echo '  2. Run cuBLAS benchmarks'
if [ \"\$USE_CUTLASS\" = 'ON' ]; then
    echo '  3. Run CUTLASS benchmarks'
    echo '  4. Generate performance analysis and visualizations'
else
    echo '  3. Generate performance analysis and visualizations'
fi
echo ''
echo 'Results will be saved to:'
echo '  - CSV files for raw data'
echo '  - PNG files for visualizations'
echo '  - performance_report.txt for detailed analysis'
"