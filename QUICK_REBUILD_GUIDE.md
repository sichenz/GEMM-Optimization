# Quick Rebuild Guide for Greene

## Problem
`cmake` is not available on the login node - you need to be inside the Singularity container.

## Solution: Use SLURM Script (Easiest)

The existing script handles everything automatically:

```bash
# Just submit the job - it will rebuild and test
cd /scratch/$USER/GEMM-Optimization
sbatch scripts/run_phase4_final.sbatch

# Monitor it
squeue -u $USER
tail -f logs/phase4_final.out
```

## Alternative: Interactive Build

If you want to build interactively:

```bash
# 1. Request an interactive GPU node
srun --gres=gpu:rtx8000:1 --time=1:00:00 --mem=32GB --pty bash

# 2. Enter Singularity container
singularity exec --nv \
    --overlay /scratch/$USER/overlay-25GB-500K.ext3:rw \
    /scratch/$USER/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash

# 3. Inside container, activate environment
source /ext3/env.sh
conda activate test

# 4. Navigate and build
cd /scratch/$USER/GEMM-Optimization
rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8 benchmark_gemm

# 5. Quick test
cd ..
./build/benchmark_gemm 2>&1 | head -50

# 6. Exit container when done
exit
```

## What the SLURM Script Does

The `run_phase4_final.sbatch` script:
1. Enters Singularity container automatically
2. Activates conda environment
3. Builds the project
4. Runs benchmarks
5. Generates reports

You don't need to do anything manually - just submit the job!

