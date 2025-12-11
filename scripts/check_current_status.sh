#!/bin/bash
# Check current status

echo "Current Status:"
echo "SLURM Job:"
squeue -u $USER

echo ""
echo "Check if you're in Singularity:"
if [ -n "$SINGULARITY_NAME" ]; then
    echo "You are in Singularity container: $SINGULARITY_NAME"
else
    echo "You are NOT in Singularity container"
    echo "  Run: singularity exec --nv --overlay /scratch/\$USER/overlay-25GB-500K.ext3:rw /scratch/\$USER/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif /bin/bash"
fi

echo ""
echo "Current directory:"
pwd

echo ""
echo "Check if benchmark exists:"
if [ -f build/benchmark_gemm ]; then
    echo "benchmark_gemm exists"
    ls -lh build/benchmark_gemm
else
    echo "benchmark_gemm not found - need to build"
fi

echo ""
echo "Recent commands:"
history | tail -10

