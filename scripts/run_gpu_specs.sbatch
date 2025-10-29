#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:10:00
#SBATCH --mem=8GB
#SBATCH --job-name=gpu_specs
#SBATCH --mail-user=sz4972@nyu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/gpu_specs_%j.out
#SBATCH --gres=gpu:1

module purge

# Create results directory
mkdir -p results
mkdir -p logs

cd /scratch/$USER/env

singularity exec --nv \
    --overlay /scratch/$USER/env/overlay-50G-10M.ext3:rw \
    /scratch/work/public/singularity/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "
source /ext3/env.sh
conda activate test
cd /scratch/$USER/GEMM-Optimization

echo 'Running GPU Specifications Tool...'
./build/gpu_specs

echo 'GPU specifications saved to results/gpu_specs.txt'
"