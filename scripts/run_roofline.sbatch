#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH --mem=16GB
#SBATCH --job-name=roofline_analysis
#SBATCH --mail-user=sz4972@nyu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/roofline_%j.out

module purge

# Create results directory
mkdir -p results
mkdir -p logs

cd /scratch/$USER/env

singularity exec \
    --overlay /scratch/$USER/env/overlay-50G-10M.ext3:rw \
    /scratch/work/public/singularity/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "
source /ext3/env.sh
conda activate test
cd /scratch/$USER/GEMM-Optimization

echo 'Running Roofline Analysis...'
python3 src/roofline_analysis.py

echo 'Analysis complete!'
echo 'Generated files:'
echo '  - results/roofline_plot.png'
echo '  - results/performance_comparison.png'
echo '  - results/analysis_report.txt'
"