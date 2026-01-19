#!/bin/bash
#SBATCH -J uncertainty_all
#SBATCH -p mit_normal_gpu
#SBATCH -G h200:1
#SBATCH -c 8
#SBATCH -t 12:00:00
#SBATCH --mem=64G
#SBATCH -o results/slurm_%j.out
#SBATCH -e results/slurm_%j.err

set -euo pipefail

echo "=========================================="
echo "Job started at $(date)"
echo "Running on $(hostname)"
echo "=========================================="
export PYTHONNOUSERSITE=1

# Change to your working directory
cd ~/uncertainty_routing

# Create results directory
mkdir -p results

# Load conda/mamba
module load miniforge

# Check GPU
echo "Checking GPU..."
nvidia-smi || echo "Warning: nvidia-smi failed"

# Activate environment
echo "Activating mech_interp environment..."
source activate mech_interp

# Verify Python and packages
echo "Python executable: $(which python)"
echo "Python version: $(python --version)"

# Test critical imports
echo "Testing imports..."
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')" || exit 1
python -c "import transformers; print(f'Transformers {transformers.__version__}')" || exit 1
python -c "import pandas; import numpy; import matplotlib; print('Core packages OK')" || exit 1

# Run the complete pipeline
echo "=========================================="
echo "Starting experiments..."
echo "=========================================="

python run_complete_pipeline.py --mode standard --model "Qwen/Qwen2.5-1.5B-Instruct"

echo "=========================================="
echo "Job completed at $(date)"
echo "=========================================="
