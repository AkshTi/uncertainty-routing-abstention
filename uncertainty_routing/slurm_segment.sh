#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH -t 05:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
mkdir -p logs

module load miniforge
source activate technical_analysis

cd "$SLURM_SUBMIT_DIR"

echo "Running: $1"
bash "$1"

# optional: validate after each segment
./check_all_results.sh || true
