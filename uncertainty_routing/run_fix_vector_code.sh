#!/bin/bash
#SBATCH --job-name=seg1_base
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

# Always run in the directory you submitted from
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Make logs in submit dir
mkdir -p logs
LOG="logs/segment1_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S).log"

# HARD log: everything printed goes to the log too
# (This is more reliable than process substitution on some clusters)
{
  echo "========================================================================"
  echo " SEGMENT 1: Base Experiments (Exp1-3)"
  echo " Started: $(date)"
  echo " Submit dir: ${SLURM_SUBMIT_DIR:-$PWD}"
  echo " PWD: $(pwd)"
  echo " Host: $(hostname)"
  echo " JobID: ${SLURM_JOB_ID:-NA}"
  echo " Log file: $LOG"
  echo "========================================================================"
  echo ""

  echo "Checking GPU availability..."
  python - <<'PY'
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
PY
  echo ""

  echo "Running your custom file ..."
  python experiment6_robustness_fixed.py
  python experiment7_safety_alignment_fixed.py


  echo ""
  echo "DONE: $(date)"
} 2>&1 | tee -a "$LOG"

