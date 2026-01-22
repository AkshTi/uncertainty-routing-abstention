#!/bin/bash
#
# SEGMENT 1: Base Experiments (Exp1-3)
# Expected runtime: 4-5 hours
# Output: Baseline results + steering vectors
#
# Run this first. It establishes the foundation for all other experiments.
#

set -e  # Exit on error

echo "========================================================================"
echo " SEGMENT 1: Base Experiments (Exp1-3)"
echo " Expected runtime: 4-5 hours"
echo " Started: $(date)"
echo "========================================================================"
echo ""

# Check GPU
echo "Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}');$
echo ""

python diagnose_and_fix_steering.py
