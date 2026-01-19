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
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
echo ""

# Run Exp1-3 only
echo "Running Experiments 1-3..."
python run_complete_pipeline_v2.py \
    --mode standard \
    --skip-exp4 \
    --skip-exp5 \
    --skip-exp6 \
    --skip-exp7 \
    --skip-exp8 \
    --skip-exp9

echo ""
echo "========================================================================"
echo " SEGMENT 1 COMPLETE!"
echo " Finished: $(date)"
echo "========================================================================"
echo ""
echo "Outputs created:"
echo "  - results/exp1_summary.json"
echo "  - results/exp2_summary.json"
echo "  - results/exp3_summary.json"
echo "  - results/steering_vectors.pt"
echo ""
echo "Next step: Run ./run_segment2.sh"
echo ""
