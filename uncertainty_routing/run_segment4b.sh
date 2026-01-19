#!/bin/bash
#
# SEGMENT 4B: Interpretability Analysis (Exp9) ⭐
# Expected runtime: 3-4 hours
# Output: Vector structure and semantic analysis
#
# THIS IS THE SECOND MOST IMPORTANT EXPERIMENT FOR ACCEPTANCE
# Split from Segment 4 if you need shorter GPU sessions
#

set -e  # Exit on error

echo "========================================================================"
echo " SEGMENT 4B: Interpretability Analysis (Exp9) ⭐"
echo " Expected runtime: 3-4 hours"
echo " Started: $(date)"
echo "========================================================================"
echo ""
echo "⚠️  THIS IS CRITICAL FOR MECHANISTIC INSIGHT (+25% acceptance)"
echo ""

# Verify prerequisites
echo "Checking prerequisites..."
if [ ! -f "results/steering_vectors_explicit.pt" ]; then
    echo "ERROR: steering_vectors_explicit.pt not found!"
    echo "Please run ./run_segment2.sh first"
    exit 1
fi

echo "✓ Prerequisites found"
echo ""

# Check GPU
echo "Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
echo ""

# Run Exp9 only
echo "Running Experiment 9 (Interpretability)..."
python experiment9_interpretability.py

echo ""
echo "========================================================================"
echo " SEGMENT 4B COMPLETE! ⭐"
echo " Finished: $(date)"
echo "========================================================================"
echo ""
echo "Outputs created:"
echo "  - results/exp9_summary.json ⭐ CRITICAL"
echo "  - results/exp9_interpretability_analysis.png"
echo ""
echo "✓ ALL EXPERIMENTS COMPLETE!"
echo ""
echo "Next step: Check results with ./check_all_results.sh"
echo ""
