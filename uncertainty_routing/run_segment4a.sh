#!/bin/bash
#
# SEGMENT 4A: Scaling Analysis (Exp8) ⭐
# Expected runtime: 4-5 hours
# Output: Multi-model scaling results
#
# THIS IS THE MOST IMPORTANT EXPERIMENT FOR ACCEPTANCE
# Split from Segment 4 if you need shorter GPU sessions
#

set -e  # Exit on error

echo "========================================================================"
echo " SEGMENT 4A: Scaling Analysis (Exp8) ⭐"
echo " Expected runtime: 4-5 hours"
echo " Started: $(date)"
echo "========================================================================"
echo ""
echo "⚠️  THIS IS THE MOST CRITICAL EXPERIMENT (+30% acceptance)"
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

# Run Exp8 only
echo "Running Experiment 8 (Scaling)..."
python experiment8_scaling_analysis.py

echo ""
echo "========================================================================"
echo " SEGMENT 4A COMPLETE! ⭐"
echo " Finished: $(date)"
echo "========================================================================"
echo ""
echo "Outputs created:"
echo "  - results/exp8_summary.json ⭐ CRITICAL"
echo "  - results/exp8_scaling_analysis.png"
echo ""
echo "Next step: Run ./run_segment4b.sh"
echo ""
