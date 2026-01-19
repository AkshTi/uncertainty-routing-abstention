#!/bin/bash
#
# SEGMENT 4: Critical Experiments (Exp8-9) ⭐⭐
# Expected runtime: 7-8 hours (LONGEST SEGMENT)
# Output: Scaling + interpretability results
#
# THIS IS THE MOST IMPORTANT SEGMENT FOR YOUR PAPER
# REQUIRES: Segments 1-2 completed (needs steering_vectors_explicit.pt, exp5_summary.json)
#
# NOTE: This is the longest segment. If you need to split further, see run_segment4a.sh and run_segment4b.sh
#

set -e  # Exit on error

echo "========================================================================"
echo " SEGMENT 4: Critical Experiments (Exp8-9) ⭐⭐"
echo " Expected runtime: 7-8 hours"
echo " Started: $(date)"
echo "========================================================================"
echo ""
echo "⚠️  THIS IS THE MOST CRITICAL SEGMENT FOR ACCEPTANCE"
echo "⚠️  This segment takes 7-8 hours - ensure GPU stability"
echo ""

# Verify prerequisites
echo "Checking prerequisites..."
if [ ! -f "results/steering_vectors_explicit.pt" ]; then
    echo "ERROR: steering_vectors_explicit.pt not found!"
    echo "Please run ./run_segment2.sh first"
    exit 1
fi

if [ ! -f "results/exp5_summary.json" ]; then
    echo "ERROR: exp5_summary.json not found!"
    echo "Please run ./run_segment2.sh first"
    exit 1
fi

echo "✓ Prerequisites found"
echo ""

# Check GPU
echo "Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
echo ""

# Run Exp8-9 only (using --only-critical mode)
echo "Running Experiments 8-9 (Critical)..."
python run_complete_pipeline_v2.py \
    --only-critical \
    --mode standard

echo ""
echo "========================================================================"
echo " SEGMENT 4 COMPLETE! ⭐⭐"
echo " Finished: $(date)"
echo "========================================================================"
echo ""
echo "Outputs created:"
echo "  - results/exp8_summary.json ⭐ CRITICAL"
echo "  - results/exp9_summary.json ⭐ CRITICAL"
echo ""
echo "✓ ALL EXPERIMENTS COMPLETE!"
echo ""
echo "Next step: Check results with ./check_all_results.sh"
echo ""
