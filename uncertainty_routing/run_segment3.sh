#!/bin/bash
#
# SEGMENT 3: Validation (Exp6-7)
# Expected runtime: 3-4 hours
# Output: Robustness + safety results
#
# REQUIRES: Segments 1-2 completed (needs steering_vectors_explicit.pt, exp5_summary.json)
#

set -e  # Exit on error

echo "========================================================================"
echo " SEGMENT 3: Validation (Exp6-7)"
echo " Expected runtime: 3-4 hours"
echo " Started: $(date)"
echo "========================================================================"
echo ""

# Verify prerequisites
echo "Checking prerequisites..."

# Check for steering vectors (primary requirement)
if [ ! -f "results/steering_vectors_explicit.pt" ]; then
    # Fallback: check if regular steering vectors exist (can be used)
    if [ ! -f "results/steering_vectors.pt" ]; then
        echo "ERROR: No steering vectors found!"
        echo "Please run ./run_segment1.sh and ./run_segment2.sh first"
        exit 1
    else
        echo "⚠️  Using steering_vectors.pt (explicit version not found)"
        echo "   This may work but exp5 should create steering_vectors_explicit.pt"
    fi
fi

# Check for Exp5 results (for optimal epsilon)
if [ ! -f "results/exp5_summary.json" ] && [ ! -f "results/exp5_raw_results.csv" ]; then
    echo "⚠️  Warning: Exp5 results not found - will use default epsilon"
else
    echo "✓ Exp5 results found"
fi

echo "✓ Prerequisites satisfied"
echo ""

# Check GPU
echo "Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
echo ""

# Run Exp6-7 only
echo "Running Experiments 6-7..."
python run_complete_pipeline_v2.py \
    --mode standard \
    --skip-exp1 \
    --skip-exp2 \
    --skip-exp3 \
    --skip-exp4 \
    --skip-exp5 \
    --skip-exp8 \
    --skip-exp9

echo ""
echo "========================================================================"
echo " SEGMENT 3 COMPLETE!"
echo " Finished: $(date)"
echo "========================================================================"
echo ""
echo "Outputs created:"
echo "  - results/exp6_summary.json"
echo "  - results/exp7_summary.json"
echo ""
echo "Next step: Run ./run_segment4.sh (CRITICAL)"
echo ""
