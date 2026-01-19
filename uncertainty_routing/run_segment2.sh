#!/bin/bash
#
# SEGMENT 2: Applications (Exp4-5)
# Expected runtime: 3-4 hours
# Output: Gate independence + trustworthiness results
#
# REQUIRES: Segment 1 completed (needs steering_vectors.pt)
#

set -e  # Exit on error

echo "========================================================================"
echo " SEGMENT 2: Applications (Exp4-5)"
echo " Expected runtime: 3-4 hours"
echo " Started: $(date)"
echo "========================================================================"
echo ""

# Verify prerequisites
echo "Checking prerequisites..."
if [ ! -f "results/steering_vectors.pt" ]; then
    echo "ERROR: steering_vectors.pt not found!"
    echo "Please run ./run_segment1.sh first"
    exit 1
fi

if [ ! -f "results/exp3_summary.json" ]; then
    echo "ERROR: exp3_summary.json not found!"
    echo "Please run ./run_segment1.sh first"
    exit 1
fi

echo "✓ Prerequisites found"
echo ""

# Check GPU
echo "Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
echo ""

# Run Exp4-5 only
echo "Running Experiments 4-5..."
python run_complete_pipeline_v2.py \
    --mode standard \
    --skip-exp1 \
    --skip-exp2 \
    --skip-exp3 \
    --skip-exp6 \
    --skip-exp7 \
    --skip-exp8 \
    --skip-exp9

echo ""
echo "========================================================================"
echo " SEGMENT 2 COMPLETE!"
echo " Finished: $(date)"
echo "========================================================================"
echo ""
echo "Outputs created:"
echo "  - results/exp4_summary.json"
echo "  - results/exp5_summary.json"
echo "  - results/steering_vectors_explicit.pt"
echo ""
echo "Next step: Run ./run_segment3.sh"
echo ""
