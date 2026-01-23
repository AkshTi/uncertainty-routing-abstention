#!/bin/bash
#
# SEGMENT 5 REGENERATE: Retrain steering vectors with VERY TEMPTING unanswerable questions
# Expected runtime: 4-6 hours
# Output: steering_vectors_explicit.pt, exp5_summary.json
#
# CRITICAL: Uses dataset_clearly_unanswerable_very_tempting.json
#           These questions are designed to cause baseline hallucination (30-50%)
#           instead of 100% abstention.
#

set -e  # Exit on error

echo "========================================================================"
echo " SEGMENT 5 REGENERATE: Steering Vector Retraining"
echo " Using VERY TEMPTING unanswerable questions"
echo " Expected runtime: 4-6 hours"
echo " Started: $(date)"
echo "========================================================================"
echo ""

# Check for updated training data
echo "Checking for new training data..."
if [ ! -f "data/dataset_clearly_unanswerable_very_tempting.json" ]; then
    echo "ERROR: data/dataset_clearly_unanswerable_very_tempting.json not found!"
    echo "This file should have been created with tempting questions."
    exit 1
fi
echo "✓ Found very tempting unanswerable questions dataset"
echo ""

# Verify experiment5_trustworthiness.py is updated to use it
echo "Verifying exp5 is configured correctly..."
if grep -q "dataset_clearly_unanswerable_very_tempting.json" experiment5_trustworthiness.py; then
    echo "✓ Exp5 configured to use very tempting questions"
else
    echo "ERROR: experiment5_trustworthiness.py not updated!"
    echo "It should load dataset_clearly_unanswerable_very_tempting.json"
    exit 1
fi
echo ""

# Delete old steering vectors to force regeneration
echo "Cleaning old steering vectors..."
rm -f results/steering_vectors.pt results/steering_vectors_explicit.pt
echo "✓ Deleted old vectors (will regenerate)"
echo ""

# Check GPU
echo "Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
echo ""

# Run Exp5
echo "Running Experiment 5 (Trustworthiness)..."
echo "This will:"
echo "  1. Load 20 answerable + 20 VERY tempting unanswerable questions"
echo "  2. Train steering vectors via contrastive learning"
echo "  3. Test epsilon sweep (-50 to +50)"
echo "  4. Select optimal epsilon and layer"
echo ""
echo "Expected outcomes:"
echo "  - Baseline abstention on unanswerables: 30-50% (NOT 100%!)"
echo "  - Baseline hallucination: 50-70%"
echo "  - Best steering abstention: 85-95%"
echo "  - Improvement: +40 to +60 percentage points"
echo ""

python experiment5_trustworthiness.py

echo ""
echo "========================================================================"
echo " SEGMENT 5 REGENERATE COMPLETE!"
echo " Finished: $(date)"
echo "========================================================================"
echo ""

# Check results
if [ -f "results/exp5_summary.json" ]; then
    echo "✓ exp5_summary.json created"

    # Extract baseline abstention rate
    baseline_abstain=$(python -c "import json; data=json.load(open('results/exp5_summary.json')); print(data['baseline_abstain_unanswerable'])")

    echo ""
    echo "CRITICAL CHECK: Baseline Abstention Rate"
    echo "  Baseline abstention: $baseline_abstain"

    if [ $(python -c "print(1 if $baseline_abstain >= 0.95 else 0)") -eq 1 ]; then
        echo ""
        echo "⚠️  WARNING: BASELINE STILL SATURATED!"
        echo "  Baseline abstention is ${baseline_abstain} (>95%)"
        echo "  The model is STILL refusing all unanswerable questions!"
        echo "  This means:"
        echo "    - Questions are still too obviously unanswerable"
        echo "    - Steering has little room to improve"
        echo "    - May need even more tempting questions"
        echo ""
        echo "  However, exp6/7 may still show improvement on their harder test sets."
        echo "  Proceed to run segments 6 and 7 to check."
    else
        echo "  ✅ GOOD! Baseline is NOT saturated (<95%)"
        echo "  Model is hallucinating on some questions - steering can improve this!"
    fi
else
    echo "❌ ERROR: exp5_summary.json not created!"
    exit 1
fi

if [ -f "results/steering_vectors_explicit.pt" ]; then
    echo "✓ steering_vectors_explicit.pt created"
else
    echo "⚠️  WARNING: steering_vectors_explicit.pt not found!"
    echo "   Checking for steering_vectors.pt instead..."
    if [ -f "results/steering_vectors.pt" ]; then
        echo "   ✓ steering_vectors.pt exists (may work for exp6)"
        echo "   ⚠️  But exp7 requires steering_vectors_explicit.pt"
    else
        echo "   ❌ No steering vectors found at all!"
        exit 1
    fi
fi

echo ""
echo "Outputs created:"
echo "  - results/exp5_summary.json"
echo "  - results/exp5_raw_results.csv"
echo "  - results/exp5_trustworthiness.png"
echo "  - results/steering_vectors_explicit.pt (or steering_vectors.pt)"
echo ""
echo "Next steps:"
echo "  1. Review exp5_summary.json to check baseline abstention rate"
echo "  2. If baseline < 95%: GOOD! Run segments 6 and 7"
echo "  3. If baseline >= 95%: SATURATED! May need even more tempting questions"
echo ""
echo "To run next segments:"
echo "  sbatch --job-name=seg6 slurm_segment.sh ./run_segment6_revalidate.sh"
echo "  sbatch --job-name=seg7 slurm_segment.sh ./run_segment7_revalidate.sh"
echo ""
