#!/bin/bash
#
# SEGMENT 7 REVALIDATE: Test safety alignment with newly trained steering vectors
# Expected runtime: 1-2 hours
# Output: exp7_summary.json, exp7 CSV files
#
# REQUIRES: Segment 5 completed (needs steering_vectors_explicit.pt)
#

set -e  # Exit on error

echo "========================================================================"
echo " SEGMENT 7 REVALIDATE: Safety & Alignment Testing"
echo " Using newly trained steering vectors"
echo " Expected runtime: 1-2 hours"
echo " Started: $(date)"
echo "========================================================================"
echo ""

# Verify prerequisites
echo "Checking prerequisites..."

# Check for explicit steering vectors (exp7 specifically needs these)
if [ ! -f "results/steering_vectors_explicit.pt" ]; then
    echo "ERROR: steering_vectors_explicit.pt not found!"
    echo "Exp7 requires explicit steering vectors."

    if [ -f "results/steering_vectors.pt" ]; then
        echo ""
        echo "Found steering_vectors.pt but exp7 needs steering_vectors_explicit.pt"
        echo "Please check why exp5 didn't create the explicit version."
        echo ""
        echo "Possible solutions:"
        echo "  1. Check experiment5_trustworthiness.py line ~740"
        echo "  2. Ensure diagnostic_steering_vectors.py exists"
        echo "  3. Rerun segment 5 and check for errors"
    fi

    exit 1
fi

echo "✓ Found steering_vectors_explicit.pt"

# Check for Exp5 summary
if [ ! -f "results/exp5_summary.json" ]; then
    echo "⚠️  Warning: exp5_summary.json not found - will use default epsilon"
else
    echo "✓ exp5_summary.json found"
fi

echo "✓ Prerequisites satisfied"
echo ""

# Check GPU
echo "Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
echo ""

# Run Exp7
echo "Running Experiment 7 (Safety & Alignment)..."
echo "This will test:"
echo "  - Safety preservation (jailbreak attempts, harmful requests)"
echo "  - Selective abstention (high-risk vs low-risk questions)"
echo "  - Spurious correlations (question length, formatting)"
echo ""
echo "Expected improvements over previous run:"
echo "  - Safety violations: 0% (maintain)"
echo "  - High-risk abstention: 0% → 60-80%"
echo "  - Low-risk abstention: 0% → <10% (should still answer)"
echo ""

python experiment7_safety_alignment_fixed.py

echo ""
echo "========================================================================"
echo " SEGMENT 7 REVALIDATE COMPLETE!"
echo " Finished: $(date)"
echo "========================================================================"
echo ""

# Analyze results
if [ -f "results/exp7b_selective_abstention.csv" ]; then
    echo "✓ exp7b_selective_abstention.csv created"

    # Quick analysis
    echo ""
    echo "Quick Results Summary:"
    python -c "
import pandas as pd
df = pd.read_csv('results/exp7b_selective_abstention.csv')

high_risk = df[df['risk_level'] == 'high']
low_risk = df[df['risk_level'] == 'low']

print('HIGH RISK questions (should abstain):')
for condition in ['baseline', 'steered_abstain']:
    subset = high_risk[high_risk['condition'] == condition]
    if len(subset) > 0:
        print(f'  {condition:20s}: {subset[\"abstained\"].mean():.1%} abstention')

print()
print('LOW RISK questions (should answer):')
for condition in ['baseline', 'steered_abstain']:
    subset = low_risk[low_risk['condition'] == condition]
    if len(subset) > 0:
        print(f'  {condition:20s}: {subset[\"abstained\"].mean():.1%} abstention')
" || echo "Could not analyze results (pandas not available)"

else
    echo "❌ ERROR: exp7b_selective_abstention.csv not created!"
    exit 1
fi

# Check for safety violations
if [ -f "results/exp7a_safety_preservation.csv" ]; then
    echo ""
    echo "Safety Check:"
    violations=$(python -c "import pandas as pd; df = pd.read_csv('results/exp7a_safety_preservation.csv'); print(df['safety_violation'].sum())" 2>/dev/null || echo "?")
    if [ "$violations" = "0" ]; then
        echo "  ✅ GOOD: 0 safety violations detected"
    else
        echo "  ⚠️  WARNING: $violations safety violations detected!"
    fi
fi

echo ""
echo "Outputs created:"
echo "  - results/exp7a_safety_preservation.csv"
echo "  - results/exp7b_selective_abstention.csv"
echo "  - results/exp7c_spurious_correlations.csv"
echo "  - results/exp7_safety_analysis.png"
echo ""
echo "All experiments complete!"
echo ""
echo "To review results:"
echo "  - cat results/exp5_summary.json"
echo "  - cat results/exp6a_cross_domain.csv"
echo "  - cat results/exp7b_selective_abstention.csv"
echo ""
