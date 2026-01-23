#!/bin/bash
#
# SEGMENT 3: Validation (Exp6-7) - UPDATED TO RUN DIRECTLY
# Expected runtime: 3-4 hours
# Output: Robustness + safety results
#
# REQUIRES: Exp5 completed (needs steering_vectors_explicit.pt, exp5_summary.json)
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
        echo "Please run exp5 first to generate steering vectors"
        echo "Run: sbatch --job-name=seg5 slurm_segment.sh ./run_segment5_regenerate.sh"
        exit 1
    else
        echo "⚠️  Using steering_vectors.pt (explicit version not found)"
        echo "   Exp6 should work but Exp7 may fail without explicit vectors"
    fi
else
    echo "✓ Found steering_vectors_explicit.pt"
fi

# Check for Exp5 results (for optimal epsilon)
if [ ! -f "results/exp5_summary.json" ]; then
    echo "⚠️  Warning: Exp5 results not found - will use default epsilon"
else
    echo "✓ Exp5 results found"

    # Show baseline abstention rate
    baseline_abstain=$(python -c "import json; data=json.load(open('results/exp5_summary.json')); print(f\"{data['baseline_abstain_unanswerable']:.1%}\")" 2>/dev/null || echo "unknown")
    echo "  Baseline abstention: $baseline_abstain"
fi

echo "✓ Prerequisites satisfied"
echo ""

# Check GPU
echo "Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
echo ""

# ============================================================================
# Run Experiment 6: Robustness Testing
# ============================================================================

echo "========================================================================"
echo " EXPERIMENT 6: Robustness Testing"
echo "========================================================================"
echo ""
echo "This will test steering on:"
echo "  - Cross-domain generalization (math, science, history, current events)"
echo "  - Prompt variations"
echo "  - Adversarial questions"
echo ""

python experiment6_robustness.py

if [ $? -ne 0 ]; then
    echo "ERROR: Experiment 6 failed!"
    exit 1
fi

echo ""
echo "✓ Experiment 6 complete!"
echo ""

# Quick results check
if [ -f "results/exp6a_cross_domain.csv" ]; then
    echo "Quick Results Summary (Exp6):"
    python -c "
import pandas as pd
df = pd.read_csv('results/exp6a_cross_domain.csv')
baseline = df[df['condition'] == 'baseline']
steered = df[df['condition'] == 'steered']

print(f'Overall Abstention:')
print(f'  Baseline: {baseline[\"abstained\"].mean():.1%}')
print(f'  Steered:  {steered[\"abstained\"].mean():.1%}')
print(f'  Improvement: {(steered[\"abstained\"].mean() - baseline[\"abstained\"].mean()):.1%}')
print()

print('By Domain:')
for domain in sorted(df['domain'].unique()):
    b_domain = baseline[baseline['domain'] == domain]
    s_domain = steered[steered['domain'] == domain]
    if len(b_domain) > 0:
        print(f'  {domain:15s}: {b_domain[\"abstained\"].mean():.1%} → {s_domain[\"abstained\"].mean():.1%} ({(s_domain[\"abstained\"].mean() - b_domain[\"abstained\"].mean()):+.1%})')
" 2>/dev/null || echo "Could not analyze results (pandas not available)"
    echo ""
fi

# ============================================================================
# Run Experiment 7: Safety & Alignment Testing
# ============================================================================

echo "========================================================================"
echo " EXPERIMENT 7: Safety & Alignment Testing"
echo "========================================================================"
echo ""
echo "This will test:"
echo "  - Safety preservation (jailbreak attempts, harmful requests)"
echo "  - Selective abstention (high-risk vs low-risk questions)"
echo "  - Spurious correlations (question length, formatting)"
echo ""

python experiment7_safety_alignment.py

if [ $? -ne 0 ]; then
    echo "ERROR: Experiment 7 failed!"
    exit 1
fi

echo ""
echo "✓ Experiment 7 complete!"
echo ""

# Quick results check
if [ -f "results/exp7b_selective_abstention.csv" ]; then
    echo "Quick Results Summary (Exp7):"
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
" 2>/dev/null || echo "Could not analyze results (pandas not available)"
    echo ""
fi

# Check for safety violations
if [ -f "results/exp7a_safety_preservation.csv" ]; then
    violations=$(python -c "import pandas as pd; df = pd.read_csv('results/exp7a_safety_preservation.csv'); print(df['safety_violation'].sum())" 2>/dev/null || echo "?")
    if [ "$violations" = "0" ]; then
        echo "Safety Check: ✅ GOOD - 0 violations detected"
    elif [ "$violations" = "?" ]; then
        echo "Safety Check: Could not check (pandas not available)"
    else
        echo "Safety Check: ⚠️  WARNING - $violations violations detected!"
    fi
    echo ""
fi

echo ""
echo "========================================================================"
echo " SEGMENT 3 COMPLETE!"
echo " Finished: $(date)"
echo "========================================================================"
echo ""
echo "Outputs created:"
echo "  Experiment 6:"
echo "    - results/exp6a_cross_domain.csv"
echo "    - results/exp6b_prompt_variations.csv"
echo "    - results/exp6c_adversarial.csv"
echo "    - results/exp6_robustness_analysis.png"
echo ""
echo "  Experiment 7:"
echo "    - results/exp7a_safety_preservation.csv"
echo "    - results/exp7b_selective_abstention.csv"
echo "    - results/exp7c_spurious_correlations.csv"
echo "    - results/exp7_safety_analysis.png"
echo ""
echo "To review detailed results:"
echo "  cat results/exp6a_cross_domain.csv | head -20"
echo "  cat results/exp7b_selective_abstention.csv | head -20"
echo ""
