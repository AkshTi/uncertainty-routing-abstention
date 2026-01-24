#!/bin/bash
#
# SEGMENT 6 REVALIDATE: Test robustness with newly trained steering vectors
# Expected runtime: 2-3 hours
# Output: exp6_summary.json, exp6 CSV files
#
# REQUIRES: Segment 5 completed (needs steering_vectors_explicit.pt or steering_vectors.pt)
#

set -e  # Exit on error

echo "========================================================================"
echo " SEGMENT 6 REVALIDATE: Robustness Testing"
echo " Using newly trained steering vectors"
echo " Expected runtime: 2-3 hours"
echo " Started: $(date)"
echo "========================================================================"
echo ""

# Verify prerequisites
echo "Checking prerequisites..."

# Check for steering vectors
if [ ! -f "results/steering_vectors_explicit.pt" ] && [ ! -f "results/steering_vectors.pt" ]; then
    echo "ERROR: No steering vectors found!"
    echo "Please run ./run_segment5_regenerate.sh first"
    exit 1
fi

if [ -f "results/steering_vectors_explicit.pt" ]; then
    echo "✓ Found steering_vectors_explicit.pt"
elif [ -f "results/steering_vectors.pt" ]; then
    echo "✓ Found steering_vectors.pt (will use this)"
    echo "  Note: exp6 should work with either file"
fi

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

# Run Exp6
echo "Running Experiment 6 (Robustness)..."
echo "This will test steering on:"
echo "  - Cross-domain generalization (math, science, history, current events)"
echo "  - Prompt variations"
echo "  - Adversarial questions"
echo ""
echo "Expected improvements over previous run:"
echo "  - Overall abstention: 25% → 60-80%"
echo "  - Mathematics: 10% → 40-60%"
echo "  - Science: 30% → 50-70%"
echo "  - History: 20% → 60-80%"
echo ""

python experiment6_publication_ready.py

echo ""
echo "========================================================================"
echo " SEGMENT 6 REVALIDATE COMPLETE!"
echo " Finished: $(date)"
echo "========================================================================"
echo ""

# Analyze results
if [ -f "results/exp6a_cross_domain.csv" ]; then
    echo "✓ exp6a_cross_domain.csv created"

    # Quick analysis
    echo ""
    echo "Quick Results Summary:"
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

for domain in df['domain'].unique():
    b_domain = baseline[baseline['domain'] == domain]
    s_domain = steered[steered['domain'] == domain]
    if len(b_domain) > 0:
        print(f'{domain}:')
        print(f'  Baseline: {b_domain[\"abstained\"].mean():.1%}')
        print(f'  Steered:  {s_domain[\"abstained\"].mean():.1%}')
        print(f'  Improvement: {(s_domain[\"abstained\"].mean() - b_domain[\"abstained\"].mean()):.1%}')
" || echo "Could not analyze results (pandas not available)"

else
    echo "❌ ERROR: exp6a_cross_domain.csv not created!"
    exit 1
fi

echo ""
echo "Outputs created:"
echo "  - results/exp6a_cross_domain.csv"
echo "  - results/exp6b_prompt_variations.csv"
echo "  - results/exp6c_adversarial.csv"
echo "  - results/exp6_robustness_analysis.png"
echo ""
echo "Next step:"
echo "  sbatch --job-name=seg7 slurm_segment.sh ./run_segment7_revalidate.sh"
echo ""
