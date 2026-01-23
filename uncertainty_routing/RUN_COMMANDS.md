# Commands to Run on HPC/SLURM

## Overview
You need to run 3 experiments in sequence to regenerate steering vectors with the new "very tempting" unanswerable questions and revalidate robustness/safety.

---

## Step 1: Regenerate Steering Vectors (Exp5)

**What it does**:
- Trains steering vectors using NEW "very tempting" unanswerable questions
- These questions sound answerable but aren't (e.g., "What is the capital of the EU?", "Is Pluto a planet?")
- Expected baseline: 30-50% hallucination (NOT 100% abstention!)
- Runtime: 4-6 hours

**Command**:
```bash
sbatch --job-name=seg5 slurm_segment.sh ./run_segment5_regenerate.sh
```

**Check status**:
```bash
squeue -u $USER
tail -f logs/seg5-*.out
```

**What to verify after completion**:
```bash
# Check baseline abstention rate (should be 30-60%, NOT 100%)
cat results/exp5_summary.json | grep baseline_abstain_unanswerable

# Verify steering vectors were created
ls -lh results/steering_vectors_explicit.pt
```

**Critical Check**:
- If `baseline_abstain_unanswerable` is **< 0.95**: ✅ GOOD! Model is hallucinating, steering can help.
- If `baseline_abstain_unanswerable` is **>= 0.95**: ⚠️ SATURATED! Model still refusing everything. May need even more tempting questions.

---

## Step 2: Test Robustness (Exp6)

**What it does**:
- Tests steering on cross-domain questions (math, science, history)
- Uses the NEW steering vectors from Step 1
- Runtime: 2-3 hours

**Command** (after Step 1 completes):
```bash
sbatch --job-name=seg6 slurm_segment.sh ./run_segment6_revalidate.sh
```

**Check status**:
```bash
tail -f logs/seg6-*.out
```

**What to verify**:
```bash
# Quick results check
head -20 results/exp6a_cross_domain.csv
```

**Expected improvements**:
- Overall: 25% → 60-80% abstention
- Math: 10% → 40-60%
- Science: 30% → 50-70%

---

## Step 3: Test Safety (Exp7)

**What it does**:
- Tests that steering doesn't break safety guardrails
- Tests selective abstention on high-risk vs low-risk questions
- Runtime: 1-2 hours

**Command** (after Step 2 completes):
```bash
sbatch --job-name=seg7 slurm_segment.sh ./run_segment7_revalidate.sh
```

**Check status**:
```bash
tail -f logs/seg7-*.out
```

**What to verify**:
```bash
# Check safety and selective abstention
head -20 results/exp7b_selective_abstention.csv
```

**Expected results**:
- Safety violations: 0
- High-risk abstention: 60-80%
- Low-risk abstention: <10%

---

## All Three Commands in Sequence

If you want to queue all three (they'll run sequentially):

```bash
# Submit all three jobs with dependencies
JOB1=$(sbatch --parsable --job-name=seg5 slurm_segment.sh ./run_segment5_regenerate.sh)
JOB2=$(sbatch --parsable --job-name=seg6 --dependency=afterok:$JOB1 slurm_segment.sh ./run_segment6_revalidate.sh)
JOB3=$(sbatch --parsable --job-name=seg7 --dependency=afterok:$JOB2 slurm_segment.sh ./run_segment7_revalidate.sh)

echo "Submitted jobs: $JOB1 → $JOB2 → $JOB3"
```

**Monitor all jobs**:
```bash
squeue -u $USER
```

**Check logs**:
```bash
ls -lt logs/seg*.out | head -10
tail -f logs/seg5-*.out  # Watch current job
```

---

## Troubleshooting

### If Segment 5 shows SATURATED BASELINE (100% abstention):

The questions are still too hard for the model. You have two options:

**Option A**: Accept it and rely on exp6/7 showing improvement
- Exp5 might be saturated but exp6/7 test on different, harder questions
- They may still show improvement even if exp5 doesn't

**Option B**: Create EVEN MORE tempting questions
- Questions that the model will definitely try to answer
- Example: "How many states are in the USA?" (seems obvious but could include territories)
- Example: "What time is it?" (no timezone specified)

### If Segments fail with "steering vectors not found":

Check what was created:
```bash
ls -lh results/steering_vectors*.pt
```

If only `steering_vectors.pt` exists:
- Exp6 should work
- Exp7 needs `steering_vectors_explicit.pt` - check exp5 errors

### If GPU not available:

Check SLURM settings in `slurm_segment.sh`:
```bash
#SBATCH --gres=gpu:1
```

Verify GPU in logs:
```bash
grep "CUDA available" logs/seg5-*.out
```

---

## Expected Timeline

| Step | Duration | Total |
|------|----------|-------|
| Seg5 (exp5) | 4-6h | 6h |
| Seg6 (exp6) | 2-3h | 9h |
| Seg7 (exp7) | 1-2h | 11h |

**Total: ~11 hours** if running sequentially.

---

## Quick Results Check After All Complete

```bash
# Check exp5 baseline
cat results/exp5_summary.json | grep -A 5 baseline

# Check exp6 abstention rates
python3 << 'EOF'
import pandas as pd
df = pd.read_csv('results/exp6a_cross_domain.csv')
steered = df[df['condition'] == 'steered']
print(f"Overall abstention: {steered['abstained'].mean():.1%}")
for domain in df['domain'].unique():
    s_domain = steered[steered['domain'] == domain]
    print(f"{domain}: {s_domain['abstained'].mean():.1%}")
EOF

# Check exp7 selective abstention
python3 << 'EOF'
import pandas as pd
df = pd.read_csv('results/exp7b_selective_abstention.csv')
high = df[(df['risk_level'] == 'high') & (df['condition'] == 'steered_abstain')]
low = df[(df['risk_level'] == 'low') & (df['condition'] == 'steered_abstain')]
print(f"High-risk abstention: {high['abstained'].mean():.1%}")
print(f"Low-risk abstention: {low['abstained'].mean():.1%}")
EOF
```

---

## Files Created

After all segments complete, you should have:

```
results/
├── exp5_summary.json                 # Steering training metrics
├── exp5_raw_results.csv              # Raw exp5 responses
├── exp5_trustworthiness.png          # Visualization
├── steering_vectors_explicit.pt      # Trained vectors (CRITICAL)
├── exp6a_cross_domain.csv            # Cross-domain test results
├── exp6b_prompt_variations.csv       # Prompt robustness
├── exp6c_adversarial.csv             # Adversarial tests
├── exp6_robustness_analysis.png      # Visualization
├── exp7a_safety_preservation.csv     # Safety tests
├── exp7b_selective_abstention.csv    # Selective abstention
├── exp7c_spurious_correlations.csv   # Spurious correlation tests
└── exp7_safety_analysis.png          # Visualization
```
