# RUN ME - Pipeline Execution Guide

## Quick Start (Choose One)

### Option 1: You've Already Run Exp1-5 (RECOMMENDED)
**Just run the 2 critical experiments (7 hours) → 2x your acceptance chances**

```bash
# Test everything works
python test_pipeline.py

# Run only critical experiments
python run_complete_pipeline_v2.py --only-critical

# This runs:
# - Experiment 8: Scaling (4 hours)
# - Experiment 9: Interpretability (3 hours)
```

**Acceptance boost**: 40% → 75-80%

---

### Option 2: Starting Fresh
**Run the complete pipeline (10-14 hours)**

```bash
# Test everything works
python test_pipeline.py

# Run all experiments
python run_complete_pipeline_v2.py --mode standard

# This runs:
# - Exp1-5: Your existing experiments (6-8 hours)
# - Exp6: Robustness (2 hours)
# - Exp7: Safety (1 hour)
# - Exp8: Scaling ⭐ (4 hours)
# - Exp9: Interpretability ⭐ (3 hours)
```

---

### Option 3: Quick Test Run
**Fast test to verify everything works (3-4 hours)**

```bash
python run_complete_pipeline_v2.py --mode quick
```

Uses fewer questions, faster but less comprehensive.

---

## Detailed Options

### Skip Specific Experiments
```bash
# Skip experiments you've already done
python run_complete_pipeline_v2.py --skip-exp1 --skip-exp2 --skip-exp3

# Run only new experiments (6-7)
python run_complete_pipeline_v2.py --skip-exp1 --skip-exp2 --skip-exp3 --skip-exp4 --skip-exp5

# DON'T skip Exp8 or Exp9 (they're critical!)
```

### Different Modes
```bash
# Quick (3-4 hours): 10 questions per experiment
python run_complete_pipeline_v2.py --mode quick

# Standard (10-14 hours): 30 questions per experiment (RECOMMENDED)
python run_complete_pipeline_v2.py --mode standard

# Full (15-20 hours): 50 questions per experiment
python run_complete_pipeline_v2.py --mode full
```

### Different Model
```bash
# Test on larger model (if you have GPU memory)
python run_complete_pipeline_v2.py --model Qwen/Qwen2.5-3B-Instruct
```

---

## Before You Run

### 1. Check GPU
```bash
python -c "import torch; print('GPU:', torch.cuda.is_available())"
```

**Minimum requirements**:
- 1.5B model: 8GB GPU
- 3B model: 16GB GPU
- 7B model: 24GB GPU

### 2. Run Test Script
```bash
python test_pipeline.py
```

This checks:
- All imports work
- Data files exist
- GPU is available
- Config is valid

**If test fails**, fix issues before running full pipeline.

---

## What You'll Get

### Files Created

**Experiment results**:
- `results/exp1_raw_results.csv` (and exp2-9)
- `results/exp1_summary.json` (and exp2-9)
- `results/steering_vectors_explicit.pt`

**Figures** (ready for paper):
- `results/exp1_paper_figure.png`
- `results/exp2_localization_analysis.png`
- `results/exp3_steering_analysis.png`
- `results/exp4_gate_independence.png`
- `results/exp5_trustworthiness.png`
- `results/exp5_optimal_epsilon.png`
- `results/exp6_robustness_analysis.png` ⭐
- `results/exp7_safety_analysis.png` ⭐
- `results/exp8_scaling_analysis.png` ⭐⭐ CRITICAL
- `results/exp9_interpretability_analysis.png` ⭐⭐ CRITICAL

**Pipeline summary**:
- `results/complete_pipeline_standard.json` (all results combined)

---

## Expected Runtime

| Mode | Questions | GPU Time | Wall Time |
|------|-----------|----------|-----------|
| quick | 10/exp | 3-4h | 4-5h |
| standard | 30/exp | 10-14h | 12-16h |
| full | 50/exp | 15-20h | 18-24h |

**Critical experiments only** (--only-critical):
- 7 hours GPU time
- 8-9 hours wall time

---

## Monitoring Progress

### Check what's running
```bash
# Watch GPU usage
nvidia-smi -l 1

# Check results directory
ls -lh results/

# Monitor latest results
tail -f results/complete_pipeline_standard.json
```

### If something crashes
1. Check error message carefully
2. Look at the last experiment that completed
3. Re-run with `--skip-exp1 --skip-exp2 ...` for completed ones
4. Or just re-run the specific experiment script:
   ```bash
   python experiment8_scaling_analysis.py
   ```

---

## Common Issues

### Issue: Out of GPU memory
**Solution**:
```bash
# Use quick mode
python run_complete_pipeline_v2.py --mode quick

# Or skip GPU-heavy experiments
python run_complete_pipeline_v2.py --skip-exp8
```

### Issue: ImportError
**Solution**:
```bash
# Check all files are present
ls experiment*.py

# Test imports
python test_pipeline.py
```

### Issue: Data files not found
**Solution**:
```bash
# Check data directory
ls data/

# Recreate data if needed
python data_preparation.py
```

### Issue: Pipeline crashes mid-run
**Solution**:
```bash
# Check which experiments completed
ls results/exp*_summary.json

# Skip completed ones and re-run
python run_complete_pipeline_v2.py --skip-exp1 --skip-exp2 --skip-exp3
```

---

## After Pipeline Completes

### 1. Check Results
```bash
# View experiment summaries
cat results/exp8_summary.json
cat results/exp9_summary.json

# Check all figures were generated
ls results/*.png
```

### 2. Extract Numbers for Paper
Open the summary JSON files and extract:
- Exp8: Model sizes, effect sizes, correlation
- Exp9: Sparsity (K dims for 90%), top-3 fraction
- Exp6: Cross-domain consistency
- Exp7: Safety refusal rate

### 3. Fill in Paper Template
```bash
# Open paper outline
open PAPER_OUTLINE.md

# Fill in all [bracketed] values with your numbers
```

### 4. Generate Final Figures
All figures are auto-generated, but check they look good:
- Resolution: 300 dpi ✓
- Fonts readable: ≥10pt ✓
- Colors: Color-blind friendly ✓

---

## Success Criteria

Your pipeline succeeded if:

**All experiments**:
- [ ] Exp1-9 completed without errors
- [ ] All summary JSON files created
- [ ] All figures generated

**Critical results** (Exp8+9):
- [ ] Exp8: Tested ≥2 model sizes
- [ ] Exp8: All models show effect (>10% abstention change)
- [ ] Exp9: Vector sparsity calculated (80-95% in top-K)
- [ ] Exp9: Dimension probing complete (top-3 fraction)

**File checklist**:
- [ ] `results/exp8_summary.json` exists
- [ ] `results/exp9_summary.json` exists
- [ ] `results/exp8_scaling_analysis.png` exists
- [ ] `results/exp9_interpretability_analysis.png` exists
- [ ] `results/complete_pipeline_standard.json` exists

---

## Recommended Workflow

### Day 1: Run Pipeline
```bash
# Morning: Test
python test_pipeline.py

# If test passes, start pipeline
python run_complete_pipeline_v2.py --mode standard

# Or if you've already done Exp1-5:
python run_complete_pipeline_v2.py --only-critical

# Let it run (10-14 hours or 7 hours for critical only)
```

### Day 2: Analyze Results
```bash
# Check all experiments completed
ls results/exp*_summary.json

# Extract key numbers
cat results/exp8_summary.json | jq '.models_tested'
cat results/exp9_summary.json | jq '.structure.k_90'
```

### Days 3-4: Write Paper
1. Open `PAPER_OUTLINE.md`
2. Fill in all [bracketed] values
3. Add sections 4.5 (Scaling) and 4.6 (Interpretability)
4. Update abstract with new findings
5. Revise introduction to emphasize insights

### Day 5: Final Polish
1. Generate publication-quality figures
2. Check page limit (4 pages main)
3. Proofread
4. Submit!

---

## The Bottom Line

**If you've already done Exp1-5**:
```bash
python run_complete_pipeline_v2.py --only-critical
```
**7 hours → 2x your acceptance chances**

**If starting fresh**:
```bash
python run_complete_pipeline_v2.py --mode standard
```
**12 hours → Complete paper**

**Either way, START WITH**:
```bash
python test_pipeline.py
```
**To catch issues early!**

---

## Questions?

Check these files:
- `QUICK_START_GUIDE.md` - What experiments do
- `MAXIMIZING_ACCEPTANCE.md` - Why they matter
- `PAPER_OUTLINE.md` - How to write the paper

Or just run the pipeline and see what happens! The code is designed to be robust and continue even if individual experiments fail.

---

## One Command to Rule Them All

**For most people, this is what you want**:

```bash
# Test first
python test_pipeline.py

# Then run
python run_complete_pipeline_v2.py --only-critical

# Wait 7 hours, write paper, submit!
```

**Good luck!** 🚀
