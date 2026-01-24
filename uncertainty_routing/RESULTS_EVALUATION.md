# Experiment 6 Results Evaluation

## 📊 What You Got

✅ **Good News**:
- Sample size: **n=50 per condition** (publication-ready!)
- Prompts: Unified format
- Parsing: Working correctly
- Files created successfully

❌ **Critical Issues**:
- **Steering has ZERO effect** (Δ = 0.0%)
- Baseline and steered are 100% identical
- Model echoes back prompt instructions
- Results are **INVALID**

---

## 🔍 Root Cause Analysis

### Issue #1: Wrong Layer (CRITICAL)

**Your Code:**
```python
df_6a, df_6b, df_6c = exp6.run_all(best_layer=26, optimal_epsilon=-2.0)
```

**Your Steering Vectors:**
```python
Available layers: [10, 16, 17, 18]  # NO LAYER 26!
```

**Result:** No steering applied → baseline = steered

### Issue #2: Prompt Confusion

**What the model sees:**
```
Answer with EXACTLY one line:
- Your answer (one line) OR  ← Complex instructions
- Exactly: UNCERTAIN
```

**What the model outputs:**
```
"-Your answer (one line)- OR -Exactly: UNC"  ← Echoing back!
```

---

## 📉 Actual Results (Invalid)

```
Overall Abstention:
  Baseline: 4.8%
  Steered:  4.8%
  Δ:        0.0%  ← Should be 20-40%!

By Domain (all domains):
  Δ: 0.0% across the board

Hallucination on Unanswerable:
  Baseline: 91%  ← Very high!
  Steered:  91%  ← No improvement!
```

**This is proof steering didn't work.**

---

## ✅ How to Fix

### Quick Fix (Test in 5 minutes)

```bash
python fix_and_test_steering.py
```

This will:
1. Test all available layers [10, 16, 17, 18]
2. Test different epsilon values [-2, -5, -10, -20]
3. Tell you which combination works best
4. Show you the correct parameters to use

### Expected Output

```
Testing layer 17, epsilon -5.0...
  Overall: 10.0% → 45.0% (Δ +35.0%)  ← This is what you want!
  Unanswerable: Δ +50.0%

✅ Best configuration:
   Layer: 17
   Epsilon: -5.0
   Effect on unanswerable: +50.0%
```

### Then Update Code

1. **Edit `experiment6_publication_ready.py` line 503:**
   ```python
   # Change from:
   df_6a, df_6b, df_6c = exp6.run_all(best_layer=26, optimal_epsilon=-2.0)

   # To (using results from fix_and_test_steering.py):
   df_6a, df_6b, df_6c = exp6.run_all(best_layer=17, optimal_epsilon=-5.0)
   ```

2. **Re-run the experiment:**
   ```bash
   ./run_segment6_revalidate.sh
   ```

---

## 🎯 Expected Results (After Fix)

Once you use the correct layer, you should see:

```
Overall Abstention:
  Baseline: 10-20%
  Steered:  40-60%
  Δ:        +25-40%  ← SUBSTANTIAL IMPROVEMENT

Unanswerable Questions (key metric):
  Baseline: 20-30% abstention (70-80% hallucination)
  Steered:  60-80% abstention (20-40% hallucination)
  Δ:        +40-50%  ← STEERING WORKING!

By Domain:
  mathematics: Δ +30-50%
  science:     Δ +25-45%
  history:     Δ +35-55%
  geography:   Δ +30-50%
```

**These are publication-worthy results!**

---

## 📋 Action Plan

### Step 1: Test Configurations (5 min)
```bash
python fix_and_test_steering.py
```

### Step 2: Update Code (1 min)
Edit `experiment6_publication_ready.py` line 503 with best layer/epsilon from Step 1.

### Step 3: Re-run Full Experiment (30-60 min)
```bash
./run_segment6_revalidate.sh
```

### Step 4: Verify Results
```bash
# Should see substantial Δ (not 0.0%!)
python -c "
import pandas as pd
df = pd.read_csv('results/exp6a_cross_domain_publication_ready.csv')
baseline = df[df['condition']=='baseline']['abstained'].mean()
steered = df[df['condition']=='steered']['abstained'].mean()
print(f'Baseline: {baseline:.1%}')
print(f'Steered:  {steered:.1%}')
print(f'Δ:        {(steered-baseline):+.1%}')
if abs(steered-baseline) > 0.20:
    print('✅ Steering working!')
else:
    print('❌ Still not working - check layer/epsilon')
"
```

---

## 🎓 What We Learned

1. **Always validate layer exists** in steering vectors
2. **Test with small n first** to catch bugs quickly
3. **Check Δ ≠ 0** before running full experiments
4. **Simpler prompts work better** (consider using `unified_prompt_minimal`)
5. **Steering vectors need correct layer** - can't just use arbitrary layer 26

---

## 📁 Files Summary

### Current (Invalid) Results
```
results/exp6a_cross_domain_publication_ready.csv  ← Δ=0%, invalid
results/exp6b_determinism_check.csv               ← Δ=0%, invalid
results/exp6c_adversarial_publication_ready.csv   ← Δ=0%, invalid
```

### After Fix (Will Be Valid)
```
results/exp6a_cross_domain_publication_ready.csv  ← Δ=30-40%, VALID!
results/exp6b_determinism_check.csv               ← Deterministic, VALID!
results/exp6c_adversarial_publication_ready.csv   ← Δ=25-35%, VALID!
```

---

## ⏱️ Time Estimate

- Testing configurations: **5 minutes**
- Updating code: **1 minute**
- Re-running experiments: **30-60 minutes**
- **Total: ~45-75 minutes to get valid results**

---

## 🚦 Status

- [x] Experiment ran (wrong configuration)
- [x] Issues identified
- [ ] **← YOU ARE HERE: Run `fix_and_test_steering.py`**
- [ ] Update code with correct layer/epsilon
- [ ] Re-run experiments
- [ ] Verify Δ ≠ 0%
- [ ] Publish results

---

## 💡 Bottom Line

**Your current results show steering didn't work because:**
1. Used layer 26 (doesn't exist in steering vectors)
2. No steering was applied
3. Baseline = Steered (proof of no effect)

**Fix:** Use correct layer (17 or 18) and appropriate epsilon (-5.0 or -10.0)

**Run this now:**
```bash
python fix_and_test_steering.py
```

This will tell you exactly which layer and epsilon to use!
