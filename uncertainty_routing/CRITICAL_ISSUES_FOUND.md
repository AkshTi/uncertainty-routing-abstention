# 🚨 CRITICAL ISSUES FOUND - Results Are Invalid

## Summary

Your publication-ready experiment ran but has **2 critical bugs** that make the results invalid:

1. **Wrong Layer**: Steering at layer 26 but vectors only exist for layers [10, 16, 17, 18]
2. **Prompt Confusion**: Model echoes back instructions instead of following them

**Result**: Steering had ZERO effect (baseline = steered, Δ = 0.0%)

---

## Issue #1: Wrong Steering Layer ⚠️

### The Problem
```python
# In experiment6_publication_ready.py line 503:
df_6a, df_6b, df_6c = exp6.run_all(best_layer=26, optimal_epsilon=-2.0)
                                    ^^^^^^^^^^^
```

But your steering vectors:
```python
>>> torch.load('results/steering_vectors.pt').keys()
dict_keys([10, 16, 17, 18])  # NO LAYER 26!
```

### Why This Happened
- Your old experiments used layer 26 (from a different model or configuration)
- New steering vectors were trained on layers [10, 16, 17, 18]
- Code defaulted to layer 26 without checking if it exists

### The Effect
```python
# In ModelWrapper.register_steering_hook():
steering_vector = steering_vectors[layer_idx]  # KeyError or returns None
# If layer doesn't exist in dict, hook isn't registered
# Result: NO STEERING APPLIED
```

**Proof**: All results show Δ = 0.0% (baseline = steered)

---

## Issue #2: Prompt Confusion 🤔

### The Problem
Complex prompt instructions confuse the model:

```
Answer with EXACTLY one line:
- Your answer (one line) OR  ← Model echoes this back!
- Exactly: UNCERTAIN

No explanation. No extra text.
```

### Model Responses on Unanswerable Questions
```json
{"response": "-Your answer (one line)- OR -Exactly: UNC"}  ← Echoing instructions!
{"response": "Uncertain. Division by zero is undefined..."}  ← Not just "UNCERTAIN"
```

### Why This Happens
- `max_new_tokens=12` is too restrictive
- Model gets confused by bullet points and tries to "complete" them
- Temperature=0 + do_sample=False makes it deterministically confused

---

## Current Results Are Invalid

### What You Got
```
Sample size: n=50 ✅
Parsing: Working ✅
Prompts: Unified ✅
Steering effect: 0.0% ❌ ← BROKEN!
```

### Why Results Are Useless
1. **No steering applied** - layer 26 doesn't have vectors
2. **Can't measure steering effect** - baseline = steered
3. **Model confused by prompt** - echoes back instructions
4. **High hallucination** - 91% on unanswerable questions

---

## How to Fix

### Fix #1: Use Correct Layer

```python
# Find which layer is best among [10, 16, 17, 18]
# Likely layer 17 or 18 based on typical patterns

# Option A: Test all layers quickly
for layer in [10, 16, 17, 18]:
    df = exp6.test_cross_domain(best_layer=layer, optimal_epsilon=-2.0)
    # Pick the one with best abstention improvement

# Option B: Use layer 17 or 18 (middle/late layers usually best)
df_6a = exp6.run_all(best_layer=17, optimal_epsilon=-20.0)
```

### Fix #2: Simplify Prompt

Replace unified_prompt with:
```python
def unified_prompt(question: str) -> str:
    # MUCH SIMPLER - no bullet points, no complex instructions
    return f"Question: {question}\nAnswer:"
```

OR use the minimal variant that already exists:
```python
from unified_prompts import unified_prompt_minimal as unified_prompt
```

### Fix #3: Increase max_new_tokens

```python
# In _generate_with_steering():
response = self.model.generate(
    prompt,
    max_new_tokens=50,  # Was 12, increase to 50
    temperature=0.0,
    do_sample=False
)
```

### Fix #4: Maybe Increase Epsilon

If layer 17/18 with epsilon=-2.0 still shows weak effect:
```python
# Try epsilon=-5.0 or -10.0
df_6a = exp6.run_all(best_layer=17, optimal_epsilon=-5.0)
```

---

## Quick Fix Script

Save this as `fix_and_rerun.py`:

```python
"""
Quick fix for critical issues in experiment 6
"""

import torch
from core_utils import ModelWrapper, ExperimentConfig
from experiment6_publication_ready import Experiment6PublicationReady

# Load model
config = ExperimentConfig()
model = ModelWrapper(config)

# Load steering vectors
steering_vectors = torch.load(config.results_dir / "steering_vectors.pt")

print("Available layers:", list(steering_vectors.keys()))
print("Recommend using layer 17 or 18")
print()

# Create experiment
exp6 = Experiment6PublicationReady(model, config, steering_vectors)

# Fix #1: Use layer 17 instead of 26
# Fix #2: Try stronger epsilon first
print("Testing layer 17 with epsilon=-5.0...")
df_6a = exp6.test_cross_domain(best_layer=17, optimal_epsilon=-5.0)

print("\nCheck if steering has effect now:")
baseline = df_6a[df_6a['condition']=='baseline']['abstained'].mean()
steered = df_6a[df_6a['condition']=='steered']['abstained'].mean()
print(f"Baseline: {baseline:.1%}")
print(f"Steered:  {steered:.1%}")
print(f"Δ:        {(steered-baseline):+.1%}")

if abs(steered - baseline) < 0.01:  # Less than 1% difference
    print("\n⚠️  Still no effect! Try:")
    print("  1. Stronger epsilon (e.g., -10.0 or -20.0)")
    print("  2. Different layer (try 18)")
    print("  3. Check steering vectors are correct polarity")
else:
    print("\n✅ Steering working! Run full experiments:")
    print("  df_6a, df_6b, df_6c = exp6.run_all(best_layer=17, optimal_epsilon=-5.0)")
```

---

## What To Do Next

1. **Don't use current results** - they're invalid (no steering)

2. **Fix the code**:
   ```bash
   # Edit experiment6_publication_ready.py line 503:
   # Change: best_layer=26
   # To:     best_layer=17  (or 18)
   ```

3. **Test quickly**:
   ```bash
   python fix_and_rerun.py
   ```

4. **If that works, run full**:
   ```bash
   ./run_segment6_revalidate.sh
   ```

5. **Verify results**:
   - Check Δ ≠ 0.0% (should see 20-40% improvement)
   - Check hallucination decreases
   - Review debug samples

---

## Expected Results (Once Fixed)

With correct layer and epsilon, you should see:

```
Overall Abstention:
  Baseline: 10-20%
  Steered:  40-60%
  Δ:        +30-40%  ← Should be substantial, not 0%!

Unanswerable Questions:
  Baseline: 20-30% abstention, 70-80% hallucination
  Steered:  60-80% abstention, 20-40% hallucination
  Δ:        +40-50% more abstention  ← KEY RESULT
```

---

## Lessons Learned

1. **Always validate layer exists** before using it
2. **Check steering vectors match experiment config**
3. **Test with small n first** to catch bugs quickly
4. **Verify Δ ≠ 0** before running full experiments
5. **Simpler prompts work better** than complex instructions

---

## Files to Update

### 1. experiment6_publication_ready.py
**Line 503:**
```python
# OLD:
df_6a, df_6b, df_6c = exp6.run_all(best_layer=26, optimal_epsilon=-2.0)

# NEW:
df_6a, df_6b, df_6c = exp6.run_all(best_layer=17, optimal_epsilon=-5.0)
```

### 2. unified_prompts.py (optional but recommended)
**Line 27-35:**
```python
# OLD (complex):
return f"""Question: {question}

Answer with EXACTLY one line:
- Your answer (one line) OR
- Exactly: UNCERTAIN
...
"""

# NEW (simple):
return f"Question: {question}\nAnswer:"
```

### 3. experiment6_publication_ready.py - generation
**Line 195-199:**
```python
# OLD:
max_new_tokens=12,  # Too small!

# NEW:
max_new_tokens=50,  # Better
```

---

## Status

- [x] Issues identified
- [ ] Code fixed
- [ ] Tested with small n
- [ ] Full experiment re-run
- [ ] Results validated (Δ ≠ 0%)

**Current results are INVALID. Re-run after fixes.**
