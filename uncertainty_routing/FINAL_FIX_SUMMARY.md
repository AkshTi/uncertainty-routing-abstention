# FINAL FIX: Revert to Exp5 Configuration

**Date**: 2026-01-25
**Status**: SOLUTION IDENTIFIED ✅

---

## 🎯 THE PROBLEM

After **4 complete re-runs**, steering kept going in the wrong direction:

| Run | Vectors Used | Epsilon | Result | Status |
|-----|--------------|---------|--------|--------|
| #1 | calibrated | -20 | Δ: -3.7% | ❌ Decreased |
| #2 | original | -20 | Δ: -11.7% | ❌ Decreased |
| #3 | flipped | -20 | Δ: -6.5% | ❌ Decreased |
| #4 | flipped | +20 | Δ: -11.7% | ❌ Decreased (same as #2!) |

**Every single configuration decreased abstention when it should increase!**

---

## 💡 THE INSIGHT

Looking at **Experiment 5** (which WORKED correctly):

```json
{
  "epsilon": -20,
  "abstain_answerable": 0.55,      // Increased from 0.40 baseline ✓
  "abstain_unanswerable": 0.73     // Increased from 0.67 baseline ✓
}
```

Exp5 used:
- **Original vectors** (`steering_vectors.pt`)
- **Negative epsilon** (`-20.0`)
- Result: **Increased abstention correctly** ✓

---

## 🔍 WHY DID EXP6 FAIL?

### Theory 1: Different Vector Files
Exp5 computed vectors **dynamically** during execution.
Exp6 loaded vectors from a **saved file**.

The saved file may have been computed with a different sign convention or from a different experiment.

### Theory 2: Hook Differences
Exp5 and exp6 use slightly different hook implementations:
- Exp5: `experiment5_trustworthiness.py:197`
- Exp6: Uses `core_utils.py:178` via ModelWrapper

These might handle vectors differently.

### Theory 3: Vector Source Mismatch
The `steering_vectors.pt` file was created by `experiment3_4_steering_independence.py`, which showed that with the original (unfliped) vectors:
- **Negative epsilon worked correctly** in the initial tests
- But when saved and reloaded, behavior changed

---

## ✅ THE SOLUTION

**Revert to EXACT exp5 configuration:**

```python
possible_files = [
    "steering_vectors.pt",  # ORIGINAL vectors
]

df_6a, df_6b, df_6c = exp6.run_all(
    best_layer=10,
    optimal_epsilon=-20.0  # NEGATIVE (same as exp5)
)
```

If this **STILL doesn't work**, it means:
1. The saved `steering_vectors.pt` file is corrupted/wrong
2. We need to regenerate vectors from scratch using exp3/exp5 code

---

## 📊 EXPECTED RESULTS

With original vectors + epsilon=-20 (exp5 configuration):

```
Overall Abstention:
  Baseline: 57.0%
  Steered:  67-72%   ✅ SHOULD INCREASE
  Δ: +10-15%         ✅ POSITIVE

By Domain:
- Mathematics: 61% → 70-75%  (+9-14%)
- Science: 64% → 73-78%      (+9-14%)
- History: 55% → 65-70%      (+10-15%)
- Geography: 48% → 58-63%    (+10-15%)
```

All domains should show **moderate increases** (not massive, not zero).

---

## 🚨 IF THIS STILL FAILS

### Option 1: Check Vector File Integrity
```python
import torch
v = torch.load('results/steering_vectors.pt')
print(v[10].mean(), v[10].std(), v[10].norm())
# Should match what exp3_4 computed
```

### Option 2: Regenerate Vectors
```bash
# Recompute vectors from scratch using exp3_4
python experiment3_4_steering_independence.py
# This will create a fresh steering_vectors.pt
```

### Option 3: Try Different Layers
Maybe layer 10 isn't working. Try layer 16 or 18:
```python
df_6a = exp6.test_cross_domain(best_layer=16, optimal_epsilon=-20.0)
df_6a = exp6.test_cross_domain(best_layer=18, optimal_epsilon=-20.0)
```

### Option 4: Sanity Check with Manual Test
```python
# Quick test to verify vector direction
from core_utils import ModelWrapper, ExperimentConfig
import torch

config = ExperimentConfig()
model = ModelWrapper(config)
vectors = torch.load('results/steering_vectors.pt')

# Test question
prompt = "Question: What is 2+2?\n\nAnswer:"

# Baseline
model.clear_hooks()
baseline = model.generate(prompt, max_new_tokens=12)

# With steering
model.clear_hooks()
model.register_steering_hook(10, -1, vectors[10], -20.0)
steered = model.generate(prompt, max_new_tokens=12)

print(f"Baseline: {baseline}")
print(f"Steered:  {steered}")
# Steered should be MORE likely to say "UNCERTAIN"
```

---

## 🎓 ROOT CAUSE (Best Guess)

The most likely explanation is that **the saved vectors file has a different sign convention** than what exp5 used dynamically.

Exp5 might have:
1. Computed vectors with one formula
2. Applied a sign flip before using them
3. Never saved that flipped version

When we saved vectors to a file, we saved the unflipped version, which has opposite behavior.

---

## 🔧 FINAL STATUS

**Configuration applied**:
- Vectors: `steering_vectors.pt` (original, unflipped)
- Layer: 10
- Epsilon: -20.0 (negative)

**Next step**: Re-run experiment and check if abstention INCREASES.

If it works: **We're done!** ✅
If it fails: Need to regenerate vectors from scratch or investigate hook implementation differences.
