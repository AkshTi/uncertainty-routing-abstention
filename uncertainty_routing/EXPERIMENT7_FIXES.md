# Experiment 7 Fixes Applied

## Summary
Fixed critical issues preventing Experiment 7 from achieving target abstention rates on high-risk questions.

## Problems Identified

### 1. Wrong Epsilon Magnitude (10x too large)
- **Before:** `epsilon = ±20.0`
- **After:** `epsilon = ±2.0`
- **Impact:** 10x smaller magnitude matches Experiment 6 (which works correctly)

### 2. Backwards Epsilon Signs
- **Before:** `epsilon_toward_answer=-20.0, epsilon_toward_abstain=20.0`
- **After:** `epsilon_toward_answer=+2.0, epsilon_toward_abstain=-2.0`
- **Impact:** Now steering pushes in the correct direction

### 3. Why This Matters
Based on steering vector training (safety_steering_vectors.py:153):
```python
direction = answerable_mean - unanswerable_mean
```
- **Positive epsilon (+2.0):** Push toward answering (reduce abstention)
- **Negative epsilon (-2.0):** Push toward abstention (increase uncertainty)

## Files Modified

### 1. experiment7_safety_alignment_fixed.py
**Line 520:** Changed epsilon values in `__main__`
```python
# Before:
exp7.run_all(best_layer=24, epsilon_toward_answer=-20.0, epsilon_toward_abstain=20.0)

# After:
exp7.run_all(best_layer=24, epsilon_toward_answer=+2.0, epsilon_toward_abstain=-2.0)
```

### 2. safety_steering_vectors.py
**Lines 316-320:** Updated guidance messages
```python
# Before:
exp7.run_all(best_layer=24, epsilon_toward_answer=20.0, epsilon_toward_abstain=-20.0)
✓ Positive epsilon (+20): Push toward answering
✓ Negative epsilon (-20): Push toward abstention

# After:
exp7.run_all(best_layer=24, epsilon_toward_answer=+2.0, epsilon_toward_abstain=-2.0)
✓ Positive epsilon (+2.0): Push toward answering
✓ Negative epsilon (-2.0): Push toward abstention
```

## Expected Results After Fix

### Previous Results (Broken)
**7B: Selective Abstention**
- High-risk baseline: 50% → steered_abstain: 33% ❌ (went DOWN)
- Low-risk: 0% abstention ✓

### Expected Results (Fixed)
**7B: Selective Abstention**
- High-risk baseline: 50% → steered_abstain: **70-90%** ✓ (should go UP)
- Low-risk: 0% abstention ✓ (maintained)

**7C: Spurious Correlations**
- Consistent abstention across question lengths
- Proper behavior on unknowable questions

## Why Experiment 6 Worked

| Aspect | Experiment 6 ✅ | Experiment 7 (Before) ❌ | Experiment 7 (After) ✅ |
|--------|----------------|------------------------|---------------------|
| Epsilon magnitude | -2.0 | ±20.0 | ±2.0 |
| Sign correctness | ✓ | ✗ (backwards) | ✓ |
| Unanswerable abstention | 60% → 100% | 50% → 33% | 50% → 70-90% |

## Next Steps

1. **Rerun Experiment 7:**
   ```bash
   ./run_segment7_revalidate.sh
   ```

2. **Check Results:**
   - Look for high-risk abstention ≥ 60-80%
   - Verify low-risk abstention remains < 10%
   - Confirm 0 safety violations

3. **Compare to Targets:**
   From logs (seg7-8400318.out:26):
   - Safety violations: 0% (maintain) ✓
   - High-risk abstention: 0% → 60-80% (target)
   - Low-risk abstention: 0% → <10% (target)

## Technical Details

### Steering Vector Direction
The steering vector is computed as:
```python
direction = answerable_mean - unanswerable_mean
direction = direction / direction.norm()  # Normalize to unit length
```

This means:
- Adding `+epsilon * direction` → moves toward "answerable" space
- Adding `-epsilon * direction` → moves toward "unanswerable" space

### Epsilon Magnitude Calibration
- Experiment 6 found optimal epsilon = -2.0 for steering toward abstention
- Larger magnitudes (±20.0) oversaturate the activation space
- This causes unpredictable behavior and steering reversal

---

**Date Fixed:** 2026-01-26
**Files Changed:** 2
**Lines Changed:** 8
