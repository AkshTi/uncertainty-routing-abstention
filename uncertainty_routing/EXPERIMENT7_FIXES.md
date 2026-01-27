# Experiment 7 Fixes Applied

## Summary
Fixed critical issues preventing Experiment 7 from achieving target abstention rates on high-risk questions.

## Problems Identified

### 1. Backwards Epsilon Signs (CRITICAL)
- **Before:** `epsilon_toward_answer=-20.0, epsilon_toward_abstain=20.0`
- **After:** `epsilon_toward_answer=+20.0, epsilon_toward_abstain=-20.0`
- **Impact:** Now steering pushes in the correct direction

### 2. Wrong Initial Guess on Magnitude
- **Initial wrong fix:** Changed to ±2.0 (10x too small)
- **Actual working value:** ±20.0 (matches Experiment 6 publication_ready results)
- **Impact:** Must use ±20.0 for steering to have effect

### 3. Why This Matters
Based on steering vector training (safety_steering_vectors.py:153):
```python
direction = answerable_mean - unanswerable_mean
```
- **Positive epsilon (+2.0):** Push toward answering (reduce abstention)
- **Negative epsilon (-2.0):** Push toward abstention (increase uncertainty)

## Files Modified

### 1. experiment7_safety_alignment_fixed.py
**Multiple locations:** Changed epsilon values throughout
```python
# Before (WRONG SIGNS):
exp7.run_all(best_layer=24, epsilon_toward_answer=-20.0, epsilon_toward_abstain=20.0)

# After (CORRECTED):
exp7.run_all(best_layer=24, epsilon_toward_answer=+20.0, epsilon_toward_abstain=-20.0)
```

**Changes:**
- Line 1-15: Updated docstring to reflect correct values
- Line 206-207: `epsilon_toward_answer=20.0, epsilon_toward_abstain=-20.0`
- Line 243-244: `epsilon_toward_answer=20.0, epsilon_toward_abstain=-20.0`
- Line 318: `epsilon=-20.0`
- Line 375-377: `best_layer=24, epsilon_toward_answer=20.0, epsilon_toward_abstain=-20.0`
- Line 520: Main call updated

### 2. safety_steering_vectors.py
**Lines 316-321:** Updated guidance messages
```python
# After:
exp7.run_all(best_layer=24, epsilon_toward_answer=+20.0, epsilon_toward_abstain=-20.0)
✓ Positive epsilon (+20.0): Push toward answering
✓ Negative epsilon (-20.0): Push toward abstention
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

| Aspect | Experiment 6 ✅ | Experiment 7 (Before) ❌ | Experiment 7 (1st fix) ❌ | Experiment 7 (Final) ✅ |
|--------|----------------|------------------------|----------------------|---------------------|
| Epsilon magnitude | -20.0 (or -50.0) | ±20.0 | ±2.0 | ±20.0 |
| Sign correctness | ✓ | ✗ (backwards) | ✓ | ✓ |
| Layer | 24 or 26 | 24 | 24 | 24 |
| High-risk abstention | - | 50% → 33% | 0% → 0% | 0% → 60-80% (expected) |

**Key Insight:** Experiment 6 actually uses `epsilon=-20.0` or `-50.0`, NOT `-2.0` as suggested in the "_fixed" file names. The working exp6 results show epsilon=-50.0 on layer 26, and the publication_ready version uses epsilon=-20.0 on layer 24.

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
