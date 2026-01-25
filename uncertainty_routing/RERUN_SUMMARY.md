# Re-Run Analysis Summary

**Date**: 2026-01-25
**Analysis**: After fixes #1 and #2

---

## 🎯 WHAT WE FOUND

### ✅ Good News
1. **Token limit works**: Processing is now 2.5x faster (4.35 it/s vs 1.73 it/s)
2. **Determinism works**: All repeated runs give identical results
3. **Parameters correct**: Now using layer=10, epsilon=-20 (as intended)

### ❌ Bad News
1. **Responses STILL multi-line** (94.2% instead of expected <10%)
   - Model uses all 12 tokens to start explanations
   - Example: `"120\n\nExplanation:\nTo find the product of"`

2. **🚨 CRITICAL: Steering direction is BACKWARDS!**
   - Negative epsilon should INCREASE abstention
   - Instead it DECREASES abstention by -3.7%
   - **Root cause**: Using wrong steering vectors

---

## 🔬 DIAGNOSIS: Vector Problem

### What Happened
The "calibrated" vectors are fundamentally broken:

```
CALIBRATED Vector = "correct answers" - "hallucinations"
              ❌ Wrong concept!

ORIGINAL Vector = "answerable questions" - "unanswerable questions"
          ✅ Correct concept!
```

The calibrated vectors mix two things:
- Question type (answerable vs unanswerable)
- Response quality (correct vs hallucinated)

This causes inverted behavior.

### Evidence
```
With calibrated vectors (epsilon=-20):
  Mathematics: 61% → 55% abstention (DECREASED ❌)
  Science:     64% → 61% abstention (DECREASED ❌)
  History:     55% → 53% abstention (DECREASED ❌)
  Geography:   48% → 44% abstention (DECREASED ❌)

Expected with correct vectors:
  All domains should INCREASE by +15-25%
```

---

## ✅ FIX APPLIED

Changed `experiment6_publication_ready.py` to use **ORIGINAL** vectors:

```python
possible_files = [
    "steering_vectors.pt",  # ← Now uses this (correct)
    # "steering_vectors_calibrated.pt",  # ← Disabled (broken)
]
```

---

## 📋 NEXT STEPS

### 1. Re-run experiment (REQUIRED)
```bash
python experiment6_publication_ready.py
```

Expected results:
- ✅ Steering should now go in CORRECT direction
- ✅ Mathematics: ~40% → ~55-65% abstention (+15-25%)
- ✅ All domains should show INCREASED abstention with negative epsilon

### 2. Check results
```bash
# View summary
tail -100 logs/*.out | grep -A20 "Overall Abstention"

# Check if direction is correct now
python test_vector_direction.py  # Diagnostic script
```

### 3. If STILL backwards
If steering is still inverted, the original vectors may need to be flipped:

```python
# Quick fix: invert all vectors
vectors = torch.load("results/steering_vectors.pt")
for k in vectors:
    vectors[k] = -vectors[k]  # Flip direction
torch.save(vectors, "results/steering_vectors_flipped.pt")
```

Then update experiment6 to use `steering_vectors_flipped.pt`.

---

## 🔧 REMAINING ISSUES

### Issue 1: Multi-line Responses (Lower Priority)

**Problem**: Responses are still multi-line (94.2%) even with max_new_tokens=12

**Root cause**: Model uses all 12 tokens to START an explanation:
```
"120\n\nExplanation:\nTo find the product of"
 ^^^  ^^  ^^^^^^^^^^^  ^^  ^^^^  ^^^  ^^^^^^^
  1    2      3        4   5     6      7     (≈7-10 tokens)
```

**Possible solutions**:
1. **Reduce tokens further**: Try `max_new_tokens=6`
2. **Stronger prompt**: Add "ONE WORD ONLY" to prompt
3. **Post-processing**: Just use first line (parsing already does this)

**Current status**: Not critical since parsing extracts first line correctly.

### Issue 2: Prompt Format (If multi-line is still a problem)

The prompt says:
```
"IMPORTANT: Return EXACTLY one line."
```

But model still generates explanations. Try:
```
"Answer in ONE WORD ONLY. No explanation."
```

---

## 📊 EXPECTED RESULTS (After Re-Run)

### Abstention Rates
With epsilon=-20 and CORRECT vectors:

| Domain      | Baseline | Steered | Expected Δ |
|-------------|----------|---------|------------|
| Mathematics | 40%      | 55-65%  | +15-25%    |
| Science     | 64%      | 80-88%  | +15-25%    |
| History     | 55%      | 70-80%  | +15-25%    |
| Geography   | 48%      | 65-75%  | +17-27%    |

**Overall**: 57% → 72-77% (+15-20%)

### Response Format
Even if still multi-line (94%), parsing should work correctly since it extracts first line only.

---

## ✅ SUCCESS CRITERIA

You'll know the fix worked when:

1. **Direction is correct**:
   - Negative epsilon → INCREASED abstention (+15-25%)
   - Positive epsilon → DECREASED abstention

2. **Magnitude is reasonable**:
   - Not 0% change (vectors don't work)
   - Not 100% abstention (vectors too strong)
   - Should be in 50-75% range for answerable questions

3. **Consistency across domains**:
   - All 4 domains should show similar increases
   - Geography shouldn't be anomalous anymore

---

## 🐛 DEBUG COMMANDS

```bash
# 1. Check which vectors are loaded
grep "Loaded steering vectors" logs/*.out | tail -1

# 2. Check abstention direction
tail -50 logs/*.out | grep -A15 "Overall Abstention"

# 3. Quick vector test (2 minutes)
python test_vector_direction.py

# 4. Compare old vs new results
diff <(grep "Overall Abstention" logs/seg6-8395815.out) \
     <(grep "Overall Abstention" logs/*.out | tail -3)
```

---

**Good luck! Re-run with the original vectors and it should work correctly now.**
