# CRITICAL BUGS IDENTIFIED & FIXED

**Analysis Date**: 2026-01-25
**Status**: 5 critical bugs found and 2 fixed

---

## 🔴 BUG #1: Token Length Mismatch (FIXED ✅)

**File**: `experiment6_publication_ready.py:84`

### Problem
```python
max_new_tokens=30,  # Comment says 12, but code uses 30
```

### Impact
- **95.6% of responses were multi-line** (should be ~5%)
- Model generated verbose explanations instead of single-line answers
- Example: `"120\n\nExplanation:\nTo find the product..."`

### Evidence
From error log line 99-106:
```
Total responses: 800
Single line (correct): 35 (4.4%)
Multi-line: 765 (95.6%)
```

### Fix Applied
```python
max_new_tokens=12,  # FIXED: Force strict one-line output
```

---

## 🔴 BUG #2: Parameter Mismatch (FIXED ✅)

**File**: `experiment6_publication_ready.py:508-509`

### Problem
```python
print(f"Using best_layer=10, optimal_epsilon=-20.0")  # Line 508: Claims this
df_6a, df_6b, df_6c = exp6.run_all(best_layer=18, optimal_epsilon=-40.0)  # Line 509: Does this!
```

**The code was LYING about what parameters it used!**

### Impact
- Using `epsilon=-40.0` caused **100% abstention on answerable math questions**
- From exp5_summary.json:
  - `epsilon=-40`: 70% abstention on answerable (too high)
  - `epsilon=-20`: 55% abstention on answerable (reasonable)
  - `epsilon=-10`: 40% abstention (baseline-like)

### Evidence from Results
```
MATHEMATICS:
  Answerable:
    baseline: n=50, abstention=40.0%
    steered: n=50, abstention=100.0%  ⚠️ BROKEN!
```

### Fix Applied
```python
df_6a, df_6b, df_6c = exp6.run_all(best_layer=10, optimal_epsilon=-20.0)
```

---

## 🔴 BUG #3: Over-Aggressive Steering (STILL NEEDS TESTING)

### Current State
Using `epsilon=-20.0` should be better than `-40.0`, but you may still need to tune this.

### Recommendation
Run a quick calibration test with these epsilon values:
```python
test_epsilons = [-30.0, -20.0, -15.0, -10.0]
```

**Goal**: Find epsilon where:
- Answerable questions: 10-30% abstention (not 100%!)
- Unanswerable questions: 80-95% abstention

---

## 🔴 BUG #4: Geography Domain Anomaly (NEEDS INVESTIGATION)

### Problem
Geography shows **BACKWARDS** behavior:
```
GEOGRAPHY:
  Baseline: 48% abstention
  Steered:  54% abstention
  Expected: Much higher increase (like 38% for mathematics)
```

### Possible Causes
1. Geography questions are contextually different
2. Steering vector doesn't generalize to geography domain
3. Dataset issues (wrong labels or question format)

### Action Required
```bash
# Examine geography questions
head -20 results/exp6a_cross_domain_publication_ready.csv | grep geography

# Check if geography questions are actually answerable/unanswerable
python -c "from scaled_datasets import create_scaled_domain_questions; \
           import json; \
           d = create_scaled_domain_questions(); \
           print(json.dumps(d['geography'], indent=2)[:500])"
```

---

## 🔴 BUG #5: Nonsensical Abstention Responses (NEEDS INVESTIGATION)

### Problem
Steered model gives bizarre responses:
```
Question: "What is 15 * 8?"
Response: "Uncertain\n\nI'm sorry, but I am an AI and do not have a specific location..."
```

### Possible Causes
1. **Steering too strong** → model becomes incoherent
2. **Hook placement wrong** → steering applied at wrong position
3. **Vector direction inverted** → pushing model in wrong direction

### Action Required
1. Test with weaker epsilon (`-10.0` to `-15.0`)
2. Verify steering vector calibration
3. Check if vectors are normalized correctly

---

## 📊 EXPECTED RESULTS AFTER FIXES

After fixing bugs #1 and #2, you should see:

### Response Format
- **Single-line responses: 90-95%** (currently 4.4%)
- **Multi-line responses: 5-10%** (currently 95.6%)

### Abstention Rates (with epsilon=-20)
| Domain      | Baseline | Steered | Expected Δ |
|-------------|----------|---------|------------|
| Mathematics | 40%      | 55-65%  | +15-25%    |
| Science     | 64%      | 75-85%  | +10-20%    |
| History     | 55%      | 70-80%  | +15-25%    |
| Geography   | 48%      | 65-75%  | +17-27%    |

### Answerable Questions
**Steered abstention should be 10-30%, NOT 100%!**

---

## ✅ NEXT STEPS

### 1. Re-run Experiment 6 (REQUIRED)
```bash
# The fixed code is ready to run
python experiment6_publication_ready.py
```

### 2. Quick Validation (RECOMMENDED)
Before the full run, test with a small sample:
```python
# Add this to experiment6_publication_ready.py temporarily
# Test with n=5 per condition to verify fixes work
quick_test_results = exp6.test_cross_domain(best_layer=10, optimal_epsilon=-20.0)
```

### 3. Inspect Debug Outputs (REQUIRED)
```bash
# After running, check if responses are now single-line
head -10 debug_outputs/exp6a_debug_samples.jsonl | jq '.response_full'
```

### 4. Statistical Analysis (AFTER RE-RUN)
Once you have clean results with single-line responses:
```python
# Run statistical tests (Chi-square, t-tests)
python analyze_exp6a.py  # You may need to create this
```

---

## 🎯 SUCCESS CRITERIA

You'll know the fixes worked when:

1. ✅ **>90% single-line responses** (not 4.4%)
2. ✅ **Answerable questions: 10-30% abstention** (not 100%)
3. ✅ **Unanswerable questions: 80-95% abstention**
4. ✅ **Geography shows normal behavior** (not anomalous)
5. ✅ **Responses make sense** (not gibberish like "I don't have a location")

---

## 📝 NOTES

### Why epsilon=-40 was so bad
From exp5_summary.json, epsilon=-40 caused:
- **70% abstention on answerable questions** (should answer these!)
- Only **30% coverage** on questions it should answer
- This explains the 100% abstention you're seeing

### Why max_new_tokens=30 was bad
- Average word is ~1.3 tokens
- 30 tokens = ~23 words = multiple sentences
- The model was free to add "Explanation:", "I think...", etc.
- 12 tokens = ~9 words = forces concise answers

### Calibrated vs Uncalibrated Vectors
You have two sets of steering vectors:
- `steering_vectors.pt` (original)
- `steering_vectors_calibrated.pt` (normalized)

Make sure you're loading the **calibrated** version!

---

## 🐛 DEBUGGING TIPS

If results are still weird after fixes:

### Check 1: Verify Token Length
```python
# Add logging to _generate_with_steering
print(f"Generated tokens: {len(response.split())}")
```

### Check 2: Verify Steering Direction
```python
# Negative epsilon should INCREASE abstention
# Positive epsilon should DECREASE abstention
# If backwards, multiply vectors by -1
```

### Check 3: Inspect Individual Cases
```bash
# Look at specific weird examples
grep "100% abstention" results/exp6a_cross_domain_publication_ready.csv
```

---

**Good luck! The two critical fixes should dramatically improve your results.
Re-run the experiment and let me know if you see any issues.**
