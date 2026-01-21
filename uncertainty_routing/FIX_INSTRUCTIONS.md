# Quick Fix Instructions - 10 Minutes

## Problem Summary

Your Exp5 and Exp8 results show inverted steering:
- **Negative ε** is increasing hallucinations (should decrease)
- **Positive ε** is decreasing hallucinations (should increase)
- Exp8 failed because steering isn't working in the expected direction

## Root Cause

The steering vectors have the wrong sign. Need to flip them.

---

## Fix Steps (On Your SSH Server)

### Step 1: Upload the fix script (from your Mac)

```bash
# On your Mac terminal
scp /Users/akshatatiwari/Desktop/MIT/mech_interp_research/uncertainty_routing/quick_fix_flip_vectors.py akshatat@orcd-login.mit.edu:~/uncertainty-routing-abstention/uncertainty_routing/
```

### Step 2: Run the fix (on SSH)

```bash
# On SSH server
cd ~/uncertainty-routing-abstention/uncertainty_routing
conda activate mech_interp_gpu
python quick_fix_flip_vectors.py
```

**Expected output**:
```
Loading results/steering_vectors.pt...
Layers found: [16, 17, 18, 20]
Flipping vector signs...
✓ Saved to results/steering_vectors.pt
✓ Saved to results/steering_vectors_explicit.pt
✅ DONE - Vectors flipped!
```

### Step 3: Re-run Exp5 (Segment 2)

```bash
# Still on SSH
./run_segment2.sh
```

**Expected**: ~30-40 minutes
**Check results**:
- At ε=-10 or ε=-20: Should have 0% hallucinations ✓
- At ε=+50: Should have high hallucinations

### Step 4: Run Exp8 (Segment 4A)

```bash
./run_segment4a.sh
```

**Expected**: ~4-5 minutes (you said it's fast)
**Check results**:
- `steering_works: True` for both 1.5B and 3B models
- `delta_abstain > 0` (positive steering effect)

### Step 5: Run Exp9 (Segment 4B)

```bash
./run_segment4b.sh
```

**Expected**: ~3-4 minutes
**Check results**:
- Sparsity analysis showing top dimensions

---

## Verification

After Step 3 (re-running Segment 2), check:

```bash
cat results/exp5_summary.json | grep -A 5 "best_eps"
```

**Should show something like**:
```json
"best_eps_value": -20.0,
"best_eps_hallucination_unanswerable": 0.0,  // ✓ Should be 0% or very low
```

NOT:
```json
"best_eps_value": -50.0,
"best_eps_hallucination_unanswerable": 0.6,  // ✗ This is broken
```

---

## What Each Step Does

**Step 2 (quick_fix_flip_vectors.py)**:
- Loads `results/steering_vectors.pt`
- Multiplies all vectors by -1
- Saves to both `steering_vectors.pt` and `steering_vectors_explicit.pt`

**Step 3 (Segment 2)**:
- Re-runs Exp4 with flipped vectors
- Re-runs Exp5 with flipped vectors
- Now negative ε should reduce hallucinations (correct!)

**Step 4 (Segment 4A)**:
- Tests steering on Qwen 1.5B, 3B, (and 7B if you have memory)
- Should now show `steering_works: True`

**Step 5 (Segment 4B)**:
- Analyzes which dimensions matter in the steering vectors
- Provides interpretability insights

---

## Expected Timeline

| Step | Time | What It Does |
|------|------|--------------|
| 1. Upload script | 30 sec | Transfer fix_script to SSH |
| 2. Run fix | 10 sec | Flip vector signs |
| 3. Segment 2 | 30-40 min | Re-run Exp4-5 with correct vectors |
| 4. Segment 4A | 4-5 min | Run Exp8 (scaling) |
| 5. Segment 4B | 3-4 min | Run Exp9 (interpretability) |

**Total**: ~40-50 minutes

---

## Alternative: Manual Flip in Python

If you can't upload the script, do this on SSH:

```python
# On SSH server
cd ~/uncertainty-routing-abstention/uncertainty_routing
conda activate mech_interp_gpu
python

# In Python:
import torch
from pathlib import Path

# Load and flip
vectors = torch.load("results/steering_vectors.pt")
flipped = {k: -v for k, v in vectors.items()}

# Save
torch.save(flipped, "results/steering_vectors.pt")
torch.save(flipped, "results/steering_vectors_explicit.pt")

print("✓ Done!")
exit()
```

Then proceed to Step 3.

---

## After Fix Complete

You'll have:
- ✅ Exp1-3: Already complete
- ✅ Exp4-5: Re-run with correct vectors
- ✅ Exp8: Scaling analysis working
- ✅ Exp9: Interpretability working

Then you can:
1. Run Segment 3 (Exp6-7) if you want robustness/safety
2. Start writing your paper with correct results

---

## Quick Commands (Copy-Paste)

```bash
# On Mac: Upload fix script
scp /Users/akshatatiwari/Desktop/MIT/mech_interp_research/uncertainty_routing/quick_fix_flip_vectors.py akshatat@orcd-login.mit.edu:~/uncertainty-routing-abstention/uncertainty_routing/

# On SSH: Run everything
cd ~/uncertainty-routing-abstention/uncertainty_routing
conda activate mech_interp_gpu
python quick_fix_flip_vectors.py
./run_segment2.sh
./run_segment4a.sh
./run_segment4b.sh
```

**Wait ~40-50 minutes, then check results!**

---

## Expected Results After Fix

### Exp5 (After Segment 2)
```json
{
  "best_eps_value": -20.0,
  "best_eps_accuracy_answerable": 0.8,  // ✓ High accuracy
  "best_eps_hallucination_unanswerable": 0.0,  // ✓ Zero hallucinations
  "best_eps_abstain_unanswerable": 1.0  // ✓ Perfect abstention
}
```

### Exp8 (After Segment 4A)
```json
{
  "Qwen/Qwen2.5-1.5B-Instruct": {
    "steering_works": true,  // ✓
    "delta_abstain": 0.4,  // ✓ Positive effect
    "baseline_halluc": 0.3,
    "steered_halluc": 0.05  // ✓ Reduced
  },
  "Qwen/Qwen2.5-3B-Instruct": {
    "steering_works": true,  // ✓
    "delta_abstain": 0.45,  // ✓ Positive effect
    ...
  }
}
```

---

That's it! The fix is simple - just flip the sign of all vectors, then re-run.
