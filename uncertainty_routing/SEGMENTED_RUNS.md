# Segmented Pipeline Runs - GPU Stability Guide

**Problem**: Your GPU might not last 10-14 hours continuously.

**Solution**: Break pipeline into 6-hour chunks that can be run independently.

---

## Two Options for Segmentation

### Option A: 4 Segments (Recommended)

**Safest approach** - No segment exceeds 5 hours:

```bash
./run_segment1.sh   # Exp1-3:  4-5 hours
./run_segment2.sh   # Exp4-5:  3-4 hours
./run_segment3.sh   # Exp6-7:  3-4 hours
./run_segment4a.sh  # Exp8:    4-5 hours ⭐
./run_segment4b.sh  # Exp9:    3-4 hours ⭐
```

**Total**: 5 runs, each under 5 hours

---

### Option B: 4 Segments (Faster)

**Faster but one long segment**:

```bash
./run_segment1.sh   # Exp1-3:  4-5 hours
./run_segment2.sh   # Exp4-5:  3-4 hours
./run_segment3.sh   # Exp6-7:  3-4 hours
./run_segment4.sh   # Exp8-9:  7-8 hours ⭐⚠️ LONG
```

**Total**: 4 runs, but last one is 7-8 hours

---

## Detailed Segment Breakdown

### Segment 1: Base Experiments (4-5 hours)
```bash
chmod +x run_segment1.sh
./run_segment1.sh
```

**What it does**:
- Exp1: Behavior-belief dissociation (1h)
- Exp2: Gate localization (1.5h)
- Exp3: Steering extraction (1.5h)

**Outputs**:
- `results/exp1_summary.json`
- `results/exp2_summary.json`
- `results/exp3_summary.json`
- `results/steering_vectors.pt` (needed for later segments)

**Can stop after**: Yes, results saved

---

### Segment 2: Applications (3-4 hours)
```bash
chmod +x run_segment2.sh
./run_segment2.sh
```

**Prerequisites**: Segment 1 complete (needs `steering_vectors.pt`)

**What it does**:
- Exp4: Gate independence (1h)
- Exp5: Trustworthiness application (2h)

**Outputs**:
- `results/exp4_summary.json`
- `results/exp5_summary.json`
- `results/steering_vectors_explicit.pt` (needed for Exp6-9)

**Can stop after**: Yes, results saved

---

### Segment 3: Validation (3-4 hours)
```bash
chmod +x run_segment3.sh
./run_segment3.sh
```

**Prerequisites**: Segments 1-2 complete (needs `steering_vectors_explicit.pt`, `exp5_summary.json`)

**What it does**:
- Exp6: Robustness testing (2h)
- Exp7: Safety alignment (1h)

**Outputs**:
- `results/exp6_summary.json`
- `results/exp7_summary.json`

**Can stop after**: Yes, but Exp8+9 are critical for paper

---

### Segment 4A: Scaling Analysis (4-5 hours) ⭐
```bash
chmod +x run_segment4a.sh
./run_segment4a.sh
```

**Prerequisites**: Segment 2 complete (needs `steering_vectors_explicit.pt`)

**What it does**:
- Exp8: Multi-model scaling (4-5h)
- Tests Qwen 1.5B, 3B, 7B

**Outputs**:
- `results/exp8_summary.json` ⭐ CRITICAL
- `results/exp8_scaling_analysis.png`

**Acceptance impact**: +30%

---

### Segment 4B: Interpretability (3-4 hours) ⭐
```bash
chmod +x run_segment4b.sh
./run_segment4b.sh
```

**Prerequisites**: Segment 2 complete (needs `steering_vectors_explicit.pt`)

**What it does**:
- Exp9: Vector interpretability (3-4h)
- Sparsity, dimension probing, semantic analysis

**Outputs**:
- `results/exp9_summary.json` ⭐ CRITICAL
- `results/exp9_interpretability_analysis.png`

**Acceptance impact**: +25%

---

## Alternative: Combined Segment 4 (7-8 hours)

If your GPU can handle 7-8 hours:

```bash
chmod +x run_segment4.sh
./run_segment4.sh
```

Runs both Exp8+9 in one go.

---

## Recommended Timeline

### Conservative (5 sessions)
```
Day 1 Morning:  ./run_segment1.sh   (4-5h)
Day 1 Evening:  ./run_segment2.sh   (3-4h)
Day 2 Morning:  ./run_segment3.sh   (3-4h)
Day 2 Evening:  ./run_segment4a.sh  (4-5h)
Day 3 Morning:  ./run_segment4b.sh  (3-4h)
```

### Faster (4 sessions)
```
Day 1 Morning:  ./run_segment1.sh   (4-5h)
Day 1 Evening:  ./run_segment2.sh   (3-4h)
Day 2 Morning:  ./run_segment3.sh   (3-4h)
Day 2 Overnight: ./run_segment4.sh  (7-8h) ⚠️ LONG
```

### Minimal (Critical only, 2 sessions)
If you've already run Exp1-5:
```
Day 1: ./run_segment4a.sh  (4-5h)
Day 2: ./run_segment4b.sh  (3-4h)
```

---

## What If GPU Crashes Mid-Segment?

Each segment script uses `set -e` to stop on errors. If GPU crashes:

**Check what completed**:
```bash
./check_all_results.sh
```

**Re-run the failed segment**:
```bash
# If segment 2 failed, just re-run it:
./run_segment2.sh
```

The scripts skip already-completed experiments automatically.

---

## Verification

After each segment:
```bash
# Check files were created
ls -lh results/

# Verify specific segment
cat results/exp3_summary.json  # After segment 1
cat results/exp5_summary.json  # After segment 2
cat results/exp8_summary.json  # After segment 4a
```

After all segments:
```bash
./check_all_results.sh
```

Should show all 9 experiments complete.

---

## Quick Mode (If GPU is Really Unstable)

Use `--mode quick` to reduce runtime by 70%:

Edit each segment script and replace:
```bash
--mode standard
```
with:
```bash
--mode quick
```

**New runtimes**:
- Segment 1: 1.5-2h (was 4-5h)
- Segment 2: 1-1.5h (was 3-4h)
- Segment 3: 1-1.5h (was 3-4h)
- Segment 4a: 1.5-2h (was 4-5h)
- Segment 4b: 1-1.5h (was 3-4h)

**Total**: 6-8 hours (vs 17-22 hours for standard)

---

## File Structure

All scripts created:

```
run_segment1.sh    - Exp1-3 (base)
run_segment2.sh    - Exp4-5 (applications)
run_segment3.sh    - Exp6-7 (validation)
run_segment4.sh    - Exp8-9 (critical, combined)
run_segment4a.sh   - Exp8 only (scaling)
run_segment4b.sh   - Exp9 only (interpretability)
check_all_results.sh - Verify completion
```

---

## Priority Order (If Time Constrained)

**Must have** for paper:
1. Segment 1 (Exp1-3) - Foundation
2. Segment 2 (Exp4-5) - Core results
3. Segment 4A (Exp8) - Scaling ⭐ +30% acceptance

**Should have**:
4. Segment 4B (Exp9) - Interpretability ⭐ +25% acceptance

**Nice to have**:
5. Segment 3 (Exp6-7) - Robustness/safety +15% acceptance

---

## Commands Summary

**Make executable**:
```bash
chmod +x run_segment*.sh check_all_results.sh
```

**Run all segments** (conservative approach):
```bash
./run_segment1.sh   # Wait for completion
./run_segment2.sh   # Wait for completion
./run_segment3.sh   # Wait for completion
./run_segment4a.sh  # Wait for completion
./run_segment4b.sh  # Wait for completion
./check_all_results.sh
```

**Run critical only** (if Exp1-5 done):
```bash
./run_segment4a.sh
./run_segment4b.sh
./check_all_results.sh
```

---

## Bottom Line

**For maximum GPU safety** (no segment >5 hours):
```bash
chmod +x *.sh
./run_segment1.sh    # Stop after 5h
./run_segment2.sh    # Stop after 4h
./run_segment3.sh    # Stop after 4h
./run_segment4a.sh   # Stop after 5h ⭐ CRITICAL
./run_segment4b.sh   # Stop after 4h ⭐ CRITICAL
```

Each segment is independent - if GPU crashes, just re-run that segment!
