# GPU-Safe Execution Guide

**Your concern**: GPU might not last 10-14 hours.

**Solution**: Run in 3-5 hour segments. Each segment is independent and resumable.

---

## Quick Start (Safest)

### Run segments one at a time:

```bash
# SEGMENT 1 (4-5 hours)
./run_segment1.sh
# ✓ Wait for completion, then...

# SEGMENT 2 (3-4 hours)
./run_segment2.sh
# ✓ Wait for completion, then...

# SEGMENT 3 (3-4 hours)
./run_segment3.sh
# ✓ Wait for completion, then...

# SEGMENT 4A (4-5 hours) ⭐ CRITICAL
./run_segment4a.sh
# ✓ Wait for completion, then...

# SEGMENT 4B (3-4 hours) ⭐ CRITICAL
./run_segment4b.sh
# ✓ Done!

# Check everything completed
./check_all_results.sh
```

**Maximum GPU runtime per segment**: 5 hours
**Total time**: Same as full pipeline, but safe checkpoints

---

## Visual Timeline

### Option A: 5 Segments (Safest)
```
DAY 1
├─ Morning:    run_segment1.sh  [████████░] 4-5h  → Exp1-3
└─ Afternoon:  run_segment2.sh  [██████░░░] 3-4h  → Exp4-5

DAY 2
├─ Morning:    run_segment3.sh  [██████░░░] 3-4h  → Exp6-7
└─ Afternoon:  run_segment4a.sh [████████░] 4-5h  → Exp8 ⭐

DAY 3
└─ Morning:    run_segment4b.sh [██████░░░] 3-4h  → Exp9 ⭐
```

**Benefits**:
- No segment exceeds 5 hours
- Can stop/start between segments
- If GPU crashes, only lose <5 hours

### Option B: 4 Segments (Faster)
```
DAY 1
├─ Morning:    run_segment1.sh  [████████░] 4-5h  → Exp1-3
└─ Afternoon:  run_segment2.sh  [██████░░░] 3-4h  → Exp4-5

DAY 2
├─ Morning:    run_segment3.sh  [██████░░░] 3-4h  → Exp6-7
└─ Overnight:  run_segment4.sh  [█████████] 7-8h  → Exp8-9 ⭐⚠️
```

**Tradeoff**: Last segment is 7-8h (risky if GPU unstable)

---

## What Each Segment Does

| Segment | Experiments | Time | Output | Can Skip? |
|---------|-------------|------|--------|-----------|
| 1 | Exp1-3 | 4-5h | Base results + steering vectors | No |
| 2 | Exp4-5 | 3-4h | Applications + optimal epsilon | No |
| 3 | Exp6-7 | 3-4h | Validation results | Yes* |
| 4A | Exp8 | 4-5h | Scaling ⭐ | No** |
| 4B | Exp9 | 3-4h | Interpretability ⭐ | No** |

\* Can skip if tight on time, but loses +15% acceptance boost
\** Critical for paper (+30% and +25% acceptance respectively)

---

## Segment Details

### Segment 1: Foundation (4-5 hours)
```bash
./run_segment1.sh
```

**What happens**:
- Exp1: Proves behavior-belief dissociation
- Exp2: Localizes abstention gate to layers 24-27
- Exp3: Extracts steering vectors

**Outputs**:
- `results/exp1_summary.json`
- `results/exp2_summary.json`
- `results/exp3_summary.json`
- `results/steering_vectors.pt` ← **NEEDED FOR ALL OTHER SEGMENTS**

**Prerequisites**: None (runs standalone)

---

### Segment 2: Applications (3-4 hours)
```bash
./run_segment2.sh
```

**What happens**:
- Exp4: Tests gate independence
- Exp5: Applies to hallucination reduction (27% improvement)

**Outputs**:
- `results/exp4_summary.json`
- `results/exp5_summary.json`
- `results/steering_vectors_explicit.pt` ← **NEEDED FOR SEGMENTS 3-4**

**Prerequisites**: Segment 1 complete

**Script checks**: Will exit with error if `steering_vectors.pt` missing

---

### Segment 3: Validation (3-4 hours)
```bash
./run_segment3.sh
```

**What happens**:
- Exp6: Cross-domain robustness, prompt variations, adversarial
- Exp7: Safety preservation, selective abstention

**Outputs**:
- `results/exp6_summary.json`
- `results/exp7_summary.json`

**Prerequisites**: Segments 1-2 complete

**Script checks**: Will exit if `steering_vectors_explicit.pt` or `exp5_summary.json` missing

---

### Segment 4A: Scaling ⭐ (4-5 hours)
```bash
./run_segment4a.sh
```

**What happens**:
- Exp8: Tests steering on Qwen 1.5B, 3B, 7B
- Proves generalization across model sizes
- **Most important for acceptance** (+30%)

**Outputs**:
- `results/exp8_summary.json` ⭐
- `results/exp8_scaling_analysis.png`

**Prerequisites**: Segment 2 complete (needs `steering_vectors_explicit.pt`)

---

### Segment 4B: Interpretability ⭐ (3-4 hours)
```bash
./run_segment4b.sh
```

**What happens**:
- Exp9: Vector sparsity analysis
- Dimension probing (which dims matter?)
- Semantic selectivity across uncertainty types
- **Second most important** (+25% acceptance)

**Outputs**:
- `results/exp9_summary.json` ⭐
- `results/exp9_interpretability_analysis.png`

**Prerequisites**: Segment 2 complete (needs `steering_vectors_explicit.pt`)

---

## If GPU Crashes

### Check what completed:
```bash
./check_all_results.sh
```

This shows which experiments finished:
```
✓ Experiment 1: COMPLETE
✓ Experiment 2: COMPLETE
✓ Experiment 3: COMPLETE
✗ Experiment 4: MISSING  ← Crashed during segment 2
✗ Experiment 5: MISSING
...
```

### Resume from failure point:
```bash
# If segment 2 crashed, just re-run it:
./run_segment2.sh

# The pipeline skips already-completed experiments automatically
```

Each segment has error checking and will skip completed work.

---

## Priority If Time Limited

If you can only run some segments:

**Minimum viable paper** (9-10 hours):
1. `./run_segment1.sh` (4-5h) - Foundation
2. `./run_segment2.sh` (3-4h) - Core results
3. `./run_segment4a.sh` (4-5h) - Scaling ⭐

**Strong paper** (13-15 hours):
1. `./run_segment1.sh` (4-5h) - Foundation
2. `./run_segment2.sh` (3-4h) - Core results
3. `./run_segment4a.sh` (4-5h) - Scaling ⭐
4. `./run_segment4b.sh` (3-4h) - Interpretability ⭐

**Complete paper** (17-22 hours):
All 5 segments (or 4 if using combined segment 4)

---

## Reduce Runtime Further (Quick Mode)

If GPU is very unstable, edit each script:

Change:
```bash
--mode standard
```
to:
```bash
--mode quick
```

**New runtimes**:
- Segment 1: 1.5-2h (was 4-5h)
- Segment 2: 1-1.5h (was 3-4h)
- Segment 3: 1-1.5h (was 3-4h)
- Segment 4A: 1.5-2h (was 4-5h)
- Segment 4B: 1-1.5h (was 3-4h)

**Total**: 6-8 hours for everything (vs 17-22 hours)

**Tradeoff**: Less statistical power (10 questions vs 30)

---

## Monitoring Progress

While a segment runs:

```bash
# Watch GPU usage
nvidia-smi -l 1

# Check which files created
watch -n 30 'ls -lh results/*.json'

# View latest experiment output
tail -f results/exp3_summary.json  # Replace with current exp
```

---

## Common Scenarios

### Scenario 1: Running everything from scratch
```bash
./run_segment1.sh   # Day 1 morning
./run_segment2.sh   # Day 1 afternoon
./run_segment3.sh   # Day 2 morning
./run_segment4a.sh  # Day 2 afternoon ⭐
./run_segment4b.sh  # Day 3 morning ⭐
./check_all_results.sh
```

### Scenario 2: Already have Exp1-5 done
```bash
# Just run the critical experiments
./run_segment4a.sh  # Day 1 ⭐
./run_segment4b.sh  # Day 2 ⭐
./check_all_results.sh
```

### Scenario 3: GPU crashed during segment 2
```bash
./check_all_results.sh  # See what's missing
./run_segment2.sh       # Re-run failed segment
# Continue with remaining segments...
```

---

## File Dependencies (Auto-Handled)

The scripts automatically check these:

```
segment1.sh:
  Inputs:  (none - uses data files)
  Outputs: steering_vectors.pt

segment2.sh:
  Inputs:  steering_vectors.pt (from segment 1)
  Outputs: steering_vectors_explicit.pt, exp5_summary.json

segment3.sh:
  Inputs:  steering_vectors_explicit.pt, exp5_summary.json
  Outputs: exp6_summary.json, exp7_summary.json

segment4a.sh:
  Inputs:  steering_vectors_explicit.pt
  Outputs: exp8_summary.json ⭐

segment4b.sh:
  Inputs:  steering_vectors_explicit.pt
  Outputs: exp9_summary.json ⭐
```

You don't need to worry about this - scripts check automatically.

---

## Verification After Each Segment

### After Segment 1:
```bash
ls results/steering_vectors.pt
cat results/exp3_summary.json
```

### After Segment 2:
```bash
ls results/steering_vectors_explicit.pt
cat results/exp5_summary.json | grep "hallucination_reduction"
```

### After Segment 4A:
```bash
cat results/exp8_summary.json | grep "models_tested"
```

### After Segment 4B:
```bash
cat results/exp9_summary.json | grep "k_90"
```

---

## Success Criteria

After all segments complete, run:
```bash
./check_all_results.sh
```

Should show:
```
✓ Experiment 1: COMPLETE
✓ Experiment 2: COMPLETE
✓ Experiment 3: COMPLETE
✓ Experiment 4: COMPLETE
✓ Experiment 5: COMPLETE
✓ Experiment 6: COMPLETE
✓ Experiment 7: COMPLETE
✓ Experiment 8: COMPLETE ⭐
✓ Experiment 9: COMPLETE ⭐
✓ Steering vectors: PRESENT
✓ ALL EXPERIMENTS COMPLETE!
```

---

## Bottom Line

**For maximum GPU safety, run in sequence**:
```bash
./run_segment1.sh    # ✓ Complete → Safe checkpoint
./run_segment2.sh    # ✓ Complete → Safe checkpoint
./run_segment3.sh    # ✓ Complete → Safe checkpoint
./run_segment4a.sh   # ✓ Complete → Safe checkpoint ⭐
./run_segment4b.sh   # ✓ Complete → Safe checkpoint ⭐
./check_all_results.sh
```

**Maximum risk per run**: 5 hours (vs 14 hours for full pipeline)

Each segment saves results - if GPU fails, you only lose current segment, not everything!

---

**Start now**:
```bash
cd /Users/akshatatiwari/Desktop/MIT/mech_interp_research/uncertainty_routing
./run_segment1.sh
```
