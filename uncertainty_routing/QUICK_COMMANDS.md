# Quick Commands Reference

## GPU-Safe Segmented Runs (RECOMMENDED)

### Run all experiments in safe chunks:
```bash
./run_segment1.sh    # 4-5h: Exp1-3 (base)
./run_segment2.sh    # 3-4h: Exp4-5 (applications)
./run_segment3.sh    # 3-4h: Exp6-7 (validation)
./run_segment4a.sh   # 4-5h: Exp8 (scaling) ⭐
./run_segment4b.sh   # 3-4h: Exp9 (interpretability) ⭐
```

**Max GPU time per run**: 5 hours
**Total**: 17-22 hours across 5 sessions

---

## Alternative: Full Pipeline (Single Run)

### If GPU is stable for 10-14 hours:
```bash
python run_complete_pipeline_v2.py --mode standard
```

---

## Critical Experiments Only

### If you already have Exp1-5 done:
```bash
./run_segment4a.sh   # Exp8: Scaling ⭐
./run_segment4b.sh   # Exp9: Interpretability ⭐
```

**Or single command**:
```bash
python run_complete_pipeline_v2.py --only-critical
```

---

## Quick Mode (Fast Testing)

### Reduce runtime by 70%:
Edit scripts and change `--mode standard` to `--mode quick`

**Or run directly**:
```bash
python run_complete_pipeline_v2.py --mode quick
```

---

## Verification

### Before starting:
```bash
python verify_complete_pipeline.py
```

### After each segment:
```bash
./check_all_results.sh
```

### Check specific results:
```bash
cat results/exp8_summary.json
cat results/exp9_summary.json
ls results/*.png
```

---

## If Something Crashes

### Check what completed:
```bash
./check_all_results.sh
```

### Re-run failed segment:
```bash
./run_segment2.sh  # Example: if segment 2 failed
```

### Skip completed experiments:
```bash
python run_complete_pipeline_v2.py --skip-exp1 --skip-exp2 --skip-exp3
```

---

## Priority Order

**Must do** (for publication):
1. Segments 1+2 (Exp1-5) - Base results
2. Segment 4A (Exp8) - Scaling ⭐

**Should do** (strong paper):
3. Segment 4B (Exp9) - Interpretability ⭐

**Nice to have** (complete paper):
4. Segment 3 (Exp6-7) - Robustness

---

## Time Estimates

| Command | Time | What It Does |
|---------|------|--------------|
| `./run_segment1.sh` | 4-5h | Exp1-3: Foundation |
| `./run_segment2.sh` | 3-4h | Exp4-5: Applications |
| `./run_segment3.sh` | 3-4h | Exp6-7: Validation |
| `./run_segment4a.sh` | 4-5h | Exp8: Scaling ⭐ |
| `./run_segment4b.sh` | 3-4h | Exp9: Interpretability ⭐ |
| **Total (segments)** | **17-22h** | **All 9 experiments** |
| `run_complete_pipeline_v2.py` | 10-14h | All 9 (single run) |
| `--only-critical` | 7h | Just Exp8+9 |
| `--mode quick` | 3-4h | Fast test (all 9) |

---

## Files to Read

| File | Purpose |
|------|---------|
| [GPU_SAFE_GUIDE.md](GPU_SAFE_GUIDE.md) | Detailed segmented runs guide |
| [SEGMENTED_RUNS.md](SEGMENTED_RUNS.md) | Full segmentation documentation |
| [START_HERE.md](START_HERE.md) | Simple 3-step start |
| [FINAL_CHECKLIST.md](FINAL_CHECKLIST.md) | Complete verification |
| [RUN_ME.md](RUN_ME.md) | Execution options |

---

## One-Liners

**Start segmented runs**:
```bash
./run_segment1.sh && ./run_segment2.sh && ./run_segment3.sh && ./run_segment4a.sh && ./run_segment4b.sh
```
*(Only if GPU is stable - otherwise run separately)*

**Verify before starting**:
```bash
python verify_complete_pipeline.py && ./run_segment1.sh
```

**Check everything completed**:
```bash
./check_all_results.sh && ls -lh results/*.png
```

---

## Bottom Line

**Safest approach** (GPU might crash):
```bash
./run_segment1.sh   # Run, wait for completion
./run_segment2.sh   # Run, wait for completion
./run_segment3.sh   # Run, wait for completion
./run_segment4a.sh  # Run, wait for completion ⭐
./run_segment4b.sh  # Run, wait for completion ⭐
```

**Fastest approach** (GPU is stable):
```bash
python run_complete_pipeline_v2.py --mode standard
```

**Critical only** (already have Exp1-5):
```bash
./run_segment4a.sh && ./run_segment4b.sh
```

---

Start now:
```bash
cd /Users/akshatatiwari/Desktop/MIT/mech_interp_research/uncertainty_routing
./run_segment1.sh
```
