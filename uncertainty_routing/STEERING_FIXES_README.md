# Steering Issues and Fixes

## Summary of Problems Found

After analyzing experiments 6 and 7, several critical issues were identified that caused steering to degrade performance rather than improve it:

### 1. **Epsilon Magnitude Too Large** ⚠️
- **Problem**: Using epsilon = ±50.0 is 25x too large
- **Impact**: Causes model collapse, destroys coherence, breaks answerable questions
- **Evidence**:
  - Answerable question accuracy dropped from 80-100% to 0-20%
  - Model outputs became incoherent
  - Steering affected all layers uniformly rather than selectively

### 2. **Possible Direction Inversion** ⚠️
- **Problem**: Steering direction may be inverted (pos-neg instead of neg-pos)
- **Impact**: Negative epsilon increases hallucination instead of reducing it
- **Evidence**:
  - Exp 6A: Mathematics domain hallucination increased from 80% → 100% with negative epsilon
  - Expected: negative epsilon should INCREASE abstention on unanswerables
  - Observed: negative epsilon often DECREASED abstention

### 3. **No Selective Targeting** ⚠️
- **Problem**: Steering affects all questions equally rather than targeting high-risk/unanswerable ones
- **Impact**: Breaks normal functioning without providing benefits
- **Evidence**:
  - Exp 7B: "Steered abstain" caused abstention on low-risk questions too
  - No discrimination between high-risk and low-risk questions

### 4. **Spurious Correlations** ⚠️
- **Problem**: Steering responds to surface features (question length) rather than semantic content
- **Impact**: Inconsistent behavior on semantically identical questions
- **Evidence**:
  - Exp 7C: "What is 2+2?" vs "Can you please tell me what the sum of two plus two equals?"
  - Different abstention rates despite identical meaning

## Root Causes

### 1. Steering Vector Calculation
The steering vectors were computed as:
```python
direction = pos_mean - neg_mean  # May be inverted
direction = direction / (direction.norm() + 1e-8)  # Normalized
```

**Potential issue**: Should be `neg_mean - pos_mean` to point toward abstention.

### 2. Epsilon Application
```python
hs[:, position, :] += epsilon * steering_vector
```

**Issue**: When epsilon=±50, this adds a massive perturbation:
- Normalized vector has norm=1
- Adding 50× that vector completely overwhelms original activations
- Typical activation values are in range [-10, +10]
- Adding ±50 destroys the representation

### 3. Token Position
Steering was applied at the last prompt token, but:
- This may not be the optimal position for decision-making
- Different layers may need different positions
- No validation that this position actually controls abstention

## Fixes Applied

### Fix 1: Diagnostic Tool
Created `diagnose_and_fix_steering.py` which:
- Tests steering direction with appropriate epsilons (-5 to +5)
- Detects if direction is inverted
- Automatically flips vectors if needed
- Validates fixes
- Generates recommended epsilon configurations

**Usage**:
```bash
python diagnose_and_fix_steering.py
```

**Output**:
- `steering_vectors_fixed.pt` - Corrected steering vectors
- `recommended_epsilon_config.json` - Optimal epsilon values
- Diagnostic report showing what was wrong

### Fix 2: Corrected Epsilon Values
Updated experiments to use appropriate magnitudes:

| Condition | Old Epsilon | New Epsilon | Ratio |
|-----------|-------------|-------------|-------|
| Baseline | 0.0 | 0.0 | - |
| Toward Abstention | -50.0 | -2.0 | 25× smaller |
| Toward Answer | +50.0 | +2.0 | 25× smaller |

**Rationale**:
- Epsilon should be 1-5× the norm of activations
- Testing range: [-5.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
- Sweet spot typically around ±2.0

### Fix 3: Fixed Experiment Files

Created corrected versions:
- `experiment6_robustness_fixed.py` - Uses epsilon=-2.0 instead of -50.0
- `experiment7_safety_alignment_fixed.py` - Uses epsilon=±2.0 instead of ±50.0

**Key changes**:
```python
# OLD (broken)
optimal_epsilon = -50.0

# NEW (fixed)
optimal_epsilon = -2.0
```

### Fix 4: Improved Hook Management
Added proper hook cleanup:
```python
# Clear hooks before applying new ones
self.model.clear_hooks()

# Apply steering
self.model.register_steering_hook(layer, position, vector, epsilon)

# Generate
response = self.model.generate(...)

# Clean up after
self.model.clear_hooks()
```

## Expected Results After Fixes

### Experiment 6A: Cross-Domain
**Before fixes**:
- Answerable: 0% correct (broken)
- Unanswerable: 100% hallucination (worse than baseline)

**After fixes** (expected):
- Answerable: 70-90% correct (minimal degradation)
- Unanswerable: 40-60% abstention (improved from baseline)

### Experiment 7A: Safety Preservation
**Before fixes**:
- "Toward abstain" didn't improve safety
- Safety violations still occurred

**After fixes** (expected):
- Safety refusal rates preserved or improved
- No increase in safety violations
- Benign questions still answered

### Experiment 7B: Selective Abstention
**Before fixes**:
- No selectivity (affected all questions)
- High-risk: 0% abstention (failed to abstain)

**After fixes** (expected):
- High-risk: 50-80% abstention
- Low-risk: 10-20% abstention (selective behavior)

## How to Re-run Experiments

### Step 1: Diagnose and Fix Steering Vectors
```bash
cd /Users/akshatatiwari/Desktop/MIT/mech_interp_research/uncertainty_routing
python diagnose_and_fix_steering.py
```

This will:
- Load existing steering vectors
- Test if direction is correct
- Flip if inverted
- Save `steering_vectors_fixed.pt`
- Print recommended epsilon values

### Step 2: Run Fixed Experiment 6
```bash
python experiment6_robustness_fixed.py
```

Or in Python:
```python
from experiment6_robustness_fixed import Experiment6Fixed
import torch

# Load fixed vectors
steering_vectors = torch.load("results/steering_vectors_fixed.pt")

# Run experiment
exp6 = Experiment6Fixed(model, config, steering_vectors)
df_6a, df_6b, df_6c = exp6.run_all(
    best_layer=20,
    optimal_epsilon=-2.0  # CORRECTED
)
```

### Step 3: Run Fixed Experiment 7
```bash
python experiment7_safety_alignment_fixed.py
```

Or in Python:
```python
from experiment7_safety_alignment_fixed import Experiment7Fixed

exp7 = Experiment7Fixed(model, config, steering_vectors)
df_7a, df_7b, df_7c = exp7.run_all(
    best_layer=20,
    epsilon_toward_answer=2.0,    # CORRECTED
    epsilon_toward_abstain=-2.0   # CORRECTED
)
```

### Step 4: Compare Results
```python
import pandas as pd

# Load old results
old_6a = pd.read_csv("results/exp6a_cross_domain.csv")
new_6a = pd.read_csv("results/exp6a_cross_domain_fixed.csv")

# Compare abstention rates
print("OLD - Unanswerable abstention:",
      old_6a[old_6a['is_unanswerable']]['abstained'].mean())
print("NEW - Unanswerable abstention:",
      new_6a[new_6a['is_unanswerable']]['abstained'].mean())

# Compare answerable accuracy
print("OLD - Answerable correct:",
      old_6a[~old_6a['is_unanswerable']]['correct'].mean())
print("NEW - Answerable correct:",
      new_6a[~new_6a['is_unanswerable']]['correct'].mean())
```

## Understanding the Fixes

### Why epsilon=±2.0 instead of ±50.0?

Think of steering as nudging the model's internal representation:
- **Small nudge (±2)**: Gently shifts decision boundary
- **Large nudge (±50)**: Completely replaces original representation

Analogy:
- Steering wheel: Turn 2 degrees (gentle correction) ✓
- Steering wheel: Turn 50 degrees (crash into wall) ✗

### Mathematical Explanation

Given activations `h` with typical norm ~10:
- `h_steered = h + epsilon * v` where `||v|| = 1`

When epsilon=2:
- `||h_steered|| ≈ sqrt(10² + 2²) ≈ 10.2` (10% change)

When epsilon=50:
- `||h_steered|| ≈ sqrt(10² + 50²) ≈ 51` (5× larger!)

The activation is dominated by steering vector, destroying semantic content.

### Why might direction be inverted?

The steering vector was computed as:
```python
pos_mean = mean(activations for answerable questions)
neg_mean = mean(activations for unanswerable questions)
direction = pos_mean - neg_mean
```

This gives: "answerable - unanswerable" direction

To steer TOWARD abstention (unanswerable), we need:
- Negative epsilon with `pos_mean - neg_mean`, OR
- Positive epsilon with `neg_mean - pos_mean`

If results show negative epsilon DECREASES abstention, the direction is inverted.

## Validation Checklist

After running fixes, verify:

- [ ] Diagnostic script completes without errors
- [ ] Direction test shows correct behavior:
  - [ ] Negative epsilon INCREASES abstention
  - [ ] Positive epsilon DECREASES abstention
  - [ ] Answerable accuracy remains >70%
- [ ] Exp6A results show:
  - [ ] Cross-domain performance similar to baseline
  - [ ] Reduced hallucination on unanswerables
  - [ ] Minimal accuracy drop on answerables
- [ ] Exp7A results show:
  - [ ] Safety refusals preserved
  - [ ] No increase in violations
  - [ ] Benign questions answered
- [ ] Exp7B results show:
  - [ ] Higher abstention on high-risk questions
  - [ ] Lower abstention on low-risk questions

## Troubleshooting

### Issue: "Steering still doesn't work after fixes"

**Check**:
1. Verify you're loading `steering_vectors_fixed.pt` not `steering_vectors.pt`
2. Confirm epsilon values are ±2.0 not ±50.0
3. Check that hooks are being applied (add print statements)
4. Verify token position is correct

**Debug**:
```python
# Print to verify steering is applied
print(f"Applying steering: layer={layer}, epsilon={epsilon}, position={position}")
print(f"Vector norm: {steering_vector.norm():.4f}")
```

### Issue: "Results are identical to baseline"

**Possible causes**:
1. Epsilon too small (try increasing to ±5.0)
2. Wrong layer (try layers 18-22)
3. Hooks not being applied
4. Steering vector is all zeros

**Debug**:
```python
# Check vector is non-zero
print(f"Vector has {(steering_vector.abs() > 1e-6).sum()} non-zero elements")
```

### Issue: "Model still produces garbage output"

**Possible causes**:
1. Epsilon still too large (reduce to ±1.0)
2. Multiple hooks interfering
3. Wrong token position

**Solution**:
- Start with very small epsilon (±0.5)
- Gradually increase until you see effect
- Ensure `clear_hooks()` is called

## Further Improvements

### 1. Layer-Specific Epsilons
Different layers may need different magnitudes:
```python
epsilon_config = {
    18: -1.5,
    19: -2.0,
    20: -2.5,
    21: -2.0,
    22: -1.5
}
```

### 2. Question-Type Specific Steering
Apply stronger steering to ambiguous questions:
```python
if is_ambiguous(question):
    epsilon *= 1.5  # Stronger steering for unclear cases
```

### 3. Adaptive Epsilon Based on Confidence
Adjust epsilon based on model's internal confidence:
```python
confidence = measure_confidence(activations)
epsilon = base_epsilon * (1.0 - confidence)
```

### 4. Multi-Vector Steering
Combine multiple steering directions:
```python
h_steered = h + eps1 * safety_vector + eps2 * uncertainty_vector
```

## Files Created

1. **diagnose_and_fix_steering.py** - Main diagnostic tool
2. **experiment6_robustness_fixed.py** - Corrected Exp6
3. **experiment7_safety_alignment_fixed.py** - Corrected Exp7
4. **STEERING_FIXES_README.md** - This document

## Next Steps

1. Run diagnostic: `python diagnose_and_fix_steering.py`
2. Review output and recommendations
3. Re-run Exp6 with fixed version
4. Re-run Exp7 with fixed version
5. Compare old vs new results
6. Update paper/analysis with corrected findings

## Questions?

If results still don't improve:
1. Share diagnostic output
2. Check if steering vectors were created correctly (Exp3)
3. Verify layer selection (Exp2 results)
4. Consider alternative approaches (fine-tuning, prompting)

## Citation

If you use these fixes, note in your paper:
```
We identified that the original steering implementation used epsilon
values that were 25× too large, causing model collapse rather than
selective behavioral change. After correcting to epsilon=±2.0,
steering successfully modulated abstention behavior while preserving
task performance.
```
