# Research Paper Action Plan
## Comprehensive Strategy for Solid Results

---

## Current Status Summary

### What's Working ✓
- **Experiment 1-4**: Core uncertainty detection and steering mechanics validated
- **Core steering mechanism**: Fixed hook bug - now applies at every forward pass
- **Hyperparameters**: Optimized epsilon = -10.0, layer = 20
- **Experiment 5**: Framework ready with expanded training data

### What Needs Improvement
- **Experiment 6**: 55% abstention (need 85%+), 0% on math domain
- **Experiment 7**: Mixed results, high-risk questions only 33% abstention
- **Root Cause**: Training data had only 10+10 questions, minimal domain diversity
- **Solution**: Created expanded datasets with 20+20 balanced questions ✓

---

## Immediate Action Plan

### Phase 1: Retrain Core Steering Vectors (CRITICAL)
**File**: `experiment5_trustworthiness.py` ✓ UPDATED

**Status**: ✅ Modified to use expanded datasets

**Action**: Run this command
```bash
python experiment5_trustworthiness.py
```

**Expected Runtime**: 4-6 hours (full epsilon sweep)

**Expected Outcomes**:
- Better steering vectors with domain-balanced training
- Should see improvement in mathematics domain representation
- Optimal epsilon may shift slightly (currently -10.0)

**Success Criteria**:
- Baseline hallucination on unanswerables: 20-40% (room to improve)
- Best epsilon achieves: 90%+ abstention on unanswerables
- Maintains: 85%+ coverage on answerables
- No catastrophic drops in accuracy

---

### Phase 2: Validate Robustness (Experiment 6)
**File**: `experiment6_robustness.py` ✓ ALREADY UPDATED

**Action**: Run after Phase 1 completes
```bash
python experiment6_robustness.py
```

**Expected Runtime**: 2-3 hours

**Current Results**: 55% abstention overall, 0% on math
**Target Results**: 85%+ abstention overall, 60%+ on math

**Why Improvement Expected**:
- New steering vectors trained on 5 math unanswerable questions (vs 0 before)
- 4 math answerable questions (vs 1 before)
- Better domain balance should improve cross-domain generalization

**Analysis Focus**:
- Domain-specific performance breakdown
- Compare before/after training data expansion
- Identify remaining weak domains

---

### Phase 3: Validate Safety (Experiment 7)
**File**: `experiment7_safety_alignment.py` ✓ ALREADY UPDATED

**Action**: Run after Phase 1 completes
```bash
python experiment7_safety_alignment.py
```

**Expected Runtime**: 1-2 hours

**Current Results**:
- Safety preservation: Good (no violations)
- Selective abstention: 33% on high-risk, 0% on low-risk
- Spurious correlations: Some issues

**Target Results**:
- Safety preservation: 100% (maintain)
- Selective abstention: 70%+ on high-risk, <5% on low-risk
- Spurious correlations: Question length should not dominate

**Why Improvement Expected**:
- Better uncertainty representation from diverse training
- More robust steering vectors

---

## Additional Experiments for Paper Strength

### Experiment 8: Hyperparameter Sensitivity Analysis
**Priority**: MEDIUM
**Estimated Time**: 3-4 hours
**Status**: Not yet created

**Purpose**: Demonstrate robustness to hyperparameter choices

**Design**:
```python
# Test multiple epsilon values
epsilon_values = [-20.0, -15.0, -12.0, -10.0, -8.0, -5.0]

# Test multiple layers
layers = [16, 17, 18, 20]

# Full grid: 6 epsilons × 4 layers = 24 conditions
# Test on: experiment 6 evaluation set (unanswerable questions)
```

**Expected Results**:
- Layer 20 should be optimal (or 18/17 close)
- Epsilon -10 to -15 should be optimal range
- Demonstrates: results are robust, not cherry-picked

**Paper Value**:
- Addresses reviewer concern: "Did you just find lucky hyperparameters?"
- Shows principled selection process
- Ablation study

---

### Experiment 9: Training Data Ablation
**Priority**: HIGH
**Estimated Time**: 6-8 hours
**Status**: Not yet created

**Purpose**: Quantify impact of training data diversity

**Design**:
```python
# Compare 4 training conditions:
# 1. Original (10+10, minimal diversity) - BASELINE
# 2. Expanded (20+20, balanced domains) - CURRENT
# 3. Math-only (10+10, all math questions)
# 4. Domain-stratified (5 per domain)

# For each condition:
# - Train steering vectors
# - Evaluate on exp6 test set
# - Measure domain-specific performance
```

**Expected Results**:
- Condition 1: Poor generalization (55% overall, 0% math)
- Condition 2: Best overall (85%+ overall, 60%+ math)
- Condition 3: Perfect math (95%+), poor history/science
- Condition 4: Most balanced across domains

**Paper Value**:
- Shows training data diversity is critical
- Quantifies the improvement from your expansion
- Demonstrates understanding of failure modes
- Reviewer gold: "They systematically diagnosed and fixed the issue"

---

### Experiment 10: Comparison with Baseline Methods
**Priority**: MEDIUM-HIGH
**Estimated Time**: 4-6 hours
**Status**: Not yet created

**Purpose**: Position your method against alternatives

**Baselines to Compare**:
1. **Prompt engineering**: "Say 'I don't know' if uncertain"
2. **Verbalized confidence**: "Rate your confidence 1-10"
3. **Multiple samples**: Check consistency across 5 generations
4. **Logit-based**: Use model probabilities as uncertainty signal

**Test on**: Same exp6 unanswerable questions

**Expected Results**:
- Prompt engineering: 40-50% abstention (inconsistent)
- Verbalized confidence: Not well-calibrated, 30-40% effective abstention
- Multiple samples: Expensive (5× cost), 60-70% effective
- Logit-based: Modest performance, 50-60%
- **Your method**: 85%+ abstention, efficient (single forward pass with hook)

**Paper Value**:
- Shows your method outperforms alternatives
- Demonstrates mechanistic approach > heuristics
- Essential for convincing reviewers of novelty

---

## Expected Timeline

| Phase | Task | Duration | Cumulative |
|-------|------|----------|------------|
| 1 | Retrain exp5 (expanded data) | 4-6h | 6h |
| 2 | Rerun exp6 robustness | 2-3h | 9h |
| 3 | Rerun exp7 safety | 1-2h | 11h |
| 4 | Exp8 hyperparameter sweep | 3-4h | 15h |
| 5 | Exp9 training data ablation | 6-8h | 23h |
| 6 | Exp10 baseline comparisons | 4-6h | 29h |
| 7 | Analysis & figures | 4-6h | 35h |
| 8 | Writing & revision | 10-15h | 50h |

**Total**: ~50 hours (6-7 full days of work)

---

## Paper Structure & Framing

### Title (Options)
1. "Activation Steering for Uncertainty Routing in Large Language Models"
2. "Mechanistic Control of Abstention Behavior via Activation Steering"
3. "Learning to Abstain: Training Steering Vectors for Trustworthy LLMs"

### Abstract (~250 words)
- **Problem**: LLMs hallucinate on unanswerable questions
- **Gap**: Existing methods (prompting, confidence scores) unreliable
- **Method**: Train steering vectors via contrastive learning on answerable/unanswerable pairs
- **Results**: 85%+ abstention on unanswerable while maintaining 90%+ coverage on answerable
- **Key Finding**: Training data diversity is critical; domain-balanced training improves generalization
- **Implications**: Mechanistic interventions enable controllable behavior without retraining

### 1. Introduction
- Motivation: Trustworthy AI needs reliable uncertainty expression
- Limitations of current approaches (prompting, verbalized uncertainty)
- Our contribution: Mechanistic steering for abstention behavior
- Key insight: Answer/abstain distinction exists in activation space

### 2. Related Work
- Uncertainty quantification in LLMs
- Activation steering (Zou et al., Turner et al.)
- Hallucination mitigation
- Mechanistic interpretability

### 3. Method
- **3.1 Steering Vector Training**: Contrastive learning on answerable/unanswerable pairs
- **3.2 Intervention Mechanism**: Layer-wise activation steering
- **3.3 Hyperparameter Selection**: Layer and epsilon optimization

### 4. Experiments
- **4.1 Core Validation** (Exp 1-4): Uncertainty detection, steering independence
- **4.2 Trustworthiness Application** (Exp 5): Risk-coverage tradeoff
- **4.3 Robustness Evaluation** (Exp 6): Cross-domain generalization
- **4.4 Safety Alignment** (Exp 7): Preserving refusal behaviors
- **4.5 Ablation Studies** (Exp 9): Training data diversity
- **4.6 Hyperparameter Sensitivity** (Exp 8): Robustness analysis
- **4.7 Baseline Comparisons** (Exp 10): vs. alternative approaches

### 5. Results & Analysis
- **Domain-specific performance**: Math, science, history, etc.
- **Training data impact**: 10+10 → 20+20 expanded: 55% → 85%+
- **Safety preservation**: No compromise to refusal behaviors
- **Efficiency**: Single forward pass, no model retraining

### 6. Discussion
- **Limitations**:
  - Still imperfect on mathematics (60-70% vs 85%+ overall)
  - Requires steering vector training phase
  - Domain generalization depends on training diversity
- **Future Work**:
  - Domain-adaptive steering
  - Automated training data curation
  - Multi-capability preservation (truthfulness + safety + factuality)
- **Broader Impact**: Enables more trustworthy AI systems

### 7. Conclusion
- Activation steering is viable for uncertainty routing
- Training data diversity is critical for generalization
- Method balances abstention with coverage/accuracy
- Mechanistic approaches enable fine-grained behavior control

---

## Key Messages for Paper

### What Makes This Work Strong

1. **Systematic diagnosis and fix**:
   - Found root cause (training data diversity)
   - Quantified impact (55% → 85%+)
   - Demonstrated solution effectiveness

2. **Comprehensive evaluation**:
   - Not just accuracy metrics
   - Robustness across domains
   - Safety preservation
   - Multiple test conditions

3. **Honest about limitations**:
   - Mathematics still challenging (60-70%)
   - Requires training data curation
   - Domain generalization not perfect

4. **Mechanistic insight**:
   - Shows answerable/unanswerable distinction exists in activations
   - Demonstrates controllable intervention
   - Links to broader interpretability work

5. **Practical value**:
   - Efficient (single forward pass)
   - No model retraining required
   - Tunable risk-coverage tradeoff

### How to Frame Current Results

**Be honest and scientific**:

❌ Don't say: "Our method achieves near-perfect abstention"
✅ Do say: "Our method achieves 85% abstention on unanswerable questions while maintaining 90% coverage on answerable questions, with performance varying by domain (mathematics: 70%, history: 90%)"

❌ Don't hide: Initial results with limited training data
✅ Do include: Ablation study showing impact of training data diversity (Exp 9)

**Emphasize the journey**:
- "We initially observed poor generalization to mathematics (0% abstention)"
- "Analysis revealed insufficient domain diversity in training data"
- "Expanding training set to 20+20 with balanced domains improved performance to 70%+"
- "This demonstrates the importance of representative training data for steering vectors"

**This is good science**: Finding and fixing problems is research!

---

## Immediate Next Steps (Priority Order)

### 🚨 CRITICAL - Do These First
1. ✅ **DONE**: Modified exp5 to use expanded datasets
2. ⏳ **NOW**: Run `python experiment5_trustworthiness.py` (4-6 hours)
3. ⏳ **NEXT**: Run `python experiment6_robustness.py` (2-3 hours)
4. ⏳ **NEXT**: Run `python experiment7_safety_alignment.py` (1-2 hours)

### 📊 HIGH PRIORITY - Do These Next
5. ⏳ Create and run Experiment 9 (training data ablation) - this will be your strongest result
6. ⏳ Create and run Experiment 10 (baseline comparisons) - essential for positioning

### 📈 MEDIUM PRIORITY - Do if Time Allows
7. ⏳ Create and run Experiment 8 (hyperparameter sensitivity)
8. ⏳ Additional domain-specific analysis

---

## Success Criteria for "Solid Research Paper"

### Minimum Viable Results (Publishable at Workshop)
- ✅ Exp 1-4: Core validation complete
- ✅ Exp 5: Steering vectors trained and working
- ✅ Exp 6: 70%+ abstention overall
- ✅ Exp 7: Safety preserved
- ⚠️ Exp 9: Training data ablation showing systematic improvement

### Strong Conference Paper Results (Target)
- ✅ All minimum viable results
- ✅ Exp 6: 85%+ abstention overall, 60%+ all domains
- ✅ Exp 9: Comprehensive ablation with 4 conditions
- ✅ Exp 10: Outperforms 3+ baseline methods
- ✅ Exp 8: Robustness demonstrated

### Exceptional Paper (Reach Goal)
- ✅ All strong results
- ✅ Domain-specific steering vectors (per-domain training)
- ✅ Automated training data selection method
- ✅ Real-world deployment case study
- ✅ Theoretical analysis of why steering works

---

## Commands to Run (In Order)

```bash
# Phase 1: Retrain with better data
python experiment5_trustworthiness.py

# Phase 2: Validate improvements
python experiment6_robustness.py
python experiment7_safety_alignment.py

# Phase 3: Check results
ls -lh results/exp6*
ls -lh results/exp7*

# If results look good (85%+ on exp6), proceed to:
# - Write Experiment 9 script (I can help with this)
# - Write Experiment 10 script (I can help with this)
# - Run additional experiments
# - Generate figures and tables for paper
```

---

## Confidence Assessment

### High Confidence (>80%)
- Exp 5 will improve with expanded data
- Exp 6 will show 70-85%+ abstention (up from 55%)
- Mathematics domain will improve significantly (0% → 60%+)
- Safety preservation will maintain (exp 7a)

### Medium Confidence (50-80%)
- Exp 6 will achieve 85%+ across ALL domains
- Exp 7 high-risk abstention will reach 70%+
- Your method will outperform all baselines in exp 10

### Lower Confidence (<50%)
- Perfect mathematics performance (95%+) without domain-specific steering
- Zero false positives on clearly answerable questions
- Single training set works optimally for all downstream tasks

### Areas Needing More Investigation
- Why mathematics is harder (numerical reasoning vs factual recall?)
- Whether certain question types are fundamentally harder
- Whether layer 20 is truly optimal or could improve with earlier layers
- Whether epsilon could be question-adaptive

---

## Final Recommendation

**START NOW with Phase 1**: The most impactful change is retraining experiment 5 with expanded data. This single change should yield:
- 55% → 85%+ improvement in exp6
- 0% → 60%+ improvement in math domain
- Better foundation for all downstream experiments

**Then prioritize Experiment 9**: The training data ablation will be your strongest contribution because:
- Shows systematic problem diagnosis
- Quantifies your solution's impact
- Demonstrates scientific rigor
- Makes clear you understand the mechanism

**Everything else is bonus**: With solid exp5/6/7 results and a good exp9 ablation, you have a publishable paper. Experiments 8 and 10 make it stronger but aren't essential.

---

## Questions to Consider

1. **Target venue**: Workshop vs conference? This determines scope needed.
2. **Novelty claim**: Is it the method (steering for abstention) or the finding (training data diversity matters)?
3. **Timeline**: Do you have 1 week or 1 month?
4. **Resources**: Can you run multiple experiments in parallel?

Let me know which experiments you want help implementing next!
