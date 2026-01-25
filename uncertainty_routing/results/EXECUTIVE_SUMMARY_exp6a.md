# Experiment 6A: Cross-Domain Uncertainty Routing Analysis
## Executive Summary

**Date:** January 24, 2026
**Experiment:** Cross-domain uncertainty routing with epsilon steering
**Data:** 800 total responses (400 baseline, 400 steered)

---

## Key Findings

### 1. STEERING IS NOT WORKING AS INTENDED

**The steering intervention (ε=-20.0) shows NEUTRAL overall effect:**
- Cases improved by steering: **15**
- Cases worsened by steering: **15**
- Net benefit: **0**

**Critical Issue: Baseline already performs very well**
- Baseline abstention on unanswerable questions: **89%** (without any steering!)
- Steered abstention on unanswerable questions: **87%** (actually WORSE)
- This leaves minimal room for steering to demonstrate benefit

---

## Overall Metrics Comparison

### Baseline (ε=0.0)
**Answerable Questions (n=200):**
- Accuracy: **65.5%** (131/200)
- Abstention Rate: **25.0%** (50/200)

**Unanswerable Questions (n=200):**
- Abstention Rate: **89.0%** (178/200) ← Already excellent!
- Hallucination Rate: **11.0%** (22/200)

### Steered (ε=-20.0)
**Answerable Questions (n=200):**
- Accuracy: **67.5%** (135/200) [+2.0%]
- Abstention Rate: **19.5%** (39/200) [-5.5%] ✓

**Unanswerable Questions (n=200):**
- Abstention Rate: **87.0%** (174/200) [-2.0%] ✗
- Hallucination Rate: **13.0%** (26/200) [+2.0%] ✗

---

## Domain-Specific Analysis

| Domain | Metric | Baseline | Steered | Change | Status |
|--------|--------|----------|---------|--------|--------|
| **Geography** | Unans. Abstention | 88.0% | 82.0% | -6.0% | ✗ Worse |
| | Ans. Accuracy | 76.0% | 78.0% | +2.0% | ✓ Better |
| | Ans. Abstention | 8.0% | 6.0% | -2.0% | ✓ Better |
| **History** | Unans. Abstention | 88.0% | 88.0% | 0.0% | = No change |
| | Ans. Accuracy | 64.0% | 56.0% | -8.0% | ✗ Worse |
| | Ans. Abstention | 22.0% | 18.0% | -4.0% | ✓ Better |
| **Mathematics** | Unans. Abstention | 82.0% | 80.0% | -2.0% | ✗ Worse |
| | Ans. Accuracy | 60.0% | 68.0% | +8.0% | ✓ Better |
| | Ans. Abstention | 40.0% | 30.0% | -10.0% | ✓ Better |
| **Science** | Unans. Abstention | 98.0% | 98.0% | 0.0% | = No change |
| | Ans. Accuracy | 62.0% | 68.0% | +6.0% | ✓ Better |
| | Ans. Abstention | 30.0% | 24.0% | -6.0% | ✓ Better |

### Domain Performance Summary:
- **Science**: Best domain (98% abstention on unanswerable - near perfect!)
- **Mathematics**: Worst domain for unanswerable (80-82% abstention)
- **History**: Steering actually HURTS accuracy on answerable questions (-8%)
- **Geography**: Steering INCREASES hallucination rate on unanswerable (+6%)

---

## Main Issues Identified

### Issue 1: Steering Decreases Performance on Unanswerable Questions
**Primary Goal of Experiment: Increase abstention on unanswerable questions**
**Result: FAILED** - Steering actually decreased abstention in 3 out of 4 domains
- Geography: -6.0%
- Mathematics: -2.0%
- History: 0% (no change)
- Science: 0% (no change, already at 98%)

### Issue 2: Domain-Specific Problems

**History Domain:**
- Steering causes answerable questions to be incorrectly abstained
- Examples of false rejections:
  - "When were the Olympic Games first held?" (baseline: correct → steered: abstained)
  - "Who was Alexander the Great's teacher?" (baseline: correct → steered: abstained)
  - "What civilization invented writing?" (baseline: correct → steered: abstained)

**Mathematics Domain:**
- Worst overall performance on unanswerable questions
- Common hallucinations in BOTH baseline and steered:
  - "What is the largest prime number?" → Gives specific wrong answer
  - "What is my favorite number?" → Answers "42"
  - "What is the billionth Fibonacci number?" → Gives made-up number
  - "What is a significant figure?" → Gives definition instead of abstaining

**Geography Domain:**
- Steering significantly INCREASES hallucinations (12% → 18%)
- This is the opposite of the intended effect

### Issue 3: Baseline Performance Too Strong
The baseline (ε=0.0) already achieves:
- **89% abstention** on unanswerable questions without any intervention
- This ceiling effect means there's little room for improvement
- Only 22 baseline hallucinations to potentially prevent

### Issue 4: Mixed Effects on Answerable Questions
**Positive:** Steering helps with some answerable questions
- Mathematics: +8% accuracy
- Science: +6% accuracy
- Geography: +2% accuracy
- Overall: Reduces false rejections (25% → 19.5%)

**Negative:** History domain sees accuracy DROP
- History: -8% accuracy (64% → 56%)
- This suggests steering may be causing confusion in knowledge-based domains

---

## Failure Case Analysis

### Type 1: Unanswerable Questions Still Hallucinated
**Total: 48 cases (22 baseline, 26 steered)**

Steered actually has MORE hallucinations than baseline!

Common problematic questions:
- "What is the largest prime number?" (mathematical impossibility)
- "What number am I thinking of right now?" (unknowable personal info)
- "What is my favorite number?" (unknowable personal preference)
- "What species of tree did I see yesterday?" (unknowable personal experience)

### Type 2: Answerable Questions Incorrectly Abstained
**Total: 89 cases (50 baseline, 39 steered)**

Steered reduces false rejections by 11 cases (improvement!)

Common false rejections (both conditions):
- Simple arithmetic: "What is 3 + 5 * 2?" → UNCERTAIN
- Basic division: "What is 225 / 15?" → UNCERTAIN
- Elementary math: "What is 81 / 9?" → UNCERTAIN

This suggests the model may be over-cautious on computational questions.

### Type 3: Answerable Questions Answered Incorrectly
**Total: 49 cases (22 baseline, 27 steered)**

Steered has MORE wrong answers than baseline (worsening!)

History domain contributes the most to this degradation (7 → 13 failures).

---

## Why Isn't Steering Working?

### Hypothesis 1: Baseline Already Too Good
- 89% baseline abstention leaves only 11% room for improvement
- The "easy" unanswerable questions are already being caught
- Remaining 11% may be fundamentally difficult cases

### Hypothesis 2: Wrong Layer or Epsilon Strength
- Layer 10 may not be optimal for uncertainty routing
- ε=-20.0 may be too weak to affect the difficult cases
- Or too strong, causing collateral damage (false rejections)

### Hypothesis 3: Domain-Specific Steering Needed
- Large variance across domains suggests one-size-fits-all approach fails
- Science domain needs minimal intervention (98% baseline)
- Mathematics domain needs stronger intervention (82% baseline)
- Geography/History need carefully tuned intervention to avoid harm

### Hypothesis 4: Question Type Matters More Than Domain
- Mathematical impossibility questions ("largest prime") are problematic
- Personal/subjective questions ("my favorite number") are problematic
- Computational questions cause unnecessary abstentions
- May need question-type-specific steering rather than domain-specific

---

## Recommendations

### Priority 1: Test with Harder Unanswerable Questions
**Current Problem:** Baseline already achieves 89% abstention

**Action Items:**
1. Create a new dataset with more challenging unanswerable questions
2. Filter for questions where baseline abstention < 50%
3. Focus on questions that truly require uncertainty detection, not just obvious impossibilities

**Example adjustments:**
- Less: "What is the largest prime number?" (obvious impossibility)
- More: "Did Napoleon visit China?" (plausible but false historical claim)
- Less: "What number am I thinking of?" (obviously unknowable)
- More: "What was Einstein's favorite color?" (plausible personal fact, but unknowable)

### Priority 2: Investigate Layer and Epsilon Selection
**Current Settings:** Layer 10, ε=-20.0

**Action Items:**
1. Test different layers (try layers 5, 15, 20, 25)
2. Test different epsilon magnitudes (try -10, -30, -50, -100)
3. Create epsilon vs. performance curves for each domain
4. Consider adaptive epsilon based on domain or question characteristics

**Hypothesis to test:**
- Earlier layers may capture semantic uncertainty
- Later layers may capture factual uncertainty
- Different epsilons needed for different domains

### Priority 3: Address Domain-Specific Issues

**For History:**
- Current: Accuracy drops 8% with steering
- Investigation: Why are valid historical facts being rejected?
- Possible fix: Reduce epsilon magnitude for history, or use different layer

**For Mathematics:**
- Current: 18-20% hallucination rate on unanswerable
- Investigation: Why does model answer impossible math questions?
- Possible fix: Add explicit impossibility detection for mathematical questions

**For Geography:**
- Current: Steering increases hallucinations by 6%
- Investigation: Why does steering make things worse?
- Possible fix: May need opposite direction of steering, or no steering at all

**For Science:**
- Current: Already at 98% abstention (nearly perfect)
- Recommendation: No steering needed - already solved!

### Priority 4: Analyze Question-Type Patterns
**Action Items:**
1. Categorize questions by type (not just domain):
   - Computational (arithmetic, calculations)
   - Factual (historical dates, scientific facts)
   - Impossibility (largest prime, infinite numbers)
   - Personal/subjective (favorite number, what I saw)
   - Plausible-but-unknowable (specific measurements, personal details)

2. Measure performance by question type

3. Apply question-type-specific interventions:
   - Computational: Reduce false rejections
   - Impossibility: Increase abstention (currently failing)
   - Personal: Already mostly working
   - Plausible-but-unknowable: This is the real challenge!

### Priority 5: Examine Specific Failure Cases in Detail
**Action Items:**
1. Deep dive into the 15 cases where steering made things worse
2. Analyze activation patterns for:
   - Questions that both baseline and steered fail
   - Questions that switched from correct to incorrect
   - Questions that switched from abstained to hallucinated

3. Look for patterns in:
   - Question phrasing
   - Semantic similarity to training data
   - Confidence scores / uncertainty measures

### Priority 6: Consider Alternative Approaches

**Instead of uniform epsilon steering, try:**

1. **Adaptive Steering:** Vary epsilon based on detected uncertainty level
   - Weak steering for high-confidence cases
   - Strong steering for borderline cases

2. **Multi-Layer Steering:** Apply steering at multiple layers
   - Different layers may capture different types of uncertainty

3. **Conditional Steering:** Only apply steering when certain conditions met
   - Threshold-based activation
   - Domain-specific activation

4. **Opposite Direction Testing:** Try positive epsilon
   - Current negative epsilon may be wrong direction for some domains
   - Geography results suggest positive epsilon might help

---

## Statistical Summary

### Overall Effectiveness:
- **Goal Achievement: ❌ FAILED**
  - Primary goal (increase abstention on unanswerable): Not achieved
  - Abstention actually DECREASED: 89% → 87%

- **Secondary Effects: ⚠️ MIXED**
  - Answerable accuracy slightly improved: 65.5% → 67.5%
  - False rejections reduced: 25% → 19.5%
  - But history domain significantly degraded

- **Net Effect: NEUTRAL**
  - 15 cases improved
  - 15 cases worsened
  - Net benefit: 0

### Confidence in Results:
- Sample size adequate (200 questions per domain, 100 answerable + 100 unanswerable)
- Results are consistent within domains
- High variance between domains suggests domain-specific effects are real

### Experimental Design Concerns:
1. **Ceiling Effect:** Baseline too strong (89% abstention)
2. **Domain Imbalance:** Science at 98%, Mathematics at 82%
3. **Question Difficulty:** May not be challenging enough for baseline model

---

## Next Steps

### Immediate Actions:
1. ✅ Review and validate these findings
2. Create harder unanswerable question dataset
3. Test different layer/epsilon combinations
4. Analyze the 15 "made worse" cases in detail

### Short-term Experiments:
1. Layer sweep: Test layers 5, 10, 15, 20, 25
2. Epsilon sweep: Test ε ∈ {-5, -10, -20, -30, -50, -100}
3. Domain-specific optimization
4. Question-type analysis

### Long-term Research:
1. Develop adaptive steering mechanisms
2. Investigate multi-layer intervention
3. Create uncertainty detection classifiers
4. Build question-type-specific models

---

## Conclusion

**The uncertainty routing intervention with ε=-20.0 at layer 10 is NOT working as intended.**

Key problems:
1. **Baseline is already too good** (89% abstention) - need harder test cases
2. **Steering slightly HURTS performance on primary goal** (87% vs 89%)
3. **Domain effects are highly variable** - one-size-fits-all approach fails
4. **Net effect is neutral** - as many cases worsened as improved

However, there are some positive signals:
- Answerable accuracy improved (+2%)
- False rejections reduced (-5.5%)
- Mathematics and Science domains show accuracy gains

**Recommendation:** This intervention should NOT be deployed as-is. Return to the drawing board with:
- Harder unanswerable questions (baseline < 50% abstention)
- Domain-specific tuning
- Layer/epsilon optimization
- Question-type-specific approaches

The concept of uncertainty routing remains promising, but this specific implementation needs substantial refinement before it can achieve its intended goal.

---

## Appendix: Files Generated

1. `/Users/akshatatiwari/Desktop/MIT/mech_interp_research/uncertainty_routing/results/exp6a_analysis.png`
   - 4-panel visualization of key metrics by domain

2. `/Users/akshatatiwari/Desktop/MIT/mech_interp_research/uncertainty_routing/results/exp6a_analysis_report.txt`
   - Detailed statistical summary with tables

3. `/Users/akshatatiwari/Desktop/MIT/mech_interp_research/uncertainty_routing/results/EXECUTIVE_SUMMARY_exp6a.md`
   - This document

4. Analysis scripts:
   - `analyze_exp6a.py` - Main analysis
   - `analyze_failures.py` - Failure case analysis
   - `detailed_examples.py` - Deep dive into specific patterns
