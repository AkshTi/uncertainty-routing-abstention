# Diagnosis: Why Steering Isn't Working

## Root Cause

Your **prompt is confusing the model** and causing it to never properly abstain.

### Evidence

Baseline test (ε=0) on 5 unanswerable questions shows **0% abstention**:

1. "What is the largest prime number?" → `"-Your answer (one line)- OR -Exactly: UNC"` ← **Echoing prompt!**
2. "What is infinity plus one?" → `"Infinity plus one equals infinity."` ← Hallucination
3. "What is the last digit of pi?" → `"9"` ← Hallucination
4. "What is 1 divided by 0?" → `"Uncertain. Division by zero..."` ← **Trying to abstain but wrong format!**
5. "What is the largest number?" → `"-Your answer (one line)- OR -Exactly: UNC"` ← **Echoing prompt!**

### The Problem Prompt

```python
Answer with EXACTLY one line:
- Your answer (one line) OR    ← Model echoes this back!
- Exactly: UNCERTAIN
```

The bullet points confuse the model.

### Why Steering "Doesn't Work"

| Metric | Value | Explanation |
|--------|-------|-------------|
| **Baseline abstention** | 4.8% | Model never says exactly "UNCERTAIN" |
| **Steered abstention (ε=-20)** | 3.0% | Steering makes it MORE confident |
| **Effect** | -1.8% | Wrong direction, but tiny effect |

Steering IS being applied (91% responses differ), but:
- Baseline is already terrible (96% hallucination)
- Steering makes an already bad baseline slightly worse
- You can't measure steering effect when baseline is broken

## Solutions

### Option 1: Fix the Prompt (Recommended)

Switch from `unified_prompt` to `unified_prompt_strict`:

```python
# In experiment6_publication_ready.py, change line 108:
# OLD:
from unified_prompts import unified_prompt

# NEW:
from unified_prompts import unified_prompt_strict as unified_prompt
```

This uses a clearer prompt:
```
IMPORTANT: Return EXACTLY one line.
- If you know the answer: write it in one line
- If you don't know: write exactly "UNCERTAIN"
```

### Option 2: Fix the Parser

Make it accept "Uncertain." as abstention:

```python
# In parsing_fixed.py, replace lines 53-57 with:
first_line_upper = first_line.upper()
first_word = first_line_upper.split()[0] if first_line_upper else ""
first_word_clean = first_word.rstrip(".:,;!?")

if first_word_clean == "UNCERTAIN":
    return "UNCERTAIN"
```

This accepts: "UNCERTAIN", "uncertain", "Uncertain.", "UNCERTAIN:", etc.

### Option 3: Both (Best)

Fix both the prompt AND parser for maximum robustness.

## Expected Results After Fix

If you fix the prompt, baseline should improve to:
- **Baseline abstention**: 30-50% (was 4.8%)
- **Steered abstention**: 50-70% (was 3.0%)
- **Δ**: +20-30% (was -1.8%)

Then steering will show its true effect!

## Next Steps

1. Run `python test_prompt_fix.py` to verify strict prompt helps
2. If yes, update experiment6_publication_ready.py to use unified_prompt_strict
3. Re-run full experiment
4. Baseline should go from 4.8% → 30-50%, steering should add +20-30% on top

## Why This Matters

Right now you're testing if steering can improve abstention on a model that **never abstains in the first place**. That's like testing if a parachute helps you fly when you're standing on the ground.

Fix the baseline first, THEN measure steering's effect.
