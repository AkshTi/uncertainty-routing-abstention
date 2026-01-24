"""
Test if using unified_prompt_strict fixes the baseline abstention
"""

import torch
from core_utils import ModelWrapper, ExperimentConfig
from unified_prompts import unified_prompt, unified_prompt_strict
from parsing_fixed import is_abstention
from scaled_datasets import create_scaled_domain_questions

config = ExperimentConfig()
model = ModelWrapper(config)

domains = create_scaled_domain_questions()
math = domains["mathematics"]

print("="*70)
print("TESTING PROMPT VARIANTS")
print("="*70)
print()

# Test 3 unanswerable questions with both prompts
unanswerable = math["unanswerable"][:3]

for i, q in enumerate(unanswerable, 1):
    print(f"\n{i}. {q['q']}")
    print("-" * 70)

    # Original prompt
    prompt_old = unified_prompt(q['q'])
    response_old = model.generate(prompt_old, max_new_tokens=12, temperature=0.0, do_sample=False)
    abstained_old = is_abstention(response_old)

    # Strict prompt
    prompt_strict = unified_prompt_strict(q['q'])
    response_strict = model.generate(prompt_strict, max_new_tokens=12, temperature=0.0, do_sample=False)
    abstained_strict = is_abstention(response_strict)

    print(f"ORIGINAL PROMPT:")
    print(f'  Response: "{response_old}"')
    print(f'  Abstained: {abstained_old}')
    print()
    print(f"STRICT PROMPT:")
    print(f'  Response: "{response_strict}"')
    print(f'  Abstained: {abstained_strict}')

print("\n" + "="*70)
print("If STRICT shows more abstentions, switch to unified_prompt_strict!")
print("="*70)
