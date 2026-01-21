#!/usr/bin/env python3
"""
Fix Steering Direction

This script diagnoses and fixes inverted steering vectors.

Problem: Negative epsilon is supposed to reduce hallucinations but is increasing them.
Solution: Test steering direction and flip if needed.

Usage:
    python fix_steering_direction.py
"""

import torch
import json
from pathlib import Path
from core_utils import ModelWrapper, ExperimentConfig
from data_preparation import create_clearly_answerable_questions, create_clearly_unanswerable_questions

print("\n" + "="*80)
print(" STEERING DIRECTION DIAGNOSTIC & FIX")
print("="*80 + "\n")

# Load config
config = ExperimentConfig()

# Check which steering vectors exist
results_dir = Path("results")
vectors_regular = results_dir / "steering_vectors.pt"
vectors_explicit = results_dir / "steering_vectors_explicit.pt"

if vectors_explicit.exists():
    print(f"✓ Found steering_vectors_explicit.pt")
    vectors_path = vectors_explicit
    output_path = vectors_explicit
elif vectors_regular.exists():
    print(f"✓ Found steering_vectors.pt")
    vectors_path = vectors_regular
    output_path = results_dir / "steering_vectors_explicit.pt"
else:
    print("✗ No steering vectors found!")
    print("Run Segment 1 first to create steering vectors")
    exit(1)

# Load steering vectors
print(f"\nLoading steering vectors from {vectors_path}...")
steering_vectors = torch.load(vectors_path)
print(f"Layers: {list(steering_vectors.keys())}")

# Convert keys to int if needed
steering_vectors = {int(k): v.to(config.device) for k, v in steering_vectors.items()}

# Pick the best layer (highest index or from exp2 results)
best_layer = max(steering_vectors.keys())
print(f"Using layer {best_layer} for testing")

# Load model
print(f"\nLoading model {config.model_name}...")
model = ModelWrapper(config)
print("✓ Model loaded")

# Load test questions
answerable = create_clearly_answerable_questions()[:5]
unanswerable = create_clearly_unanswerable_questions()[:5]

print(f"\nTest set: {len(answerable)} answerable, {len(unanswerable)} unanswerable")

# Test steering direction
print("\n" + "="*80)
print(" TESTING STEERING DIRECTION")
print("="*80 + "\n")

from experiment5_trustworthiness import Experiment5

exp5 = Experiment5(model, config, steering_vectors)

# Test a few epsilons
test_epsilons = [-30.0, 0.0, +30.0]
results = []

for eps in test_epsilons:
    print(f"\nTesting epsilon = {eps}...")

    # Test on unanswerable (should abstain with negative epsilon)
    abstentions = 0
    hallucinations = 0

    for q in unanswerable[:3]:
        response = exp5.model.generate(
            q['question'],
            temperature=0.0,
            do_sample=False,
            max_new_tokens=50,
            steering_layer=best_layer if eps != 0 else None,
            steering_epsilon=eps if eps != 0 else None,
            steering_vector=steering_vectors[best_layer] if eps != 0 else None
        )

        is_abstain = exp5._detect_abstention(response)
        abstentions += int(is_abstain)
        if not is_abstain:
            hallucinations += 1

        print(f"  Q: {q['question'][:50]}...")
        print(f"  A: {response[:80]}...")
        print(f"  Abstained: {is_abstain}")

    abstain_rate = abstentions / 3.0
    halluc_rate = hallucinations / 3.0

    results.append({
        'epsilon': eps,
        'abstain_rate': abstain_rate,
        'halluc_rate': halluc_rate
    })

    print(f"\n  Summary: {abstain_rate*100:.0f}% abstention, {halluc_rate*100:.0f}% hallucination")

print("\n" + "="*80)
print(" DIAGNOSIS")
print("="*80 + "\n")

# Analyze results
baseline = [r for r in results if r['epsilon'] == 0.0][0]
negative = [r for r in results if r['epsilon'] == -30.0][0]
positive = [r for r in results if r['epsilon'] == +30.0][0]

print(f"Baseline (ε=0):   {baseline['abstain_rate']*100:.0f}% abstention, {baseline['halluc_rate']*100:.0f}% hallucination")
print(f"Negative (ε=-30): {negative['abstain_rate']*100:.0f}% abstention, {negative['halluc_rate']*100:.0f}% hallucination")
print(f"Positive (ε=+30): {positive['abstain_rate']*100:.0f}% abstention, {positive['halluc_rate']*100:.0f}% hallucination")

# Expected behavior:
# - Negative epsilon should INCREASE abstention (reduce hallucination)
# - Positive epsilon should DECREASE abstention (increase hallucination on unanswerables)

needs_flip = False

if negative['abstain_rate'] < baseline['abstain_rate']:
    print("\n⚠️  PROBLEM DETECTED: Negative epsilon DECREASES abstention (should increase)")
    print("    This means steering direction is INVERTED")
    needs_flip = True
elif positive['abstain_rate'] > baseline['abstain_rate']:
    print("\n⚠️  PROBLEM DETECTED: Positive epsilon INCREASES abstention (should decrease)")
    print("    This means steering direction is INVERTED")
    needs_flip = True
else:
    print("\n✓ Steering direction appears CORRECT")
    print("  - Negative epsilon increases abstention")
    print("  - Positive epsilon decreases abstention")

# Apply fix if needed
if needs_flip:
    print("\n" + "="*80)
    print(" APPLYING FIX")
    print("="*80 + "\n")

    print("Flipping steering vector signs...")
    fixed_vectors = {k: -v for k, v in steering_vectors.items()}

    # Save fixed vectors
    torch.save(fixed_vectors, output_path)
    print(f"✓ Saved corrected steering vectors to {output_path}")

    # Also update steering_vectors.pt if that's what we loaded
    if vectors_path == vectors_regular:
        torch.save(fixed_vectors, vectors_regular)
        print(f"✓ Also updated {vectors_regular}")

    print("\n" + "="*80)
    print(" VERIFICATION")
    print("="*80 + "\n")

    # Re-test with flipped vectors
    print("Re-testing with flipped vectors...")

    exp5_fixed = Experiment5(model, config, fixed_vectors)

    for eps in [-30.0, 0.0, +30.0]:
        abstentions = 0
        for q in unanswerable[:3]:
            response = exp5_fixed.model.generate(
                q['question'],
                temperature=0.0,
                do_sample=False,
                max_new_tokens=50,
                steering_layer=best_layer if eps != 0 else None,
                steering_epsilon=eps if eps != 0 else None,
                steering_vector=fixed_vectors[best_layer] if eps != 0 else None
            )
            is_abstain = exp5_fixed._detect_abstention(response)
            abstentions += int(is_abstain)

        print(f"  ε={eps:+.0f}: {abstentions}/3 abstentions ({abstentions/3.0*100:.0f}%)")

    print("\n✅ FIX APPLIED!")
    print("\nNext steps:")
    print("  1. Re-run Segment 2: ./run_segment2.sh")
    print("  2. Re-run Segment 4A: ./run_segment4a.sh")
    print("  3. Re-run Segment 4B: ./run_segment4b.sh")

else:
    print("\nNo fix needed - steering vectors are already correct.")

    # Still save as _explicit if needed
    if not vectors_explicit.exists():
        print(f"\nSaving steering_vectors_explicit.pt for Exp6-9...")
        torch.save(steering_vectors, output_path)
        print(f"✓ Saved to {output_path}")

    print("\nPossible issues:")
    print("  1. The test set is too small (only 3 questions)")
    print("  2. Epsilon magnitude is too small/large")
    print("  3. The model isn't responding to steering")
    print("\nCheck your Exp5 results manually to see which epsilon works best.")

print("\n" + "="*80)
print(" DONE")
print("="*80 + "\n")
