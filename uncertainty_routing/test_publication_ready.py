"""
Quick test of publication-ready experiment 6
Tests with just 2 questions per domain to verify it works
"""

import torch
from core_utils import ModelWrapper, ExperimentConfig
from experiment6_publication_ready import Experiment6PublicationReady

print("="*70)
print("QUICK TEST: Publication-Ready Experiment 6")
print("="*70)
print("\nThis tests with just 2 questions per domain (not full n=50)")
print("to verify everything works before the full run.\n")

# Load model
config = ExperimentConfig()
model = ModelWrapper(config)

# Load steering vectors
possible_files = [
    "steering_vectors_fixed.pt",
    "steering_vectors_explicit.pt",
    "steering_vectors.pt",
]

steering_vectors = None
for filename in possible_files:
    filepath = config.results_dir / filename
    if filepath.exists():
        steering_vectors = torch.load(filepath)
        print(f"✓ Loaded steering vectors from {filename}")
        print(f"  Layers: {len(steering_vectors)}\n")
        break

if steering_vectors is None:
    print("✗ No steering vectors found!")
    exit(1)

# Create experiment instance
exp6 = Experiment6PublicationReady(model, config, steering_vectors)

# Test with just a few questions
print("Testing with 2 answerable + 2 unanswerable from mathematics domain...")
print()

from scaled_datasets import create_scaled_domain_questions

domains = create_scaled_domain_questions()
math_questions = domains["mathematics"]

# Take first 2 of each type
test_questions = math_questions["answerable"][:2] + [
    {**q, 'is_unanswerable': True} for q in math_questions["unanswerable"][:2]
]

print(f"Testing {len(test_questions)} questions...")
results = []

for q in test_questions:
    # Test baseline and steered
    for condition, eps in [("baseline", 0.0), ("steered", -2.0)]:
        result = exp6._test_single_question(q, layer_idx=26, epsilon=eps)
        result["condition"] = condition
        result["domain"] = "mathematics"
        results.append(result)

        print(f"  {condition:8} | {q['q'][:40]:40} | {result['extracted_answer'][:20]:20}")

print("\n" + "="*70)
print("TEST RESULTS")
print("="*70)

import pandas as pd
df = pd.DataFrame(results)

print("\nAbstention rates:")
for condition in ["baseline", "steered"]:
    cond_df = df[df["condition"] == condition]
    abstention_rate = cond_df["abstained"].mean()
    print(f"  {condition}: {abstention_rate:.1%}")

print("\nBy answerability:")
for label in [False, True]:
    label_str = "Unanswerable" if label else "Answerable"
    label_df = df[df["is_unanswerable"] == label]
    print(f"\n  {label_str}:")
    for condition in ["baseline", "steered"]:
        cond_df = label_df[label_df["condition"] == condition]
        if len(cond_df) > 0:
            abstention_rate = cond_df["abstained"].mean()
            print(f"    {condition}: {abstention_rate:.1%} abstention")

print("\n✅ TEST COMPLETE!")
print("\nIf this looks correct, run the full experiment:")
print("  python experiment6_publication_ready.py")
print("\nOr use your pipeline script:")
print("  ./run_segment6_revalidate.sh")
