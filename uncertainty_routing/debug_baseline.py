"""
Debug: Check what baseline (ε=0) actually produces
"""

import torch
from core_utils import ModelWrapper, ExperimentConfig
from experiment6_publication_ready import Experiment6PublicationReady
from scaled_datasets import create_scaled_domain_questions

config = ExperimentConfig()
model = ModelWrapper(config)
steering_vectors = torch.load(config.results_dir / "steering_vectors.pt")

exp6 = Experiment6PublicationReady(model, config, steering_vectors)

# Get test questions
domains = create_scaled_domain_questions()
math = domains["mathematics"]

print("="*70)
print("BASELINE TEST (ε=0, no steering)")
print("="*70)
print()

# Test 5 unanswerable questions with NO steering
print("Testing 5 unanswerable questions with ε=0...")
print()

for i, q in enumerate(math["unanswerable"][:5]):
    q_copy = q.copy()
    q_copy['is_unanswerable'] = True

    result = exp6._test_single_question(q_copy, 10, 0.0)  # ε=0 = no steering

    print(f"{i+1}. {q['q'][:60]}")
    print(f"   Response: \"{result['response']}\"")
    print(f"   Abstained: {result['abstained']}")
    print()

print("="*70)
print("If abstained=False for most, baseline abstention should be low (~0-20%)")
print("If abstained=True for most, baseline abstention would be high (~60-100%)")
