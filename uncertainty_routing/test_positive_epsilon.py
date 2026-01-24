"""
Quick test: Try POSITIVE epsilon values
"""

import torch
import pandas as pd
from core_utils import ModelWrapper, ExperimentConfig
from experiment6_publication_ready import Experiment6PublicationReady
from scaled_datasets import create_scaled_domain_questions

print("="*70)
print("TESTING POSITIVE EPSILON (flipped direction)")
print("="*70)

# Load
config = ExperimentConfig()
model = ModelWrapper(config)
steering_vectors = torch.load(config.results_dir / "steering_vectors.pt")

exp6 = Experiment6PublicationReady(model, config, steering_vectors)

# Get test set (just mathematics for speed)
domains = create_scaled_domain_questions()
math = domains["mathematics"]

test_questions = []
for q in math["answerable"][:10]:
    test_questions.append({"q_data": q, "is_unanswerable": False})
for q in math["unanswerable"][:10]:
    q_copy = q.copy()
    q_copy['is_unanswerable'] = True
    test_questions.append({"q_data": q_copy, "is_unanswerable": True})

print(f"Test set: {len(test_questions)} questions (10 answerable + 10 unanswerable)")
print()

# Test positive epsilon values
epsilons_to_test = [+2.0, +5.0, +10.0, +20.0]
results = []

for epsilon in epsilons_to_test:
    print(f"\nTesting ε={epsilon:+.1f}...")

    for item in test_questions:
        q = item["q_data"]

        # Baseline
        result_b = exp6._test_single_question(q, 10, 0.0)
        result_b["condition"] = "baseline"
        result_b["test_epsilon"] = epsilon
        results.append(result_b)

        # Steered
        result_s = exp6._test_single_question(q, 10, epsilon)
        result_s["condition"] = "steered"
        result_s["test_epsilon"] = epsilon
        results.append(result_s)

    df = pd.DataFrame(results)
    df_eps = df[df['test_epsilon'] == epsilon]

    baseline = df_eps[df_eps['condition'] == 'baseline']
    steered = df_eps[df_eps['condition'] == 'steered']

    # Overall
    b_abs = baseline['abstained'].mean()
    s_abs = steered['abstained'].mean()
    delta = s_abs - b_abs

    # Unanswerable only
    b_unans = baseline[baseline['is_unanswerable'] == True]['abstained'].mean()
    s_unans = steered[steered['is_unanswerable'] == True]['abstained'].mean()
    delta_unans = s_unans - b_unans

    print(f"  Overall: {b_abs:.1%} → {s_abs:.1%} (Δ {delta:+.1%})")
    print(f"  Unanswerable: {b_unans:.1%} → {s_unans:.1%} (Δ {delta_unans:+.1%})")

print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)

# Compare all
for epsilon in epsilons_to_test:
    df_eps = pd.DataFrame(results)
    df_eps = df_eps[df_eps['test_epsilon'] == epsilon]

    baseline = df_eps[df_eps['condition'] == 'baseline']
    steered = df_eps[df_eps['condition'] == 'steered']

    b_unans = baseline[baseline['is_unanswerable'] == True]['abstained'].mean()
    s_unans = steered[steered['is_unanswerable'] == True]['abstained'].mean()
    delta_unans = s_unans - b_unans

    print(f"ε={epsilon:+6.1f}: Δ_unanswerable={delta_unans:+6.1%}")

print()
print("If you see POSITIVE Δ with positive epsilon, that's the right direction!")
