"""
Quick Fix & Test - Experiment 6 Steering Issues

This script:
1. Tests all available layers to find which works best
2. Tests different epsilon values
3. Shows you which combination gives best results
4. Then you can run full experiment with correct parameters

Usage:
    python fix_and_test_steering.py
"""

import torch
import pandas as pd
from core_utils import ModelWrapper, ExperimentConfig
from experiment6_publication_ready import Experiment6PublicationReady
from scaled_datasets import create_scaled_domain_questions

print("="*70)
print("FIXING & TESTING STEERING")
print("="*70)
print("\nThis will test different layers and epsilon values")
print("to find what works best.\n")

# Load model
config = ExperimentConfig()
model = ModelWrapper(config)

# Load steering vectors
print("Loading steering vectors...")
steering_vectors = torch.load(config.results_dir / "steering_vectors.pt")
available_layers = list(steering_vectors.keys())
print(f"✓ Available layers: {available_layers}")
print()

# Create experiment
exp6 = Experiment6PublicationReady(model, config, steering_vectors)

# Get small test set (just 5 questions per type for speed)
print("Creating small test set (5 answerable + 5 unanswerable)...")
domains = create_scaled_domain_questions()
test_questions = []

# Just use mathematics for quick test
math = domains["mathematics"]
for q in math["answerable"][:5]:
    test_questions.append({"q_data": q, "is_unanswerable": False})
for q in math["unanswerable"][:5]:
    test_questions.append({"q_data": q, "is_unanswerable": True})

print(f"✓ Test set: {len(test_questions)} questions")
print()

# Test function
def test_config(layer, epsilon):
    """Test a specific layer/epsilon configuration"""
    results = []

    for item in test_questions:
        q = item["q_data"]
        q['is_unanswerable'] = item['is_unanswerable']

        # Test baseline and steered
        for condition, eps in [("baseline", 0.0), ("steered", epsilon)]:
            result = exp6._test_single_question(q, layer, eps)
            result["condition"] = condition
            results.append(result)

    df = pd.DataFrame(results)

    # Calculate metrics
    baseline = df[df["condition"]=="baseline"]
    steered = df[df["condition"]=="steered"]

    baseline_abstain = baseline["abstained"].mean()
    steered_abstain = steered["abstained"].mean()
    delta = steered_abstain - baseline_abstain

    # Check on unanswerable only
    baseline_unans = baseline[baseline["is_unanswerable"]==True]["abstained"].mean()
    steered_unans = steered[steered["is_unanswerable"]==True]["abstained"].mean()
    delta_unans = steered_unans - baseline_unans

    return {
        "layer": layer,
        "epsilon": epsilon,
        "baseline_abstain": baseline_abstain,
        "steered_abstain": steered_abstain,
        "delta": delta,
        "delta_unanswerable": delta_unans,
        "df": df
    }

# Test all combinations
print("Testing different configurations...")
print("-"*70)

results = []

# Test each available layer
epsilons_to_test = [-2.0, -5.0, -10.0, -20.0]

for layer in available_layers:
    for epsilon in epsilons_to_test:
        print(f"\nTesting layer {layer}, epsilon {epsilon}...")

        result = test_config(layer, epsilon)
        results.append(result)

        print(f"  Overall: {result['baseline_abstain']:.1%} → {result['steered_abstain']:.1%} (Δ {result['delta']:+.1%})")
        print(f"  Unanswerable: Δ {result['delta_unanswerable']:+.1%}")

print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)

# Find best configuration
results.sort(key=lambda x: abs(x['delta_unanswerable']), reverse=True)

print("\nRanked by effect on unanswerable questions:\n")
for i, r in enumerate(results, 1):
    print(f"{i}. Layer {r['layer']:2d}, ε={r['epsilon']:6.1f}: Δ_unans={r['delta_unanswerable']:+6.1%}, Δ_all={r['delta']:+6.1%}")

print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)

best = results[0]
print(f"\n✅ Best configuration:")
print(f"   Layer: {best['layer']}")
print(f"   Epsilon: {best['epsilon']}")
print(f"   Effect on unanswerable: {best['delta_unanswerable']:+.1%}")
print(f"   Effect overall: {best['delta']:+.1%}")

if abs(best['delta']) < 0.05:  # Less than 5% effect
    print("\n⚠️  WARNING: Effect is still very weak!")
    print("   Possible issues:")
    print("   - Steering vectors may have wrong polarity (try positive epsilon)")
    print("   - Vectors may not be strong enough")
    print("   - Model may need different approach")
    print("\n   Try:")
    print("   1. Test with POSITIVE epsilon (flip direction)")
    print("   2. Check how vectors were created")
    print("   3. Try even larger epsilon (±50)")
else:
    print("\n✅ Steering is working!")
    print("\n   To run full experiment, update experiment6_publication_ready.py:")
    print(f"   df_6a, df_6b, df_6c = exp6.run_all(best_layer={best['layer']}, optimal_epsilon={best['epsilon']})")
    print("\n   Then run:")
    print("   ./run_segment6_revalidate.sh")

print("\n" + "="*70)

# Save test results
test_results_path = config.results_dir / "steering_test_results.csv"
pd.DataFrame(results).to_csv(test_results_path, index=False)
print(f"\n✓ Test results saved to {test_results_path}")
