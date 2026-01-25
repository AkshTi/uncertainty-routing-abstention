"""
Quick test to verify steering vector direction is correct.

This script tests a few questions with different epsilons to check:
1. Does negative epsilon INCREASE abstention? (correct behavior)
2. Does positive epsilon DECREASE abstention? (correct behavior)

If backwards, the vectors are inverted.
"""

import torch
from core_utils import ModelWrapper, ExperimentConfig
from unified_prompts import unified_prompt_strict
from parsing_fixed import is_abstention

def test_vector_direction():
    """Test if steering vectors work in the correct direction"""

    print("="*70)
    print("STEERING VECTOR DIRECTION TEST")
    print("="*70)

    config = ExperimentConfig()
    model = ModelWrapper(config)

    # Test with both vector sets
    vector_files = [
        ("ORIGINAL", "results/steering_vectors.pt"),
        ("CALIBRATED", "results/steering_vectors_calibrated.pt"),
    ]

    # Simple test questions
    test_questions = [
        {"q": "What is 2+2?", "answerable": True},
        {"q": "What is 10*5?", "answerable": True},
        {"q": "What color is my shirt?", "answerable": False},
        {"q": "What mountain can I see from here?", "answerable": False},
    ]

    for vec_name, vec_path in vector_files:
        print(f"\n{'='*70}")
        print(f"Testing: {vec_name} ({vec_path})")
        print(f"{'='*70}")

        try:
            vectors = torch.load(vec_path)
            print(f"✓ Loaded {len(vectors)} layers")
        except:
            print(f"✗ Could not load {vec_path}")
            continue

        layer = 10 if 10 in vectors else list(vectors.keys())[0]
        print(f"Using layer: {layer}")

        # Test different epsilons
        epsilons = [-20.0, 0.0, 20.0]
        results = {eps: {"abstain_count": 0, "answer_count": 0} for eps in epsilons}

        for eps in epsilons:
            print(f"\n  Testing epsilon={eps}:")

            for q in test_questions:
                prompt = unified_prompt_strict(q["q"])

                # Clear hooks
                model.clear_hooks()

                # Apply steering if needed
                if eps != 0.0:
                    model.register_steering_hook(layer, -1, vectors[layer], eps)

                # Generate
                response = model.generate(
                    prompt,
                    max_new_tokens=12,
                    temperature=0.0,
                    do_sample=False
                )

                # Check if abstained
                abstained = is_abstention(response)

                if abstained:
                    results[eps]["abstain_count"] += 1
                else:
                    results[eps]["answer_count"] += 1

                status = "ABSTAIN" if abstained else "ANSWER"
                print(f"    {q['q'][:40]:40s} → {status:8s} ({response[:30]}...)")

                # Clear hooks
                model.clear_hooks()

        # Analyze results
        print(f"\n  RESULTS:")
        for eps in epsilons:
            total = results[eps]["abstain_count"] + results[eps]["answer_count"]
            abstain_rate = results[eps]["abstain_count"] / total * 100
            print(f"    eps={eps:5.1f}: {abstain_rate:5.1f}% abstention ({results[eps]['abstain_count']}/{total})")

        # Check if direction is correct
        baseline_rate = results[0.0]["abstain_count"] / len(test_questions)
        neg_rate = results[-20.0]["abstain_count"] / len(test_questions)
        pos_rate = results[20.0]["abstain_count"] / len(test_questions)

        print(f"\n  DIRECTION CHECK:")
        if neg_rate > baseline_rate:
            print(f"    ✓ Negative epsilon INCREASES abstention ({neg_rate:.1%} > {baseline_rate:.1%})")
        else:
            print(f"    ✗ Negative epsilon DECREASES abstention ({neg_rate:.1%} < {baseline_rate:.1%}) - WRONG!")

        if pos_rate < baseline_rate:
            print(f"    ✓ Positive epsilon DECREASES abstention ({pos_rate:.1%} < {baseline_rate:.1%})")
        else:
            print(f"    ✗ Positive epsilon INCREASES abstention ({pos_rate:.1%} > {baseline_rate:.1%}) - WRONG!")

        if neg_rate > baseline_rate and pos_rate < baseline_rate:
            print(f"\n  ✅ VERDICT: {vec_name} vectors work CORRECTLY")
        else:
            print(f"\n  ❌ VERDICT: {vec_name} vectors are INVERTED or BROKEN")

if __name__ == "__main__":
    test_vector_direction()
