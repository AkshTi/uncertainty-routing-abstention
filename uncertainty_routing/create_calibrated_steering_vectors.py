"""
Create Calibrated Steering Vectors - Epistemic State (not confidence tone)

KEY DIFFERENCE:
- OLD: "I'm confident" vs "I'm uncertain" (tone/disposition)
- NEW: Questions model KNOWS vs questions model DOESN'T KNOW (epistemic state)

Strategy:
1. Find questions model answers correctly → extract activations (KNOWS)
2. Find questions model hallucinates on → extract activations (DOESN'T KNOW)
3. Vector = KNOWS - DOESN'T_KNOW
4. Negative epsilon → push toward "doesn't know" state → abstain more
"""

import torch
import json
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

from core_utils import ModelWrapper, ExperimentConfig
from scaled_datasets import create_scaled_domain_questions
from unified_prompts import unified_prompt_minimal  # Simple prompt, no instructions


def get_questions_by_knowledge_state(
    model: ModelWrapper,
    n_per_type: int = 20
) -> tuple[List[Dict], List[Dict]]:
    """
    Find questions where model demonstrates knowledge vs lack of knowledge.

    Returns:
        (knows_questions, doesnt_know_questions)
    """
    print("\n" + "="*70)
    print("STEP 1: Finding questions by knowledge state")
    print("="*70)

    domains = create_scaled_domain_questions()

    knows = []  # Model gets these right
    doesnt_know = []  # Model hallucinates on these

    # Test answerable questions to find what model knows
    print("\nTesting answerable questions...")
    for domain_name, data in tqdm(domains.items()):
        for q in data["answerable"][:30]:  # Test more than we need
            # Use minimal prompt - no confidence instructions
            prompt = unified_prompt_minimal(q['q'])
            response = model.generate(prompt, max_new_tokens=20, temperature=0.0, do_sample=False)

            # Check if model got it right
            expected = q.get('a', '').lower()
            if expected and expected in response.lower():
                knows.append({
                    "question": q['q'],
                    "answer": q['a'],
                    "domain": domain_name,
                    "response": response
                })

                if len(knows) >= n_per_type:
                    break

        if len(knows) >= n_per_type:
            break

    # Test unanswerable questions to find what model doesn't know
    print("\nTesting unanswerable questions...")
    for domain_name, data in tqdm(domains.items()):
        for q in data["unanswerable"][:30]:
            prompt = unified_prompt_minimal(q['q'])
            response = model.generate(prompt, max_new_tokens=20, temperature=0.0, do_sample=False)

            # If model gives a confident answer (not UNCERTAIN), it's hallucinating
            response_lower = response.lower()
            if "uncertain" not in response_lower and len(response.strip()) > 0:
                doesnt_know.append({
                    "question": q['q'],
                    "domain": domain_name,
                    "response": response,
                    "hallucinated": True
                })

                if len(doesnt_know) >= n_per_type:
                    break

        if len(doesnt_know) >= n_per_type:
            break

    print(f"\n✓ Found {len(knows)} questions model KNOWS")
    print(f"✓ Found {len(doesnt_know)} questions model DOESN'T KNOW (hallucinates)")

    return knows[:n_per_type], doesnt_know[:n_per_type]


def compute_calibrated_steering_vectors(
    model: ModelWrapper,
    knows_questions: List[Dict],
    doesnt_know_questions: List[Dict],
    layers: List[int]
) -> Dict[int, torch.Tensor]:
    """
    Compute steering vectors from epistemic state contrast.

    Args:
        knows_questions: Questions model answers correctly
        doesnt_know_questions: Questions model hallucinates on
        layers: Which layers to compute vectors for

    Returns:
        Dict mapping layer_idx to steering vector
    """
    print("\n" + "="*70)
    print("STEP 2: Computing steering vectors")
    print("="*70)

    print(f"\nLayers to process: {layers}")
    print(f"KNOWS examples: {len(knows_questions)}")
    print(f"DOESN'T KNOW examples: {len(doesnt_know_questions)}")
    print()

    steering_vectors = {}

    for layer_idx in layers:
        print(f"\nProcessing layer {layer_idx}...")

        # Get activations for KNOWS state
        knows_acts = []
        print("  Extracting KNOWS activations...")
        for q in tqdm(knows_questions, leave=False):
            prompt = unified_prompt_minimal(q['question'])
            acts = model.get_layer_activations(prompt, layer_idx)

            # Use last token activation
            if acts.ndim == 3:
                acts = acts[0, -1, :]
            elif acts.ndim == 2:
                acts = acts[-1, :]

            knows_acts.append(acts.cpu())

        # Get activations for DOESN'T KNOW state
        doesnt_know_acts = []
        print("  Extracting DOESN'T KNOW activations...")
        for q in tqdm(doesnt_know_questions, leave=False):
            prompt = unified_prompt_minimal(q['question'])
            acts = model.get_layer_activations(prompt, layer_idx)

            if acts.ndim == 3:
                acts = acts[0, -1, :]
            elif acts.ndim == 2:
                acts = acts[-1, :]

            doesnt_know_acts.append(acts.cpu())

        # Compute mean difference
        knows_mean = torch.stack(knows_acts).mean(dim=0)
        doesnt_know_mean = torch.stack(doesnt_know_acts).mean(dim=0)

        # Steering vector = KNOWS - DOESN'T_KNOW
        # So:
        # - Positive epsilon → push toward KNOWS (more confident, answer more)
        # - Negative epsilon → push toward DOESN'T_KNOW (less confident, abstain more)
        steering_vec = knows_mean - doesnt_know_mean

        # Normalize to unit norm
        steering_vec = steering_vec / (steering_vec.norm() + 1e-8)

        steering_vectors[layer_idx] = steering_vec

        print(f"  ✓ Vector norm (before norm): {(knows_mean - doesnt_know_mean).norm():.3f}")
        print(f"  ✓ Vector norm (after norm): {steering_vec.norm():.3f}")

    return steering_vectors


def test_steering_vectors(
    model: ModelWrapper,
    steering_vectors: Dict[int, torch.Tensor],
    test_layer: int = 10
):
    """Quick sanity check of the vectors"""
    print("\n" + "="*70)
    print("STEP 3: Testing steering vectors")
    print("="*70)

    # Get a few test questions
    domains = create_scaled_domain_questions()
    math = domains["mathematics"]

    print(f"\nTesting layer {test_layer} with different epsilon values...")
    print()

    # Test on one unanswerable question
    q = math["unanswerable"][0]
    prompt = unified_prompt_minimal(q['q'])

    print(f"Test question: {q['q']}")
    print()

    for epsilon in [0.0, -10.0, -20.0, +10.0]:
        model.clear_hooks()

        if epsilon != 0.0:
            model.register_steering_hook(
                test_layer,
                -1,
                steering_vectors[test_layer],
                epsilon
            )

        response = model.generate(prompt, max_new_tokens=20, temperature=0.0, do_sample=False)
        model.clear_hooks()

        abstained = "uncertain" in response.lower()

        print(f"ε={epsilon:+6.1f}: \"{response[:50]}...\" (abstained={abstained})")

    print()
    print("Expected behavior:")
    print("  ε=0:    Baseline (some abstention)")
    print("  ε=-20:  MORE abstention (pushed toward doesn't-know state)")
    print("  ε=+20:  LESS abstention (pushed toward knows state)")


def main():
    print("="*70)
    print("CREATING CALIBRATED STEERING VECTORS")
    print("="*70)
    print()
    print("These vectors contrast:")
    print("  KNOWS: Questions model answers correctly")
    print("  DOESN'T KNOW: Questions model hallucinates on")
    print()
    print("This captures epistemic state, not confidence tone.")
    print()

    # Load model
    config = ExperimentConfig()
    model = ModelWrapper(config)

    # Find questions by knowledge state
    knows, doesnt_know = get_questions_by_knowledge_state(model, n_per_type=15)

    # Save examples for reference
    examples = {
        "knows": knows,
        "doesnt_know": doesnt_know
    }

    examples_path = config.results_dir / "steering_examples_calibrated.json"
    with open(examples_path, 'w') as f:
        json.dump(examples, f, indent=2)
    print(f"\n✓ Saved examples to {examples_path}")

    # Compute steering vectors
    layers = [10, 16, 17, 18]  # Same as before
    steering_vectors = compute_calibrated_steering_vectors(
        model, knows, doesnt_know, layers
    )

    # Save vectors
    output_path = config.results_dir / "steering_vectors_calibrated.pt"
    torch.save(steering_vectors, output_path)
    print(f"\n✓ Saved steering vectors to {output_path}")

    # Test the vectors
    test_steering_vectors(model, steering_vectors, test_layer=10)

    print("\n" + "="*70)
    print("DONE! Next steps:")
    print("="*70)
    print()
    print("1. Update experiment6_publication_ready.py:")
    print("   possible_files = [")
    print("       \"steering_vectors_calibrated.pt\",  # NEW!")
    print("       \"steering_vectors.pt\",")
    print("       ...")
    print("   ]")
    print()
    print("2. Re-run with NEGATIVE epsilon:")
    print("   optimal_epsilon = -20.0  # Should now INCREASE abstention")
    print()
    print("3. Expected results:")
    print("   Baseline (unanswerable): ~89% abstention")
    print("   Steered (ε=-20): ~95-100% abstention (Δ +6-11%)")
    print()


if __name__ == "__main__":
    main()
