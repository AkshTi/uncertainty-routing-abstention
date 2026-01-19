"""
Diagnostic: Check and recompute steering vectors for trustworthiness

This script will:
1. Verify how steering vectors were computed
2. Recompute them with explicit answerable/unanswerable contrast
3. Test the new vectors on a few examples
"""

import json
import torch
from pathlib import Path
from typing import List, Dict
import numpy as np

from core_utils import ModelWrapper, ExperimentConfig


def compute_steering_vectors_explicit(
    model: ModelWrapper,
    answerable_questions: List[Dict],
    unanswerable_questions: List[Dict],
    layers: List[int],
    n_examples: int = 15,
) -> Dict[int, torch.Tensor]:
    """
    Compute steering vectors with EXPLICIT prompting for epistemic state.
    
    Key insight: We want to contrast:
    - POS: "I can answer this" (confident, answerable)
    - NEG: "I cannot answer this" (uncertain, should abstain)
    """
    
    print("\n" + "=" * 70)
    print("COMPUTING STEERING VECTORS")
    print("=" * 70)
    
    # Create explicit prompts that emphasize the epistemic difference
    pos_prompts = []
    for ex in answerable_questions[:n_examples]:
        q = ex["question"]
        a = ex.get("answer", "")
        # Prompt that leads to confident answering
        prompt = (
            f"Question: {q}\n\n"
            f"This question has a clear answer that I know. "
            f"I will provide the answer confidently.\n\n"
            f"Answer: {a}"
        )
        pos_prompts.append(prompt)
    
    neg_prompts = []
    for ex in unanswerable_questions[:n_examples]:
        q = ex["question"]
        # Prompt that leads to abstention
        prompt = (
            f"Question: {q}\n\n"
            f"This question cannot be answered with certainty. "
            f"I should abstain rather than guess.\n\n"
            f"Answer: UNCERTAIN"
        )
        neg_prompts.append(prompt)
    
    print(f"Positive (answerable) examples: {len(pos_prompts)}")
    print(f"Negative (unanswerable) examples: {len(neg_prompts)}")
    
    print("\nExample positive prompt:")
    print("-" * 50)
    print(pos_prompts[0][:200])
    print("\nExample negative prompt:")
    print("-" * 50)
    print(neg_prompts[0][:200])
    
    # Compute vectors
    steering_vectors = {}
    
    for layer_idx in layers:
        print(f"\nProcessing layer {layer_idx}...")
        
        # Get activations for positive examples
        pos_acts = []
        for prompt in pos_prompts:
            acts = model.get_layer_activations(prompt, layer_idx)
            # Use LAST token (where decision is made)
            if acts.ndim == 3:
                acts = acts[0, -1, :]  # [batch, seq, hidden] -> [hidden]
            elif acts.ndim == 2:
                acts = acts[-1, :]
            elif acts.ndim == 1:
                acts = acts  # Already just [hidden]
            pos_acts.append(acts.cpu())
        
        # Get activations for negative examples  
        neg_acts = []
        for prompt in neg_prompts:
            acts = model.get_layer_activations(prompt, layer_idx)
            if acts.ndim == 3:
                acts = acts[0, -1, :]
            elif acts.ndim == 2:
                acts = acts[-1, :]
            elif acts.ndim == 1:
                acts = acts
            neg_acts.append(acts.cpu())
        
        # Compute mean difference
        pos_mean = torch.stack(pos_acts).mean(dim=0)
        neg_mean = torch.stack(neg_acts).mean(dim=0)
        
        # Steering vector = POS - NEG
        # So positive epsilon pushes toward answering (POS)
        # Negative epsilon pushes toward abstaining (NEG)
        steering_vec = pos_mean - neg_mean
        
        # Normalize
        steering_vec = steering_vec / (steering_vec.norm() + 1e-8)
        
        steering_vectors[layer_idx] = steering_vec
        
        print(f"  Vector norm (before normalization): {(pos_mean - neg_mean).norm():.3f}")
        print(f"  Vector norm (after normalization): {steering_vec.norm():.3f}")
    
    return steering_vectors


def test_steering_quick(
    model: ModelWrapper,
    steering_vectors: Dict[int, torch.Tensor],
    test_layer: int,
    epsilon: float,
):
    """Quick test of steering effect"""
    from experiment5_trustworthiness import Experiment5
    
    config = ExperimentConfig()
    exp5 = Experiment5(model, config, steering_vectors)
    
    # Test questions
    test_cases = [
        {
            "question": "What is 2+2?",
            "answer": "4",
            "is_unanswerable": False,
        },
        {
            "question": "What was Socrates' favorite breakfast?",
            "answer": None,
            "is_unanswerable": True,
        },
    ]
    
    print(f"\n{'=' * 70}")
    print(f"QUICK TEST: Layer {test_layer}, ε={epsilon}")
    print("=" * 70)
    
    for tc in test_cases:
        result = exp5.test_one(tc, test_layer, epsilon)
        q_type = "UNANSWERABLE" if tc["is_unanswerable"] else "ANSWERABLE"
        print(f"\n{q_type}: {tc['question']}")
        print(f"  Abstained: {result['abstained']}")
        print(f"  Response: {result['response_preview'][:100]}")


def main():
    """Recompute and test steering vectors"""
    
    config = ExperimentConfig()
    print("Initializing model...")
    model = ModelWrapper(config)
    
    # Load datasets
    print("\nLoading datasets...")
    with open("./data/dataset_clearly_answerable.json", "r") as f:
        answerable = json.load(f)
    with open("./data/dataset_clearly_unanswerable.json", "r") as f:
        unanswerable = json.load(f)
    
    answerable = [
        {**ex, "is_unanswerable": False}
        for ex in answerable
        if ex.get("question") and ex.get("answer")
    ][:30]
    
    # Add tempting unanswerables
    tempting = [
        {"question": "What year was Socrates born?", "answer": None, "is_unanswerable": True},
        {"question": "What was Cleopatra's exact height in centimeters?", "answer": None, "is_unanswerable": True},
        {"question": "What was Julius Caesar's favorite food?", "answer": None, "is_unanswerable": True},
        {"question": "What time did Shakespeare wake up on his 30th birthday?", "answer": None, "is_unanswerable": True},
        {"question": "What was Aristotle's childhood nickname?", "answer": None, "is_unanswerable": True},
        {"question": "What color were Napoleon's eyes?", "answer": None, "is_unanswerable": True},
        {"question": "What was Plato's mother's maiden name?", "answer": None, "is_unanswerable": True},
        {"question": "What was the exact date Confucius was born?", "answer": None, "is_unanswerable": True},
        {"question": "What was Genghis Khan's weight at age 40?", "answer": None, "is_unanswerable": True},
        {"question": "How many books were in the Library of Alexandria?", "answer": None, "is_unanswerable": True},
    ]
    
    original_unans = [
        {**ex, "is_unanswerable": True, "answer": None}
        for ex in unanswerable
        if ex.get("question")
    ][:20]
    
    unanswerable = (original_unans + tempting)[:30]
    
    print(f"Answerable: {len(answerable)}")
    print(f"Unanswerable: {len(unanswerable)}")
    
    # Compute new steering vectors
    print("\n" + "=" * 70)
    print("RECOMPUTING STEERING VECTORS WITH EXPLICIT PROMPTING")
    print("=" * 70)
    
    steering_vectors = compute_steering_vectors_explicit(
        model=model,
        answerable_questions=answerable,
        unanswerable_questions=unanswerable,
        layers=config.target_layers,
        n_examples=15,
    )
    
    # Save
    out_path = config.results_dir / "steering_vectors_explicit.pt"
    torch.save({k: v.cpu() for k, v in steering_vectors.items()}, out_path)
    print(f"\n✓ Saved to {out_path}")
    
    # Quick test
    test_layer = config.target_layers[len(config.target_layers) // 2]  # Middle layer
    
    print("\n" + "=" * 70)
    print("TESTING NEW VECTORS")
    print("=" * 70)
    
    for eps in [-30.0, 0.0, 30.0]:
        test_steering_quick(model, steering_vectors, test_layer, eps)
    
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("\nWith these explicit vectors:")
    print("  POS = confident answering state")
    print("  NEG = uncertain abstaining state")
    print("  Vector = POS - NEG")
    print("\nTherefore:")
    print("  +epsilon = push toward POS (more answering, less abstention)")
    print("  -epsilon = push toward NEG (less answering, more abstention)")
    print("\nFor trustworthiness, you want:")
    print("  -epsilon on unanswerables (increase abstention)")
    print("  Small epsilon on answerables (maintain coverage)")
    
    print("\n✓ Run experiment5 with these new vectors:")
    print(f"  steering_vectors = torch.load('{out_path}')")
    print("  Use NEGATIVE epsilon values to increase abstention")


if __name__ == "__main__":
    main()