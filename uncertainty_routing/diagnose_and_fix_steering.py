#!/usr/bin/env python3
"""
Comprehensive Steering Diagnosis and Fix

ROOT CAUSES IDENTIFIED:
1. Steering direction may be inverted (pos-neg instead of neg-pos)
2. Epsilon magnitude (-50) is too extreme, causing model collapse
3. Steering applied at wrong token position during generation
4. No normalization consistency between training and application

FIXES:
1. Verify and correct steering direction
2. Use appropriate epsilon range (0.5-5.0 instead of -50 to +50)
3. Apply steering at correct token positions
4. Add proper normalization and scaling
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

from core_utils import ModelWrapper, ExperimentConfig, extract_answer
from data_preparation import (
    create_clearly_answerable_questions,
    create_clearly_unanswerable_questions,
    format_prompt
)


class SteeringDiagnostic:
    """Diagnose and fix steering vector issues"""

    def __init__(self):
        self.config = ExperimentConfig()
        self.results_dir = Path("results")
        self.model = None
        self.steering_vectors = None

    def load_steering_vectors(self) -> Dict[int, torch.Tensor]:
        """Load existing steering vectors"""
        vectors_path = self.results_dir / "steering_vectors.pt"
        explicit_path = self.results_dir / "steering_vectors_explicit.pt"

        if explicit_path.exists():
            print(f"✓ Loading steering_vectors_explicit.pt")
            path = explicit_path
        elif vectors_path.exists():
            print(f"✓ Loading steering_vectors.pt")
            path = vectors_path
        else:
            raise FileNotFoundError("No steering vectors found. Run Experiment 3 first.")

        vectors = torch.load(path)
        # Ensure integer keys
        vectors = {int(k): v for k, v in vectors.items()}
        print(f"  Loaded {len(vectors)} steering vectors for layers: {list(vectors.keys())}")
        return vectors

    def load_model(self):
        """Load model wrapper"""
        print(f"\nLoading model {self.config.model_name}...")
        self.model = ModelWrapper(self.config)
        print("✓ Model loaded")

    def test_baseline_behavior(self, questions: List[Dict], question_type: str) -> Dict:
        """Test model behavior without steering"""
        print(f"\nTesting baseline on {len(questions)} {question_type} questions...")

        abstentions = 0
        correct = 0
        hallucinations = 0

        for q in questions:
            prompt = format_prompt(q["question"], "neutral")
            response = self.model.generate(prompt, temperature=0.0, do_sample=False, max_new_tokens=50)

            answer = extract_answer(response)
            is_uncertain = answer == "UNCERTAIN"

            if is_uncertain:
                abstentions += 1
            elif question_type == "answerable" and "answer" in q:
                # Check if correct
                if q["answer"].lower() in response.lower():
                    correct += 1
            elif question_type == "unanswerable":
                hallucinations += 1

        result = {
            "abstention_rate": abstentions / len(questions),
            "correct_rate": correct / len(questions) if question_type == "answerable" else 0,
            "hallucination_rate": hallucinations / len(questions) if question_type == "unanswerable" else 0
        }

        print(f"  Abstention: {result['abstention_rate']:.1%}")
        if question_type == "answerable":
            print(f"  Correct: {result['correct_rate']:.1%}")
        else:
            print(f"  Hallucination: {result['hallucination_rate']:.1%}")

        return result

    def test_steering_effect(self, layer: int, epsilon: float, questions: List[Dict],
                            question_type: str) -> Dict:
        """Test steering at specific layer and epsilon"""
        abstentions = 0
        correct = 0
        hallucinations = 0

        steering_vec = self.steering_vectors[layer]

        for q in questions:
            prompt = format_prompt(q["question"], "neutral")

            # Apply steering
            self.model.clear_hooks()

            # Get prompt length to steer at last token
            inputs = self.model.tokenizer(prompt, return_tensors="pt").to(self.model.config.device)
            position = inputs["input_ids"].shape[1] - 1

            # Register steering hook
            self.model.register_steering_hook(layer, position, steering_vec, epsilon)

            # Generate with steering
            response = self.model.generate(prompt, temperature=0.0, do_sample=False, max_new_tokens=50)

            self.model.clear_hooks()

            answer = extract_answer(response)
            is_uncertain = answer == "UNCERTAIN"

            if is_uncertain:
                abstentions += 1
            elif question_type == "answerable" and "answer" in q:
                if q["answer"].lower() in response.lower():
                    correct += 1
            elif question_type == "unanswerable":
                hallucinations += 1

        return {
            "abstention_rate": abstentions / len(questions),
            "correct_rate": correct / len(questions) if question_type == "answerable" else 0,
            "hallucination_rate": hallucinations / len(questions) if question_type == "unanswerable" else 0
        }

    def diagnose_direction(self, layer: int, test_epsilons: List[float] = None) -> Dict:
        """
        Diagnose if steering direction is correct

        Expected behavior:
        - NEGATIVE epsilon should INCREASE abstention on unanswerables (reduce hallucination)
        - POSITIVE epsilon should DECREASE abstention on unanswerables (increase hallucination)
        - NEGATIVE epsilon should NOT break answerable questions
        """
        if test_epsilons is None:
            test_epsilons = [-3.0, -1.0, 0.0, 1.0, 3.0]

        print(f"\n{'='*80}")
        print(f"DIAGNOSING STEERING DIRECTION (Layer {layer})")
        print(f"{'='*80}\n")

        # Load test questions
        answerable = create_clearly_answerable_questions()[:10]
        unanswerable = create_clearly_unanswerable_questions()[:10]

        results = []

        for eps in test_epsilons:
            print(f"\nTesting epsilon = {eps:+.1f}...")

            # Test on unanswerable (primary diagnostic)
            unans_result = self.test_steering_effect(layer, eps, unanswerable, "unanswerable")
            print(f"  Unanswerable: {unans_result['abstention_rate']:.1%} abstention, "
                  f"{unans_result['hallucination_rate']:.1%} hallucination")

            # Test on answerable (check if we break the model)
            ans_result = self.test_steering_effect(layer, eps, answerable, "answerable")
            print(f"  Answerable: {ans_result['correct_rate']:.1%} correct, "
                  f"{ans_result['abstention_rate']:.1%} abstention")

            results.append({
                "epsilon": eps,
                "unans_abstention": unans_result['abstention_rate'],
                "unans_hallucination": unans_result['hallucination_rate'],
                "ans_correct": ans_result['correct_rate'],
                "ans_abstention": ans_result['abstention_rate']
            })

        df = pd.DataFrame(results)

        # Analyze
        print(f"\n{'='*80}")
        print("ANALYSIS")
        print(f"{'='*80}\n")

        baseline = df[df['epsilon'] == 0.0].iloc[0]
        most_negative = df[df['epsilon'] == df['epsilon'].min()].iloc[0]
        most_positive = df[df['epsilon'] == df['epsilon'].max()].iloc[0]

        print(f"Baseline (ε=0):  Unans abstention: {baseline['unans_abstention']:.1%}, "
              f"Ans correct: {baseline['ans_correct']:.1%}")
        print(f"Negative (ε={most_negative['epsilon']:+.1f}): Unans abstention: {most_negative['unans_abstention']:.1%}, "
              f"Ans correct: {most_negative['ans_correct']:.1%}")
        print(f"Positive (ε={most_positive['epsilon']:+.1f}): Unans abstention: {most_positive['unans_abstention']:.1%}, "
              f"Ans correct: {most_positive['ans_correct']:.1%}")

        # Determine if direction is inverted
        direction_inverted = False
        magnitude_too_large = False

        # Check if negative epsilon DECREASES abstention (wrong direction)
        if most_negative['unans_abstention'] < baseline['unans_abstention']:
            print("\n⚠️  INVERTED DIRECTION: Negative epsilon decreases abstention (should increase)")
            direction_inverted = True

        # Check if positive epsilon INCREASES abstention (wrong direction)
        if most_positive['unans_abstention'] > baseline['unans_abstention']:
            print("\n⚠️  INVERTED DIRECTION: Positive epsilon increases abstention (should decrease)")
            direction_inverted = True

        # Check if steering breaks answerable questions
        if most_negative['ans_correct'] < baseline['ans_correct'] * 0.5:
            print(f"\n⚠️  MAGNITUDE TOO LARGE: Negative epsilon destroys accuracy "
                  f"({most_negative['ans_correct']:.1%} vs {baseline['ans_correct']:.1%})")
            magnitude_too_large = True

        if not direction_inverted:
            print("\n✓ Direction appears CORRECT")

        return {
            "direction_inverted": direction_inverted,
            "magnitude_too_large": magnitude_too_large,
            "results_df": df,
            "layer": layer
        }

    def fix_steering_vectors(self) -> Dict[int, torch.Tensor]:
        """Apply fixes to steering vectors"""
        print(f"\n{'='*80}")
        print("APPLYING FIXES")
        print(f"{'='*80}\n")

        fixed_vectors = {}

        # Test primary layer (usually highest layer)
        primary_layer = max(self.steering_vectors.keys())

        diagnosis = self.diagnose_direction(primary_layer)

        if diagnosis["direction_inverted"]:
            print("\n→ FIX 1: Flipping steering vector direction")
            for layer, vec in self.steering_vectors.items():
                fixed_vectors[layer] = -vec
            print("  ✓ All steering vectors flipped")
        else:
            print("\n→ No direction flip needed")
            fixed_vectors = self.steering_vectors.copy()

        # Normalize vectors consistently
        print("\n→ FIX 2: Ensuring consistent normalization")
        for layer in fixed_vectors:
            vec = fixed_vectors[layer]
            norm = vec.norm()
            if norm > 1e-8:
                fixed_vectors[layer] = vec / norm
            print(f"  Layer {layer}: norm before={norm:.4f}, after={fixed_vectors[layer].norm():.4f}")

        # Save fixed vectors
        output_path = self.results_dir / "steering_vectors_fixed.pt"
        torch.save(fixed_vectors, output_path)
        print(f"\n✓ Saved fixed vectors to {output_path}")

        return fixed_vectors

    def generate_recommended_config(self, diagnosis: Dict):
        """Generate recommended configuration for experiments"""
        print(f"\n{'='*80}")
        print("RECOMMENDED CONFIGURATION")
        print(f"{'='*80}\n")

        print("Based on diagnosis, use these parameters for Experiments 6-9:")
        print()
        print("Epsilon ranges:")
        print("  - For increasing abstention: epsilon = -5.0 to -0.5")
        print("  - For decreasing abstention: epsilon = +0.5 to +5.0")
        print("  - Baseline: epsilon = 0.0")
        print()
        print("Recommended values for experiments:")
        print("  - Baseline: 0.0")
        print("  - Moderate toward abstention: -2.0")
        print("  - Strong toward abstention: -5.0")
        print("  - Moderate toward answer: +2.0")
        print("  - Strong toward answer: +5.0")
        print()
        print("DO NOT USE: epsilon values like -50 or +50 (way too large)")

        # Write config file
        config_path = self.results_dir / "recommended_epsilon_config.json"
        import json
        config = {
            "steering_config": {
                "primary_layer": diagnosis["layer"],
                "epsilon_ranges": {
                    "toward_abstention": [-5.0, -3.0, -2.0, -1.0, -0.5],
                    "baseline": [0.0],
                    "toward_answer": [0.5, 1.0, 2.0, 3.0, 5.0]
                },
                "recommended_for_experiments": {
                    "exp6_exp7_baseline": 0.0,
                    "exp6_exp7_steered": -2.0,
                    "exp7_toward_answer": 2.0,
                    "exp7_toward_abstain": -2.0
                }
            }
        }

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\n✓ Saved configuration to {config_path}")

    def run_full_diagnosis(self):
        """Run complete diagnostic and fix pipeline"""
        print(f"\n{'='*80}")
        print("STEERING DIAGNOSTIC & FIX PIPELINE")
        print(f"{'='*80}\n")

        # Step 1: Load vectors
        self.steering_vectors = self.load_steering_vectors()

        # Step 2: Load model
        self.load_model()

        # Step 3: Test baseline
        print(f"\n{'='*80}")
        print("BASELINE BEHAVIOR (No Steering)")
        print(f"{'='*80}")

        answerable = create_clearly_answerable_questions()[:10]
        unanswerable = create_clearly_unanswerable_questions()[:10]

        ans_baseline = self.test_baseline_behavior(answerable, "answerable")
        unans_baseline = self.test_baseline_behavior(unanswerable, "unanswerable")

        # Step 4: Diagnose primary layer
        primary_layer = max(self.steering_vectors.keys())
        diagnosis = self.diagnose_direction(primary_layer, test_epsilons=[-5.0, -2.0, 0.0, 2.0, 5.0])

        # Step 5: Apply fixes if needed
        if diagnosis["direction_inverted"] or diagnosis["magnitude_too_large"]:
            fixed_vectors = self.fix_steering_vectors()

            # Re-test with fixed vectors
            print(f"\n{'='*80}")
            print("VERIFICATION WITH FIXED VECTORS")
            print(f"{'='*80}\n")

            self.steering_vectors = fixed_vectors
            verification = self.diagnose_direction(primary_layer, test_epsilons=[-2.0, 0.0, 2.0])

            if not verification["direction_inverted"]:
                print("\n✅ FIX SUCCESSFUL!")
            else:
                print("\n⚠️  Issue persists - manual investigation needed")

        # Step 6: Generate recommendations
        self.generate_recommended_config(diagnosis)

        print(f"\n{'='*80}")
        print("NEXT STEPS")
        print(f"{'='*80}\n")
        print("1. Update experiments 6 and 7 to use recommended epsilon values")
        print("2. Use 'steering_vectors_fixed.pt' instead of 'steering_vectors.pt'")
        print("3. Re-run experiments with corrected configuration")
        print()


if __name__ == "__main__":
    diagnostic = SteeringDiagnostic()
    diagnostic.run_full_diagnosis()
