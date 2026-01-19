#!/usr/bin/env python3
"""
Complete Pipeline Runner v2 - Bug-Free Edition

Runs ALL experiments including critical new ones (Exp6-9) for paper submission.

Usage:
    python run_complete_pipeline_v2.py --mode quick    # 3-4 hours, minimal tests
    python run_complete_pipeline_v2.py --mode standard # 8-10 hours, recommended
    python run_complete_pipeline_v2.py --mode full     # 15-20 hours, comprehensive

    # Skip specific experiments
    python run_complete_pipeline_v2.py --skip-exp6 --skip-exp7

    # Run only critical experiments (Exp8+9)
    python run_complete_pipeline_v2.py --only-critical
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
import argparse

# Set environment variables BEFORE importing torch
os.environ['TRANSFORMERS_CACHE'] = str(Path.home() / '.cache' / 'huggingface')
os.environ['HF_HOME'] = str(Path.home() / '.cache' / 'huggingface')

import torch
import numpy as np

# Import experiments
from core_utils import ModelWrapper, ExperimentConfig, set_seed
from data_preparation import (
    create_ambiguous_prompts,
    create_clearly_answerable_questions,
    create_clearly_unanswerable_questions
)


def _json_sanitize(obj):
    """Convert results dict into JSON-serializable primitives."""
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        new = {}
        for k, v in obj.items():
            if not isinstance(k, (str, int, float, bool)) and k is not None:
                k = str(k)
            new[k] = _json_sanitize(v)
        return new
    elif isinstance(obj, (list, tuple)):
        return [_json_sanitize(x) for x in obj]
    else:
        return obj


def run_exp1(model, config, questions):
    """Experiment 1: Behavior-Belief Dissociation"""
    print("\n" + "="*80)
    print("[1/9] EXPERIMENT 1: Behavior-Belief Dissociation")
    print("="*80)

    try:
        from experiment1_behavior_belief import Experiment1

        exp1 = Experiment1(model, config)
        exp1_df = exp1.run(questions)
        exp1_summary = exp1.analyze(exp1_df)

        print("✓ Experiment 1 complete")
        return exp1_summary
    except Exception as e:
        print(f"✗ Experiment 1 failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_exp2(model, config, answerable, unanswerable, n_pairs, mode):
    """Experiment 2: Gate Localization"""
    print("\n" + "="*80)
    print("[2/9] EXPERIMENT 2: Gate Localization")
    print("="*80)

    try:
        from experiment2_localization import Experiment2

        exp2 = Experiment2(model, config)
        exp2_df = exp2.run(
            pos_examples=answerable[:n_pairs],
            neg_examples=unanswerable[:n_pairs],
            n_pairs=n_pairs,
            layer_stride=2 if mode == 'quick' else 1
        )
        exp2_summary = exp2.analyze(exp2_df)

        # Update target layers based on results
        critical_layers = exp2_summary.get('critical_layers', [])
        if len(critical_layers) >= 4:
            config.target_layers = critical_layers[-4:]
        elif len(critical_layers) > 0:
            config.target_layers = critical_layers

        print(f"✓ Experiment 2 complete - Target layers: {config.target_layers}")
        return exp2_summary
    except Exception as e:
        print(f"✗ Experiment 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_exp3(model, config, answerable, unanswerable, test_questions):
    """Experiment 3: Steering Control"""
    print("\n" + "="*80)
    print("[3/9] EXPERIMENT 3: Steering Control")
    print("="*80)

    try:
        from experiment3_4_steering_independence import Experiment3

        exp3 = Experiment3(model, config)
        exp3_df = exp3.run(
            pos_examples=answerable,
            neg_examples=unanswerable,
            test_examples=test_questions
        )
        exp3_summary = exp3.analyze(exp3_df)

        best_layer = exp3_summary.get('best_layer', config.target_layers[-1] if config.target_layers else 24)

        print(f"✓ Experiment 3 complete - Best layer: {best_layer}")
        return exp3_summary, exp3.steering_vectors, best_layer
    except Exception as e:
        print(f"✗ Experiment 3 failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def run_exp4(model, config, steering_vectors, questions, best_layer):
    """Experiment 4: Gate Independence"""
    print("\n" + "="*80)
    print("[4/9] EXPERIMENT 4: Gate Independence")
    print("="*80)

    try:
        from experiment3_4_steering_independence import Experiment4

        exp4 = Experiment4(model, config, steering_vectors)
        exp4_df = exp4.run(
            questions=questions,
            best_layer=best_layer,
            epsilon=12.0,
            n_uncert_samples=20,
            reverse_steering=True  # Use negative steering for abstention
        )
        exp4_summary = exp4.analyze(exp4_df)

        print("✓ Experiment 4 complete")
        return exp4_summary
    except Exception as e:
        print(f"✗ Experiment 4 failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_exp5(model, config, steering_vectors, answerable, unanswerable):
    """Experiment 5: Trustworthiness Application"""
    print("\n" + "="*80)
    print("[5/9] EXPERIMENT 5: Trustworthiness Application")
    print("="*80)

    try:
        from experiment5_trustworthiness import Experiment5

        # Prepare test set
        test_set = [
            {**q, "is_unanswerable": False} for q in answerable
        ] + [
            {**q, "is_unanswerable": True, "answer": None} for q in unanswerable
        ]

        exp5 = Experiment5(model, config, steering_vectors)
        exp5_df = exp5.run(
            questions=test_set,
            test_layers=list(steering_vectors.keys()),
            epsilon_test=20.0,
            epsilon_values=[-50.0, -40.0, -30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0],
            n_probe=15,
        )
        exp5_summary = exp5.analyze(exp5_df)

        print("✓ Experiment 5 complete")
        return exp5_summary
    except Exception as e:
        print(f"✗ Experiment 5 failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_exp6(model, config, steering_vectors, best_layer, optimal_epsilon):
    """Experiment 6: Robustness Testing"""
    print("\n" + "="*80)
    print("[6/9] EXPERIMENT 6: Robustness Testing")
    print("="*80)

    try:
        from experiment6_robustness import Experiment6

        exp6 = Experiment6(model, config, steering_vectors)

        # Cross-domain testing
        print("\n[6A] Cross-domain generalization...")
        df_domains = exp6.test_cross_domain(best_layer, optimal_epsilon)

        # Prompt variations
        print("\n[6B] Prompt variation robustness...")
        with open("./data/dataset_clearly_answerable.json", 'r') as f:
            answerable = json.load(f)
        with open("./data/dataset_clearly_unanswerable.json", 'r') as f:
            unanswerable = json.load(f)

        test_questions = [
            {**q, "is_unanswerable": False} for q in answerable[:5]
        ] + [
            {**q, "is_unanswerable": True, "answer": None} for q in unanswerable[:5]
        ]

        df_prompts = exp6.test_prompt_variations(test_questions, best_layer, optimal_epsilon)

        # Adversarial questions
        print("\n[6C] Adversarial question testing...")
        df_adversarial = exp6.test_adversarial_questions(best_layer, optimal_epsilon)

        # Analyze
        exp6_summary = exp6.analyze_robustness(df_domains, df_prompts, df_adversarial)

        print("✓ Experiment 6 complete")
        return exp6_summary
    except Exception as e:
        print(f"✗ Experiment 6 failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_exp7(model, config, steering_vectors, best_layer, optimal_epsilon):
    """Experiment 7: Safety & Alignment"""
    print("\n" + "="*80)
    print("[7/9] EXPERIMENT 7: Safety & Alignment Testing")
    print("="*80)

    try:
        from experiment7_safety_alignment import Experiment7

        exp7 = Experiment7(model, config, steering_vectors)

        # Safety preservation
        print("\n[7A] Safety preservation testing...")
        df_preservation = exp7.test_safety_preservation(best_layer, optimal_epsilon)

        # Selective abstention
        print("\n[7B] Selective abstention preservation...")
        df_selective = exp7.test_selective_abstention(best_layer, optimal_epsilon)

        # Spurious correlations
        print("\n[7C] Spurious correlation testing...")
        df_spurious = exp7.test_spurious_correlations(best_layer, optimal_epsilon)

        # Analyze
        exp7_summary = exp7.analyze_safety(df_preservation, df_selective, df_spurious)

        print("✓ Experiment 7 complete")
        return exp7_summary
    except Exception as e:
        print(f"✗ Experiment 7 failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_exp8(config, test_questions):
    """Experiment 8: Scaling Analysis"""
    print("\n" + "="*80)
    print("[8/9] EXPERIMENT 8: Scaling Analysis ⭐ CRITICAL")
    print("="*80)

    try:
        from experiment8_scaling_analysis import Experiment8

        exp8 = Experiment8(config)
        summary_df, all_results = exp8.run_scaling_analysis(test_questions)

        if summary_df is not None:
            exp8_summary = exp8.analyze_scaling(summary_df, all_results)
            print("✓ Experiment 8 complete")
            return exp8_summary
        else:
            print("✗ Experiment 8 failed: No models successfully tested")
            return None
    except Exception as e:
        print(f"✗ Experiment 8 failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_exp9(model, config, steering_vectors, best_layer):
    """Experiment 9: Interpretability Analysis"""
    print("\n" + "="*80)
    print("[9/9] EXPERIMENT 9: Interpretability Analysis ⭐ CRITICAL")
    print("="*80)

    try:
        from experiment9_interpretability import Experiment9

        exp9 = Experiment9(model, config, steering_vectors)
        exp9_summary = exp9.run_interpretability_analysis(best_layer)

        # Visualize
        exp9.visualize_interpretability(exp9_summary)

        print("✓ Experiment 9 complete")
        return exp9_summary
    except Exception as e:
        print(f"✗ Experiment 9 failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Run complete experimental pipeline")
    parser.add_argument('--mode', default='standard', choices=['quick', 'standard', 'full'],
                       help='Experiment thoroughness level')
    parser.add_argument('--model', default='Qwen/Qwen2.5-1.5B-Instruct',
                       help='Model to use')

    # Skip options
    parser.add_argument('--skip-exp1', action='store_true', help='Skip Experiment 1')
    parser.add_argument('--skip-exp2', action='store_true', help='Skip Experiment 2')
    parser.add_argument('--skip-exp3', action='store_true', help='Skip Experiment 3')
    parser.add_argument('--skip-exp4', action='store_true', help='Skip Experiment 4')
    parser.add_argument('--skip-exp5', action='store_true', help='Skip Experiment 5')
    parser.add_argument('--skip-exp6', action='store_true', help='Skip Experiment 6')
    parser.add_argument('--skip-exp7', action='store_true', help='Skip Experiment 7')
    parser.add_argument('--skip-exp8', action='store_true', help='Skip Experiment 8 (NOT RECOMMENDED)')
    parser.add_argument('--skip-exp9', action='store_true', help='Skip Experiment 9 (NOT RECOMMENDED)')

    # Special mode: only run critical experiments
    parser.add_argument('--only-critical', action='store_true',
                       help='Run only Exp8+9 (assumes Exp1-5 already done)')

    args = parser.parse_args()

    print("\n" + "="*80)
    print(f" COMPLETE PIPELINE v2: {args.mode.upper()} MODE")
    print(f" Model: {args.model}")
    print(f" Time: {datetime.now()}")
    print("="*80 + "\n")

    # Check GPU
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")

    # Setup config
    config = ExperimentConfig()
    config.model_name = args.model

    # Mode-specific settings
    if args.mode == 'quick':
        config.n_force_guess_samples = 10
        n_questions = 10
        n_pairs = 5
        n_test = 15
    elif args.mode == 'standard':
        config.n_force_guess_samples = 20
        n_questions = 30
        n_pairs = 10
        n_test = 30
    else:  # full
        config.n_force_guess_samples = 25
        n_questions = 50
        n_pairs = 15
        n_test = 40

    set_seed(config.seed)
    start_time = time.time()

    # Prepare data
    print("Loading datasets...")
    ambiguous = create_ambiguous_prompts()[:n_questions]
    answerable = create_clearly_answerable_questions()[:n_pairs]
    unanswerable = create_clearly_unanswerable_questions()[:n_pairs]

    print(f"  Ambiguous: {len(ambiguous)}")
    print(f"  Answerable: {len(answerable)}")
    print(f"  Unanswerable: {len(unanswerable)}\n")

    results = {}

    # Special mode: only critical experiments
    if args.only_critical:
        print("ONLY-CRITICAL MODE: Running Exp8+9 only")
        print("Assumes Exp1-5 already completed\n")

        # Load model
        print("Loading model...")
        model = ModelWrapper(config)

        # Load existing steering vectors
        vectors_path = config.results_dir / "steering_vectors_explicit.pt"
        if not vectors_path.exists():
            print(f"ERROR: Steering vectors not found at {vectors_path}")
            print("Please run Exp1-5 first or remove --only-critical flag")
            sys.exit(1)

        steering_vectors = torch.load(vectors_path)
        steering_vectors = {int(k): v.to(config.device) for k, v in steering_vectors.items()}

        # Load Exp5 summary for optimal epsilon
        exp5_summary_path = config.results_dir / "exp5_summary.json"
        if exp5_summary_path.exists():
            with open(exp5_summary_path, 'r') as f:
                exp5_summary = json.load(f)
            optimal_epsilon = exp5_summary.get('best_eps_value', -50.0)
            best_layer = 27  # Or extract from summary
        else:
            optimal_epsilon = -50.0
            best_layer = 27

        print(f"Using best_layer={best_layer}, optimal_epsilon={optimal_epsilon}\n")

        # Prepare test questions for Exp8
        test_questions_exp8 = [
            {**q, "is_unanswerable": False} for q in answerable[:15]
        ] + [
            {**q, "is_unanswerable": True, "answer": None} for q in unanswerable[:15]
        ]

        # Run Exp8
        if not args.skip_exp8:
            results['experiment8'] = run_exp8(config, test_questions_exp8)

        # Run Exp9
        if not args.skip_exp9:
            results['experiment9'] = run_exp9(model, config, steering_vectors, best_layer)

    else:
        # Full pipeline
        # Load model (do this once)
        print("Loading model...")
        model = ModelWrapper(config)
        print(f"✓ Model loaded\n")

        # Experiment 1
        if not args.skip_exp1:
            results['experiment1'] = run_exp1(model, config, ambiguous)
        else:
            print("[1/9] Skipping Experiment 1")

        # Experiment 2
        if not args.skip_exp2:
            results['experiment2'] = run_exp2(model, config, answerable, unanswerable, n_pairs, args.mode)
        else:
            print("[2/9] Skipping Experiment 2")

        # Experiment 3
        if not args.skip_exp3:
            exp3_summary, steering_vectors, best_layer = run_exp3(
                model, config, answerable, unanswerable, ambiguous[:n_questions//2]
            )
            results['experiment3'] = exp3_summary
        else:
            print("[3/9] Skipping Experiment 3")
            # Load from existing results
            vectors_path = config.results_dir / "steering_vectors.pt"
            if vectors_path.exists():
                steering_vectors = torch.load(vectors_path)
                steering_vectors = {int(k): v.to(config.device) for k, v in steering_vectors.items()}
                best_layer = max(steering_vectors.keys())
            else:
                print("ERROR: Steering vectors not found. Cannot skip Exp3.")
                sys.exit(1)

        # Experiment 4
        if not args.skip_exp4 and steering_vectors is not None:
            results['experiment4'] = run_exp4(model, config, steering_vectors, ambiguous[:20], best_layer)
        else:
            print("[4/9] Skipping Experiment 4")

        # Experiment 5
        if not args.skip_exp5 and steering_vectors is not None:
            results['experiment5'] = run_exp5(model, config, steering_vectors, answerable[:n_test], unanswerable[:n_test])

            # Get optimal epsilon from Exp5
            if results['experiment5']:
                optimal_epsilon = results['experiment5'].get('best_eps_value', -50.0)
            else:
                optimal_epsilon = -50.0
        else:
            print("[5/9] Skipping Experiment 5")
            optimal_epsilon = -50.0

        # Experiment 6
        if not args.skip_exp6 and steering_vectors is not None:
            results['experiment6'] = run_exp6(model, config, steering_vectors, best_layer, optimal_epsilon)
        else:
            print("[6/9] Skipping Experiment 6")

        # Experiment 7
        if not args.skip_exp7 and steering_vectors is not None:
            results['experiment7'] = run_exp7(model, config, steering_vectors, best_layer, optimal_epsilon)
        else:
            print("[7/9] Skipping Experiment 7")

        # Experiment 8 (Scaling) - CRITICAL
        if not args.skip_exp8:
            # Prepare test questions
            test_questions_exp8 = [
                {**q, "is_unanswerable": False} for q in answerable[:15]
            ] + [
                {**q, "is_unanswerable": True, "answer": None} for q in unanswerable[:15]
            ]

            results['experiment8'] = run_exp8(config, test_questions_exp8)
        else:
            print("[8/9] Skipping Experiment 8 ⚠️  NOT RECOMMENDED")

        # Experiment 9 (Interpretability) - CRITICAL
        if not args.skip_exp9 and steering_vectors is not None:
            results['experiment9'] = run_exp9(model, config, steering_vectors, best_layer)
        else:
            print("[9/9] Skipping Experiment 9 ⚠️  NOT RECOMMENDED")

    # Save results
    total_time = time.time() - start_time

    results['metadata'] = {
        'mode': args.mode,
        'model': args.model,
        'n_questions': n_questions,
        'n_pairs': n_pairs,
        'timestamp': datetime.now().isoformat(),
        'total_time_minutes': total_time / 60,
        'only_critical': args.only_critical,
    }

    # Sanitize and save
    output_path = config.results_dir / f"complete_pipeline_{args.mode}.json"
    with open(output_path, 'w') as f:
        safe_results = _json_sanitize(results)
        json.dump(safe_results, f, indent=2)

    print("\n" + "="*80)
    print(" PIPELINE COMPLETE!")
    print("="*80)
    print(f"\nTotal time: {total_time/60:.1f} minutes")
    print(f"Results saved to: {output_path}")

    # Summary
    print("\nExperiments completed:")
    for exp_name, exp_result in results.items():
        if exp_name == 'metadata':
            continue
        status = "✓" if exp_result is not None else "✗"
        print(f"  {status} {exp_name}")

    print("\n✓ All requested experiments completed!\n")

    return results


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ PIPELINE ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
