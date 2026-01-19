#!/usr/bin/env python3
"""
Quick runner for critical experiments (Exp6 + Exp7)

Usage:
    python run_critical_experiments.py --mode quick     # 1-2 hours, minimal tests
    python run_critical_experiments.py --mode standard  # 3-4 hours, recommended
    python run_critical_experiments.py --mode full      # 6-8 hours, comprehensive

This will run:
- Experiment 6: Robustness (cross-domain, prompt variations, adversarial)
- Experiment 7: Safety (preservation, selective abstention, spurious correlations)
"""

import argparse
import json
import time
from pathlib import Path
import torch

from core_utils import ModelWrapper, ExperimentConfig, set_seed
from experiment6_robustness import Experiment6
from experiment7_safety_alignment import Experiment7


def main():
    parser = argparse.ArgumentParser(description="Run critical experiments for paper")
    parser.add_argument(
        '--mode',
        choices=['quick', 'standard', 'full'],
        default='standard',
        help='Experiment thoroughness level'
    )
    parser.add_argument(
        '--skip-exp6',
        action='store_true',
        help='Skip Experiment 6 (robustness) - NOT recommended'
    )
    parser.add_argument(
        '--skip-exp7',
        action='store_true',
        help='Skip Experiment 7 (safety)'
    )
    args = parser.parse_args()

    print("\n" + "="*80)
    print(f" RUNNING CRITICAL EXPERIMENTS ({args.mode.upper()} MODE)")
    print("="*80)
    print()

    # Setup
    config = ExperimentConfig()
    set_seed(config.seed)

    # Mode-specific settings
    mode_config = {
        'quick': {
            'n_cross_domain': 5,  # questions per domain
            'n_prompt_var': 5,
            'n_adversarial': 3,
            'n_safety': 5,
            'n_selective': 3,
            'n_spurious': 2,
        },
        'standard': {
            'n_cross_domain': 10,
            'n_prompt_var': 10,
            'n_adversarial': 6,
            'n_safety': 15,
            'n_selective': 5,
            'n_spurious': 3,
        },
        'full': {
            'n_cross_domain': 15,
            'n_prompt_var': 15,
            'n_adversarial': 10,
            'n_safety': 20,
            'n_selective': 8,
            'n_spurious': 6,
        }
    }

    cfg = mode_config[args.mode]

    # Load model
    print("Loading model...")
    start_time = time.time()
    model = ModelWrapper(config)
    print(f"✓ Model loaded in {time.time() - start_time:.1f}s")

    # Load steering vectors
    print("\nLoading steering vectors...")
    vectors_path = config.results_dir / "steering_vectors_explicit.pt"

    if not vectors_path.exists():
        print(f"ERROR: Steering vectors not found at {vectors_path}")
        print("Please run diagnostic_steering_vectors.py first!")
        return

    steering_vectors = torch.load(vectors_path)
    steering_vectors = {int(k): v.to(config.device) for k, v in steering_vectors.items()}
    print(f"✓ Loaded {len(steering_vectors)} steering vectors")

    # Get optimal parameters from Exp5
    print("\nLoading Exp5 results...")
    exp5_summary_path = config.results_dir / "exp5_summary.json"
    if not exp5_summary_path.exists():
        print("WARNING: Exp5 summary not found. Using default layer=27, epsilon=-50")
        best_layer = 27
        optimal_epsilon = -50.0
    else:
        with open(exp5_summary_path, 'r') as f:
            exp5_summary = json.load(f)
        best_layer = 27  # Or extract from exp5_summary if available
        optimal_epsilon = exp5_summary.get('best_eps_value', -50.0)

    print(f"✓ Using layer {best_layer}, epsilon {optimal_epsilon}")

    results_summary = {
        'mode': args.mode,
        'config': cfg,
        'best_layer': best_layer,
        'optimal_epsilon': optimal_epsilon,
    }

    # =========================================================================
    # EXPERIMENT 6: ROBUSTNESS
    # =========================================================================
    if not args.skip_exp6:
        print("\n" + "="*80)
        print("EXPERIMENT 6: ROBUSTNESS TESTING")
        print("="*80)

        exp6_start = time.time()
        exp6 = Experiment6(model, config, steering_vectors)

        # 6A: Cross-domain
        print("\n[6A] Cross-domain generalization...")
        try:
            df_domains = exp6.test_cross_domain(best_layer, optimal_epsilon)
            print(f"✓ Cross-domain complete: {len(df_domains)} tests")
            results_summary['exp6_cross_domain_tests'] = len(df_domains)
        except Exception as e:
            print(f"ERROR in Exp6A: {e}")
            df_domains = None

        # 6B: Prompt variations
        print("\n[6B] Prompt variation robustness...")
        try:
            # Load some test questions
            with open("./data/dataset_clearly_answerable.json", 'r') as f:
                answerable = json.load(f)
            with open("./data/dataset_clearly_unanswerable.json", 'r') as f:
                unanswerable = json.load(f)

            test_questions = [
                {**q, "is_unanswerable": False} for q in answerable[:cfg['n_prompt_var']//2]
            ] + [
                {**q, "is_unanswerable": True, "answer": None}
                for q in unanswerable[:cfg['n_prompt_var']//2]
            ]

            df_prompts = exp6.test_prompt_variations(test_questions, best_layer, optimal_epsilon)
            print(f"✓ Prompt variations complete: {len(df_prompts)} tests")
            results_summary['exp6_prompt_variation_tests'] = len(df_prompts)
        except Exception as e:
            print(f"ERROR in Exp6B: {e}")
            df_prompts = None

        # 6C: Adversarial
        print("\n[6C] Adversarial question testing...")
        try:
            df_adversarial = exp6.test_adversarial_questions(best_layer, optimal_epsilon)
            print(f"✓ Adversarial testing complete: {len(df_adversarial)} tests")
            results_summary['exp6_adversarial_tests'] = len(df_adversarial)
        except Exception as e:
            print(f"ERROR in Exp6C: {e}")
            df_adversarial = None

        # Analyze
        if df_domains is not None or df_prompts is not None or df_adversarial is not None:
            print("\n[6] Analyzing robustness results...")
            try:
                exp6_summary = exp6.analyze_robustness(
                    df_domains if df_domains is not None else None,
                    df_prompts if df_prompts is not None else None,
                    df_adversarial if df_adversarial is not None else None
                )

                # Save summary
                with open(config.results_dir / "exp6_summary.json", 'w') as f:
                    json.dump(exp6_summary, f, indent=2)

                results_summary['exp6'] = exp6_summary
                print(f"✓ Exp6 analysis complete")
            except Exception as e:
                print(f"ERROR in Exp6 analysis: {e}")

        exp6_time = time.time() - exp6_start
        print(f"\n✓ Experiment 6 complete in {exp6_time/60:.1f} minutes")
    else:
        print("\nSkipping Experiment 6 (robustness)")

    # =========================================================================
    # EXPERIMENT 7: SAFETY
    # =========================================================================
    if not args.skip_exp7:
        print("\n" + "="*80)
        print("EXPERIMENT 7: SAFETY & ALIGNMENT TESTING")
        print("="*80)

        exp7_start = time.time()
        exp7 = Experiment7(model, config, steering_vectors)

        # 7A: Safety preservation
        print("\n[7A] Safety preservation testing...")
        try:
            df_preservation = exp7.test_safety_preservation(best_layer, optimal_epsilon)
            print(f"✓ Safety preservation complete: {len(df_preservation)} tests")
            results_summary['exp7_preservation_tests'] = len(df_preservation)
        except Exception as e:
            print(f"ERROR in Exp7A: {e}")
            df_preservation = None

        # 7B: Selective abstention
        print("\n[7B] Selective abstention preservation...")
        try:
            df_selective = exp7.test_selective_abstention(best_layer, optimal_epsilon)
            print(f"✓ Selective abstention complete: {len(df_selective)} tests")
            results_summary['exp7_selective_tests'] = len(df_selective)
        except Exception as e:
            print(f"ERROR in Exp7B: {e}")
            df_selective = None

        # 7C: Spurious correlations
        print("\n[7C] Spurious correlation testing...")
        try:
            df_spurious = exp7.test_spurious_correlations(best_layer, optimal_epsilon)
            print(f"✓ Spurious correlation testing complete: {len(df_spurious)} tests")
            results_summary['exp7_spurious_tests'] = len(df_spurious)
        except Exception as e:
            print(f"ERROR in Exp7C: {e}")
            df_spurious = None

        # Analyze
        if df_preservation is not None or df_selective is not None or df_spurious is not None:
            print("\n[7] Analyzing safety results...")
            try:
                exp7_summary = exp7.analyze_safety(
                    df_preservation if df_preservation is not None else None,
                    df_selective if df_selective is not None else None,
                    df_spurious if df_spurious is not None else None
                )

                # Save summary
                with open(config.results_dir / "exp7_summary.json", 'w') as f:
                    json.dump(exp7_summary, f, indent=2)

                results_summary['exp7'] = exp7_summary
                print(f"✓ Exp7 analysis complete")
            except Exception as e:
                print(f"ERROR in Exp7 analysis: {e}")

        exp7_time = time.time() - exp7_start
        print(f"\n✓ Experiment 7 complete in {exp7_time/60:.1f} minutes")
    else:
        print("\nSkipping Experiment 7 (safety)")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    total_time = time.time() - start_time
    results_summary['total_time_minutes'] = total_time / 60

    print("\n" + "="*80)
    print(" EXPERIMENTS COMPLETE!")
    print("="*80)
    print(f"\nTotal time: {total_time/60:.1f} minutes")
    print(f"\nResults saved to: {config.results_dir}")

    # Save combined summary
    with open(config.results_dir / f"critical_experiments_summary_{args.mode}.json", 'w') as f:
        json.dump(results_summary, f, indent=2)

    print("\n✓ All critical experiments complete!")
    print("\nNext steps:")
    print("  1. Review figures in results/")
    print("  2. Check summary JSONs for key numbers")
    print("  3. Incorporate results into paper")
    print()

    # Quick stats summary
    if 'exp6' in results_summary:
        print("Exp6 (Robustness) Key Findings:")
        if 'cross_domain_consistency' in results_summary['exp6']:
            consistency = results_summary['exp6']['cross_domain_consistency']
            print(f"  - Cross-domain consistency: {consistency:.3f}")
        print()

    if 'exp7' in results_summary:
        print("Exp7 (Safety) Key Findings:")
        if 'safety_preserved' in results_summary['exp7']:
            preserved = results_summary['exp7']['safety_preserved']
            print(f"  - Safety preserved: {'YES ✓' if preserved else 'NO ✗'}")
        if 'baseline_refusal_rate' in results_summary['exp7']:
            refusal = results_summary['exp7']['baseline_refusal_rate']
            print(f"  - Baseline refusal rate: {refusal:.1%}")
        print()


if __name__ == "__main__":
    main()
