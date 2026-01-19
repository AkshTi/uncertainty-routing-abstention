#!/usr/bin/env python3
"""
Complete pipeline runner for SLURM
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Set environment variables BEFORE importing torch
os.environ['TRANSFORMERS_CACHE'] = str(Path.home() / '.cache' / 'huggingface')
os.environ['HF_HOME'] = str(Path.home() / '.cache' / 'huggingface')

import torch
import numpy as np

# Now import your modules
from core_utils import ModelWrapper, ExperimentConfig, set_seed
from data_preparation import create_ambiguous_prompts, create_clearly_answerable_questions, create_clearly_unanswerable_questions
from experiment1_behavior_belief import Experiment1
from experiment2_localization import Experiment2
from experiment3_4_steering_independence import Experiment3, Experiment4
from experiment5_trustworthiness import Experiment5


def _json_sanitize(obj):
    """Convert results dict into JSON-serializable primitives."""
    # torch / numpy scalars
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()

    # dict / list recursion
    if isinstance(obj, dict):
        new = {}
        for k, v in obj.items():
            # JSON requires primitive keys; stringify anything else (tuples, etc.)
            if not isinstance(k, (str, int, float, bool)) and k is not None:
                k = str(k)
            new[k] = _json_sanitize(v)
        return new
    elif isinstance(obj, (list, tuple)):
        return [_json_sanitize(x) for x in obj]
    else:
        return obj


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='standard', choices=['quick', 'standard', 'full'])
    parser.add_argument('--model', default='Qwen/Qwen2.5-1.5B-Instruct')
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print(f" STARTING PIPELINE: {args.mode.upper()} MODE")
    print(f" Model: {args.model}")
    print(f" Time: {datetime.now()}")
    print("="*80 + "\n")
    
    # Check GPU
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Setup config
    config = ExperimentConfig()
    config.model_name = args.model
    
    if args.mode == 'quick':
        config.n_force_guess_samples = 10
        n_questions = 10
        n_pairs = 5
    elif args.mode == 'standard':
        config.n_force_guess_samples = 20
        n_questions = 30
        n_pairs = 10
    else:  # full
        config.n_force_guess_samples = 25
        n_questions = 50
        n_pairs = 15
    
    set_seed(config.seed)
    
    # Prepare data
    print("\n[1/6] Preparing datasets...")
    ambiguous = create_ambiguous_prompts()[:n_questions]
    answerable = create_clearly_answerable_questions()[:n_pairs]
    unanswerable = create_clearly_unanswerable_questions()[:n_pairs]
    print(f"  Ambiguous: {len(ambiguous)}")
    print(f"  Answerable: {len(answerable)}")
    print(f"  Unanswerable: {len(unanswerable)}")
    
    # Load model
    print("\n[2/6] Loading model...")
    start_time = time.time()
    model = ModelWrapper(config)
    print(f"  Loaded in {time.time() - start_time:.1f}s")
    
    results = {}
    
    # # Experiment 1
    print("\n[3/6] Running Experiment 1: Behavior-Belief Dissociation...")
    exp1 = Experiment1(model, config)
    exp1_df = exp1.run(ambiguous)
    exp1_summary = exp1.analyze(exp1_df)
    results['experiment1'] = exp1_summary
    print("  ✓ Complete")
    
    # Experiment 2
    print("\n[4/6] Running Experiment 2: Gate Localization...")
    exp2 = Experiment2(model, config)
    exp2_df = exp2.run(
        pos_examples=answerable,
        neg_examples=unanswerable,
        n_pairs=n_pairs,
        layer_stride=2 if args.mode == 'quick' else 1
    )
    exp2_summary = exp2.analyze(exp2_df)
    results['experiment2'] = exp2_summary
    
    # # Update target layers
    critical_layers = exp2_summary['critical_layers']
    if len(critical_layers) >= 4:
        config.target_layers = critical_layers[-4:]
    else:
        config.target_layers = critical_layers
    print(f"  ✓ Complete - Target layers: {config.target_layers}")
    
    # Experiment 3
    print("\n[5/6] Running Experiment 3: Steering Control...")
    exp3 = Experiment3(model, config)
    exp3_df = exp3.run(
        pos_examples=answerable,
        neg_examples=unanswerable,
        test_examples=ambiguous[:n_questions//2]
    )
    exp3_summary = exp3.analyze(exp3_df)
    results['experiment3'] = exp3_summary
    
    best_layer = exp3_summary['best_layer']
    steering_vectors = exp3.steering_vectors
    print(f"  ✓ Complete - Best layer: {best_layer}")
    
    # Experiment 4 (KEY)
#     print("\n[6/6] Running Experiment 4: Gate Independence ⭐...")
    # steering_vectors = torch.load(config.results_dir / "steering_vectors.pt")
    # steering_vectors = {int(k): v.to(config.device) for k, v in steering_vectors.items()}
#     exp4 = Experiment4(model, config, steering_vectors)

#     exp4 = Experiment4(model, config, steering_vectors)
#     exp4_df = exp4.run(
#     questions=all_questions,  # your full dataset
#     best_layer=24,
#     epsilon=8.0,
#     use_exp3_questions=True  # ← finds questions that changed in Exp3
# )
    exp4_df = exp4.run(
        questions=ambiguous[:20],
        best_layer=24,
        epsilon=12.0,  # Stronger steering
        reverse_steering=False
)
    exp4_summary = exp4.analyze(exp4_df)
    results['experiment4'] = exp4_summary
    print("  ✓ Complete - KEY FINDING ESTABLISHED")
    
    # Experiment 5
    print("\nRunning Experiment 5: Trustworthiness Application...")
    exp5 = Experiment5(model, config, steering_vectors)
    
    results_df = exp5.run(
        questions=test_set,
        test_layers=list(steering_vectors.keys()),
        epsilon_test=20.0,
        epsilon_values=[-50.0, -40.0, -30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0],
        n_probe=15,
    )
    
    summary = exp5.analyze(results_df)
    # Test all available layers to find the most responsive one
    # test_layers = list(steering_vectors.keys())
    # exp5_df = exp5.run(answerable, test_layers=test_layers, epsilon_test=20.0)
    
    # exp5_summary = exp5.analyze(exp5_df)
    # results['experiment5'] = exp5_summary
    # print("  ✓ Complete")
    # print("\nRunning Experiment 5: Trustworthiness Application...")
    # exp5 = Experiment5(model, config, steering_vectors)
    # exp5_df = exp5.run(answerable, 24)
    # exp5_summary = exp5.analyze(exp5_df)
    # results['experiment5'] = exp5_summary
    # print("  ✓ Complete")
    
    # Save results
    results['metadata'] = {
        'mode': args.mode,
        'model': args.model,
        'n_questions': n_questions,
        'n_pairs': n_pairs,
        'timestamp': datetime.now().isoformat(),
        'total_time_minutes': (time.time() - start_time) / 60
    }
    
      

    output_path = config.results_dir / f"master_summary_{args.mode}.json"
    with open(output_path, 'w') as f:
        safe_results = _json_sanitize(results)
        json.dump(safe_results, f, indent=2)
        
    output_path = config.results_dir / f"master_summary_{args.mode}.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print(" PIPELINE COMPLETE!")
    print("="*80)
    print(f"\nTotal time: {results['metadata']['total_time_minutes']:.1f} minutes")
    print(f"Results saved to: {output_path}")
    print("\n✓ All experiments completed successfully!\n")
    
    return results


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)