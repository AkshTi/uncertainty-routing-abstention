"""
Experiment 8: Scaling Analysis & Model Comparison

CRITICAL FOR ACCEPTANCE: Show findings generalize across model sizes/families

Addresses reviewer concern: "Only tested on 1.5B model, will it scale?"

This is THE experiment that will significantly boost your novelty and acceptance chances.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from tqdm import tqdm
from typing import List, Dict
import json

from core_utils import ModelWrapper, ExperimentConfig, extract_answer, set_seed


class Experiment8:
    """Test steering across different model sizes and families"""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results = []
        set_seed(config.seed)

    def test_model(self, model_name: str, test_questions: List[Dict],
                   target_layers: List[int] = None) -> Dict:
        """
        Test steering on a specific model

        Args:
            model_name: HuggingFace model name
            test_questions: Questions to test (answerable + unanswerable)
            target_layers: Which layers to test (None = auto-detect late layers)
        """
        print(f"\n{'='*70}")
        print(f"Testing model: {model_name}")
        print(f"{'='*70}")

        # Load model
        config = ExperimentConfig()
        config.model_name = model_name

        try:
            model_wrapper = ModelWrapper(config)
            n_layers = model_wrapper.model.config.num_hidden_layers
            hidden_dim = model_wrapper.model.config.hidden_size

            print(f"Loaded: {n_layers} layers, {hidden_dim} hidden dim")
        except Exception as e:
            print(f"ERROR loading model: {e}")
            return None

        # Auto-detect late layers if not specified
        if target_layers is None:
            # Test last 20% of layers
            start_layer = int(n_layers * 0.8)
            target_layers = list(range(start_layer, n_layers))
            print(f"Auto-selected layers: {target_layers}")

        # Extract steering vectors (quick version - just use last layer)
        print("\nExtracting steering vectors...")

        answerable = [q for q in test_questions if not q.get('is_unanswerable', False)][:10]
        unanswerable = [q for q in test_questions if q.get('is_unanswerable', False)][:10]

        steering_vectors = {}
        for layer_idx in target_layers[:3]:  # Just test top 3 layers for speed
            try:
                from experiment3_4_steering_independence import Experiment3

                # Format prompts
                from data_preparation import format_prompt
                pos_prompts = [format_prompt(q["question"], "neutral", q.get("context"))
                              for q in answerable]
                neg_prompts = [format_prompt(q["question"], "neutral", q.get("context"))
                              for q in unanswerable]

                exp3 = Experiment3(model_wrapper, config)
                direction = exp3.compute_steering_direction(pos_prompts, neg_prompts, layer_idx)
                steering_vectors[layer_idx] = direction

            except Exception as e:
                print(f"ERROR computing steering for layer {layer_idx}: {e}")
                continue

        if not steering_vectors:
            print("ERROR: Could not extract steering vectors")
            return None

        # Test steering effect
        print("\nTesting steering effect...")
        best_layer = max(steering_vectors.keys())  # Use deepest layer

        results = []
        for eps in [-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0]:
            for q in test_questions:
                try:
                    from experiment5_trustworthiness import Experiment5
                    exp5 = Experiment5(model_wrapper, config, steering_vectors)
                    result = exp5.test_one(q, best_layer, eps)
                    result['model'] = model_name
                    result['n_layers'] = n_layers
                    result['hidden_dim'] = hidden_dim
                    results.append(result)
                except Exception as e:
                    print(f"ERROR testing question: {e}")
                    continue

        if not results:
            print("ERROR: No results collected")
            return None

        df = pd.DataFrame(results)

        # Compute metrics
        metrics = []
        for eps in df['epsilon'].unique():
            sub = df[df['epsilon'] == eps]
            ans = sub[~sub['is_unanswerable']]
            unans = sub[sub['is_unanswerable']]

            coverage = 1.0 - ans['abstained'].mean() if len(ans) else 0.0
            abstain_unans = unans['abstained'].mean() if len(unans) else 0.0
            halluc_unans = unans['hallucinated'].mean() if len(unans) else 0.0

            metrics.append({
                'model': model_name,
                'n_layers': n_layers,
                'hidden_dim': hidden_dim,
                'epsilon': eps,
                'coverage_answerable': coverage,
                'abstain_unanswerable': abstain_unans,
                'hallucination_unanswerable': halluc_unans,
            })

        metrics_df = pd.DataFrame(metrics)

        # Compute steering effectiveness
        baseline = metrics_df[metrics_df['epsilon'] == 0.0].iloc[0]
        steered = metrics_df[metrics_df['epsilon'] == -30.0].iloc[0]  # Use strong negative steering

        delta_abstain = steered['abstain_unanswerable'] - baseline['abstain_unanswerable']
        delta_coverage = steered['coverage_answerable'] - baseline['coverage_answerable']

        print(f"\nResults for {model_name}:")
        print(f"  Baseline hallucination: {baseline['hallucination_unanswerable']:.1%}")
        print(f"  Steered hallucination: {steered['hallucination_unanswerable']:.1%}")
        print(f"  Δ abstention: {delta_abstain:+.1%}")
        print(f"  Δ coverage: {delta_coverage:+.1%}")

        # Cleanup
        del model_wrapper
        torch.cuda.empty_cache()

        return {
            'model_name': model_name,
            'n_layers': n_layers,
            'hidden_dim': hidden_dim,
            'steering_effective': abs(delta_abstain) > 0.10,  # At least 10% change
            'delta_abstain_unanswerable': float(delta_abstain),
            'delta_coverage_answerable': float(delta_coverage),
            'baseline_hallucination': float(baseline['hallucination_unanswerable']),
            'steered_hallucination': float(steered['hallucination_unanswerable']),
            'metrics_df': metrics_df,
        }

    def run_scaling_analysis(self, test_questions: List[Dict]) -> pd.DataFrame:
        """
        Test steering across multiple model sizes

        Models to test (in order of priority):
        1. Qwen2.5-1.5B-Instruct (already have)
        2. Qwen2.5-3B-Instruct (or Llama-3.2-3B if available)
        3. Qwen2.5-7B-Instruct (or Llama-3.1-8B if GPU memory allows)
        """
        print("\n" + "="*70)
        print("EXPERIMENT 8: SCALING ANALYSIS")
        print("="*70)

        # Models in order of priority (you can test as many as GPU memory allows)
        models_to_test = [
            "Qwen/Qwen2.5-1.5B-Instruct",  # Already tested
            "Qwen/Qwen2.5-3B-Instruct",    # Medium size
            # "Qwen/Qwen2.5-7B-Instruct",   # Uncomment if you have GPU memory
            # "meta-llama/Llama-3.2-3B-Instruct",  # Different family
        ]

        all_results = []

        for model_name in models_to_test:
            result = self.test_model(model_name, test_questions)
            if result:
                all_results.append(result)

                # Save individual result
                model_safe_name = model_name.replace('/', '_')
                result['metrics_df'].to_csv(
                    self.config.results_dir / f"exp8_{model_safe_name}_results.csv",
                    index=False
                )

        if not all_results:
            print("ERROR: No models successfully tested")
            return None

        # Combine results
        summary_df = pd.DataFrame([
            {
                'model': r['model_name'],
                'n_layers': r['n_layers'],
                'hidden_dim': r['hidden_dim'],
                'steering_works': r['steering_effective'],
                'delta_abstain': r['delta_abstain_unanswerable'],
                'delta_coverage': r['delta_coverage_answerable'],
                'baseline_halluc': r['baseline_hallucination'],
                'steered_halluc': r['steered_hallucination'],
            }
            for r in all_results
        ])

        summary_df.to_csv(self.config.results_dir / "exp8_scaling_summary.csv", index=False)

        return summary_df, all_results

    def analyze_scaling(self, summary_df: pd.DataFrame, all_results: List[Dict]) -> Dict:
        """Analyze scaling behavior"""
        print("\n" + "="*70)
        print("EXPERIMENT 8: SCALING ANALYSIS")
        print("="*70)

        print("\nModel Comparison:")
        print(summary_df.to_string(index=False))

        # Check if steering works across all models
        all_work = summary_df['steering_works'].all()

        if all_work:
            print("\n✓ SUCCESS: Steering works across all tested models!")
        else:
            failed = summary_df[~summary_df['steering_works']]['model'].tolist()
            print(f"\n⚠️  Steering failed on: {failed}")

        # Check for scaling trends
        if len(summary_df) >= 2:
            # Correlation between model size and steering effectiveness
            corr_params = summary_df['hidden_dim'].corr(summary_df['delta_abstain'].abs())
            print(f"\nScaling correlation (size vs effect): {corr_params:.3f}")

            if abs(corr_params) < 0.3:
                print("✓ Steering effectiveness independent of model size!")
            else:
                print("⚠️  Steering effectiveness may scale with model size")

        # Visualize
        self._plot_scaling(summary_df, all_results)

        return {
            'models_tested': summary_df['model'].tolist(),
            'all_successful': bool(all_work),
            'scaling_correlation': float(corr_params) if len(summary_df) >= 2 else None,
            'summary': summary_df.to_dict('records'),
        }

    def _plot_scaling(self, summary_df: pd.DataFrame, all_results: List[Dict]):
        """Create scaling visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Panel 1: Steering effectiveness by model
        model_labels = [m.split('/')[-1].replace('-Instruct', '') for m in summary_df['model']]

        axes[0, 0].bar(range(len(model_labels)), summary_df['delta_abstain'].abs(),
                      color=['green' if x else 'red' for x in summary_df['steering_works']],
                      alpha=0.7, edgecolor='black')
        axes[0, 0].set_xticks(range(len(model_labels)))
        axes[0, 0].set_xticklabels(model_labels, rotation=45, ha='right')
        axes[0, 0].set_ylabel("Abstention Change (absolute)")
        axes[0, 0].set_title("Steering Effectiveness by Model", fontweight='bold')
        axes[0, 0].axhline(0.10, color='blue', linestyle='--', alpha=0.5,
                          label='Threshold (10%)')
        axes[0, 0].legend()
        axes[0, 0].grid(axis='y', alpha=0.3)

        # Panel 2: Tradeoff comparison
        axes[0, 1].scatter(summary_df['delta_coverage'], summary_df['delta_abstain'],
                          s=100, alpha=0.7, edgecolors='black')

        for i, model in enumerate(model_labels):
            axes[0, 1].annotate(model,
                               (summary_df.iloc[i]['delta_coverage'],
                                summary_df.iloc[i]['delta_abstain']),
                               fontsize=9, ha='center')

        axes[0, 1].axhline(0, color='gray', linestyle='-', alpha=0.3)
        axes[0, 1].axvline(0, color='gray', linestyle='-', alpha=0.3)
        axes[0, 1].set_xlabel("Δ Coverage (answerable)")
        axes[0, 1].set_ylabel("Δ Abstention (unanswerable)")
        axes[0, 1].set_title("Risk-Coverage Tradeoff by Model", fontweight='bold')
        axes[0, 1].grid(alpha=0.3)

        # Panel 3: Hallucination reduction
        x = np.arange(len(model_labels))
        width = 0.35

        axes[1, 0].bar(x - width/2, summary_df['baseline_halluc'], width,
                      label='Baseline', alpha=0.7, color='red', edgecolor='black')
        axes[1, 0].bar(x + width/2, summary_df['steered_halluc'], width,
                      label='Steered', alpha=0.7, color='green', edgecolor='black')

        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(model_labels, rotation=45, ha='right')
        axes[1, 0].set_ylabel("Hallucination Rate")
        axes[1, 0].set_title("Hallucination Reduction by Model", fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].set_ylim([0, 1])
        axes[1, 0].grid(axis='y', alpha=0.3)

        # Panel 4: Full epsilon sweep (if we have multiple models)
        if len(all_results) > 0:
            for result in all_results:
                model_name = result['model_name'].split('/')[-1].replace('-Instruct', '')
                metrics_df = result['metrics_df']

                axes[1, 1].plot(metrics_df['epsilon'],
                               metrics_df['hallucination_unanswerable'],
                               marker='o', label=model_name, linewidth=2)

            axes[1, 1].axvline(0, color='gray', linestyle='--', alpha=0.5)
            axes[1, 1].set_xlabel("Epsilon")
            axes[1, 1].set_ylabel("Hallucination Rate (unanswerable)")
            axes[1, 1].set_title("Steering Curves by Model", fontweight='bold')
            axes[1, 1].legend()
            axes[1, 1].grid(alpha=0.3)
            axes[1, 1].set_ylim([0, 1])

        plt.tight_layout()

        output_path = self.config.results_dir / "exp8_scaling_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Figure saved to {output_path}")
        plt.close()


def main():
    """Run scaling analysis"""
    config = ExperimentConfig()

    # Load test questions
    print("Loading test questions...")
    with open("./data/dataset_clearly_answerable.json", 'r') as f:
        answerable = json.load(f)
    with open("./data/dataset_clearly_unanswerable.json", 'r') as f:
        unanswerable = json.load(f)

    # Use subset for speed
    test_questions = [
        {**q, "is_unanswerable": False} for q in answerable[:15]
    ] + [
        {**q, "is_unanswerable": True, "answer": None} for q in unanswerable[:15]
    ]

    print(f"Test set: {len(test_questions)} questions")

    # Run scaling analysis
    exp8 = Experiment8(config)
    summary_df, all_results = exp8.run_scaling_analysis(test_questions)

    if summary_df is not None:
        # Analyze
        analysis = exp8.analyze_scaling(summary_df, all_results)

        # Save
        with open(config.results_dir / "exp8_summary.json", 'w') as f:
            json.dump(analysis, f, indent=2)

        print("\n✓ Experiment 8 complete!")


if __name__ == "__main__":
    main()
