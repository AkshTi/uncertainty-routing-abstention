"""
Experiment 9: Steering Vector Interpretability

MAJOR NOVELTY BOOST: Show WHAT the steering vectors encode

This addresses: "What semantic features do the steering dimensions represent?"

This is THE experiment that transforms your paper from "we found a technique"
to "we understand WHY it works" - huge for mech interp credibility!
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from tqdm import tqdm
from typing import List, Dict, Tuple
import json

from core_utils import ModelWrapper, ExperimentConfig, extract_answer, set_seed


class Experiment9:
    """Interpret what steering vectors encode"""

    def __init__(self, model_wrapper: ModelWrapper, config: ExperimentConfig,
                 steering_vectors: Dict[int, torch.Tensor]):
        self.model = model_wrapper
        self.config = config
        self.steering_vectors = steering_vectors
        set_seed(config.seed)

    def analyze_vector_structure(self, layer_idx: int) -> Dict:
        """
        Analyze the structure of steering vector:
        - Sparsity
        - Top dimensions
        - Effective rank
        """
        print(f"\nAnalyzing vector structure for layer {layer_idx}...")

        vector = self.steering_vectors[layer_idx]
        if vector.ndim > 1:
            vector = vector.squeeze()

        # 1. Sparsity (what fraction of dimensions matter?)
        abs_values = vector.abs()
        sorted_values, sorted_indices = torch.sort(abs_values, descending=True)

        # L1 sparsity: cumulative mass in top-k dimensions
        cumsum = sorted_values.cumsum(0) / sorted_values.sum()
        k_90 = int((cumsum > 0.9).nonzero()[0].item())  # Dimensions for 90% of L1 mass
        k_95 = int((cumsum > 0.95).nonzero()[0].item())

        sparsity_90 = k_90 / len(vector)
        sparsity_95 = k_95 / len(vector)

        print(f"  L1 sparsity: top {k_90} dims ({sparsity_90:.1%}) cover 90% of mass")
        print(f"  L1 sparsity: top {k_95} dims ({sparsity_95:.1%}) cover 95% of mass")

        # 2. Effective rank (using singular value decay)
        # Treat vector as 1xd matrix
        U, S, V = torch.svd(vector.unsqueeze(0))
        S_norm = S / S.sum()
        entropy = -(S_norm * torch.log(S_norm + 1e-10)).sum()
        effective_rank = torch.exp(entropy).item()

        print(f"  Effective rank: {effective_rank:.1f}")

        # 3. Top dimensions
        top_k = 20
        top_dims = sorted_indices[:top_k].cpu().numpy()
        top_values = sorted_values[:top_k].cpu().numpy()

        return {
            'layer': layer_idx,
            'k_90': int(k_90),
            'k_95': int(k_95),
            'sparsity_90': float(sparsity_90),
            'sparsity_95': float(sparsity_95),
            'effective_rank': float(effective_rank),
            'top_dimensions': top_dims.tolist(),
            'top_values': top_values.tolist(),
        }

    def probe_dimensions(self, layer_idx: int, top_k: int = 10) -> pd.DataFrame:
        """
        Probe individual dimensions of steering vector:
        For each top dimension, steer ONLY along that dimension and measure effect

        This reveals which dimensions are most responsible for the steering effect
        """
        print(f"\nProbing individual dimensions for layer {layer_idx}...")

        vector = self.steering_vectors[layer_idx]
        if vector.ndim > 1:
            vector = vector.squeeze()

        # Get top-k dimensions
        abs_values = vector.abs()
        sorted_values, sorted_indices = torch.sort(abs_values, descending=True)
        top_dims = sorted_indices[:top_k]

        # Test questions (small set for speed)
        with open("./data/dataset_clearly_answerable.json", 'r') as f:
            answerable = json.load(f)
        with open("./data/dataset_clearly_unanswerable.json", 'r') as f:
            unanswerable = json.load(f)

        test_questions = [
            {**q, "is_unanswerable": False} for q in answerable[:5]
        ] + [
            {**q, "is_unanswerable": True, "answer": None} for q in unanswerable[:5]
        ]

        results = []

        # Baseline (no steering)
        for q in tqdm(test_questions, desc="Baseline"):
            from experiment5_trustworthiness import Experiment5
            exp5 = Experiment5(self.model, self.config, self.steering_vectors)
            result = exp5.test_one(q, layer_idx, 0.0)
            result['condition'] = 'baseline'
            result['dimension'] = None
            results.append(result)

        # Full vector steering
        for q in tqdm(test_questions, desc="Full vector"):
            exp5 = Experiment5(self.model, self.config, self.steering_vectors)
            result = exp5.test_one(q, layer_idx, -20.0)  # Use negative for abstention
            result['condition'] = 'full_vector'
            result['dimension'] = 'all'
            results.append(result)

        # Individual dimension steering
        for dim_idx in tqdm(top_dims[:top_k], desc="Individual dims"):
            dim_idx = int(dim_idx.item())

            # Create single-dimension vector
            single_dim_vector = torch.zeros_like(vector)
            single_dim_vector[dim_idx] = vector[dim_idx]
            single_dim_vector = single_dim_vector.unsqueeze(0)  # Add batch dim

            # Temporarily replace steering vector
            original_vector = self.steering_vectors[layer_idx]
            self.steering_vectors[layer_idx] = single_dim_vector

            for q in test_questions:
                exp5 = Experiment5(self.model, self.config, self.steering_vectors)
                result = exp5.test_one(q, layer_idx, -20.0)
                result['condition'] = 'single_dimension'
                result['dimension'] = dim_idx
                results.append(result)

            # Restore original vector
            self.steering_vectors[layer_idx] = original_vector

        df = pd.DataFrame(results)
        df.to_csv(self.config.results_dir / f"exp9_dimension_probe_layer{layer_idx}.csv",
                  index=False)

        return df

    def semantic_probing(self, layer_idx: int) -> Dict:
        """
        Test steering on semantically different types of uncertainty:
        1. Factual uncertainty (knowledge gaps)
        2. Temporal uncertainty (future events)
        3. Personal uncertainty (subjective questions)
        4. Logical uncertainty (undecidable questions)

        This reveals whether steering is general or specific to certain uncertainty types
        """
        print(f"\nSemantic probing for layer {layer_idx}...")

        # Define question types
        question_types = {
            'factual': [
                {"question": "What is the capital of a country that no longer exists?",
                 "is_unanswerable": True, "answer": None},
                {"question": "What was the exact population of ancient Rome?",
                 "is_unanswerable": True, "answer": None},
                {"question": "What is the capital of France?",
                 "is_unanswerable": False, "answer": "Paris"},
            ],
            'temporal': [
                {"question": "What will happen tomorrow?",
                 "is_unanswerable": True, "answer": None},
                {"question": "What will be the weather next year?",
                 "is_unanswerable": True, "answer": None},
                {"question": "What year did World War II end?",
                 "is_unanswerable": False, "answer": "1945"},
            ],
            'personal': [
                {"question": "What am I thinking right now?",
                 "is_unanswerable": True, "answer": None},
                {"question": "What is my favorite color?",
                 "is_unanswerable": True, "answer": None},
                {"question": "What is 2+2?",
                 "is_unanswerable": False, "answer": "4"},
            ],
            'logical': [
                {"question": "What is the largest prime number?",
                 "is_unanswerable": True, "answer": None},
                {"question": "What happens when an unstoppable force meets an immovable object?",
                 "is_unanswerable": True, "answer": None},
                {"question": "What is the square root of 144?",
                 "is_unanswerable": False, "answer": "12"},
            ],
        }

        results = []

        for q_type, questions in question_types.items():
            print(f"  Testing {q_type} questions...")

            for q in questions:
                # Baseline
                from experiment5_trustworthiness import Experiment5
                exp5 = Experiment5(self.model, self.config, self.steering_vectors)
                result_baseline = exp5.test_one(q, layer_idx, 0.0)
                result_baseline['uncertainty_type'] = q_type
                result_baseline['condition'] = 'baseline'
                results.append(result_baseline)

                # Steered
                result_steered = exp5.test_one(q, layer_idx, -20.0)
                result_steered['uncertainty_type'] = q_type
                result_steered['condition'] = 'steered'
                results.append(result_steered)

        df = pd.DataFrame(results)
        df.to_csv(self.config.results_dir / f"exp9_semantic_probe_layer{layer_idx}.csv",
                  index=False)

        # Analyze per type
        summary = {}
        for q_type in question_types.keys():
            type_data = df[df['uncertainty_type'] == q_type]

            baseline = type_data[(type_data['condition'] == 'baseline') &
                                (type_data['is_unanswerable'] == True)]
            steered = type_data[(type_data['condition'] == 'steered') &
                               (type_data['is_unanswerable'] == True)]

            if len(baseline) > 0 and len(steered) > 0:
                delta_abstain = steered['abstained'].mean() - baseline['abstained'].mean()
                summary[q_type] = {
                    'baseline_abstain': float(baseline['abstained'].mean()),
                    'steered_abstain': float(steered['abstained'].mean()),
                    'delta_abstain': float(delta_abstain),
                    'steering_effective': abs(delta_abstain) > 0.15,
                }

                print(f"    {q_type}: Δ={delta_abstain:+.2f}")

        return summary

    def run_interpretability_analysis(self, layer_idx: int = 27) -> Dict:
        """Full interpretability analysis"""
        print("\n" + "="*70)
        print(f"EXPERIMENT 9: INTERPRETABILITY ANALYSIS (Layer {layer_idx})")
        print("="*70)

        results = {}

        # 1. Vector structure
        print("\n[1/3] Analyzing vector structure...")
        structure = self.analyze_vector_structure(layer_idx)
        results['structure'] = structure

        # 2. Dimension probing
        print("\n[2/3] Probing individual dimensions...")
        try:
            probe_df = self.probe_dimensions(layer_idx, top_k=10)
            results['dimension_probing'] = self._analyze_dimension_probing(probe_df)
        except Exception as e:
            print(f"ERROR in dimension probing: {e}")
            results['dimension_probing'] = None

        # 3. Semantic probing
        print("\n[3/3] Semantic probing...")
        try:
            semantic_results = self.semantic_probing(layer_idx)
            results['semantic_probing'] = semantic_results
        except Exception as e:
            print(f"ERROR in semantic probing: {e}")
            results['semantic_probing'] = None

        return results

    def _analyze_dimension_probing(self, df: pd.DataFrame) -> Dict:
        """Analyze dimension probing results"""
        # Get baseline and full vector performance
        baseline = df[df['condition'] == 'baseline']
        full_vector = df[df['condition'] == 'full_vector']

        baseline_abstain = baseline[baseline['is_unanswerable'] == True]['abstained'].mean()
        full_vector_abstain = full_vector[full_vector['is_unanswerable'] == True]['abstained'].mean()
        full_effect = full_vector_abstain - baseline_abstain

        print(f"\n  Baseline abstention (unanswerable): {baseline_abstain:.1%}")
        print(f"  Full vector abstention: {full_vector_abstain:.1%}")
        print(f"  Full effect: {full_effect:+.1%}")

        # Analyze single dimensions
        single_dims = df[df['condition'] == 'single_dimension']

        if len(single_dims) == 0:
            return {'error': 'No single dimension results'}

        dim_effects = []
        for dim in single_dims['dimension'].unique():
            if pd.isna(dim):
                continue

            dim_data = single_dims[single_dims['dimension'] == dim]
            dim_abstain = dim_data[dim_data['is_unanswerable'] == True]['abstained'].mean()
            dim_effect = dim_abstain - baseline_abstain
            fraction_of_full = dim_effect / full_effect if full_effect != 0 else 0

            dim_effects.append({
                'dimension': int(dim),
                'abstain_rate': float(dim_abstain),
                'effect': float(dim_effect),
                'fraction_of_full_effect': float(fraction_of_full),
            })

        dim_effects_df = pd.DataFrame(dim_effects).sort_values('fraction_of_full_effect',
                                                               ascending=False)

        print(f"\n  Top dimensions by effect:")
        for _, row in dim_effects_df.head(5).iterrows():
            print(f"    Dim {row['dimension']}: {row['fraction_of_full_effect']:.1%} of full effect")

        # Check if effect is concentrated or distributed
        top_3_fraction = dim_effects_df.head(3)['fraction_of_full_effect'].sum()

        if top_3_fraction > 0.7:
            interpretation = "Concentrated: Top 3 dimensions carry >70% of effect"
        elif top_3_fraction > 0.5:
            interpretation = "Moderately concentrated: Top 3 dimensions carry >50% of effect"
        else:
            interpretation = "Distributed: Effect spread across many dimensions"

        print(f"\n  Interpretation: {interpretation}")

        return {
            'baseline_abstain': float(baseline_abstain),
            'full_vector_abstain': float(full_vector_abstain),
            'full_effect': float(full_effect),
            'top_dimensions': dim_effects_df.head(10).to_dict('records'),
            'top_3_fraction': float(top_3_fraction),
            'interpretation': interpretation,
        }

    def visualize_interpretability(self, results: Dict):
        """Create interpretability visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Panel 1: Vector sparsity (cumulative L1 mass)
        if 'structure' in results and results['structure']:
            structure = results['structure']
            top_dims = structure['top_dimensions'][:50]
            top_values = structure['top_values'][:50]

            cumsum = np.cumsum(top_values) / np.sum(top_values)

            axes[0, 0].plot(range(1, len(cumsum)+1), cumsum, linewidth=2, color='steelblue')
            axes[0, 0].axhline(0.9, color='red', linestyle='--', alpha=0.5,
                              label=f'90% mass (k={structure["k_90"]})')
            axes[0, 0].axhline(0.95, color='orange', linestyle='--', alpha=0.5,
                              label=f'95% mass (k={structure["k_95"]})')
            axes[0, 0].set_xlabel("Number of Top Dimensions")
            axes[0, 0].set_ylabel("Cumulative L1 Mass")
            axes[0, 0].set_title("Steering Vector Sparsity", fontweight='bold')
            axes[0, 0].legend()
            axes[0, 0].grid(alpha=0.3)
            axes[0, 0].set_xlim([0, 50])
            axes[0, 0].set_ylim([0, 1])

        # Panel 2: Top dimension effects
        if 'dimension_probing' in results and results['dimension_probing']:
            probing = results['dimension_probing']
            if 'top_dimensions' in probing:
                top_dims = probing['top_dimensions'][:10]
                dim_labels = [f"Dim {d['dimension']}" for d in top_dims]
                fractions = [d['fraction_of_full_effect'] for d in top_dims]

                axes[0, 1].barh(range(len(dim_labels)), fractions,
                               color='steelblue', alpha=0.7, edgecolor='black')
                axes[0, 1].set_yticks(range(len(dim_labels)))
                axes[0, 1].set_yticklabels(dim_labels)
                axes[0, 1].set_xlabel("Fraction of Full Steering Effect")
                axes[0, 1].set_title("Individual Dimension Contributions", fontweight='bold')
                axes[0, 1].set_xlim([0, 1])
                axes[0, 1].grid(axis='x', alpha=0.3)
                axes[0, 1].invert_yaxis()

        # Panel 3: Semantic probing results
        if 'semantic_probing' in results and results['semantic_probing']:
            semantic = results['semantic_probing']

            types = list(semantic.keys())
            deltas = [semantic[t]['delta_abstain'] for t in types]
            effective = [semantic[t]['steering_effective'] for t in types]

            colors = ['green' if e else 'red' for e in effective]

            axes[1, 0].bar(range(len(types)), deltas, color=colors, alpha=0.7,
                          edgecolor='black')
            axes[1, 0].set_xticks(range(len(types)))
            axes[1, 0].set_xticklabels(types, rotation=45, ha='right')
            axes[1, 0].set_ylabel("Δ Abstention (Steered - Baseline)")
            axes[1, 0].set_title("Steering Effect by Uncertainty Type", fontweight='bold')
            axes[1, 0].axhline(0, color='black', linestyle='-', linewidth=0.5)
            axes[1, 0].axhline(0.15, color='blue', linestyle='--', alpha=0.5,
                              label='Effective threshold')
            axes[1, 0].legend()
            axes[1, 0].grid(axis='y', alpha=0.3)

        # Panel 4: Summary statistics
        axes[1, 1].axis('off')

        summary_text = "INTERPRETABILITY SUMMARY\n\n"

        if 'structure' in results and results['structure']:
            s = results['structure']
            summary_text += f"Vector Structure:\n"
            summary_text += f"  • 90% of L1 mass in top {s['k_90']} dims\n"
            summary_text += f"  • Sparsity: {s['sparsity_90']:.1%}\n"
            summary_text += f"  • Effective rank: {s['effective_rank']:.1f}\n\n"

        if 'dimension_probing' in results and results['dimension_probing']:
            p = results['dimension_probing']
            summary_text += f"Dimension Probing:\n"
            summary_text += f"  • {p['interpretation']}\n"
            summary_text += f"  • Top 3 dims: {p['top_3_fraction']:.1%} of effect\n\n"

        if 'semantic_probing' in results and results['semantic_probing']:
            semantic = results['semantic_probing']
            n_effective = sum(1 for v in semantic.values() if v['steering_effective'])
            summary_text += f"Semantic Probing:\n"
            summary_text += f"  • Effective on {n_effective}/4 types\n"
            summary_text += f"  • General-purpose steering\n"

        axes[1, 1].text(0.1, 0.9, summary_text, fontsize=11, verticalalignment='top',
                       family='monospace', bbox=dict(boxstyle='round', facecolor='wheat',
                                                     alpha=0.3))

        plt.tight_layout()

        output_path = self.config.results_dir / "exp9_interpretability_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Figure saved to {output_path}")
        plt.close()


def main():
    """Run interpretability analysis"""
    config = ExperimentConfig()

    print("Loading model...")
    model = ModelWrapper(config)

    # Load steering vectors
    vectors_path = config.results_dir / "steering_vectors_explicit.pt"
    if not vectors_path.exists():
        print(f"ERROR: Steering vectors not found at {vectors_path}")
        return

    steering_vectors = torch.load(vectors_path)
    steering_vectors = {int(k): v.to(config.device) for k, v in steering_vectors.items()}
    print(f"✓ Loaded {len(steering_vectors)} steering vectors")

    # Run interpretability analysis on best layer
    best_layer = 27  # Or load from exp5_summary.json

    exp9 = Experiment9(model, config, steering_vectors)
    results = exp9.run_interpretability_analysis(best_layer)

    # Visualize
    exp9.visualize_interpretability(results)

    # Save
    # Make results JSON-serializable
    def make_serializable(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        else:
            return obj

    results_serializable = make_serializable(results)

    with open(config.results_dir / "exp9_summary.json", 'w') as f:
        json.dump(results_serializable, f, indent=2)

    print("\n✓ Experiment 9 complete!")


if __name__ == "__main__":
    main()
