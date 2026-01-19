"""
Experiment 2: Gate Localization via Activation Patching
Save as: experiment2_localization.py

Identifies which layers causally control the abstention decision
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from tqdm import tqdm
from typing import List, Dict, Tuple
import json

from core_utils import (
    ModelWrapper, ExperimentConfig, extract_answer,
    get_decision_token_position, compute_binary_margin_simple, set_seed
)
from data_preparation import format_prompt


class Experiment2:
    """Activation Patching to Localize Decision Control"""
    
    def __init__(self, model_wrapper: ModelWrapper, config: ExperimentConfig):
        self.model = model_wrapper
        self.config = config
        self.results = []
        set_seed(config.seed)
        self.n_layers = self.model.model.config.num_hidden_layers
    
    def get_baseline_margin(self, prompt: str) -> float:
        """Get baseline answerable/unanswerable tendency"""
        self.model.clear_hooks()
        response = self.model.generate(prompt, temperature=0.0, do_sample=False)
        return compute_binary_margin_simple(response)
    
    def cache_activation(self, prompt: str, layer_idx: int) -> torch.Tensor:
        """Cache activation at specific layer for a prompt"""
        self.model.clear_hooks()
        
        # Get decision token position
        position = get_decision_token_position(self.model.tokenizer, prompt)
        
        # Register hook to cache
        self.model.register_cache_hook(layer_idx, position)
        
        # Run forward pass
        _ = self.model.generate(prompt, temperature=0.0, do_sample=False)
        
        # Extract cached activation
        activation = self.model.activation_cache[f"layer_{layer_idx}"].clone()
        
        self.model.clear_hooks()
        return activation
    
    def patch_and_measure(self, target_prompt: str, donor_activation: torch.Tensor,
                         layer_idx: int) -> float:
        """Patch donor activation into target and measure result"""
        self.model.clear_hooks()
        
        # Get decision position for target
        position = get_decision_token_position(self.model.tokenizer, target_prompt)
        
        # Register patching hook
        self.model.register_patch_hook(layer_idx, position, donor_activation)
        
        # Generate with patch
        response = self.model.generate(target_prompt, temperature=0.0, do_sample=False)
        margin = compute_binary_margin_simple(response)
        
        self.model.clear_hooks()
        return margin
    
    def test_single_layer_patch(self, pos_prompt: str, neg_prompt: str,
                                layer_idx: int) -> Dict:
        """
        Test patching at single layer:
        1. Cache activation from POS (answerable) example
        2. Patch into NEG (unanswerable) example
        3. Measure effect
        """
        # Get baseline for NEG
        baseline_margin = self.get_baseline_margin(neg_prompt)
        
        # Cache POS activation
        pos_activation = self.cache_activation(pos_prompt, layer_idx)
        
        # Patch into NEG and measure
        patched_margin = self.patch_and_measure(neg_prompt, pos_activation, layer_idx)
        
        # Calculate effect
        delta = patched_margin - baseline_margin
        flipped = (baseline_margin < 0 and patched_margin > 0)
        
        return {
            "layer": layer_idx,
            "baseline_margin": float(baseline_margin),
            "patched_margin": float(patched_margin),
            "delta_margin": float(delta),
            "flipped": flipped
        }
    
    def run(self, pos_examples: List[Dict], neg_examples: List[Dict],
            n_pairs: int = 10, layer_stride: int = 1) -> pd.DataFrame:
        """
        Run activation patching across all layers
        
        Args:
            pos_examples: Questions model tends to answer
            neg_examples: Questions model tends to abstain on
            n_pairs: Number of example pairs to test
            layer_stride: Test every Nth layer (1 = all layers, 2 = every other, etc.)
        """
        print("\n" + "="*60)
        print("EXPERIMENT 2: Gate Localization")
        print("="*60)
        print(f"Testing {n_pairs} example pairs")
        print(f"Across {self.n_layers // layer_stride} layers (stride={layer_stride})")
        print()
        
        # Prepare example pairs
        pairs = self._prepare_pairs(pos_examples, neg_examples, n_pairs)
        
        # Test each layer
        layers_to_test = range(0, self.n_layers, layer_stride)
        
        for layer_idx in tqdm(layers_to_test, desc="Layers"):
            for pair_idx, (pos_prompt, neg_prompt, pos_q, neg_q) in enumerate(pairs):
                try:
                    result = self.test_single_layer_patch(pos_prompt, neg_prompt, layer_idx)
                    
                    # Add metadata
                    result.update({
                        "pair_idx": pair_idx,
                        "pos_question": pos_q["question"][:50],
                        "neg_question": neg_q["question"][:50]
                    })
                    
                    self.results.append(result)
                    
                except Exception as e:
                    print(f"\nError at layer {layer_idx}, pair {pair_idx}: {e}")
                    continue
        
        df = pd.DataFrame(self.results)
        
        # Save results
        output_path = self.config.results_dir / "exp2_raw_results.csv"
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")
        
        return df
    
    def _prepare_pairs(self, pos_examples: List[Dict], neg_examples: List[Dict],
                      n_pairs: int) -> List[Tuple]:
        """Prepare (POS, NEG) prompt pairs"""
        pairs = []
        
        n_available = min(len(pos_examples), len(neg_examples), n_pairs)
        
        for i in range(n_available):
            pos_q = pos_examples[i]
            neg_q = neg_examples[i]
            
            pos_prompt = format_prompt(
                pos_q["question"],
                "neutral",
                pos_q.get("context")
            )
            
            neg_prompt = format_prompt(
                neg_q["question"],
                "neutral",
                neg_q.get("context")
            )
            
            pairs.append((pos_prompt, neg_prompt, pos_q, neg_q))
        
        return pairs
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        """Analyze patching results"""
        print("\n" + "="*60)
        print("EXPERIMENT 2: ANALYSIS")
        print("="*60)
        
        # Compute layer-wise statistics
        layer_stats = df.groupby("layer").agg({
            "delta_margin": ["mean", "std", "count"],
            "flipped": "mean"
        }).round(3)
        
        print("\nLayer-wise Patching Effects:")
        print(layer_stats)
        
        # Find critical layers (high positive delta)
        mean_deltas = df.groupby("layer")["delta_margin"].mean()
        significant_threshold = mean_deltas.quantile(0.75)
        
        critical_layers = mean_deltas[mean_deltas > significant_threshold].index.tolist()
        
        print(f"\nCritical Layers (delta > {significant_threshold:.3f}):")
        print(critical_layers)
        
        # Check if late layers dominate
        late_start = int(self.n_layers * 0.6)  # Last 40% of layers
        late_layers = [l for l in critical_layers if l >= late_start]
        
        print(f"\nLate-layer concentration:")
        print(f"  Critical layers after layer {late_start}: {len(late_layers)}/{len(critical_layers)}")
        print(f"  Proportion: {len(late_layers)/max(len(critical_layers), 1):.1%}")
        
        # Find layer with maximum effect
        max_effect_layer = mean_deltas.idxmax()
        max_effect_value = mean_deltas.max()
        
        print(f"\nMaximum effect at layer {max_effect_layer}: Δ={max_effect_value:.3f}")
        
        # Flip rate analysis
        flip_rates = df.groupby("layer")["flipped"].mean()
        max_flip_layer = flip_rates.idxmax()
        max_flip_rate = flip_rates.max()
        
        print(f"Maximum flip rate at layer {max_flip_layer}: {max_flip_rate:.1%}")
        
        # Visualize
        self.plot_results(df, critical_layers)
        
        return {
            "layer_stats": layer_stats.to_dict(),
            "critical_layers": critical_layers,
            "max_effect_layer": int(max_effect_layer),
            "max_effect_value": float(max_effect_value),
            "max_flip_layer": int(max_flip_layer),
            "max_flip_rate": float(max_flip_rate),
            "late_layer_proportion": len(late_layers)/max(len(critical_layers), 1)
        }
    
    def plot_results(self, df: pd.DataFrame, critical_layers: List[int]):
        """Create visualization"""
        
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Mean delta margin by layer
        layer_means = df.groupby("layer")["delta_margin"].mean()
        layer_stds = df.groupby("layer")["delta_margin"].std()
        
        axes[0, 0].plot(layer_means.index, layer_means.values, 
                       marker='o', linewidth=2, markersize=4, color='#3498db')
        axes[0, 0].fill_between(layer_means.index,
                                layer_means - layer_stds,
                                layer_means + layer_stds,
                                alpha=0.3, color='#3498db')
        axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        # Highlight critical layers
        for layer in critical_layers:
            axes[0, 0].axvline(x=layer, color='orange', alpha=0.3, linewidth=1)
        
        axes[0, 0].set_xlabel("Layer", fontsize=11)
        axes[0, 0].set_ylabel("Δ Margin (POS→NEG)", fontsize=11)
        axes[0, 0].set_title("Patching Effect by Layer", fontsize=12, fontweight='bold')
        axes[0, 0].grid(alpha=0.3)
        
        # Plot 2: Flip rate by layer
        flip_rates = df.groupby("layer")["flipped"].mean()
        
        axes[0, 1].plot(flip_rates.index, flip_rates.values,
                       marker='o', linewidth=2, markersize=4, color='#e74c3c')
        axes[0, 1].set_xlabel("Layer", fontsize=11)
        axes[0, 1].set_ylabel("Flip Rate (NEG→POS)", fontsize=11)
        axes[0, 1].set_title("Decision Flip Rate by Layer", fontsize=12, fontweight='bold')
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].grid(alpha=0.3)
        
        # Highlight critical layers
        for layer in critical_layers:
            axes[0, 1].axvline(x=layer, color='orange', alpha=0.3, linewidth=1)
        
        # Plot 3: Distribution of effects in early vs late layers
        late_start = int(self.n_layers * 0.6)
        df['layer_group'] = df['layer'].apply(
            lambda x: 'Late' if x >= late_start else 'Early'
        )
        
        sns.boxplot(data=df, x='layer_group', y='delta_margin', ax=axes[1, 0],
                   palette=['#95a5a6', '#e74c3c'])
        axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel("Layer Group", fontsize=11)
        axes[1, 0].set_ylabel("Δ Margin", fontsize=11)
        axes[1, 0].set_title("Early vs Late Layer Effects", fontsize=12, fontweight='bold')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Plot 4: Cumulative effect
        cumsum = layer_means.cumsum()
        axes[1, 1].plot(cumsum.index, cumsum.values,
                       linewidth=2, color='#9b59b6')
        axes[1, 1].set_xlabel("Layer", fontsize=11)
        axes[1, 1].set_ylabel("Cumulative Δ Margin", fontsize=11)
        axes[1, 1].set_title("Cumulative Causal Effect", fontsize=12, fontweight='bold')
        axes[1, 1].grid(alpha=0.3)
        
        # Add vertical line at late_start
        axes[1, 1].axvline(x=late_start, color='orange', linestyle='--',
                          alpha=0.7, label=f'Late layers (>{late_start})')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        # Save
        output_path = self.config.results_dir / "exp2_localization_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to {output_path}")
        plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    """Run Experiment 2"""
    import json
    
    # Setup
    config = ExperimentConfig()
    
    print("Initializing model...")
    model = ModelWrapper(config)
    
    # Load data
    print("\nLoading datasets...")
    with open("./data/dataset_clearly_answerable.json", 'r') as f:
        answerable = json.load(f)
    with open("./data/dataset_clearly_unanswerable.json", 'r') as f:
        unanswerable = json.load(f)
    
    # Run experiment
    # Use layer_stride=2 for faster testing (tests every other layer)
    exp2 = Experiment2(model, config)
    results_df = exp2.run(
        pos_examples=answerable[:10],
        neg_examples=unanswerable[:10],
        n_pairs=10,
        layer_stride=1  # Set to 2 or 3 for faster runs
    )
    
    # Analyze
    summary = exp2.analyze(results_df)
    
    # Save summary
    summary_path = config.results_dir / "exp2_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Experiment 2 complete!")
    print(f"  Results: {config.results_dir / 'exp2_raw_results.csv'}")
    print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()
