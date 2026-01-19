"""
Experiment 3: Low-Dimensional Steering Control (FIXED)
Experiment 4: Gate Independence from Uncertainty

Save as: experiment3_4_steering_independence.py

FIXES:
1. Proper steering application during generation
2. Correct measurement of effects
3. Robust margin computation
4. Better position tracking
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


# ============================================================================
# EXPERIMENT 3: Steering Control (FIXED)
# ============================================================================

def _get_blocks(hf_model):
    """
    Return the list/ModuleList of transformer blocks for many HF architectures.
    """
    # Qwen2 / Llama-style: model.model.layers
    if hasattr(hf_model, "model") and hasattr(hf_model.model, "layers"):
        return hf_model.model.layers

    # Some models: model.layers directly
    if hasattr(hf_model, "layers"):
        return hf_model.layers

    # GPT-2 style: transformer.h
    if hasattr(hf_model, "transformer") and hasattr(hf_model.transformer, "h"):
        return hf_model.transformer.h

    # GPT-NeoX style: gpt_neox.layers
    if hasattr(hf_model, "gpt_neox") and hasattr(hf_model.gpt_neox, "layers"):
        return hf_model.gpt_neox.layers

    # MPT style: transformer.blocks
    if hasattr(hf_model, "transformer") and hasattr(hf_model.transformer, "blocks"):
        return hf_model.transformer.blocks

    raise AttributeError(
        f"Don't know where transformer blocks are for model class {type(hf_model)}"
    )

def _get_blocks(hf_model):
    """
    Return the list/ModuleList of transformer blocks for many HF architectures.
    """
    # Qwen2 / Llama-style: model.model.layers
    if hasattr(hf_model, "model") and hasattr(hf_model.model, "layers"):
        return hf_model.model.layers

    # Some models: model.layers directly
    if hasattr(hf_model, "layers"):
        return hf_model.layers

    # GPT-2 style: transformer.h
    if hasattr(hf_model, "transformer") and hasattr(hf_model.transformer, "h"):
        return hf_model.transformer.h

    # GPT-NeoX style: gpt_neox.layers
    if hasattr(hf_model, "gpt_neox") and hasattr(hf_model.gpt_neox, "layers"):
        return hf_model.gpt_neox.layers

    # MPT style: transformer.blocks
    if hasattr(hf_model, "transformer") and hasattr(hf_model.transformer, "blocks"):
        return hf_model.transformer.blocks

    raise AttributeError(
        f"Don't know where transformer blocks are for model class {type(hf_model)}"
    )

class Experiment3:
    """Identify and test low-dimensional steering direction"""
    
    def __init__(self, model_wrapper: ModelWrapper, config: ExperimentConfig):
        self.model = model_wrapper
        self.config = config
        self.steering_vectors = {}
        self.results = []
        set_seed(config.seed)
    
    def compute_steering_direction(self, pos_prompts: List[str],
                                   neg_prompts: List[str],
                                   layer_idx: int) -> torch.Tensor:
        """
        Compute answerability direction as mean difference between
        POS (answerable) and NEG (unanswerable) activations
        """
        print(f"  Computing steering direction for layer {layer_idx}...")
        
        self.model.clear_hooks()
        pos_activations = []
        neg_activations = []
        
        # Collect POS activations
        for prompt in pos_prompts:
            # Get position BEFORE the actual answer starts
            inputs = self.model.tokenizer(prompt, return_tensors="pt").to(self.model.config.device)
            prompt_length = inputs["input_ids"].shape[1]
            # Use last prompt token position (before generation)
            position = prompt_length - 1
            
            self.model.register_cache_hook(layer_idx, position)
            _ = self.model.generate(prompt, temperature=0.0, do_sample=False, max_new_tokens=1)
            
            activation = self.model.activation_cache[f"layer_{layer_idx}"].clone()
            pos_activations.append(activation)
            self.model.clear_hooks()
        
        # Collect NEG activations
        for prompt in neg_prompts:
            inputs = self.model.tokenizer(prompt, return_tensors="pt").to(self.model.config.device)
            prompt_length = inputs["input_ids"].shape[1]
            position = prompt_length - 1
            
            self.model.register_cache_hook(layer_idx, position)
            _ = self.model.generate(prompt, temperature=0.0, do_sample=False, max_new_tokens=1)
            
            activation = self.model.activation_cache[f"layer_{layer_idx}"].clone()
            neg_activations.append(activation)
            self.model.clear_hooks()
        
        # Compute mean difference
        pos_mean = torch.stack(pos_activations).mean(dim=0)
        neg_mean = torch.stack(neg_activations).mean(dim=0)
        
        direction = pos_mean - neg_mean
        
        # Normalize
        direction = direction / (direction.norm() + 1e-8)
        
        print(f"    Direction norm before normalization: {(pos_mean - neg_mean).norm():.4f}")
        print(f"    POS activation norm: {pos_mean.norm():.4f}")
        print(f"    NEG activation norm: {neg_mean.norm():.4f}")
        
        return direction.to(self.model.config.device)
    
    def test_steering_robust(self, prompt: str, layer_idx: int,
                            steering_vector: torch.Tensor, epsilon: float) -> Dict:
        """
        Apply steering and measure effect - ROBUST VERSION
        
        Key fix: Apply steering to ALL tokens during generation, not just one position
        """
        self.model.clear_hooks()
        
        # Baseline generation (no steering)
        baseline_response = self.model.generate(
            prompt, 
            temperature=0.0, 
            do_sample=False, 
            max_new_tokens=50
        )
        baseline_answer = extract_answer(baseline_response)
        baseline_margin = compute_binary_margin_simple(baseline_response)
        
        if epsilon == 0:
            # No steering, return baseline
            return {
                "baseline_margin": float(baseline_margin),
                "steered_margin": float(baseline_margin),
                "delta_margin": 0.0,
                "flipped": False,
                "baseline_response": baseline_response[:100],
                "steered_response": baseline_response[:100],
                "baseline_answer": baseline_answer,
                "steered_answer": baseline_answer
            }
        
        # Steered generation - apply to all generated positions
        self.model.clear_hooks()
        
        # Get prompt length
        inputs = self.model.tokenizer(prompt, return_tensors="pt").to(self.model.config.device)
        prompt_length = inputs["input_ids"].shape[1]
        
        # Register hook that applies steering to all positions during generation
        def steering_hook_all_positions(module, input, output):
            # output may be Tensor or tuple
            if isinstance(output, tuple):
                hidden_states = output[0]
                rest = output[1:]
            else:
                hidden_states = output
                rest = None
        
            hidden_states = hidden_states.clone()
        
            # steering_vector is shape [1, d]; make it [d] for clean broadcast
            sv = steering_vector.to(hidden_states.device)
            if sv.ndim == 2 and sv.shape[0] == 1:
                sv = sv[0]
        
            hidden_states[:, -1, :] += epsilon * sv
        
            if rest is None:
                return hidden_states
            return (hidden_states,) + rest
    
            
        layer = _get_blocks(self.model.model)[layer_idx]
        handle = layer.register_forward_hook(steering_hook_all_positions)
        self.model.hooks.append(handle)
        
        # Generate with steering
        steered_response = self.model.generate(
            prompt,
            temperature=0.0,
            do_sample=False,
            max_new_tokens=50
        )
        steered_answer = extract_answer(steered_response)
        steered_margin = compute_binary_margin_simple(steered_response)
        
        self.model.clear_hooks()
        
        # Check if decision flipped
        flipped = (baseline_margin > 0 and steered_margin < 0) or \
                 (baseline_margin < 0 and steered_margin > 0)
        
        # Also check if answer changed
        answer_changed = (baseline_answer != steered_answer)
        
        return {
            "baseline_margin": float(baseline_margin),
            "steered_margin": float(steered_margin),
            "delta_margin": float(steered_margin - baseline_margin),
            "flipped": flipped,
            "answer_changed": answer_changed,
            "baseline_response": baseline_response[:100],
            "steered_response": steered_response[:100],
            "baseline_answer": baseline_answer,
            "steered_answer": steered_answer
        }

    def run(self, pos_examples: List[Dict], neg_examples: List[Dict],
            test_examples: List[Dict]) -> pd.DataFrame:
        """Run steering experiments"""
        print("\n" + "="*60)
        print("EXPERIMENT 3: Steering Control (FIXED)")
        print("="*60)
        
        # Format prompts for computing direction
        pos_prompts = [format_prompt(ex["question"], "neutral", ex.get("context"))
                      for ex in pos_examples[:10]]
        neg_prompts = [format_prompt(ex["question"], "neutral", ex.get("context"))
                      for ex in neg_examples[:10]]
        
        # Compute steering directions for target layers
        print("\nComputing steering directions...")
        for layer_idx in self.config.target_layers:
            direction = self.compute_steering_direction(pos_prompts, neg_prompts, layer_idx)
            self.steering_vectors[layer_idx] = direction
            print(f"  Layer {layer_idx}: direction shape = {direction.shape}")
        
        # Test steering across epsilons
        print("\nTesting steering effects...")
        for layer_idx in tqdm(self.config.target_layers, desc="Layers"):
            steering_vec = self.steering_vectors[layer_idx]
            
            for epsilon in self.config.steering_epsilon_range:
                for ex in test_examples:
                    prompt = format_prompt(ex["question"], "neutral", ex.get("context"))
                    
                    try:
                        result = self.test_steering_robust(prompt, layer_idx, steering_vec, epsilon)
                        
                        self.results.append({
                            "layer": layer_idx,
                            "epsilon": epsilon,
                            "question": ex["question"][:50],
                            "true_answerability": ex.get("answerability", "unknown"),
                            **result
                        })
                    except Exception as e:
                        print(f"\nError on layer {layer_idx}, epsilon {epsilon}: {e}")
                        continue
        
        torch.save(
            {k: v.detach().cpu() for k, v in self.steering_vectors.items()},
            self.config.results_dir / "steering_vectors.pt"
        )
        print("Saved steering vectors to results/steering_vectors.pt")

        df = pd.DataFrame(self.results)
        
        # Diagnostic output
        print("\n=== STEERING DIAGNOSTIC ===")
        print(f"Total tests: {len(df)}")
        print(f"Non-zero deltas: {(df['delta_margin'] != 0).sum()}")
        print(f"Flips: {df['flipped'].sum()}")
        print(f"Answer changes: {df['answer_changed'].sum()}")
        
        # Show example of steering working
        changed = df[df['answer_changed'] == True]
        if len(changed) > 0:
            print("\nExample of steering effect:")
            example = changed.iloc[0]
            print(f"  Layer: {example['layer']}, Epsilon: {example['epsilon']}")
            print(f"  Question: {example['question']}")
            print(f"  Baseline answer: {example['baseline_answer']}")
            print(f"  Steered answer: {example['steered_answer']}")
            print(f"  Delta margin: {example['delta_margin']:.3f}")
        
        # Save
        output_path = self.config.results_dir / "exp3_raw_results.csv"
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")
        
        return df
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        """Analyze steering results - using answer changes as primary metric"""
        print("\n" + "="*60)
        print("EXPERIMENT 3: ANALYSIS")
        print("="*60)
        
        # Primary metric: Answer changes (more reliable than margin)
        has_effects = df['answer_changed'].any()
        
        if not has_effects:
            print("\n⚠️  WARNING: No steering effects detected!")
            return {
                "error": "No steering effects detected",
                "best_layer": self.config.target_layers[-1]  # Default fallback
            }
        
        print(f"\n✓ Steering is working: {df['answer_changed'].sum()} answer changes detected")
        
        # Change rates by epsilon and layer
        change_summary = df.pivot_table(
            index='layer',
            columns='epsilon',
            values='answer_changed',
            aggfunc='mean'
        )
        
        print("\nAnswer Change Rates by Layer and Epsilon:")
        print(change_summary.round(3))
        
        # Find most effective layer (highest change rate at max epsilon)
        max_eps = df[df['epsilon'] > 0]['epsilon'].max()
        if max_eps in change_summary.columns:
            best_layer_idx = change_summary[max_eps].idxmax()
            best_layer_effect = change_summary.loc[best_layer_idx, max_eps]
        else:
            # Fallback: use overall change rate
            layer_effects = df.groupby('layer')['answer_changed'].mean()
            best_layer_idx = layer_effects.idxmax()
            best_layer_effect = layer_effects.max()
        
        print(f"\nMost effective layer: {best_layer_idx} (change rate={best_layer_effect:.1%} at ε={max_eps})")
        
        # Test monotonicity
        print("\nEpsilon-effect correlations by layer:")
        for layer in self.config.target_layers:
            layer_data = df[df["layer"] == layer]
            if len(layer_data) > 0:
                eps_means = layer_data.groupby("epsilon")["answer_changed"].mean()
                if len(eps_means) > 1:
                    corr = eps_means.corr(pd.Series(eps_means.index))
                    print(f"  Layer {layer}: r = {corr:.3f}")
        
        # Visualize
        self.plot_results_fixed(df, change_summary)
        
        return {
            "change_rates": change_summary.to_dict(),
            "best_layer": int(best_layer_idx),
            "best_layer_effect": float(best_layer_effect),
            "total_answer_changes": int(df['answer_changed'].sum())
        }

    
    def plot_results_fixed(self, df: pd.DataFrame, change_summary):
        """Create visualizations"""
        
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Flip rate vs epsilon for each layer
        for layer in self.config.target_layers:
            layer_data = df[df["layer"] == layer]
            flip_by_eps = layer_data.groupby("epsilon")["flipped"].mean()
            
            if len(flip_by_eps) > 0:
                axes[0, 0].plot(flip_by_eps.index, flip_by_eps.values,
                              marker='o', label=f'Layer {layer}', linewidth=2)
        
        axes[0, 0].set_xlabel("Epsilon (ε)", fontsize=11)
        axes[0, 0].set_ylabel("Flip Rate", fontsize=11)
        axes[0, 0].set_title("Steering Flip Rate vs Epsilon", fontsize=12, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        axes[0, 0].set_ylim([0, 1])
        
        # Plot 2: Mean shift vs epsilon
        for layer in self.config.target_layers:
            layer_data = df[df["layer"] == layer]
            shift_by_eps = layer_data.groupby("epsilon")["delta_margin"].mean()
            
            if len(shift_by_eps) > 0:
                axes[0, 1].plot(shift_by_eps.index, shift_by_eps.values,
                              marker='o', label=f'Layer {layer}', linewidth=2)
        
        axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[0, 1].set_xlabel("Epsilon (ε)", fontsize=11)
        axes[0, 1].set_ylabel("Mean Δ Margin", fontsize=11)
        axes[0, 1].set_title("Steering Effect Magnitude", fontsize=12, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # Plot 3: Heatmap of flip rates (if we have data)
        if not change_summary.empty and change_summary.notna().any().any():
            heat = change_summary.copy()

            # Coerce every cell to numeric; non-numeric becomes NaN
            heat = heat.apply(pd.to_numeric, errors="coerce")
            
            # Optional: fill NaNs so heatmap doesn't choke / looks nicer
            heat = heat.fillna(0.0)
            
            # sns.heatmap(heat, annot=True, fmt=".2f", ax=axes[...])

            sns.heatmap(heat, annot=True, fmt='.2f', cmap='YlOrRd',
                      ax=axes[1, 0], cbar_kws={'label': 'Flip Rate'})
            axes[1, 0].set_xlabel("Epsilon (ε)", fontsize=11)
            axes[1, 0].set_ylabel("Layer", fontsize=11)
            axes[1, 0].set_title("Flip Rate Heatmap", fontsize=12, fontweight='bold')
        else:
            axes[1, 0].text(0.5, 0.5, 'No flip data available',
                           ha='center', va='center', fontsize=14)
            axes[1, 0].set_title("Flip Rate Heatmap", fontsize=12, fontweight='bold')
        
        # Plot 4: Distribution of effects at max epsilon
        max_eps = df[df['epsilon'] > 0]['epsilon'].max()
        max_eps_data = df[df["epsilon"] == max_eps]
        
        if len(max_eps_data) > 0:
            sns.violinplot(data=max_eps_data, x='layer', y='delta_margin',
                          ax=axes[1, 1], palette='Set2')
            axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            axes[1, 1].set_xlabel("Layer", fontsize=11)
            axes[1, 1].set_ylabel("Δ Margin", fontsize=11)
            axes[1, 1].set_title(f"Effect Distribution at ε={max_eps}", fontsize=12, fontweight='bold')
            axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        output_path = self.config.results_dir / "exp3_steering_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to {output_path}")
        plt.close()

    def plot_results(self, df: pd.DataFrame, flip_summary, shift_summary):
        """Create visualizations"""
        
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Flip rate vs epsilon for each layer
        for layer in self.config.target_layers:
            layer_data = df[df["layer"] == layer]
            flip_by_eps = layer_data.groupby("epsilon")["flipped"].mean()
            
            if len(flip_by_eps) > 0:
                axes[0, 0].plot(flip_by_eps.index, flip_by_eps.values,
                              marker='o', label=f'Layer {layer}', linewidth=2)
        
        axes[0, 0].set_xlabel("Epsilon (ε)", fontsize=11)
        axes[0, 0].set_ylabel("Flip Rate", fontsize=11)
        axes[0, 0].set_title("Steering Flip Rate vs Epsilon", fontsize=12, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        axes[0, 0].set_ylim([0, 1])
        
        # Plot 2: Mean shift vs epsilon
        for layer in self.config.target_layers:
            layer_data = df[df["layer"] == layer]
            shift_by_eps = layer_data.groupby("epsilon")["delta_margin"].mean()
            
            if len(shift_by_eps) > 0:
                axes[0, 1].plot(shift_by_eps.index, shift_by_eps.values,
                              marker='o', label=f'Layer {layer}', linewidth=2)
        
        axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[0, 1].set_xlabel("Epsilon (ε)", fontsize=11)
        axes[0, 1].set_ylabel("Mean Δ Margin", fontsize=11)
        axes[0, 1].set_title("Steering Effect Magnitude", fontsize=12, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # Plot 3: Heatmap of flip rates (if we have data)
        if not flip_summary.empty and flip_summary.notna().any().any():
            sns.heatmap(flip_summary, annot=True, fmt='.2f', cmap='YlOrRd',
                       ax=axes[1, 0], cbar_kws={'label': 'Flip Rate'})
            axes[1, 0].set_xlabel("Epsilon (ε)", fontsize=11)
            axes[1, 0].set_ylabel("Layer", fontsize=11)
            axes[1, 0].set_title("Flip Rate Heatmap", fontsize=12, fontweight='bold')
        else:
            axes[1, 0].text(0.5, 0.5, 'No flip data available',
                           ha='center', va='center', fontsize=14)
            axes[1, 0].set_title("Flip Rate Heatmap", fontsize=12, fontweight='bold')
        
        # Plot 4: Distribution of effects at max epsilon
        max_eps = df[df['epsilon'] > 0]['epsilon'].max()
        max_eps_data = df[df["epsilon"] == max_eps]
        
        if len(max_eps_data) > 0:
            sns.violinplot(data=max_eps_data, x='layer', y='delta_margin',
                          ax=axes[1, 1], palette='Set2')
            axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            axes[1, 1].set_xlabel("Layer", fontsize=11)
            axes[1, 1].set_ylabel("Δ Margin", fontsize=11)
            axes[1, 1].set_title(f"Effect Distribution at ε={max_eps}", fontsize=12, fontweight='bold')
            axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        output_path = self.config.results_dir / "exp3_steering_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to {output_path}")
        plt.close()


# ============================================================================
# EXPERIMENT 4: Gate Independence (UNCHANGED - Already works)
# ============================================================================
import re
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# ----------------------------
# Helpers for Exp4 (NEW)
# ----------------------------

def _exp4_prompt(question: str, context: str = None) -> str:
    """
    Strict, machine-parseable prompt that makes abstention observable.
    Forces model to output: ANSWER: YES/NO/UNCERTAIN
    """
    ctx = ""
    if context:
        ctx = f"\nContext: {context}\n"

    return (
        "You are answering a question.\n"
        "If you are not confident in a YES/NO answer, you MUST answer UNCERTAIN.\n"
        "Output EXACTLY one line in this format:\n"
        "ANSWER: YES\n"
        "or\n"
        "ANSWER: NO\n"
        "or\n"
        "ANSWER: UNCERTAIN\n"
        f"{ctx}"
        f"Question: {question}\n"
        "ANSWER:"
    )

def _parse_yes_no_uncertain(text: str) -> str:
    """
    Robust parsing: returns 'YES', 'NO', or 'UNCERTAIN' (default UNCERTAIN if ambiguous).
    """
    if text is None:
        return "UNCERTAIN"

    # Look for a line like "ANSWER: ..."
    m = re.search(r"(?im)^\s*ANSWER\s*:\s*([A-Z]+)\s*$", text)
    if m:
        tok = m.group(1).strip().upper()
        if tok in ("YES", "NO", "UNCERTAIN"):
            return tok

    # Fallback heuristics if formatting is imperfect
    t = text.lower()
    if "uncertain" in t or "not sure" in t or "i don't know" in t:
        return "UNCERTAIN"

    # If it contains a clear yes/no token early
    if re.search(r"(?i)\b(yes)\b", text[:80]):
        return "YES"
    if re.search(r"(?i)\b(no)\b", text[:80]):
        return "NO"

    return "UNCERTAIN"

def _get_model_device(hf_model) -> torch.device:
    return next(hf_model.parameters()).device

def _apply_steering_hook(layer_module, steering_vec: torch.Tensor, epsilon: float, hook_handles_list: list):
    """
    Attach a forward hook that adds epsilon * steering_vec to the last position.
    Handles output being Tensor or tuple; handles hidden_states being 2D or 3D.
    """
    def hook_fn(module, inputs, output):
        # output may be Tensor or tuple
        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            rest = None

        # If hidden_states is not a tensor, bail
        if not torch.is_tensor(hidden_states):
            return output

        hs = hidden_states.clone()

        sv = steering_vec.to(hs.device)
        if sv.ndim == 2 and sv.shape[0] == 1:
            sv = sv[0]  # [D]

        # hs can be [B, T, D] or [T, D] depending on HF internals / checkpointing
        if hs.ndim == 3:
            hs[:, -1, :] = hs[:, -1, :] + epsilon * sv
        elif hs.ndim == 2:
            hs[-1, :] = hs[-1, :] + epsilon * sv
        else:
            return output

        if rest is None:
            return hs
        return (hs,) + rest

    handle = layer_module.register_forward_hook(hook_fn)
    hook_handles_list.append(handle)

class Experiment4:
    """
    EXPERIMENT 4: Gate Independence from Uncertainty
    
    GOAL: Show that steering controls the ABSTENTION GATE independently of:
    1. Internal uncertainty (how dispersed the model's beliefs are)
    2. Actual knowledge (whether the question is answerable)
    
    We do this by:
    - Testing on questions that SHOULD trigger abstention (impossible/ambiguous)
    - Showing steering can FORCE answers even when model is uncertain
    - Testing on questions that SHOULDN'T trigger abstention (easy factual)
    - Showing steering can FORCE abstention even when model is confident
    """

    def __init__(self, model_wrapper: ModelWrapper, config: ExperimentConfig,
                 steering_vectors: Dict[int, torch.Tensor]):
        self.model = model_wrapper
        self.config = config
        self.steering_vectors = steering_vectors
        self.results = []
        set_seed(config.seed)

    # -----------------------------
    # Prompts
    # -----------------------------
    def _allow_abstention_prompt(self, question: str, context: str = None) -> str:
        """
        Prompt that allows abstention but doesn't force it.
        Key: Make it a PREFERENCE not a hard rule, so steering can override.
        """
        base = format_prompt(question, "neutral", context) if context else f"Question: {question}"
        
        # Softer version - suggests abstention as an option, doesn't demand it
        return (
            base + "\n\n"
            "If unsure, you may respond: UNCERTAIN\n"
            "Answer:"
        )

    def _force_guess_prompt(self, question: str, context: str = None) -> str:
        """For measuring internal uncertainty - must give an answer."""
        base = format_prompt(question, "neutral", context) if context else f"Question: {question}"
        return base + "\n\nYou MUST give an answer (do not say you don't know):\n"

    # -----------------------------
    # Internal uncertainty measurement
    # -----------------------------
    def measure_internal_uncertainty(self, question: str, context: str = None,
                                    n_samples: int = 20) -> Dict:
        """
        Sample forced guesses to measure how uncertain the model is internally.
        High uncertainty = model gives different answers when sampled.
        """
        print(f"    Measuring uncertainty: {question[:40]}...")
        prompt = self._force_guess_prompt(question, context)

        answers = []
        for i in range(n_samples):
            resp = self.model.generate(prompt, temperature=1.0, do_sample=True, max_new_tokens=30)
            ans = extract_answer(resp)[:50]  # First 50 chars
            answers.append(ans)

        from collections import Counter
        counts = Counter(answers)
        n = len(answers)
        probs = np.array([c / n for c in counts.values()], dtype=np.float64)

        entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
        p_max = float(np.max(probs)) if len(probs) > 0 else 0.0

        print(f"      Top answers: {counts.most_common(2)}")
        print(f"      Entropy={entropy:.2f}, p_max={p_max:.2f}")

        return {
            "entropy": entropy,
            "p_max": p_max,
            "n_unique": len(counts),
        }

    def _bin_uncertainty(self, p_max: float) -> str:
        """Bin by concentration of probability mass."""
        if p_max >= 0.65:
            return "low"       # Confident (13/20+ agree)
        elif p_max >= 0.45:
            return "medium"    # Moderate (9/20+ agree)
        else:
            return "high"      # Uncertain (dispersed)

    # -----------------------------
    # Abstention testing with steering
    # -----------------------------
    def test_abstention_with_steering(self, question: str, context: str,
                                     layer_idx: int, steering_vector: torch.Tensor,
                                     epsilon: float) -> Dict:
        """
        Test whether model abstains, with and without steering.
        
        Returns whether model said "UNCERTAIN" in each condition.
        """
        prompt = self._allow_abstention_prompt(question, context)
        
        # BASELINE (no steering)
        self.model.clear_hooks()
        baseline_resp = self.model.generate(
            prompt, 
            temperature=0.0, 
            do_sample=False, 
            max_new_tokens=50
        )
        baseline_ans = extract_answer(baseline_resp)
        baseline_abstained = ("UNCERTAIN" in baseline_resp.upper()[:50]) or (baseline_ans == "UNCERTAIN")
        
        print(f"      Baseline: {'ABSTAINED' if baseline_abstained else 'ANSWERED'}")
        print(f"        Response: {baseline_resp[:80]}")
        
        # STEERED
        self.model.clear_hooks()
        
        def steering_hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
                rest = output[1:]
            else:
                hidden_states = output
                rest = None
        
            hidden_states = hidden_states.clone()
        
            sv = steering_vector.to(hidden_states.device)
            if sv.ndim == 2 and sv.shape[0] == 1:
                sv = sv[0]
        
            # Apply steering to last token
            hidden_states[:, -1, :] += epsilon * sv
        
            if rest is None:
                return hidden_states
            return (hidden_states,) + rest
        
        layer = _get_blocks(self.model.model)[layer_idx]
        handle = layer.register_forward_hook(steering_hook)
        self.model.hooks.append(handle)
        
        steered_resp = self.model.generate(
            prompt,
            temperature=0.0,
            do_sample=False,
            max_new_tokens=50
        )
        steered_ans = extract_answer(steered_resp)
        steered_abstained = ("UNCERTAIN" in steered_resp.upper()[:50]) or (steered_ans == "UNCERTAIN")
        
        self.model.clear_hooks()
        
        print(f"      Steered:  {'ABSTAINED' if steered_abstained else 'ANSWERED'}")
        print(f"        Response: {steered_resp[:80]}")
        
        # Check if behavior changed
        behavior_flipped = (baseline_abstained != steered_abstained)
        
        if behavior_flipped:
            if baseline_abstained:
                print(f"      → GATE OPENED: Forced to answer when uncertain!")
            else:
                print(f"      → GATE CLOSED: Forced to abstain when confident!")
        
        return {
            "baseline_abstained": baseline_abstained,
            "steered_abstained": steered_abstained,
            "behavior_flipped": behavior_flipped,
            "baseline_response": baseline_resp[:200],
            "steered_response": steered_resp[:200],
            "baseline_answer": baseline_ans,
            "steered_answer": steered_ans,
        }

    # -----------------------------
    # Run
    # -----------------------------
    def run(self, questions: List[Dict], best_layer: int,
            epsilon: float = 8.0,
            n_uncert_samples: int = 20,
            reverse_steering: bool = False) -> pd.DataFrame:
        """
        Run Exp4: Show steering controls abstention gate independently of uncertainty.
        
        Args:
            reverse_steering: If True, flip the steering direction (-epsilon instead of +epsilon).
                            Use this if your vector pushes toward answering but you want abstention.
        """
        print("\n" + "="*70)
        print("EXPERIMENT 4: Gate Independence from Uncertainty")
        print("="*70)
        print("Goal: Show steering controls ABSTENTION independently of uncertainty")
        print(f"Parameters: layer={best_layer}, ε={epsilon}, samples={n_uncert_samples}")
        print(f"Reverse steering: {reverse_steering}")
        print(f"Questions: {len(questions)}\n")

        # Load steering vector
        if best_layer in self.steering_vectors:
            steering_vec = self.steering_vectors[best_layer]
        elif str(best_layer) in self.steering_vectors:
            steering_vec = self.steering_vectors[str(best_layer)]
        else:
            raise KeyError(f"No steering vector for layer {best_layer}")
        
        steering_vec = steering_vec.to(self.config.device)
        print(f"Loaded steering vector: shape={steering_vec.shape}, norm={torch.norm(steering_vec).item():.4f}\n")

        self.results = []

        # PHASE 1: Measure internal uncertainty
        print("="*70)
        print("PHASE 1: Measuring Internal Uncertainty")
        print("="*70)
        print("(Does the model have consistent beliefs when sampled?)\n")
        
        enriched = []
        for i, q_data in enumerate(questions, 1):
            print(f"  [{i}/{len(questions)}] {q_data['question'][:60]}")
            q = q_data["question"]
            ctx = q_data.get("context")

            uncert = self.measure_internal_uncertainty(q, ctx, n_samples=n_uncert_samples)
            ubin = self._bin_uncertainty(uncert["p_max"])
            
            enriched.append({
                **q_data,
                "internal_entropy": uncert["entropy"],
                "p_max": uncert["p_max"],
                "n_unique_answers": uncert["n_unique"],
                "uncertainty_bin": ubin,
            })

        # Summary
        print("\n" + "="*70)
        print("PHASE 1 SUMMARY: Uncertainty Distribution")
        print("="*70)
        from collections import Counter
        bin_dist = Counter(q["uncertainty_bin"] for q in enriched)
        for bin_name in ["low", "medium", "high"]:
            count = bin_dist.get(bin_name, 0)
            pct = 100 * count / len(enriched) if enriched else 0
            print(f"  {bin_name.upper()} uncertainty: {count} questions ({pct:.0f}%)")

        # PHASE 2: Test abstention with steering
        print("\n" + "="*70)
        print("PHASE 2: Testing Abstention Gate Control")
        print("="*70)
        print("(Can steering override the model's natural abstention behavior?)\n")
        
        for i, q_data in enumerate(enriched, 1):
            print(f"  [{i}/{len(enriched)}] Uncertainty: {q_data['uncertainty_bin'].upper()}")
            print(f"    Q: {q_data['question'][:60]}")
            
            q = q_data["question"]
            ctx = q_data.get("context")

            result = self.test_abstention_with_steering(
                q, ctx, best_layer, steering_vec, 
                epsilon if not reverse_steering else -epsilon
            )

            self.results.append({
                "question": q[:80],
                "expected_abstention": q_data.get("should_abstain", None),
                "internal_entropy": float(q_data["internal_entropy"]),
                "p_max": float(q_data["p_max"]),
                "n_unique_answers": int(q_data["n_unique_answers"]),
                "uncertainty_bin": q_data["uncertainty_bin"],
                "baseline_abstained": result["baseline_abstained"],
                "steered_abstained": result["steered_abstained"],
                "behavior_flipped": result["behavior_flipped"],
                "layer": int(best_layer),
                "epsilon": float(epsilon),
                "baseline_response": result["baseline_response"],
                "steered_response": result["steered_response"],
            })

        df = pd.DataFrame(self.results)

        # PHASE 2 Summary
        print("\n" + "="*70)
        print("PHASE 2 SUMMARY: Steering Effects")
        print("="*70)
        
        n_baseline_abstain = df["baseline_abstained"].sum()
        n_steered_abstain = df["steered_abstained"].sum()
        n_flips = df["behavior_flipped"].sum()
        
        print(f"\nBaseline abstentions: {n_baseline_abstain}/{len(df)} ({100*n_baseline_abstain/len(df):.0f}%)")
        print(f"Steered abstentions:  {n_steered_abstain}/{len(df)} ({100*n_steered_abstain/len(df):.0f}%)")
        print(f"Behavior flips:       {n_flips}/{len(df)} ({100*n_flips/len(df):.0f}%)")

        # Key transitions
        abstain_to_answer = df[
            (df["baseline_abstained"] == True) &
            (df["steered_abstained"] == False)
        ]
        answer_to_abstain = df[
            (df["baseline_abstained"] == False) &
            (df["steered_abstained"] == True)
        ]
        
        print(f"\n  Gate OPENED (abstain → answer): {len(abstain_to_answer)}")
        print(f"  Gate CLOSED (answer → abstain): {len(answer_to_abstain)}")
        
        # By uncertainty bin
        print("\nBehavior flips by uncertainty bin:")
        for bin_name in ["low", "medium", "high"]:
            bin_df = df[df["uncertainty_bin"] == bin_name]
            if len(bin_df) == 0:
                continue
            flips = bin_df["behavior_flipped"].sum()
            pct = 100 * flips / len(bin_df)
            print(f"  {bin_name.upper()}: {flips}/{len(bin_df)} flipped ({pct:.0f}%)")

        # Save
        output_path = self.config.results_dir / "exp4_raw_results.csv"
        df.to_csv(output_path, index=False)
        print(f"\n✓ Results saved to {output_path}")

        return df

    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Analyze: Does steering control abstention independently of uncertainty?
        
        KEY RESULT: If steering flips behavior across ALL uncertainty bins,
        then the gate is independent of the model's confidence.
        """
        print("\n" + "="*70)
        print("EXPERIMENT 4: ANALYSIS")
        print("="*70)

        # 1. Overall statistics
        print("\n1. OVERALL GATE CONTROL")
        print("-" * 40)
        total = len(df)
        n_flips = df["behavior_flipped"].sum()
        
        n_baseline_abstain = df["baseline_abstained"].sum()
        n_steered_abstain = df["steered_abstained"].sum()
        
        print(f"Total questions: {total}")
        print(f"Baseline abstentions: {n_baseline_abstain} ({100*n_baseline_abstain/total:.0f}%)")
        print(f"Steered abstentions: {n_steered_abstain} ({100*n_steered_abstain/total:.0f}%)")
        print(f"Behavior flips: {n_flips} ({100*n_flips/total:.0f}%)")

        # 2. Gate control by uncertainty level
        print("\n2. GATE CONTROL BY UNCERTAINTY BIN")
        print("-" * 40)
        print("(Key result: Flips should occur across ALL bins)\n")
        
        flip_by_bin = {}
        for bin_name in ["low", "medium", "high"]:
            bin_df = df[df["uncertainty_bin"] == bin_name]
            if len(bin_df) == 0:
                flip_by_bin[bin_name] = 0.0
                continue
            
            n_flips_bin = bin_df["behavior_flipped"].sum()
            rate = n_flips_bin / len(bin_df)
            flip_by_bin[bin_name] = float(rate)
            
            # Show transitions
            open_gate = bin_df[(bin_df["baseline_abstained"]) & (~bin_df["steered_abstained"])].shape[0]
            close_gate = bin_df[(~bin_df["baseline_abstained"]) & (bin_df["steered_abstained"])].shape[0]
            
            print(f"  {bin_name.upper()} uncertainty ({len(bin_df)} questions):")
            print(f"    Flips: {n_flips_bin}/{len(bin_df)} ({100*rate:.0f}%)")
            print(f"    - Opened gate (forced answer): {open_gate}")
            print(f"    - Closed gate (forced abstain): {close_gate}")

        # 3. Independence test
        print("\n3. INDEPENDENCE TEST")
        print("-" * 40)
        rates = [v for v in flip_by_bin.values() if v > 0]
        if len(rates) >= 2:
            rate_mean = np.mean(rates)
            rate_std = np.std(rates)
            cv = rate_std / rate_mean if rate_mean > 0 else float('inf')
            
            print(f"Mean flip rate across bins: {rate_mean:.2%}")
            print(f"Standard deviation: {rate_std:.3f}")
            print(f"Coefficient of variation: {cv:.3f}")
            
            if cv < 0.3:
                print("\n✓ LOW VARIANCE → Gate control is INDEPENDENT of uncertainty!")
                print("  Steering can override abstention regardless of model confidence.")
            else:
                print("\n⚠️  HIGH VARIANCE → Gate may depend on uncertainty level")
        else:
            print("Need multiple uncertainty bins to assess independence")

        # 4. Show example transitions
        print("\n4. EXAMPLE TRANSITIONS")
        print("-" * 40)
        
        # High uncertainty: forced to answer
        high_forced = df[(df["uncertainty_bin"] == "high") & 
                        (df["baseline_abstained"]) & 
                        (~df["steered_abstained"])]
        if len(high_forced) > 0:
            ex = high_forced.iloc[0]
            print(f"\nHIGH UNCERTAINTY → FORCED ANSWER:")
            print(f"  Q: {ex['question']}")
            print(f"  Baseline: {ex['baseline_response'][:60]}...")
            print(f"  Steered:  {ex['steered_response'][:60]}...")
        
        # Low uncertainty: forced to abstain
        low_forced = df[(df["uncertainty_bin"] == "low") & 
                       (~df["baseline_abstained"]) & 
                       (df["steered_abstained"])]
        if len(low_forced) > 0:
            ex = low_forced.iloc[0]
            print(f"\nLOW UNCERTAINTY → FORCED ABSTENTION:")
            print(f"  Q: {ex['question']}")
            print(f"  Baseline: {ex['baseline_response'][:60]}...")
            print(f"  Steered:  {ex['steered_response'][:60]}...")

        # Generate plots
        self.plot_results(df)

        return {
            "bin_counts": df["uncertainty_bin"].value_counts().to_dict(),
            "flip_rate_by_bin": flip_by_bin,
            "total_flips": int(n_flips),
            "total_questions": int(total),
            "baseline_abstention_rate": float(n_baseline_abstain / total) if total > 0 else 0.0,
            "steered_abstention_rate": float(n_steered_abstain / total) if total > 0 else 0.0,
            "gate_opened": int(df[(df["baseline_abstained"]) & (~df["steered_abstained"])].shape[0]),
            "gate_closed": int(df[(~df["baseline_abstained"]) & (df["steered_abstained"])].shape[0]),
        }

    def plot_results(self, df: pd.DataFrame):
        """Visualization of gate control."""
        order = ["low", "medium", "high"]
        present = [b for b in order if b in set(df["uncertainty_bin"])]

        if len(present) == 0:
            print("WARNING: No valid bins!")
            return

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Panel 1: Flip rates by uncertainty
        flip_rates = df.groupby("uncertainty_bin")["behavior_flipped"].mean()
        flip_rates = flip_rates.reindex(present).fillna(0.0)
        
        colors = ['#2ecc71', '#f39c12', '#e74c3c'][:len(present)]
        axes[0].bar(present, flip_rates.values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        axes[0].set_ylim(0, 1)
        axes[0].set_title("Gate Control Across Uncertainty Levels", fontsize=13, fontweight='bold')
        axes[0].set_xlabel("Internal Uncertainty")
        axes[0].set_ylabel("Behavior Flip Rate")
        axes[0].grid(axis="y", alpha=0.3)
        axes[0].axhline(y=flip_rates.mean(), color='red', linestyle='--', 
                       alpha=0.6, linewidth=2, label=f'Mean: {flip_rates.mean():.0%}')
        axes[0].legend()
        
        for i, val in enumerate(flip_rates.values):
            axes[0].text(i, val + 0.04, f'{100*val:.0f}%', ha='center', fontsize=12, fontweight='bold')

        # Panel 2: Abstention rates before/after
        abstain_before = df.groupby("uncertainty_bin")["baseline_abstained"].mean().reindex(present).fillna(0)
        abstain_after = df.groupby("uncertainty_bin")["steered_abstained"].mean().reindex(present).fillna(0)
        
        x = np.arange(len(present))
        width = 0.35
        
        axes[1].bar(x - width/2, abstain_before.values, width, label='Baseline', 
                   color='steelblue', alpha=0.8, edgecolor='black')
        axes[1].bar(x + width/2, abstain_after.values, width, label='Steered',
                   color='coral', alpha=0.8, edgecolor='black')
        
        axes[1].set_ylim(0, 1)
        axes[1].set_title("Abstention Rates: Before vs After Steering", fontsize=13, fontweight='bold')
        axes[1].set_xlabel("Internal Uncertainty")
        axes[1].set_ylabel("Abstention Rate")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(present)
        axes[1].legend()
        axes[1].grid(axis="y", alpha=0.3)

        # Panel 3: Transition types
        for i, bin_name in enumerate(present):
            bin_df = df[df["uncertainty_bin"] == bin_name]
            
            opened = bin_df[(bin_df["baseline_abstained"]) & (~bin_df["steered_abstained"])].shape[0]
            closed = bin_df[(~bin_df["baseline_abstained"]) & (bin_df["steered_abstained"])].shape[0]
            unchanged = len(bin_df) - opened - closed
            
            axes[2].bar(i, opened, color='#2ecc71', alpha=0.8, edgecolor='black')
            axes[2].bar(i, closed, bottom=opened, color='#e74c3c', alpha=0.8, edgecolor='black')
            axes[2].bar(i, unchanged, bottom=opened+closed, color='#95a5a6', alpha=0.5, edgecolor='black')
        
        axes[2].set_title("Types of Gate Control", fontsize=13, fontweight='bold')
        axes[2].set_xlabel("Internal Uncertainty")
        axes[2].set_ylabel("Number of Questions")
        axes[2].set_xticks(range(len(present)))
        axes[2].set_xticklabels(present)
        axes[2].legend(['Gate Opened', 'Gate Closed', 'Unchanged'], loc='upper right')
        axes[2].grid(axis="y", alpha=0.3)

        plt.tight_layout()

        outpath = self.config.results_dir / "exp4_gate_independence.png"
        plt.savefig(outpath, dpi=250, bbox_inches="tight")
        print(f"\n✓ Figure saved to {outpath}")
        plt.close()

# ----------------------------
# Experiment 4 (REPLACEMENT)
# ----------------------------

# class Experiment4:
#     """
#     EXPERIMENT 4: Gate Independence - MATCHING EXP3 EXACTLY
    
#     Key fix: Use the EXACT SAME steering logic as Exp3.
#     """

#     def __init__(self, model_wrapper: ModelWrapper, config: ExperimentConfig,
#                  steering_vectors: Dict[int, torch.Tensor]):
#         self.model = model_wrapper
#         self.config = config
#         self.steering_vectors = steering_vectors
#         self.results = []
#         set_seed(config.seed)

#     def _get_answer_prompt(self, question: str, context: str = None) -> str:
#         """Use the SAME prompt format as Exp3."""
#         return format_prompt(question, "neutral", context)

#     def _force_guess_prompt(self, question: str, context: str = None) -> str:
#         """For measuring internal uncertainty."""
#         prompt = self._get_answer_prompt(question, context)
#         return prompt + " (You must give an answer)"

#     # -----------------------------
#     # Internal uncertainty measurement
#     # -----------------------------
#     def measure_internal_uncertainty(self, question: str, context: str = None,
#                                     n_samples: int = 20) -> Dict:
#         """Sample forced guesses to measure uncertainty."""
#         print(f"\n  [MEASURING] {question[:50]}...")
#         prompt = self._force_guess_prompt(question, context)

#         responses = []
#         for i in range(n_samples):
#             resp = self.model.generate(prompt, temperature=1.0, do_sample=True, max_new_tokens=30)
#             resp_short = resp[:50].strip()
#             responses.append(resp_short)
#             if i < 3:
#                 print(f"    Sample {i+1}: {resp_short}")

#         from collections import Counter
#         counts = Counter(responses)
#         n = len(responses)
#         probs = np.array([c / n for c in counts.values()], dtype=np.float64)

#         entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
#         p_max = float(np.max(probs)) if len(probs) > 0 else 0.0

#         print(f"    Distribution: {dict(counts)}")
#         print(f"    → Entropy={entropy:.3f}, p_max={p_max:.3f}")

#         return {
#             "entropy": entropy,
#             "p_max": p_max,
#             "counts": dict(counts),
#             "n": n,
#         }

#     def _bin_uncertainty(self, p_max: float) -> str:
#         """Bin by concentration."""
#         if p_max >= 0.65:
#             return "low"
#         elif p_max >= 0.45:
#             return "medium"
#         else:
#             return "high"

#     # -----------------------------
#     # EXACT COPY OF EXP3'S test_steering_robust
#     # -----------------------------
#     def test_steering_robust(self, prompt: str, layer_idx: int,
#                             steering_vector: torch.Tensor, epsilon: float) -> Dict:
#         """
#         COPIED DIRECTLY FROM EXP3 - Apply steering and measure effect.
#         """
#         self.model.clear_hooks()
        
#         # Baseline generation (no steering)
#         baseline_response = self.model.generate(
#             prompt, 
#             temperature=0.0, 
#             do_sample=False, 
#             max_new_tokens=50
#         )
#         baseline_answer = extract_answer(baseline_response)
        
#         if epsilon == 0:
#             return {
#                 "baseline_response": baseline_response[:100],
#                 "steered_response": baseline_response[:100],
#                 "baseline_answer": baseline_answer,
#                 "steered_answer": baseline_answer,
#                 "answer_changed": False
#             }
        
#         # Steered generation - EXACT COPY FROM EXP3
#         self.model.clear_hooks()
        
#         # Get prompt length
#         inputs = self.model.tokenizer(prompt, return_tensors="pt").to(self.model.config.device)
#         prompt_length = inputs["input_ids"].shape[1]
        
#         # Register hook that applies steering to all positions during generation
#         def steering_hook_all_positions(module, input, output):
#             # output may be Tensor or tuple
#             if isinstance(output, tuple):
#                 hidden_states = output[0]
#                 rest = output[1:]
#             else:
#                 hidden_states = output
#                 rest = None
        
#             hidden_states = hidden_states.clone()
        
#             # steering_vector is shape [1, d]; make it [d] for clean broadcast
#             sv = steering_vector.to(hidden_states.device)
#             if sv.ndim == 2 and sv.shape[0] == 1:
#                 sv = sv[0]
        
#             # IN-PLACE ADDITION like Exp3
#             hidden_states[:, -1, :] += epsilon * sv
        
#             if rest is None:
#                 return hidden_states
#             return (hidden_states,) + rest
    
#         # Get the layer from the model
#         layer = _get_blocks(self.model.model)[layer_idx]
#         handle = layer.register_forward_hook(steering_hook_all_positions)
#         self.model.hooks.append(handle)
        
#         # Generate with steering
#         steered_response = self.model.generate(
#             prompt,
#             temperature=0.0,
#             do_sample=False,
#             max_new_tokens=50
#         )
#         steered_answer = extract_answer(steered_response)
        
#         self.model.clear_hooks()
        
#         # Check if answer changed
#         answer_changed = (baseline_answer != steered_answer)
        
#         print(f"    BASELINE: {baseline_answer[:40]}")
#         print(f"    STEERED:  {steered_answer[:40]}")
#         print(f"    → Changed: {answer_changed}")
        
#         return {
#             "baseline_response": baseline_response[:100],
#             "steered_response": steered_response[:100],
#             "baseline_answer": baseline_answer,
#             "steered_answer": steered_answer,
#             "answer_changed": answer_changed
#         }

#     # -----------------------------
#     # Run
#     # -----------------------------
#     def run(self, questions: List[Dict], best_layer: int,
#             epsilon: float = 8.0,
#             n_uncert_samples: int = 20) -> pd.DataFrame:
#         """
#         Run Exp4 using EXACT SAME steering as Exp3.
#         """
#         print("\n" + "="*70)
#         print("EXPERIMENT 4: Gate Independence (MATCHING EXP3)")
#         print("="*70)
#         print(f"Parameters: layer={best_layer}, ε={epsilon}, samples={n_uncert_samples}")
#         print(f"Questions: {len(questions)}\n")

#         # Get steering vector for best layer
#         if best_layer in self.steering_vectors:
#             steering_vec = self.steering_vectors[best_layer]
#         elif str(best_layer) in self.steering_vectors:
#             steering_vec = self.steering_vectors[str(best_layer)]
#         else:
#             raise KeyError(f"No steering vector for layer {best_layer}. Keys: {list(self.steering_vectors.keys())}")
        
#         steering_vec = steering_vec.to(self.config.device)
#         print(f"Loaded steering vector: shape={steering_vec.shape}, norm={torch.norm(steering_vec).item():.4f}\n")

#         self.results = []

#         # Phase 1: Measure uncertainty
#         print("="*70)
#         print("PHASE 1: Measuring Internal Uncertainty")
#         print("="*70)
        
#         enriched = []
#         for i, q_data in enumerate(questions, 1):
#             print(f"\n[Question {i}/{len(questions)}]")
#             q = q_data["question"]
#             ctx = q_data.get("context")

#             m = self.measure_internal_uncertainty(q, ctx, n_samples=n_uncert_samples)
#             ubin = self._bin_uncertainty(m["p_max"])
#             print(f"  → Bin: {ubin.upper()}")

#             enriched.append({
#                 **q_data,
#                 "internal_entropy": m["entropy"],
#                 "p_max": m["p_max"],
#                 "uncertainty_bin": ubin,
#             })

#         # Summary
#         print("\n" + "="*70)
#         print("PHASE 1 SUMMARY")
#         print("="*70)
#         from collections import Counter
#         bin_dist = Counter(q["uncertainty_bin"] for q in enriched)
#         for bin_name in ["low", "medium", "high"]:
#             count = bin_dist.get(bin_name, 0)
#             pct = 100 * count / len(enriched) if enriched else 0
#             print(f"  {bin_name.upper()}: {count}/{len(enriched)} ({pct:.0f}%)")

#         # Phase 2: Test steering (USING EXP3 LOGIC)
#         print("\n" + "="*70)
#         print("PHASE 2: Testing Steering (Using Exp3 Logic)")
#         print("="*70)
        
#         for i, q_data in enumerate(enriched, 1):
#             print(f"\n[Test {i}/{len(enriched)}] Bin: {q_data['uncertainty_bin'].upper()}")
#             q = q_data["question"]
#             ctx = q_data.get("context")
            
#             prompt = self._get_answer_prompt(q, ctx)

#             r = self.test_steering_robust(prompt, best_layer, steering_vec, epsilon)

#             self.results.append({
#                 "question": q[:80],
#                 "internal_entropy": float(q_data["internal_entropy"]),
#                 "p_max": float(q_data["p_max"]),
#                 "uncertainty_bin": q_data["uncertainty_bin"],
#                 "baseline_answer": r["baseline_answer"],
#                 "steered_answer": r["steered_answer"],
#                 "answer_changed": r["answer_changed"],
#                 "layer": int(best_layer),
#                 "epsilon": float(epsilon),
#                 "baseline_text": r["baseline_response"],
#                 "steered_text": r["steered_response"],
#             })

#         df = pd.DataFrame(self.results)

#         # Summary
#         print("\n" + "="*70)
#         print("PHASE 2 SUMMARY")
#         print("="*70)
        
#         total_changes = df["answer_changed"].sum()
#         print(f"\nTotal answer changes: {total_changes}/{len(df)} ({100*total_changes/len(df):.0f}%)")
        
#         print("\nAnswer changes by uncertainty bin:")
#         for bin_name in ["low", "medium", "high"]:
#             bin_df = df[df["uncertainty_bin"] == bin_name]
#             if len(bin_df) == 0:
#                 continue
#             changes = bin_df["answer_changed"].sum()
#             pct = 100 * changes / len(bin_df)
#             print(f"  {bin_name.upper()}: {changes}/{len(bin_df)} ({pct:.0f}%)")

#         # Save
#         output_path = self.config.results_dir / "exp4_raw_results.csv"
#         df.to_csv(output_path, index=False)
#         print(f"\n✓ Results saved to {output_path}")

#         return df

#     def analyze(self, df: pd.DataFrame) -> Dict:
#         """Analysis."""
#         print("\n" + "="*70)
#         print("EXPERIMENT 4: ANALYSIS")
#         print("="*70)

#         print("\n1. OVERALL STATISTICS")
#         print("-" * 40)
#         total = len(df)
#         total_changes = df["answer_changed"].sum()
#         print(f"Total questions: {total}")
#         print(f"Answers changed: {total_changes} ({100*total_changes/total:.1f}%)")

#         print("\n2. CHANGES BY UNCERTAINTY BIN")
#         print("-" * 40)
        
#         change_by_bin = {}
#         for bin_name in ["low", "medium", "high"]:
#             bin_df = df[df["uncertainty_bin"] == bin_name]
#             if len(bin_df) == 0:
#                 print(f"  {bin_name.upper()}: no questions")
#                 change_by_bin[bin_name] = 0.0
#                 continue
            
#             n_changes = bin_df["answer_changed"].sum()
#             rate = n_changes / len(bin_df)
#             change_by_bin[bin_name] = float(rate)
            
#             print(f"  {bin_name.upper()}: {n_changes}/{len(bin_df)} changed ({100*rate:.1f}%)")

#         print("\n3. INDEPENDENCE TEST")
#         print("-" * 40)
#         rates = [v for v in change_by_bin.values() if v > 0]
#         if len(rates) >= 2:
#             rate_std = np.std(rates)
#             rate_mean = np.mean(rates)
#             cv = rate_std / rate_mean if rate_mean > 0 else 0
#             print(f"Mean change rate: {rate_mean:.3f}")
#             print(f"Std dev: {rate_std:.3f}")
#             print(f"Coefficient of variation: {cv:.3f}")
#             if cv < 0.25:
#                 print("✓ Low variance → steering is INDEPENDENT of uncertainty")
#             else:
#                 print("⚠️  High variance → steering may depend on uncertainty")

#         print("\n4. UNCERTAINTY DISTRIBUTION")
#         print("-" * 40)
#         bin_counts = df["uncertainty_bin"].value_counts().to_dict()
#         for bin_name in ["low", "medium", "high"]:
#             count = bin_counts.get(bin_name, 0)
#             print(f"  {bin_name.upper()}: {count} questions")

#         self.plot_results(df)

#         return {
#             "bin_counts": bin_counts,
#             "answer_change_by_bin": change_by_bin,
#             "total_answer_changes": int(total_changes),
#             "total_questions": int(total),
#             "change_rate_overall": float(total_changes / total) if total > 0 else 0.0,
#         }

#     def plot_results(self, df: pd.DataFrame):
#         """Visualization."""
#         order = ["low", "medium", "high"]
#         present = [b for b in order if b in set(df["uncertainty_bin"])]

#         if len(present) == 0:
#             print("WARNING: No valid bins!")
#             return

#         change_rates = df.groupby("uncertainty_bin")["answer_changed"].mean()
#         change_rates = change_rates.reindex(present).fillna(0.0)

#         counts = df.groupby("uncertainty_bin").size()
#         counts = counts.reindex(present).fillna(0).astype(int)

#         fig, axes = plt.subplots(1, 2, figsize=(12, 4))

#         # Panel 1: Change rates
#         colors = ['#2ecc71', '#f39c12', '#e74c3c'][:len(present)]
#         bars = axes[0].bar(present, change_rates.values, color=colors, alpha=0.8, edgecolor='black')
#         axes[0].set_ylim(0, 1)
#         axes[0].set_title("Steering Changes Answers Regardless of Uncertainty", 
#                          fontsize=12, fontweight='bold')
#         axes[0].set_xlabel("Internal Uncertainty Level")
#         axes[0].set_ylabel("Answer Change Rate")
#         axes[0].grid(axis="y", alpha=0.3)
#         axes[0].axhline(y=change_rates.mean(), color='red', linestyle='--', 
#                       alpha=0.5, label=f'Mean: {change_rates.mean():.1%}')
#         axes[0].legend()
        
#         for i, (bin_name, val) in enumerate(zip(present, change_rates.values)):
#             axes[0].text(i, val + 0.03, f'{100*val:.0f}%', ha='center', fontsize=11, fontweight='bold')

#         # Panel 2: Sample sizes
#         axes[1].bar(present, counts.values, color='steelblue', alpha=0.7, edgecolor='black')
#         axes[1].set_title("Questions per Uncertainty Bin", fontsize=12, fontweight='bold')
#         axes[1].set_xlabel("Internal Uncertainty Level")
#         axes[1].set_ylabel("Number of Questions")
#         axes[1].grid(axis="y", alpha=0.3)
        
#         for i, (bin_name, count) in enumerate(zip(present, counts.values)):
#             axes[1].text(i, count + 0.2, str(count), ha='center', fontsize=11, fontweight='bold')

#         plt.tight_layout()

#         outpath = self.config.results_dir / "exp4_gate_independence.png"
#         plt.savefig(outpath, dpi=250, bbox_inches="tight")
#         print(f"\n✓ Figure saved to {outpath}")
#         plt.close()

# ============================================================================
# Main
# ============================================================================

def main():
    """Run Experiments 3 and 4"""
    import json
    from data_preparation import create_ambiguous_prompts, create_clearly_answerable_questions, create_clearly_unanswerable_questions
    
    # Setup
    config = ExperimentConfig()
    
    print("Initializing model...")
    model = ModelWrapper(config)
    
    # Load data
    print("\nLoading datasets...")
    answerable = create_clearly_answerable_questions()[:10]
    unanswerable = create_clearly_unanswerable_questions()[:10]
    ambiguous = create_ambiguous_prompts()[:30]
    
    # Experiment 3: Steering
    # exp3 = Experiment3(model, config)
    # results3 = exp3.run(
    #     pos_examples=answerable,
    #     neg_examples=unanswerable,
    #     test_examples=ambiguous[:20]
    # )
    # summary3 = exp3.analyze(results3)
    
    # # Find best layer from Exp3
    # best_layer = summary3.get('best_layer', config.target_layers[-1])
    # print(f"\n→ Using layer {best_layer} for Experiment 4")
    
    # Experiment 4: Gate Independence (THE KEY EXPERIMENT)
    
    # Create diverse questions for abstention testing
    # Create diverse questions for abstention testing
    abstention_questions = [
        # EASY (should answer, low uncertainty)
        {"question": "What is 2+2?", "should_abstain": False},
        {"question": "What is the capital of France?", "should_abstain": False},
        
        # IMPOSSIBLE (should abstain, high uncertainty)  
        {"question": "What did I have for breakfast yesterday?", "should_abstain": True},
        {"question": "What is my mother's maiden name?", "should_abstain": True},
        
        # AMBIGUOUS (from Exp3 - should abstain, high uncertainty)
        *ambiguous[:6],  # Use some from your existing set
    ]

    # Test positive epsilon (push toward answering)
    # Use the same ambiguous questions that worked in Exp3
    steering_vectors = torch.load(config.results_dir / "steering_vectors.pt")
    steering_vectors = {int(k): v.to(config.device) for k, v in steering_vectors.items()}
    exp4 = Experiment4(model, config, steering_vectors)
    results4 = exp4.run(
        questions=ambiguous[:20],  # Same as Exp3 test set
        best_layer=24,
        epsilon=12.0,
        reverse_steering=True
    )
    summary4 = exp4.analyze(results4)
        
    # # Test negative epsilon (push toward abstaining)  
    # results4_neg = exp4.run(
    #     questions=abstention_questions,
    #     best_layer=24,
    #     epsilon=8.0,
    #     reverse_steering=True  # Reversed direction
    # )
    
    # Compare which one shows more flips
    # print(f"\nPositive ε flips: {results4_pos['behavior_flipped'].sum()}")
    # print(f"Negative ε flips: {results4_neg['behavior_flipped'].sum()}")

    # Save summaries
    # with open(config.results_dir / "exp3_summary.json", 'w') as f:
    #     json.dump(summary3, f, indent=2)
    with open(config.results_dir / "exp4_summary.json", 'w') as f:
        json.dump(summary4, f, indent=2)
    
    print(f"\n✓ Experiments 3 & 4 complete!")


if __name__ == "__main__":
    main()