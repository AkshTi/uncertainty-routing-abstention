"""
Train Safety-Aware Steering Vectors for Experiment 7

This script trains steering vectors that control EPISTEMIC uncertainty
(knowing vs not knowing) while preserving safety boundaries.

Key difference from original steering vectors:
- POS examples: Benign answerable questions (safe + answerable)
- NEG examples: Benign unanswerable questions (safe + unanswerable)
- EXCLUDED: Harmful questions entirely (to preserve safety alignment)

Output: results/steering_vectors_safety_aware.pt
"""

import torch
import pandas as pd
from tqdm import tqdm
from typing import List, Dict
from pathlib import Path

from core_utils import ModelWrapper, ExperimentConfig, set_seed


def _get_blocks(hf_model):
    """Return the list/ModuleList of transformer blocks."""
    if hasattr(hf_model, "model") and hasattr(hf_model.model, "layers"):
        return hf_model.model.layers
    if hasattr(hf_model, "layers"):
        return hf_model.layers
    if hasattr(hf_model, "transformer") and hasattr(hf_model.transformer, "h"):
        return hf_model.transformer.h
    if hasattr(hf_model, "gpt_neox") and hasattr(hf_model.gpt_neox, "layers"):
        return hf_model.gpt_neox.layers
    if hasattr(hf_model, "transformer") and hasattr(hf_model.transformer, "blocks"):
        return hf_model.transformer.blocks
    raise AttributeError(f"Don't know where transformer blocks are for {type(hf_model)}")


class SafetyAwareSteeringTrainer:
    """Train steering vectors that preserve safety boundaries"""
    
    def __init__(self, model_wrapper: ModelWrapper, config: ExperimentConfig):
        self.model = model_wrapper
        self.config = config
        set_seed(config.seed)
    
    def create_training_examples(self) -> Dict[str, List[Dict]]:
        """
        Create training examples that separate epistemic uncertainty from safety.
        
        CRITICAL: All examples are BENIGN (safe to answer or abstain on)
        """
        return {
            "epistemic_answerable": [
                # Factual questions - clearly answerable, safe
                {"question": "What is the capital of France?", "context": None},
                {"question": "How many days are in a week?", "context": None},
                {"question": "What is 2 + 2?", "context": None},
                {"question": "What color is the sky on a clear day?", "context": None},
                {"question": "Who wrote Romeo and Juliet?", "context": None},
                {"question": "What is the boiling point of water at sea level?", "context": None},
                {"question": "How many continents are there?", "context": None},
                {"question": "What is the largest planet in our solar system?", "context": None},
                {"question": "What language is spoken in Brazil?", "context": None},
                {"question": "What is the chemical symbol for gold?", "context": None},
                
                # Safe practical questions with clear answers
                {"question": "How do I tie a shoelace?", "context": None},
                {"question": "What are common symptoms of the flu?", "context": None},
                {"question": "How do I change a light bulb?", "context": None},
                {"question": "What is the purpose of a firewall in computers?", "context": None},
                {"question": "How do I make scrambled eggs?", "context": None},
            ],
            
            "epistemic_unanswerable": [
                # Personal/private information - unanswerable but not harmful
                {"question": "What did I have for breakfast this morning?", "context": None},
                {"question": "What is my mother's maiden name?", "context": None},
                {"question": "What number am I thinking of right now?", "context": None},
                {"question": "What color shirt am I wearing?", "context": None},
                {"question": "What is my favorite song?", "context": None},
                {"question": "Where did I go on vacation last year?", "context": None},
                {"question": "What is my cat's name?", "context": None},
                {"question": "What time did I wake up today?", "context": None},
                
                # Ambiguous without context - unanswerable but not harmful
                {"question": "Is it raining?", "context": None},
                {"question": "What time is it?", "context": None},
                {"question": "Is the store open?", "context": None},
                {"question": "Will it be sunny tomorrow?", "context": None},
                {"question": "Is the package here yet?", "context": None},
                {"question": "Did they call back?", "context": None},
                {"question": "Is the meeting still on?", "context": None},
            ]
        }
    
    def format_prompt(self, question: str, context: str = None) -> str:
        """Format question as prompt - neutral, allows both answering and abstention"""
        if context:
            return f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        return f"Question: {question}\n\nAnswer:"
    
    def collect_activations(self, prompts: List[str], layer_idx: int) -> torch.Tensor:
        """Collect activations at last prompt token for a set of prompts"""
        activations = []
        
        for prompt in tqdm(prompts, desc=f"Layer {layer_idx}", leave=False):
            self.model.clear_hooks()
            
            # Get prompt length
            inputs = self.model.tokenizer(prompt, return_tensors="pt").to(self.config.device)
            prompt_length = inputs["input_ids"].shape[1]
            position = prompt_length - 1  # Last prompt token
            
            # Register hook to capture activation
            self.model.register_cache_hook(layer_idx, position)
            
            # Generate one token to capture activation
            _ = self.model.generate(prompt, temperature=0.0, do_sample=False, max_new_tokens=1)
            
            # Get cached activation
            activation = self.model.activation_cache[f"layer_{layer_idx}"].clone()
            activations.append(activation)
            
            self.model.clear_hooks()
        
        return torch.stack(activations)
    
    def compute_steering_vector(self, layer_idx: int,
                               answerable_prompts: List[str],
                               unanswerable_prompts: List[str]) -> torch.Tensor:
        """
        Compute steering vector: answerable_mean - unanswerable_mean
        
        Direction points from "I don't know" toward "I can answer"
        - Positive epsilon: Push toward answering
        - Negative epsilon: Push toward abstention
        """
        print(f"\nComputing steering vector for layer {layer_idx}...")
        
        # Collect activations
        print("  Collecting answerable activations...")
        answerable_acts = self.collect_activations(answerable_prompts, layer_idx)
        
        print("  Collecting unanswerable activations...")
        unanswerable_acts = self.collect_activations(unanswerable_prompts, layer_idx)
        
        # Compute means
        answerable_mean = answerable_acts.mean(dim=0)
        unanswerable_mean = unanswerable_acts.mean(dim=0)
        
        # Direction: from uncertain toward answering
        direction = answerable_mean - unanswerable_mean
        
        # Normalize
        direction = direction / (direction.norm() + 1e-8)
        
        print(f"  Stats:")
        print(f"    Raw direction norm: {(answerable_mean - unanswerable_mean).norm():.4f}")
        print(f"    Answerable mean norm: {answerable_mean.norm():.4f}")
        print(f"    Unanswerable mean norm: {unanswerable_mean.norm():.4f}")
        print(f"    Normalized direction norm: {direction.norm():.4f}")
        
        return direction
    
    def train_all_layers(self, target_layers: List[int] = None) -> Dict[int, torch.Tensor]:
        """Train steering vectors for all target layers"""
        if target_layers is None:
            target_layers = self.config.target_layers
        
        print("\n" + "="*70)
        print("TRAINING SAFETY-AWARE STEERING VECTORS")
        print("="*70)
        print(f"Target layers: {target_layers}")
        print(f"Device: {self.config.device}")
        
        # Get training examples
        examples = self.create_training_examples()
        
        # Format prompts
        answerable_prompts = [
            self.format_prompt(ex["question"], ex.get("context"))
            for ex in examples["epistemic_answerable"]
        ]
        
        unanswerable_prompts = [
            self.format_prompt(ex["question"], ex.get("context"))
            for ex in examples["epistemic_unanswerable"]
        ]
        
        print(f"\nTraining data:")
        print(f"  Answerable examples: {len(answerable_prompts)}")
        print(f"  Unanswerable examples: {len(unanswerable_prompts)}")
        
        print("\nExample answerable questions:")
        for ex in examples["epistemic_answerable"][:3]:
            print(f"  - {ex['question']}")
        
        print("\nExample unanswerable questions:")
        for ex in examples["epistemic_unanswerable"][:3]:
            print(f"  - {ex['question']}")
        
        # Train vectors for each layer
        steering_vectors = {}
        
        for layer_idx in target_layers:
            vector = self.compute_steering_vector(
                layer_idx,
                answerable_prompts,
                unanswerable_prompts
            )
            steering_vectors[layer_idx] = vector.cpu()
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print(f"Trained {len(steering_vectors)} steering vectors")
        
        return steering_vectors
    
    def save_vectors(self, steering_vectors: Dict[int, torch.Tensor], 
                    filename: str = "steering_vectors_safety_aware.pt"):
        """Save steering vectors to disk"""
        output_path = self.config.results_dir / filename
        
        torch.save(steering_vectors, output_path)
        
        print(f"\n✓ Saved to: {output_path}")
        print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")
        
        # Save metadata
        metadata = {
            "layers": list(steering_vectors.keys()),
            "vector_shape": str(steering_vectors[list(steering_vectors.keys())[0]].shape),
            "device": str(self.config.device),
            "model": self.config.model_name,
            "training_type": "safety_aware_epistemic",
            "description": "Steering vectors trained on benign answerable vs unanswerable questions"
        }
        
        import json
        metadata_path = self.config.results_dir / filename.replace(".pt", "_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Metadata saved to: {metadata_path}")
        
        return output_path
    
    def validate_vectors(self, steering_vectors: Dict[int, torch.Tensor]):
        """Quick validation that vectors look reasonable"""
        print("\n" + "="*70)
        print("VALIDATION")
        print("="*70)
        
        for layer_idx, vector in steering_vectors.items():
            norm = vector.norm().item()
            mean = vector.mean().item()
            std = vector.std().item()
            
            print(f"\nLayer {layer_idx}:")
            print(f"  Shape: {vector.shape}")
            print(f"  Norm: {norm:.4f}")
            print(f"  Mean: {mean:.6f}")
            print(f"  Std: {std:.4f}")
            
            # Sanity checks
            if norm < 0.9 or norm > 1.1:
                print(f"  ⚠️  WARNING: Norm should be ~1.0")
            if abs(mean) > 0.01:
                print(f"  ⚠️  WARNING: Mean should be close to 0")
            
        print("\n✓ Validation complete")


def main():
    """Main training script"""
    print("="*70)
    print("SAFETY-AWARE STEERING VECTOR TRAINING")
    print("="*70)
    
    # Setup
    config = ExperimentConfig()
    
    # Ensure results directory exists
    config.results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nConfiguration:")
    print(f"  Model: {config.model_name}")
    print(f"  Device: {config.device}")
    print(f"  Results dir: {config.results_dir}")
    
    # Initialize model
    print("\nLoading model...")
    model = ModelWrapper(config)
    print("✓ Model loaded")
    
    # Initialize trainer
    trainer = SafetyAwareSteeringTrainer(model, config)
    
    # Train vectors
    steering_vectors = trainer.train_all_layers()
    
    # Validate
    trainer.validate_vectors(steering_vectors)
    
    # Save
    output_path = trainer.save_vectors(steering_vectors)
    
    print("\n" + "="*70)
    print("ALL DONE!")
    print("="*70)
    print(f"\nTo use these vectors in Experiment 7:")
    print(f"  steering_vectors = torch.load('{output_path}')")
    print(f"  exp7 = Experiment7Fixed(model, config, steering_vectors)")
    print(f"  exp7.run_all(best_layer=24, epsilon_toward_answer=20.0, epsilon_toward_abstain=-20.0)")
    
    print("\nExpected behavior:")
    print("  ✓ Positive epsilon (+20): Push toward answering (reduce abstention)")
    print("  ✓ Negative epsilon (-20): Push toward abstention (increase uncertainty)")
    print("  ✓ Safety boundaries: PRESERVED (vectors trained only on benign examples)")


if __name__ == "__main__":
    main()
