"""
Complete Experiment Implementation - Part 1: Core Infrastructure
Save as: core_utils.py

Requirements:
pip install torch transformers numpy pandas matplotlib seaborn scikit-learn tqdm
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, List
from dataclasses import dataclass
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================
def _get_hidden_states(output):
    return output[0] if isinstance(output, (tuple, list)) else output

@dataclass
class ExperimentConfig:
    """Configuration for all experiments"""
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    results_dir: Path = Path("./results")
    
    # Experiment parameters
    n_force_guess_samples: int = 20
    steering_epsilon_range: List[float] = None
    target_layers: List[int] = None
    
    def __post_init__(self):
        self.results_dir.mkdir(exist_ok=True)
        if self.steering_epsilon_range is None:
            self.steering_epsilon_range = [0, 0.5, 1.0, 2.0, 4.0, 8.0]
        if self.target_layers is None:
            self.target_layers = [24, 25, 26, 27]

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

# ============================================================================
# Model Wrapper
# ============================================================================

class ModelWrapper:
    """Wrapper for model with activation caching and intervention"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        print(f"Loading model {config.model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16,
            device_map=config.device,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.activation_cache = {}
        self.hooks = []
        print("Model loaded successfully!")
    
    def clear_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activation_cache = {}
    
    def _get_hidden_states(self, output):
        # HF layers sometimes return Tensor, sometimes tuple where [0] is hidden states
        return output[0] if isinstance(output, (tuple, list)) else output

    def register_cache_hook(self, layer_idx: int, position: Optional[int] = None):
        cached = {"done": False}
    
        def hook_fn(module, input, output):
            hidden_states = self._get_hidden_states(output)
    
            if cached["done"]:
                return output
    
            if position is not None:
                seq_len = hidden_states.shape[1]
                if position >= seq_len:
                    return output  # skip cached decoding step
                self.activation_cache[f"layer_{layer_idx}"] = hidden_states[:, position, :].detach().cpu()
            else:
                self.activation_cache[f"layer_{layer_idx}"] = hidden_states.detach().cpu()
    
            cached["done"] = True
            return output
    
        layer = _get_blocks(self.model.model)[layer_idx]
        handle = layer.register_forward_hook(hook_fn)
        self.hooks.append(handle)
    
    def register_patch_hook(self, layer_idx: int, position: int, replacement: torch.Tensor):
        applied = {"done": False}
    
        def hook_fn(module, input, output):
            hidden_states = self._get_hidden_states(output)
    
            # During cached decoding, seq_len is often 1 -> skip
            seq_len = hidden_states.shape[1]
            if applied["done"] or position >= seq_len:
                return output
    
            hs = hidden_states.clone()
            hs[:, position, :] = replacement.to(hs.device)
    
            applied["done"] = True
    
            if isinstance(output, (tuple, list)):
                return (hs,) + tuple(output[1:])
            return hs
    
        layer = _get_blocks(self.model.model)[layer_idx]
        handle = layer.register_forward_hook(hook_fn)
        self.hooks.append(handle)

    def register_steering_hook(self, layer_idx: int, position: int,
                          steering_vector: torch.Tensor, epsilon: float):
        """
        Register steering hook that applies at EVERY forward pass.
        Uses position=-1 to apply at last token (for autoregressive generation).
        """
        def hook_fn(module, input, output):
            hidden_states = self._get_hidden_states(output)

            if not torch.is_tensor(hidden_states):
                return output

            hs = hidden_states.clone()
            sv = steering_vector.to(hs.device).to(hs.dtype)

            # Squeeze if needed
            if sv.ndim == 2 and sv.shape[0] == 1:
                sv = sv.squeeze(0)

            # Apply at last token position (-1) for every forward pass
            # This ensures steering works during autoregressive generation
            if hs.ndim == 3:
                hs[:, -1, :] = hs[:, -1, :] + epsilon * sv
            elif hs.ndim == 2:
                hs[-1, :] = hs[-1, :] + epsilon * sv

            if isinstance(output, (tuple, list)):
                return (hs,) + tuple(output[1:])
            return hs
    
        layer = _get_blocks(self.model.model)[layer_idx]
        handle = layer.register_forward_hook(hook_fn)
        self.hooks.append(handle)

    
    def generate(self, prompt: str, max_new_tokens: int = 50, 
                 temperature: float = 1.0, do_sample: bool = True) -> str:
        """Generate text completion"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.config.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = full_text[len(prompt):].strip()
        return generated
    
    def get_logits(self, prompt: str) -> torch.Tensor:
        """Get model logits"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.config.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.logits
        
        """
Add these methods to your ModelWrapper class
"""
    
    def get_layer_activations(self, prompt: str, layer_idx: int, position: Optional[int] = None) -> torch.Tensor:
        """
        Get activations from a specific layer for a given prompt.
        
        Args:
            prompt: Input text
            layer_idx: Which layer to extract from
            position: Which token position to extract (None = all positions)
        
        Returns:
            Activations tensor with shape:
            - [seq_len, hidden_dim] if position=None
            - [hidden_dim] if position is specified
        """
        # Clear any existing hooks and cache
        self.clear_hooks()
        
        # Register hook to cache activations
        self.register_cache_hook(layer_idx, position)
        
        # Run forward pass to populate cache
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.config.device)
        
        with torch.no_grad():
            _ = self.model(**inputs)
        
        # Retrieve cached activations
        cache_key = f"layer_{layer_idx}"
        if cache_key not in self.activation_cache:
            raise RuntimeError(f"Failed to cache activations for layer {layer_idx}")
        
        activations = self.activation_cache[cache_key]
        
        # Clean up
        self.clear_hooks()
        
        return activations
    
    
    # Also add this helper function at module level (outside the class)
    # if you don't already have it:
    
    def _get_blocks(hf_model):
        """Get transformer blocks from model"""
        # Check common model architectures
        if hasattr(hf_model, "model") and hasattr(hf_model.model, "layers"):
            return hf_model.model.layers  # Llama, Mistral
        if hasattr(hf_model, "layers"):
            return hf_model.layers
        if hasattr(hf_model, "transformer") and hasattr(hf_model.transformer, "h"):
            return hf_model.transformer.h  # GPT-2 style
        if hasattr(hf_model, "gpt_neox") and hasattr(hf_model.gpt_neox, "layers"):
            return hf_model.gpt_neox.layers  # GPT-NeoX
        if hasattr(hf_model, "transformer") and hasattr(hf_model.transformer, "blocks"):
            return hf_model.transformer.blocks
        raise AttributeError(f"Can't find transformer blocks for {type(hf_model)}")


# ============================================================================
# Utilities
# ============================================================================

def extract_answer(response: str) -> str:
    """Extract final answer, detecting uncertainty"""
    response = response.strip()

    uncertainty_markers = [
        "UNCERTAIN", "I don't know", "cannot answer",
        "can't answer", "unsure", "unclear", "not sure",
        "impossible to", "don't have enough",
        # Additional markers for better detection
        "cannot determine", "cannot be determined",
        "no way to know", "unable to answer",
        "not possible to", "cannot provide",
        "insufficient information", "not enough information",
        "I'm not able", "I am not able",
        "would need more", "cannot say",
        "there is no way", "there's no way",
        "it is impossible", "it's impossible",
        "I cannot", "I can't",
        "unknown", "unknowable",
        "not known", "isn't known",
        "no definitive", "no certain",
        "would be speculation", "purely speculative",
        "cannot predict", "impossible to predict",
        "beyond my", "outside my",
    ]
    
    response_lower = response.lower()
    for marker in uncertainty_markers:
        if marker.lower() in response_lower:
            return "UNCERTAIN"
    
    # Get first substantial line
    lines = [l.strip() for l in response.split('\n') if l.strip()]
    if lines:
        # Check first line for uncertainty
        if any(m.lower() in lines[0].lower() for m in uncertainty_markers):
            return "UNCERTAIN"
        return lines[0]
    
    return response if response else "UNCERTAIN"


def get_decision_token_position(tokenizer, prompt: str) -> int:
    """
    Returns the token position corresponding to the *last content token*
    BEFORE the model begins generating the answer.
    This is usually right before 'Answer:' or 'A:'.
    """
    toks = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
    text_tokens = [tokenizer.decode([t]) for t in toks]

    # Look for common answer delimiters
    for i in range(len(text_tokens) - 1, -1, -1):
        tok = text_tokens[i].strip().lower()

        if tok in {":", "answer", "answer:", "a", "a:"}:
            # Steer the token *before* the delimiter
            if i > 0:
                return i - 1

    # Fallback: steer last token of prompt
    return len(toks) - 1



def compute_binary_margin_simple(response: str) -> float:
    """
    Simplified binary margin:
    +1 if answers, -1 if abstains
    """
    answer = extract_answer(response)
    return -1.0 if answer == "UNCERTAIN" else 1.0


def set_seed(seed: int):
    """Set random seeds"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
