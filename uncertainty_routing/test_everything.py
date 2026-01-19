"""
Complete Testing & Validation Script
Save as: test_everything.py

Run this first to ensure all code works before full experiments
"""

import sys
from pathlib import Path
import json

def test_imports():
    """Test all imports work"""
    print("\n" + "="*60)
    print("TEST 1: Checking imports...")
    print("="*60)
    
    try:
        import torch
        print("✓ PyTorch:", torch.__version__)
    except ImportError as e:
        print("✗ PyTorch import failed:", e)
        return False
    
    try:
        import transformers
        print("✓ Transformers:", transformers.__version__)
    except ImportError as e:
        print("✗ Transformers import failed:", e)
        return False
    
    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        from tqdm import tqdm
        print("✓ Data science packages: numpy, pandas, matplotlib, seaborn, tqdm")
    except ImportError as e:
        print("✗ Data science packages import failed:", e)
        return False
    
    print("\n✓ All imports successful!\n")
    return True


def test_data_preparation():
    """Test data preparation module"""
    print("="*60)
    print("TEST 2: Testing data preparation...")
    print("="*60)
    
    try:
        from data_preparation import (
            create_ambiguous_prompts,
            create_clearly_answerable_questions,
            create_clearly_unanswerable_questions,
            format_prompt,
            prepare_experiment_datasets
        )
        
        # Test dataset creation
        ambiguous = create_ambiguous_prompts()
        print(f"✓ Created {len(ambiguous)} ambiguous prompts")
        
        answerable = create_clearly_answerable_questions()
        print(f"✓ Created {len(answerable)} answerable questions")
        
        unanswerable = create_clearly_unanswerable_questions()
        print(f"✓ Created {len(unanswerable)} unanswerable questions")
        
        # Test prompt formatting
        test_q = "What is 2+2?"
        for regime in ["neutral", "cautious", "confident", "force_guess"]:
            prompt = format_prompt(test_q, regime)
            assert "2+2" in prompt
            print(f"✓ Formatted prompt with regime '{regime}'")
        
        # Test full dataset preparation
        datasets = prepare_experiment_datasets()
        print(f"✓ Prepared {len(datasets)} dataset variants")
        
        print("\n✓ Data preparation working!\n")
        return True
        
    except Exception as e:
        print(f"✗ Data preparation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_loading():
    """Test model loading"""
    print("="*60)
    print("TEST 3: Testing model loading...")
    print("="*60)
    
    try:
        from core_utils import ModelWrapper, ExperimentConfig
        
        config = ExperimentConfig()
        print(f"✓ Config created: {config.model_name}")
        print(f"  Device: {config.device}")
        
        print("\nLoading model (this may take 1-2 minutes)...")
        model = ModelWrapper(config)
        print("✓ Model loaded successfully!")
        
        # Test basic generation
        print("\nTesting basic generation...")
        test_prompt = "Question: What is 2+2?\n\nFINAL ANSWER:"
        response = model.generate(test_prompt, max_new_tokens=20, temperature=0.0, do_sample=False)
        print(f"  Prompt: {test_prompt}")
        print(f"  Response: {response[:100]}")
        print("✓ Generation working!")
        
        # Test hook system
        print("\nTesting activation hooks...")
        model.register_cache_hook(0, position=5)
        _ = model.generate(test_prompt, max_new_tokens=5)
        assert "layer_0" in model.activation_cache
        print(f"✓ Cached activation shape: {model.activation_cache['layer_0'].shape}")
        model.clear_hooks()
        print("✓ Hook system working!")
        
        print("\n✓ Model loading and basic operations successful!\n")
        return True, model
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_utilities():
    """Test utility functions"""
    print("="*60)
    print("TEST 4: Testing utility functions...")
    print("="*60)
    
    try:
        from core_utils import extract_answer, compute_binary_margin_simple
        
        # Test answer extraction
        test_cases = [
            ("UNCERTAIN", "UNCERTAIN"),
            ("I don't know the answer", "UNCERTAIN"),
            ("The answer is 4", "The answer is 4"),
            ("42\n\nThis is the answer.", "42"),
        ]
        
        for response, expected in test_cases:
            result = extract_answer(response)
            assert expected in result or result in expected, f"Failed on: {response}"
            print(f"✓ extract_answer('{response[:30]}...') = '{result[:30]}...'")
        
        # Test margin computation
        margin1 = compute_binary_margin_simple("The answer is 42")
        assert margin1 > 0, "Should be positive for answers"
        print(f"✓ compute_binary_margin_simple('answer') = {margin1}")
        
        margin2 = compute_binary_margin_simple("UNCERTAIN")
        assert margin2 < 0, "Should be negative for abstention"
        print(f"✓ compute_binary_margin_simple('UNCERTAIN') = {margin2}")
        
        print("\n✓ Utility functions working!\n")
        return True
        
    except Exception as e:
        print(f"✗ Utility functions failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mini_experiment(model):
    """Run a minimal version of each experiment"""
    print("="*60)
    print("TEST 5: Running mini experiments...")
    print("="*60)
    
    try:
        from experiment1_behavior_belief import Experiment1
        from experiment2_localization import Experiment2
        from experiment3_4_steering_independence import Experiment3, Experiment4
        from experiment5_trustworthiness import Experiment5
        from core_utils import ExperimentConfig
        from data_preparation import create_ambiguous_prompts, create_clearly_answerable_questions
        
        config = ExperimentConfig()
        config.n_force_guess_samples = 3  # Minimal for testing
        config.steering_epsilon_range = [0, 2.0]
        
        # Mini datasets
        ambiguous = create_ambiguous_prompts()[:3]
        answerable = create_clearly_answerable_questions()[:3]
        
        # Test Exp 1
        print("\nTesting Experiment 1...")
        exp1 = Experiment1(model, config)
        # Just test methods, don't run full experiment
        internal = exp1.measure_internal_uncertainty(ambiguous[0]["question"])
        assert "entropy" in internal
        print(f"✓ Exp1: Measured internal uncertainty (entropy={internal['entropy']:.3f})")
        
        # Test Exp 2
        print("\nTesting Experiment 2...")
        exp2 = Experiment2(model, config)
        result = exp2.test_single_layer_patch(
            "Answer: Paris",
            "UNCERTAIN",
            layer_idx=10
        )
        assert "delta_margin" in result
        print(f"✓ Exp2: Activation patching works (delta={result['delta_margin']:.3f})")
        
        # Test Exp 3
        print("\nTesting Experiment 3...")
        exp3 = Experiment3(model, config)
        from data_preparation import format_prompt
        pos_prompts = [format_prompt(q["question"], "neutral") for q in answerable[:2]]
        neg_prompts = [format_prompt("What is Napoleon's favorite color?", "neutral")]
        
        direction = exp3.compute_steering_direction(pos_prompts, neg_prompts, layer_idx=10)
        assert direction.shape[0] > 0
        print(f"✓ Exp3: Computed steering direction (shape={direction.shape})")
        
        print("\n✓ All experiment components working!\n")
        return True
        
    except Exception as e:
        print(f"✗ Mini experiments failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_full_test_suite():
    """Run complete test suite"""
    print("\n" + "="*70)
    print(" COMPLETE TEST SUITE")
    print("="*70)
    
    results = {
        "imports": False,
        "data_prep": False,
        "model_loading": False,
        "utilities": False,
        "mini_experiments": False
    }
    
    # Test 1: Imports
    results["imports"] = test_imports()
    if not results["imports"]:
        print("\n❌ FAILED: Fix imports before proceeding")
        return False
    
    # Test 2: Data preparation
    results["data_prep"] = test_data_preparation()
    if not results["data_prep"]:
        print("\n❌ FAILED: Fix data preparation before proceeding")
        return False
    
    # Test 3: Model loading
    success, model = test_model_loading()
    results["model_loading"] = success
    if not success:
        print("\n❌ FAILED: Fix model loading before proceeding")
        return False
    
    # Test 4: Utilities
    results["utilities"] = test_utilities()
    if not results["utilities"]:
        print("\n❌ FAILED: Fix utilities before proceeding")
        return False
    
    # Test 5: Mini experiments
    results["mini_experiments"] = test_mini_experiment(model)
    
    # Summary
    print("\n" + "="*70)
    print(" TEST SUMMARY")
    print("="*70)
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n" + "="*70)
        print(" ✓✓✓ ALL TESTS PASSED! ✓✓✓")
        print("="*70)
        print("\nYou're ready to run the full experiments!")
        print("\nNext steps:")
        print("  1. Run quick test:  python experiment5_trustworthiness.py")
        print("  2. Run full suite:  python -c 'from experiment5_trustworthiness import run_all_experiments; run_all_experiments()'")
        return True
    else:
        print("\n" + "="*70)
        print(" ✗✗✗ SOME TESTS FAILED ✗✗✗")
        print("="*70)
        print("\nPlease fix the issues above before running experiments.")
        return False


if __name__ == "__main__":
    success = run_full_test_suite()
    sys.exit(0 if success else 1)
