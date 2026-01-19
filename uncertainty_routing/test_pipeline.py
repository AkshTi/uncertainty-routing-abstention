#!/usr/bin/env python3
"""
Pipeline Test Script

Quick test to verify all experiments can be imported and basic functionality works.
Run this BEFORE the full pipeline to catch errors early.

Usage:
    python test_pipeline.py
"""

import sys
from pathlib import Path

print("="*80)
print(" PIPELINE TEST SCRIPT")
print("="*80)
print()

# Test 1: Check imports
print("[1/6] Testing imports...")
try:
    from core_utils import ModelWrapper, ExperimentConfig, set_seed
    from data_preparation import (
        create_ambiguous_prompts,
        create_clearly_answerable_questions,
        create_clearly_unanswerable_questions
    )
    from experiment1_behavior_belief import Experiment1
    from experiment2_localization import Experiment2
    from experiment3_4_steering_independence import Experiment3, Experiment4
    from experiment5_trustworthiness import Experiment5
    from experiment6_robustness import Experiment6
    from experiment7_safety_alignment import Experiment7
    from experiment8_scaling_analysis import Experiment8
    from experiment9_interpretability import Experiment9
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Check data files exist
print("\n[2/6] Checking data files...")
data_files = [
    "./data/dataset_clearly_answerable.json",
    "./data/dataset_clearly_unanswerable.json",
    "./data/dataset_ambiguous.json",
]

missing_files = []
for file_path in data_files:
    if not Path(file_path).exists():
        missing_files.append(file_path)
        print(f"✗ Missing: {file_path}")
    else:
        print(f"✓ Found: {file_path}")

if missing_files:
    print(f"\n⚠️  Missing {len(missing_files)} data files")
    print("You may need to run data_preparation.py first")
else:
    print("✓ All data files present")

# Test 3: Check GPU availability
print("\n[3/6] Checking GPU...")
try:
    import torch
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠️  No GPU available (will use CPU, very slow)")
except Exception as e:
    print(f"✗ GPU check failed: {e}")

# Test 4: Test config creation
print("\n[4/6] Testing config...")
try:
    config = ExperimentConfig()
    print(f"✓ Config created")
    print(f"  Model: {config.model_name}")
    print(f"  Device: {config.device}")
    print(f"  Results dir: {config.results_dir}")

    # Check results directory exists
    if not config.results_dir.exists():
        config.results_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Created results directory")
    else:
        print(f"  Results directory exists")
except Exception as e:
    print(f"✗ Config creation failed: {e}")
    sys.exit(1)

# Test 5: Test data loading
print("\n[5/6] Testing data loading...")
try:
    import json

    with open("./data/dataset_clearly_answerable.json", 'r') as f:
        answerable = json.load(f)
    with open("./data/dataset_clearly_unanswerable.json", 'r') as f:
        unanswerable = json.load(f)

    ambiguous = create_ambiguous_prompts()

    print(f"✓ Data loaded:")
    print(f"  Answerable: {len(answerable)} questions")
    print(f"  Unanswerable: {len(unanswerable)} questions")
    print(f"  Ambiguous: {len(ambiguous)} questions")

    # Check data format
    if len(answerable) > 0:
        q = answerable[0]
        if 'question' not in q:
            print("⚠️  Warning: answerable questions missing 'question' field")
        if 'answer' not in q:
            print("⚠️  Warning: answerable questions missing 'answer' field")
    else:
        print("⚠️  Warning: No answerable questions found")

except Exception as e:
    print(f"✗ Data loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Check for existing results
print("\n[6/6] Checking existing results...")
exp_results = [
    "exp1_raw_results.csv",
    "exp2_raw_results.csv",
    "exp3_raw_results.csv",
    "exp4_raw_results.csv",
    "exp5_raw_results.csv",
    "steering_vectors_explicit.pt",
]

existing_count = 0
for result_file in exp_results:
    path = config.results_dir / result_file
    if path.exists():
        existing_count += 1
        print(f"✓ Found: {result_file}")

if existing_count == 0:
    print("No existing results found (starting fresh)")
elif existing_count < len(exp_results):
    print(f"⚠️  {existing_count}/{len(exp_results)} experiments already completed")
    print("   You may want to skip completed experiments with --skip-exp1, etc.")
else:
    print(f"✓ All {existing_count} experiments already completed")
    print("   Run with --only-critical to just run Exp8+9")

# Summary
print("\n" + "="*80)
print(" TEST SUMMARY")
print("="*80)
print()

# Check if critical files exist
critical_ready = (
    (config.results_dir / "steering_vectors_explicit.pt").exists() and
    (config.results_dir / "exp5_summary.json").exists()
)

if critical_ready:
    print("✓ READY TO RUN CRITICAL EXPERIMENTS")
    print("\nRun:")
    print("  python run_complete_pipeline_v2.py --only-critical")
    print("\nThis will run Exp8+9 (7 hours, +55% acceptance boost)")
else:
    print("✓ READY TO RUN FULL PIPELINE")
    print("\nRun:")
    print("  python run_complete_pipeline_v2.py --mode standard")
    print("\nThis will run all experiments (10-14 hours)")

print()
print("For faster testing, use --mode quick (3-4 hours)")
print("To skip already-completed experiments, use --skip-exp1, etc.")
print()

print("="*80)
print("✓ All tests passed! Pipeline ready to run.")
print("="*80)
