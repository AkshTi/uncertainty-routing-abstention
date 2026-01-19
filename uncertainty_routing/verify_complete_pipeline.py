#!/usr/bin/env python3
"""
Complete Pipeline Verification Script

This script performs comprehensive checks to ensure the complete pipeline (Exp1-9)
will run without errors. Run this BEFORE starting the full pipeline.

Usage:
    python verify_complete_pipeline.py
"""

import sys
import json
from pathlib import Path

def color_print(msg, status):
    """Print with color coding"""
    colors = {
        'success': '\033[92m',  # Green
        'warning': '\033[93m',  # Yellow
        'error': '\033[91m',    # Red
        'info': '\033[94m',     # Blue
        'end': '\033[0m'
    }
    print(f"{colors.get(status, '')}{msg}{colors['end']}")

print("\n" + "="*80)
print(" COMPLETE PIPELINE VERIFICATION")
print("="*80 + "\n")

all_checks_passed = True
warnings = []

# ============================================================================
# CHECK 1: Import Verification
# ============================================================================
print("[1/8] Verifying imports...")

try:
    # Core utilities
    from core_utils import ModelWrapper, ExperimentConfig, set_seed
    color_print("  ✓ Core utilities imported", 'success')
except ImportError as e:
    color_print(f"  ✗ Core utilities failed: {e}", 'error')
    all_checks_passed = False

try:
    # Data preparation
    from data_preparation import (
        create_ambiguous_prompts,
        create_clearly_answerable_questions,
        create_clearly_unanswerable_questions
    )
    color_print("  ✓ Data preparation imported", 'success')
except ImportError as e:
    color_print(f"  ✗ Data preparation failed: {e}", 'error')
    all_checks_passed = False

experiments_ok = True
for i in range(1, 10):
    try:
        if i == 1:
            from experiment1_behavior_belief import Experiment1
        elif i == 2:
            from experiment2_localization import Experiment2
        elif i in [3, 4]:
            from experiment3_4_steering_independence import Experiment3, Experiment4
        elif i == 5:
            from experiment5_trustworthiness import Experiment5
        elif i == 6:
            from experiment6_robustness import Experiment6
        elif i == 7:
            from experiment7_safety_alignment import Experiment7
        elif i == 8:
            from experiment8_scaling_analysis import Experiment8
        elif i == 9:
            from experiment9_interpretability import Experiment9
    except ImportError as e:
        color_print(f"  ✗ Experiment {i} failed: {e}", 'error')
        all_checks_passed = False
        experiments_ok = False
        break

if experiments_ok:
    color_print("  ✓ All experiments (1-9) imported successfully", 'success')

# ============================================================================
# CHECK 2: Data Files
# ============================================================================
print("\n[2/8] Checking data files...")

data_files = [
    "./data/dataset_clearly_answerable.json",
    "./data/dataset_clearly_unanswerable.json",
    "./data/dataset_ambiguous.json",
]

missing_files = []
for file_path in data_files:
    if not Path(file_path).exists():
        missing_files.append(file_path)
        color_print(f"  ✗ Missing: {file_path}", 'error')
    else:
        # Verify format
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            if isinstance(data, list) and len(data) > 0:
                color_print(f"  ✓ {file_path} ({len(data)} items)", 'success')
            else:
                color_print(f"  ⚠ {file_path} is empty or invalid format", 'warning')
                warnings.append(f"Data file {file_path} may be invalid")
        except Exception as e:
            color_print(f"  ✗ Cannot read {file_path}: {e}", 'error')
            all_checks_passed = False

if missing_files:
    color_print(f"\n  ⚠ Missing {len(missing_files)} data files", 'warning')
    color_print(f"  You may need to run: python data_preparation.py", 'info')
    all_checks_passed = False

# ============================================================================
# CHECK 3: GPU Availability
# ============================================================================
print("\n[3/8] Checking GPU...")

try:
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        color_print(f"  ✓ CUDA available: {gpu_name}", 'success')
        color_print(f"  ✓ GPU Memory: {gpu_memory:.2f} GB", 'success')

        # Memory recommendations
        if gpu_memory < 8:
            color_print(f"  ⚠ Low GPU memory. Use --mode quick or smaller models", 'warning')
            warnings.append("Low GPU memory - may need to use quick mode")
        elif gpu_memory >= 24:
            color_print(f"  ✓ Sufficient memory for all model sizes (1.5B-7B)", 'success')
        elif gpu_memory >= 16:
            color_print(f"  ✓ Sufficient memory for 1.5B and 3B models", 'success')
        else:
            color_print(f"  ✓ Sufficient memory for 1.5B model", 'success')
    else:
        color_print("  ⚠ No GPU available (will use CPU - VERY SLOW)", 'warning')
        warnings.append("No GPU - pipeline will be extremely slow")
except Exception as e:
    color_print(f"  ✗ GPU check failed: {e}", 'error')

# ============================================================================
# CHECK 4: Configuration
# ============================================================================
print("\n[4/8] Testing configuration...")

try:
    config = ExperimentConfig()
    color_print(f"  ✓ Config created successfully", 'success')
    color_print(f"    Model: {config.model_name}", 'info')
    color_print(f"    Device: {config.device}", 'info')
    color_print(f"    Results dir: {config.results_dir}", 'info')

    # Check/create results directory
    if not config.results_dir.exists():
        config.results_dir.mkdir(parents=True, exist_ok=True)
        color_print(f"  ✓ Created results directory", 'success')
    else:
        color_print(f"  ✓ Results directory exists", 'success')
except Exception as e:
    color_print(f"  ✗ Config creation failed: {e}", 'error')
    all_checks_passed = False

# ============================================================================
# CHECK 5: Data Loading Test
# ============================================================================
print("\n[5/8] Testing data loading...")

try:
    answerable = create_clearly_answerable_questions()
    unanswerable = create_clearly_unanswerable_questions()
    ambiguous = create_ambiguous_prompts()

    color_print(f"  ✓ Loaded {len(answerable)} answerable questions", 'success')
    color_print(f"  ✓ Loaded {len(unanswerable)} unanswerable questions", 'success')
    color_print(f"  ✓ Loaded {len(ambiguous)} ambiguous prompts", 'success')

    # Verify format
    if len(answerable) > 0:
        q = answerable[0]
        if 'question' in q and 'answer' in q:
            color_print(f"  ✓ Data format valid", 'success')
        else:
            color_print(f"  ✗ Data format invalid (missing required fields)", 'error')
            all_checks_passed = False

except Exception as e:
    color_print(f"  ✗ Data loading failed: {e}", 'error')
    import traceback
    traceback.print_exc()
    all_checks_passed = False

# ============================================================================
# CHECK 6: Existing Results
# ============================================================================
print("\n[6/8] Checking existing results...")

exp_files = [
    ("exp1_summary.json", "Experiment 1: Behavior-Belief"),
    ("exp2_summary.json", "Experiment 2: Localization"),
    ("exp3_summary.json", "Experiment 3: Steering"),
    ("exp4_summary.json", "Experiment 4: Gate Independence"),
    ("exp5_summary.json", "Experiment 5: Trustworthiness"),
    ("steering_vectors_explicit.pt", "Steering Vectors (for Exp6-9)"),
]

completed = []
for file_name, description in exp_files:
    path = config.results_dir / file_name
    if path.exists():
        color_print(f"  ✓ {description} completed", 'success')
        completed.append(file_name)

if len(completed) == 0:
    color_print(f"  → No experiments completed yet (starting fresh)", 'info')
    color_print(f"  → Run: python run_complete_pipeline_v2.py --mode standard", 'info')
elif len(completed) == len(exp_files):
    color_print(f"  ✓ All base experiments (1-5) completed!", 'success')
    color_print(f"  → You can run critical experiments only:", 'info')
    color_print(f"  → python run_complete_pipeline_v2.py --only-critical", 'info')
else:
    color_print(f"  → {len(completed)}/{len(exp_files)} experiments completed", 'info')
    color_print(f"  → You may want to use --skip-exp1, --skip-exp2, etc.", 'info')

# ============================================================================
# CHECK 7: Pipeline Script
# ============================================================================
print("\n[7/8] Checking pipeline script...")

pipeline_script = Path("run_complete_pipeline_v2.py")
if pipeline_script.exists():
    color_print(f"  ✓ Pipeline script exists", 'success')

    # Check it's executable (or at least readable)
    try:
        with open(pipeline_script, 'r') as f:
            content = f.read()
        if 'def main()' in content and 'run_exp1' in content:
            color_print(f"  ✓ Pipeline script appears valid", 'success')
        else:
            color_print(f"  ⚠ Pipeline script may be incomplete", 'warning')
    except Exception as e:
        color_print(f"  ✗ Cannot read pipeline script: {e}", 'error')
        all_checks_passed = False
else:
    color_print(f"  ✗ Pipeline script not found", 'error')
    all_checks_passed = False

# ============================================================================
# CHECK 8: Dependencies
# ============================================================================
print("\n[8/8] Checking Python dependencies...")

required_packages = [
    ('torch', 'PyTorch'),
    ('transformers', 'HuggingFace Transformers'),
    ('pandas', 'Pandas'),
    ('matplotlib', 'Matplotlib'),
    ('seaborn', 'Seaborn'),
    ('numpy', 'NumPy'),
    ('tqdm', 'TQDM'),
]

missing_packages = []
for package, name in required_packages:
    try:
        __import__(package)
        color_print(f"  ✓ {name}", 'success')
    except ImportError:
        color_print(f"  ✗ {name} not installed", 'error')
        missing_packages.append(package)
        all_checks_passed = False

if missing_packages:
    color_print(f"\n  Install missing packages with:", 'info')
    color_print(f"  pip install {' '.join(missing_packages)}", 'info')

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print(" VERIFICATION SUMMARY")
print("="*80 + "\n")

if all_checks_passed and len(warnings) == 0:
    color_print("✓ ALL CHECKS PASSED - READY TO RUN PIPELINE!", 'success')
    print()

    if len(completed) >= 5:
        color_print("RECOMMENDED COMMAND (critical experiments only):", 'info')
        print("  python run_complete_pipeline_v2.py --only-critical")
        print()
        color_print("This will run Exp8+9 (7 hours, +55% acceptance boost)", 'info')
    else:
        color_print("RECOMMENDED COMMAND (full pipeline):", 'info')
        print("  python run_complete_pipeline_v2.py --mode standard")
        print()
        color_print("This will run all experiments (10-14 hours)", 'info')

    print()
    print("MODES:")
    print("  --mode quick     : 3-4 hours, 10 questions per experiment")
    print("  --mode standard  : 10-14 hours, 30 questions per experiment (RECOMMENDED)")
    print("  --mode full      : 15-20 hours, 50 questions per experiment")

elif all_checks_passed and len(warnings) > 0:
    color_print("⚠ CHECKS PASSED WITH WARNINGS", 'warning')
    print()
    print("Warnings:")
    for w in warnings:
        color_print(f"  • {w}", 'warning')
    print()
    color_print("You can proceed but may encounter issues.", 'info')

else:
    color_print("✗ SOME CHECKS FAILED", 'error')
    print()
    print("Please fix the errors above before running the pipeline.")
    print()
    sys.exit(1)

print()
print("="*80)
print()
