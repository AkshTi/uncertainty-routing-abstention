#!/usr/bin/env python3
"""
Fix Optimal Epsilon in Exp5 Summary

Problem: Exp5 selects epsilon=-50 as "best", but this breaks the model.
Solution: Override best_eps_value to -10 (the actual optimal from analysis).

Usage:
    python fix_optimal_epsilon.py
"""

import json
from pathlib import Path

print("\n" + "="*80)
print(" FIXING OPTIMAL EPSILON IN EXP5 SUMMARY")
print("="*80 + "\n")

# Load exp5_summary.json
summary_path = Path("results/exp5_summary.json")

if not summary_path.exists():
    print(f"✗ {summary_path} not found!")
    print("Run Exp5 first")
    exit(1)

print(f"Loading {summary_path}...")
with open(summary_path, 'r') as f:
    summary = json.load(f)

# Show current "best" epsilon
current_best = summary['best_eps_value']
print(f"\nCurrent best_eps_value: {current_best}")

# Find metrics for epsilon=-10
epsilon_minus_10 = None
for metric in summary['metrics']:
    if metric['epsilon'] == -10.0:
        epsilon_minus_10 = metric
        break

if not epsilon_minus_10:
    print("✗ Could not find epsilon=-10 metrics!")
    exit(1)

print(f"\nMetrics at epsilon=-10:")
print(f"  Coverage: {epsilon_minus_10['coverage_answerable']:.1%}")
print(f"  Accuracy: {epsilon_minus_10['accuracy_answerable']:.1%}")
print(f"  Abstention on unanswerable: {epsilon_minus_10['abstain_unanswerable']:.1%}")
print(f"  Hallucination on unanswerable: {epsilon_minus_10['hallucination_unanswerable']:.1%}")

# Update to use epsilon=-10 as optimal
print(f"\nUpdating best_eps_value from {current_best} to -10.0...")

summary['best_eps_value'] = -10.0
summary['best_eps_coverage_answerable'] = epsilon_minus_10['coverage_answerable']
summary['best_eps_accuracy_answerable'] = epsilon_minus_10['accuracy_answerable']
summary['best_eps_abstain_unanswerable'] = epsilon_minus_10['abstain_unanswerable']
summary['best_eps_hallucination_unanswerable'] = epsilon_minus_10['hallucination_unanswerable']

# Recalculate deltas
baseline_coverage = summary['baseline_coverage_answerable']
baseline_abstain = summary['baseline_abstain_unanswerable']

summary['delta_coverage_answerable'] = epsilon_minus_10['coverage_answerable'] - baseline_coverage
summary['delta_abstain_unanswerable'] = epsilon_minus_10['abstain_unanswerable'] - baseline_abstain
summary['delta_accuracy_answerable'] = epsilon_minus_10['accuracy_answerable'] - summary['baseline_accuracy_answerable']
summary['delta_hallucination_unanswerable'] = epsilon_minus_10['hallucination_unanswerable'] - summary['baseline_hallucination_unanswerable']

# Save updated summary
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"✓ Saved updated summary to {summary_path}")

print("\n" + "="*80)
print(" SUMMARY OF CHANGES")
print("="*80 + "\n")

print(f"Best epsilon: {current_best} → -10.0")
print(f"\nNew optimal performance:")
print(f"  Coverage improvement: {summary['delta_coverage_answerable']:+.1%}")
print(f"  Accuracy change: {summary['delta_accuracy_answerable']:+.1%}")
print(f"  Abstention improvement: {summary['delta_abstain_unanswerable']:+.1%}")
print(f"  Hallucination change: {summary['delta_hallucination_unanswerable']:+.1%}")

print("\n" + "="*80)
print(" ✅ DONE!")
print("="*80 + "\n")

print("Next steps:")
print("  1. Copy this updated file to SSH:")
print("     scp results/exp5_summary.json user@ssh:~/path/results/")
print("  2. Re-run Exp6-7 (they will now use epsilon=-10):")
print("     ./run_segment3.sh")
print("  3. Run Exp8-9:")
print("     ./run_segment4a.sh")
print("     ./run_segment4b.sh")
print()
