#!/usr/bin/env python3
"""
Quick Fix: Flip Steering Vectors

Based on your Exp5 results, negative epsilon is making things worse.
This script simply flips the sign of all steering vectors.

After running this:
- Negative epsilon will reduce hallucinations (correct behavior)
- Re-run Exp5 and Exp8 to get correct results

Usage:
    python quick_fix_flip_vectors.py
"""

import torch
from pathlib import Path

print("\n" + "="*80)
print(" QUICK FIX: FLIP STEERING VECTORS")
print("="*80 + "\n")

results_dir = Path("results")
vectors_path = results_dir / "steering_vectors.pt"
vectors_explicit_path = results_dir / "steering_vectors_explicit.pt"

# Load existing vectors
if not vectors_path.exists():
    print(f"✗ {vectors_path} not found!")
    exit(1)

print(f"Loading {vectors_path}...")
vectors = torch.load(vectors_path)

print(f"Layers found: {list(vectors.keys())}")
print(f"Vector shapes: {[(k, v.shape) for k, v in vectors.items()]}")

# Flip signs
print("\nFlipping vector signs...")
flipped_vectors = {k: -v for k, v in vectors.items()}

# Save both files
print(f"\nSaving flipped vectors...")
torch.save(flipped_vectors, vectors_path)
print(f"✓ Saved to {vectors_path}")

torch.save(flipped_vectors, vectors_explicit_path)
print(f"✓ Saved to {vectors_explicit_path}")

print("\n" + "="*80)
print(" ✅ DONE - Vectors flipped!")
print("="*80 + "\n")

print("What changed:")
print("  - OLD: negative ε increased hallucinations (wrong!)")
print("  - NEW: negative ε will reduce hallucinations (correct!)")
print()
print("Next steps:")
print("  1. Copy this script to your SSH server")
print("  2. Run: python quick_fix_flip_vectors.py")
print("  3. Re-run: ./run_segment2.sh (will re-do Exp5 with correct vectors)")
print("  4. Run: ./run_segment4a.sh (Exp8 will now work)")
print("  5. Run: ./run_segment4b.sh (Exp9 will use correct vectors)")
print()
