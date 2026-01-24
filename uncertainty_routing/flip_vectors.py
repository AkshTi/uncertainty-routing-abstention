"""
Flip the steering vectors (multiply by -1)

This creates new vectors where:
- +ε pushes toward UNCERTAINTY (abstain more)
- -ε pushes toward CONFIDENCE (abstain less)
"""

import torch
from pathlib import Path

results_dir = Path("results")

# Load original vectors
steering_vectors = torch.load(results_dir / "steering_vectors.pt")

print("="*70)
print("FLIPPING STEERING VECTORS")
print("="*70)
print()

print(f"Original vectors: {list(steering_vectors.keys())}")
print()

# Flip all vectors
flipped_vectors = {}
for layer, vec in steering_vectors.items():
    flipped_vectors[layer] = -vec  # Multiply by -1
    print(f"Layer {layer}:")
    print(f"  Original norm: {vec.norm():.4f}")
    print(f"  Flipped norm: {flipped_vectors[layer].norm():.4f}")

# Save flipped vectors
output_path = results_dir / "steering_vectors_flipped.pt"
torch.save(flipped_vectors, output_path)

print()
print(f"✓ Saved flipped vectors to {output_path}")
print()
print("Now re-run experiment6_publication_ready.py and change:")
print("  possible_files = [")
print("      \"steering_vectors_flipped.pt\",  # Try flipped first")
print("      \"steering_vectors.pt\",")
print("      ...")
print("  ]")
