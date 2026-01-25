"""
Fix inverted steering vectors by flipping their sign.

PROBLEM: Current vectors point from "unanswerable" → "answerable"
NEEDED: Vectors should point from "answerable" → "unanswerable"

SOLUTION: Multiply all vectors by -1
"""

import torch
from pathlib import Path

def flip_vectors(input_file, output_file):
    """Flip all steering vectors by multiplying by -1"""

    print(f"Loading vectors from {input_file}...")
    vectors = torch.load(input_file)

    print(f"Flipping {len(vectors)} vectors...")
    flipped = {}
    for layer_idx, vec in vectors.items():
        flipped[layer_idx] = -vec  # Flip sign
        print(f"  Layer {layer_idx}: norm before={torch.norm(vec).item():.3f}, norm after={torch.norm(flipped[layer_idx]).item():.3f}")

    print(f"\nSaving flipped vectors to {output_file}...")
    torch.save(flipped, output_file)

    print(f"✓ Done! Flipped vectors saved to {output_file}")

    # Verify
    print("\nVerification:")
    reloaded = torch.load(output_file)
    for layer_idx in vectors.keys():
        v1 = vectors[layer_idx].flatten()
        v2 = reloaded[layer_idx].flatten()
        cos_sim = (v1 @ v2) / (torch.norm(v1) * torch.norm(v2))
        print(f"  Layer {layer_idx}: cos_sim = {cos_sim.item():.3f} (should be close to -1.0)")

if __name__ == "__main__":
    results_dir = Path("results")

    # Flip both vector sets
    print("="*70)
    print("FLIPPING ORIGINAL VECTORS")
    print("="*70)
    flip_vectors(
        results_dir / "steering_vectors.pt",
        results_dir / "steering_vectors_flipped.pt"
    )

    print("\n" + "="*70)
    print("FLIPPING CALIBRATED VECTORS")
    print("="*70)
    flip_vectors(
        results_dir / "steering_vectors_calibrated.pt",
        results_dir / "steering_vectors_calibrated_flipped.pt"
    )

    print("\n" + "="*70)
    print("✅ ALL VECTORS FLIPPED")
    print("="*70)
    print("\nNext step:")
    print("  Update experiment6_publication_ready.py to use:")
    print("    - steering_vectors_flipped.pt")
    print("  or")
    print("    - steering_vectors_calibrated_flipped.pt")
