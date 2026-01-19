"""
Quick script to find optimal epsilon value from your results
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load your results
df = pd.read_csv("results/exp5_raw_results.csv")

# Compute metrics per epsilon
metrics = []
for eps in sorted(df["epsilon"].unique()):
    sub = df[df["epsilon"] == eps]
    
    ans = sub[~sub["is_unanswerable"]]
    unans = sub[sub["is_unanswerable"]]
    
    coverage_ans = 1.0 - ans["abstained"].mean() if len(ans) else 0.0
    accuracy_ans = ans[~ans["abstained"]]["correct"].mean() if len(ans[~ans["abstained"]]) else 0.0
    abstain_unans = unans["abstained"].mean() if len(unans) else 0.0
    halluc_unans = unans["hallucinated"].mean() if len(unans) else 0.0
    
    metrics.append({
        "epsilon": eps,
        "coverage_answerable": coverage_ans,
        "accuracy_answerable": accuracy_ans,
        "abstain_unanswerable": abstain_unans,
        "hallucination_unanswerable": halluc_unans,
    })

mdf = pd.DataFrame(metrics)

print("=" * 80)
print("FINDING OPTIMAL EPSILON")
print("=" * 80)

# Baseline
baseline = mdf[mdf["epsilon"] == 0.0].iloc[0]
print(f"\nBaseline (ε=0):")
print(f"  Coverage: {baseline['coverage_answerable']:.1%}")
print(f"  Hallucination: {baseline['hallucination_unanswerable']:.1%}")

# Only consider negative epsilons (increase abstention)
negative_eps = mdf[mdf["epsilon"] < 0].copy()

print("\n" + "=" * 80)
print("CANDIDATES (negative epsilon only):")
print("=" * 80)

# Calculate deltas and score each option
for idx, row in negative_eps.iterrows():
    delta_cov = row["coverage_answerable"] - baseline["coverage_answerable"]
    delta_halluc = row["hallucination_unanswerable"] - baseline["hallucination_unanswerable"]
    
    # Improved scoring: balance both metrics equally
    # Normalize to [0, 1] scale:
    # - halluc_reduction: higher is better (0.367 -> 0 is best possible)
    # - cov_retention: higher is better (0 loss is best)
    
    halluc_reduction = -delta_halluc / baseline["hallucination_unanswerable"]  # 0 to 1
    cov_retention = max(0, 1 + delta_cov / baseline["coverage_answerable"])  # 0 to 1
    
    # Balanced score: equal weight to both objectives
    # But require minimum coverage threshold (>40% = >0.67 retention)
    if cov_retention < 0.67:
        score = -999  # Disqualify if coverage drops too much
    else:
        score = 0.5 * halluc_reduction + 0.5 * cov_retention
    
    print(f"\nε = {row['epsilon']:.0f}:")
    print(f"  Coverage: {row['coverage_answerable']:.1%} (Δ={delta_cov:+.1%})")
    print(f"  Accuracy: {row['accuracy_answerable']:.1%}")
    print(f"  Abstain unans: {row['abstain_unanswerable']:.1%}")
    print(f"  Hallucinate: {row['hallucination_unanswerable']:.1%} (Δ={delta_halluc:+.1%})")
    print(f"  Halluc reduction: {halluc_reduction:.1%}, Cov retention: {cov_retention:.1%}")
    print(f"  Score: {score:.3f}")

# Find best by score
negative_eps["delta_cov"] = negative_eps["coverage_answerable"] - baseline["coverage_answerable"]
negative_eps["delta_halluc"] = negative_eps["hallucination_unanswerable"] - baseline["hallucination_unanswerable"]
negative_eps["halluc_reduction"] = -negative_eps["delta_halluc"] / baseline["hallucination_unanswerable"]
negative_eps["cov_retention"] = (1 + negative_eps["delta_cov"] / baseline["coverage_answerable"]).apply(lambda x: max(0, x))

# Disqualify if coverage drops below 40%
negative_eps["score"] = negative_eps.apply(
    lambda row: -999 if row["cov_retention"] < 0.67 
    else 0.5 * row["halluc_reduction"] + 0.5 * row["cov_retention"],
    axis=1
)

best = negative_eps.loc[negative_eps["score"].idxmax()]

print("\n" + "=" * 80)
print("RECOMMENDATION (Balanced Coverage + Hallucination Reduction)")
print("=" * 80)
print(f"\n🎯 Optimal epsilon: {best['epsilon']:.0f}")
print(f"\n   Coverage: {best['coverage_answerable']:.1%} "
      f"(Δ={best['delta_cov']:+.1%} from baseline)")
print(f"   Accuracy: {best['accuracy_answerable']:.1%}")
print(f"   Hallucination: {best['hallucination_unanswerable']:.1%} "
      f"(Δ={best['delta_halluc']:+.1%} from baseline)")
print(f"   Abstain on unanswerables: {best['abstain_unanswerable']:.1%}")

print("\n📊 Tradeoff summary:")
halluc_reduction_pct = best['halluc_reduction'] * 100
cov_loss_pct = (1 - best['cov_retention']) * 100
print(f"   ✓ Reduces hallucination by {halluc_reduction_pct:.0f}%")
if cov_loss_pct > 0:
    print(f"   ⚠ Costs {cov_loss_pct:.0f}% of coverage on answerables")
else:
    print(f"   ✓ Maintains coverage on answerables")

print("\n💡 Alternative options:")
print("=" * 80)

# Show top 3 alternatives
top3 = negative_eps.nlargest(3, "score")
for idx, row in top3.iterrows():
    if row["epsilon"] == best["epsilon"]:
        continue
    print(f"\nε = {row['epsilon']:.0f} (score: {row['score']:.3f}):")
    print(f"   Coverage: {row['coverage_answerable']:.1%}, "
          f"Hallucination: {row['hallucination_unanswerable']:.1%}")
    print(f"   {row['halluc_reduction']*100:.0f}% halluc reduction, "
          f"{(1-row['cov_retention'])*100:.0f}% coverage loss")

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: Coverage-Hallucination tradeoff
ax1.scatter(negative_eps["coverage_answerable"], 
           negative_eps["hallucination_unanswerable"],
           c=negative_eps["epsilon"], cmap="RdYlGn_r", s=100, edgecolors="black")
ax1.scatter(baseline["coverage_answerable"], 
           baseline["hallucination_unanswerable"],
           c="blue", s=200, marker="*", edgecolors="black", 
           label="Baseline", zorder=5)
ax1.scatter(best["coverage_answerable"],
           best["hallucination_unanswerable"],
           c="green", s=200, marker="D", edgecolors="black",
           label="Optimal", zorder=5)

ax1.set_xlabel("Coverage on Answerable (%)", fontsize=12)
ax1.set_ylabel("Hallucination on Unanswerable (%)", fontsize=12)
ax1.set_title("Risk-Coverage Tradeoff", fontsize=13, fontweight="bold")
ax1.legend()
ax1.grid(alpha=0.3)

# Add colorbar
cbar = plt.colorbar(ax1.collections[0], ax=ax1)
cbar.set_label("Epsilon", fontsize=11)

# Panel 2: Metrics vs epsilon
ax2.plot(negative_eps["epsilon"], negative_eps["coverage_answerable"], 
        marker="o", label="Coverage (ans)", linewidth=2, color="#3498db")
ax2.plot(negative_eps["epsilon"], negative_eps["abstain_unanswerable"],
        marker="s", label="Abstain (unans)", linewidth=2, color="#9b59b6")
ax2.plot(negative_eps["epsilon"], negative_eps["hallucination_unanswerable"],
        marker="^", label="Hallucinate (unans)", linewidth=2, color="#e74c3c")

ax2.axvline(best["epsilon"], color="green", linestyle="--", alpha=0.7, linewidth=2,
           label=f"Optimal (ε={best['epsilon']:.0f})")
ax2.axvline(0, color="gray", linestyle=":", alpha=0.5)

ax2.set_xlabel("Epsilon", fontsize=12)
ax2.set_ylabel("Rate", fontsize=12)
ax2.set_title("Metrics vs Steering Strength", fontsize=13, fontweight="bold")
ax2.legend()
ax2.grid(alpha=0.3)
ax2.set_ylim([0, 1])

plt.tight_layout()
plt.savefig("results/exp5_optimal_epsilon.png", dpi=300, bbox_inches="tight")
print(f"\n📈 Visualization saved to results/exp5_optimal_epsilon.png")