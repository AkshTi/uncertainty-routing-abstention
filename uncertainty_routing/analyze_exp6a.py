import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Read the CSV file
csv_path = "/Users/akshatatiwari/Desktop/MIT/mech_interp_research/uncertainty_routing/results/exp6a_cross_domain_publication_ready.csv"
df = pd.read_csv(csv_path)

# Display basic info
print("=" * 80)
print("DATASET OVERVIEW")
print("=" * 80)
print(f"Total rows: {len(df)}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nConditions: {df['condition'].unique()}")
print(f"Domains: {df['domain'].unique()}")
print(f"Epsilon values: {df['epsilon'].unique()}")
print(f"\nIs_unanswerable distribution:\n{df['is_unanswerable'].value_counts()}")
print(f"\nCondition distribution:\n{df['condition'].value_counts()}")
print(f"\nDomain distribution:\n{df['domain'].value_counts()}")

# Separate answerable and unanswerable questions
answerable = df[df['is_unanswerable'] == False]
unanswerable = df[df['is_unanswerable'] == True]

print("\n" + "=" * 80)
print("1. OVERALL METRICS BY CONDITION")
print("=" * 80)

# Overall metrics
for condition in ['baseline', 'steered']:
    print(f"\n{condition.upper()} (epsilon={df[df['condition']==condition]['epsilon'].iloc[0]}):")
    print("-" * 40)

    # Answerable questions metrics
    ans_cond = answerable[answerable['condition'] == condition]
    print(f"\nAnswerable Questions (n={len(ans_cond)}):")
    print(f"  Accuracy: {ans_cond['correct'].mean():.3f} ({ans_cond['correct'].sum()}/{len(ans_cond)})")
    print(f"  Abstention Rate: {ans_cond['abstained'].mean():.3f} ({ans_cond['abstained'].sum()}/{len(ans_cond)})")

    # Unanswerable questions metrics
    unans_cond = unanswerable[unanswerable['condition'] == condition]
    print(f"\nUnanswerable Questions (n={len(unans_cond)}):")
    print(f"  Abstention Rate: {unans_cond['abstained'].mean():.3f} ({unans_cond['abstained'].sum()}/{len(unans_cond)})")
    print(f"  Hallucination Rate: {unans_cond['hallucinated'].mean():.3f} ({unans_cond['hallucinated'].sum()}/{len(unans_cond)})")

print("\n" + "=" * 80)
print("2. METRICS BY DOMAIN")
print("=" * 80)

# Create a summary dataframe
summary_data = []

for domain in sorted(df['domain'].unique()):
    for condition in ['baseline', 'steered']:
        domain_cond_df = df[(df['domain'] == domain) & (df['condition'] == condition)]

        # Answerable metrics
        ans_domain = domain_cond_df[domain_cond_df['is_unanswerable'] == False]
        ans_accuracy = ans_domain['correct'].mean() if len(ans_domain) > 0 else 0
        ans_abstention = ans_domain['abstained'].mean() if len(ans_domain) > 0 else 0

        # Unanswerable metrics
        unans_domain = domain_cond_df[domain_cond_df['is_unanswerable'] == True]
        unans_abstention = unans_domain['abstained'].mean() if len(unans_domain) > 0 else 0
        unans_hallucination = unans_domain['hallucinated'].mean() if len(unans_domain) > 0 else 0

        summary_data.append({
            'domain': domain,
            'condition': condition,
            'ans_n': len(ans_domain),
            'ans_accuracy': ans_accuracy,
            'ans_abstention': ans_abstention,
            'unans_n': len(unans_domain),
            'unans_abstention': unans_abstention,
            'unans_hallucination': unans_hallucination
        })

summary_df = pd.DataFrame(summary_data)

# Print by domain
for domain in sorted(df['domain'].unique()):
    print(f"\n{domain.upper()}:")
    print("-" * 60)
    domain_summary = summary_df[summary_df['domain'] == domain]

    for _, row in domain_summary.iterrows():
        print(f"\n  {row['condition'].capitalize()}:")
        print(f"    Answerable (n={int(row['ans_n'])}): Acc={row['ans_accuracy']:.3f}, Abst={row['ans_abstention']:.3f}")
        print(f"    Unanswerable (n={int(row['unans_n'])}): Abst={row['unans_abstention']:.3f}, Hall={row['unans_hallucination']:.3f}")

print("\n" + "=" * 80)
print("3. COMPARISON: BASELINE vs STEERED")
print("=" * 80)

baseline_summary = summary_df[summary_df['condition'] == 'baseline']
steered_summary = summary_df[summary_df['condition'] == 'steered']

print("\nChange in Abstention Rate on Unanswerable Questions:")
print("-" * 60)
for domain in sorted(df['domain'].unique()):
    base_rate = baseline_summary[baseline_summary['domain'] == domain]['unans_abstention'].values[0]
    steer_rate = steered_summary[steered_summary['domain'] == domain]['unans_abstention'].values[0]
    change = steer_rate - base_rate
    print(f"{domain:15s}: {base_rate:.3f} -> {steer_rate:.3f} (Δ={change:+.3f})")

print("\nChange in Accuracy on Answerable Questions:")
print("-" * 60)
for domain in sorted(df['domain'].unique()):
    base_acc = baseline_summary[baseline_summary['domain'] == domain]['ans_accuracy'].values[0]
    steer_acc = steered_summary[steered_summary['domain'] == domain]['ans_accuracy'].values[0]
    change = steer_acc - base_acc
    print(f"{domain:15s}: {base_acc:.3f} -> {steer_acc:.3f} (Δ={change:+.3f})")

print("\nChange in Abstention Rate on Answerable Questions (should stay low):")
print("-" * 60)
for domain in sorted(df['domain'].unique()):
    base_rate = baseline_summary[baseline_summary['domain'] == domain]['ans_abstention'].values[0]
    steer_rate = steered_summary[steered_summary['domain'] == domain]['ans_abstention'].values[0]
    change = steer_rate - base_rate
    print(f"{domain:15s}: {base_rate:.3f} -> {steer_rate:.3f} (Δ={change:+.3f})")

print("\n" + "=" * 80)
print("4. DETAILED ANALYSIS")
print("=" * 80)

# Check for problematic patterns
print("\nPotential Issues:")
print("-" * 60)

# Issue 1: Steering causing abstention on answerable questions
overall_ans_baseline = answerable[answerable['condition'] == 'baseline']['abstained'].mean()
overall_ans_steered = answerable[answerable['condition'] == 'steered']['abstained'].mean()
if overall_ans_steered > overall_ans_baseline + 0.1:  # More than 10% increase
    print(f"⚠ ISSUE: Steering increases abstention on ANSWERABLE questions")
    print(f"  Baseline: {overall_ans_baseline:.3f}, Steered: {overall_ans_steered:.3f} (Δ={overall_ans_steered - overall_ans_baseline:+.3f})")

# Issue 2: Steering not effective on unanswerable
overall_unans_baseline = unanswerable[unanswerable['condition'] == 'baseline']['abstained'].mean()
overall_unans_steered = unanswerable[unanswerable['condition'] == 'steered']['abstained'].mean()
if overall_unans_steered < 0.5:  # Less than 50% abstention
    print(f"⚠ ISSUE: Steering not effective enough on UNANSWERABLE questions")
    print(f"  Baseline: {overall_unans_baseline:.3f}, Steered: {overall_unans_steered:.3f}")
    print(f"  Target should be close to 1.0")

# Issue 3: Loss of accuracy on answerable
overall_acc_baseline = answerable[answerable['condition'] == 'baseline']['correct'].mean()
overall_acc_steered = answerable[answerable['condition'] == 'steered']['correct'].mean()
if overall_acc_steered < overall_acc_baseline - 0.05:  # More than 5% drop
    print(f"⚠ ISSUE: Steering reduces accuracy on ANSWERABLE questions")
    print(f"  Baseline: {overall_acc_baseline:.3f}, Steered: {overall_acc_steered:.3f} (Δ={overall_acc_steered - overall_acc_baseline:+.3f})")

# Issue 4: Domain-specific problems
print("\nDomain-specific concerns:")
for domain in sorted(df['domain'].unique()):
    base_unans_abst = baseline_summary[baseline_summary['domain'] == domain]['unans_abstention'].values[0]
    steer_unans_abst = steered_summary[steered_summary['domain'] == domain]['unans_abstention'].values[0]
    improvement = steer_unans_abst - base_unans_abst

    if improvement < 0.2:  # Less than 20% improvement
        print(f"  {domain}: Poor steering effect (Δ={improvement:+.3f})")

print("\n" + "=" * 80)
print("5. RECOMMENDATIONS")
print("=" * 80)

recommendations = []

# Rec 1: Check epsilon strength
if overall_unans_steered < 0.7:
    recommendations.append("Consider increasing |epsilon| (currently -20.0) to strengthen the uncertainty signal for unanswerable questions")

# Rec 2: Check if hurting answerable too much
if overall_ans_steered > 0.2:
    recommendations.append("High abstention on answerable questions suggests steering may be too aggressive or not well-targeted")

# Rec 3: Domain-specific tuning
domain_variation = steered_summary['unans_abstention'].std()
if domain_variation > 0.15:
    recommendations.append("High variance across domains suggests domain-specific epsilon values might help")

# Rec 4: Check layer selection
recommendations.append("Verify that layer 10 is optimal for uncertainty routing - try other layers")

# Rec 5: Analyze failure cases
recommendations.append("Examine specific questions where steering fails (both false abstentions and false answers)")

if not recommendations:
    recommendations.append("Steering appears to be working reasonably well - consider fine-tuning epsilon for optimal balance")

for i, rec in enumerate(recommendations, 1):
    print(f"\n{i}. {rec}")

# Create visualizations
print("\n" + "=" * 80)
print("Creating visualizations...")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Abstention rates on unanswerable questions by domain
ax = axes[0, 0]
domains = sorted(df['domain'].unique())
x = np.arange(len(domains))
width = 0.35

baseline_unans = [baseline_summary[baseline_summary['domain'] == d]['unans_abstention'].values[0] for d in domains]
steered_unans = [steered_summary[steered_summary['domain'] == d]['unans_abstention'].values[0] for d in domains]

ax.bar(x - width/2, baseline_unans, width, label='Baseline', alpha=0.8)
ax.bar(x + width/2, steered_unans, width, label='Steered', alpha=0.8)
ax.set_ylabel('Abstention Rate', fontsize=11)
ax.set_title('Abstention on Unanswerable Questions\n(Higher is Better)', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(domains, rotation=45, ha='right')
ax.legend()
ax.set_ylim([0, 1])
ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.3, label='50% threshold')
ax.grid(axis='y', alpha=0.3)

# Plot 2: Accuracy on answerable questions by domain
ax = axes[0, 1]
baseline_acc = [baseline_summary[baseline_summary['domain'] == d]['ans_accuracy'].values[0] for d in domains]
steered_acc = [steered_summary[steered_summary['domain'] == d]['ans_accuracy'].values[0] for d in domains]

ax.bar(x - width/2, baseline_acc, width, label='Baseline', alpha=0.8)
ax.bar(x + width/2, steered_acc, width, label='Steered', alpha=0.8)
ax.set_ylabel('Accuracy', fontsize=11)
ax.set_title('Accuracy on Answerable Questions\n(Higher is Better)', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(domains, rotation=45, ha='right')
ax.legend()
ax.set_ylim([0, 1])
ax.grid(axis='y', alpha=0.3)

# Plot 3: Abstention on answerable questions (should stay low)
ax = axes[1, 0]
baseline_ans_abst = [baseline_summary[baseline_summary['domain'] == d]['ans_abstention'].values[0] for d in domains]
steered_ans_abst = [steered_summary[steered_summary['domain'] == d]['ans_abstention'].values[0] for d in domains]

ax.bar(x - width/2, baseline_ans_abst, width, label='Baseline', alpha=0.8)
ax.bar(x + width/2, steered_ans_abst, width, label='Steered', alpha=0.8)
ax.set_ylabel('Abstention Rate', fontsize=11)
ax.set_title('Abstention on Answerable Questions\n(Lower is Better - False Rejections)', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(domains, rotation=45, ha='right')
ax.legend()
ax.set_ylim([0, 1])
ax.grid(axis='y', alpha=0.3)

# Plot 4: Hallucination rate on unanswerable
ax = axes[1, 1]
baseline_hall = [baseline_summary[baseline_summary['domain'] == d]['unans_hallucination'].values[0] for d in domains]
steered_hall = [steered_summary[steered_summary['domain'] == d]['unans_hallucination'].values[0] for d in domains]

ax.bar(x - width/2, baseline_hall, width, label='Baseline', alpha=0.8)
ax.bar(x + width/2, steered_hall, width, label='Steered', alpha=0.8)
ax.set_ylabel('Hallucination Rate', fontsize=11)
ax.set_title('Hallucination on Unanswerable Questions\n(Lower is Better)', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(domains, rotation=45, ha='right')
ax.legend()
ax.set_ylim([0, 1])
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
output_path = "/Users/akshatatiwari/Desktop/MIT/mech_interp_research/uncertainty_routing/results/exp6a_analysis.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nSaved visualization to: {output_path}")

# Create summary table
print("\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)
summary_table = summary_df.pivot_table(
    index='domain',
    columns='condition',
    values=['ans_accuracy', 'ans_abstention', 'unans_abstention', 'unans_hallucination']
)
print(summary_table.to_string())

# Save detailed report
report_path = "/Users/akshatatiwari/Desktop/MIT/mech_interp_research/uncertainty_routing/results/exp6a_analysis_report.txt"
with open(report_path, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("EXPERIMENT 6A: CROSS-DOMAIN UNCERTAINTY ROUTING ANALYSIS\n")
    f.write("=" * 80 + "\n\n")

    f.write("EXPERIMENT SETUP:\n")
    f.write(f"- Baseline: epsilon = 0.0\n")
    f.write(f"- Steered: epsilon = -20.0\n")
    f.write(f"- Layer: 10\n")
    f.write(f"- Domains: {', '.join(sorted(df['domain'].unique()))}\n")
    f.write(f"- Total questions: {len(df)}\n")
    f.write(f"- Answerable: {len(answerable)}\n")
    f.write(f"- Unanswerable: {len(unanswerable)}\n\n")

    f.write("=" * 80 + "\n")
    f.write("KEY FINDINGS\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"OVERALL PERFORMANCE:\n\n")
    f.write(f"Baseline (ε=0.0):\n")
    f.write(f"  Answerable: Accuracy={overall_acc_baseline:.3f}, Abstention={overall_ans_baseline:.3f}\n")
    f.write(f"  Unanswerable: Abstention={overall_unans_baseline:.3f}, Hallucination={(1-overall_unans_baseline):.3f}\n\n")

    f.write(f"Steered (ε=-20.0):\n")
    f.write(f"  Answerable: Accuracy={overall_acc_steered:.3f}, Abstention={overall_ans_steered:.3f}\n")
    f.write(f"  Unanswerable: Abstention={overall_unans_steered:.3f}, Hallucination={(1-overall_unans_steered):.3f}\n\n")

    f.write(f"CHANGES:\n")
    f.write(f"  Δ Abstention (unanswerable): {overall_unans_steered - overall_unans_baseline:+.3f}\n")
    f.write(f"  Δ Accuracy (answerable): {overall_acc_steered - overall_acc_baseline:+.3f}\n")
    f.write(f"  Δ Abstention (answerable): {overall_ans_steered - overall_ans_baseline:+.3f}\n\n")

    f.write("=" * 80 + "\n")
    f.write("RECOMMENDATIONS\n")
    f.write("=" * 80 + "\n\n")
    for i, rec in enumerate(recommendations, 1):
        f.write(f"{i}. {rec}\n\n")

    f.write("=" * 80 + "\n")
    f.write("DETAILED METRICS BY DOMAIN\n")
    f.write("=" * 80 + "\n\n")
    f.write(summary_table.to_string())

print(f"\nSaved detailed report to: {report_path}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
