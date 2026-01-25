import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
csv_path = "/Users/akshatatiwari/Desktop/MIT/mech_interp_research/uncertainty_routing/results/exp6a_cross_domain_publication_ready.csv"
df = pd.read_csv(csv_path)

# Create a comprehensive insight visualization
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

# Main title
fig.suptitle('Experiment 6A: Why Uncertainty Steering is NOT Working',
             fontsize=16, fontweight='bold', y=0.98)

# Plot 1: The Key Insight - Baseline Ceiling Effect
ax1 = fig.add_subplot(gs[0, :])
conditions = ['Baseline\n(ε=0.0)', 'Steered\n(ε=-20.0)', 'Ideal\nTarget']
abstention_rates = [0.890, 0.870, 1.000]
colors = ['#3498db', '#e74c3c', '#2ecc71']
bars = ax1.bar(conditions, abstention_rates, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

# Add percentage labels on bars
for bar, rate in zip(bars, abstention_rates):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{rate*100:.0f}%',
             ha='center', va='bottom', fontsize=14, fontweight='bold')

ax1.set_ylabel('Abstention Rate on Unanswerable Questions', fontsize=12, fontweight='bold')
ax1.set_title('KEY ISSUE: Baseline Already Achieves 89% - Steering Actually WORSENS to 87%',
              fontsize=13, fontweight='bold', color='red', pad=15)
ax1.set_ylim([0, 1.1])
ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3, label='50% threshold')
ax1.axhline(y=0.89, color='blue', linestyle=':', alpha=0.5, linewidth=2)
ax1.text(2.5, 0.91, 'Baseline performance\n(89% without steering!)',
         fontsize=10, style='italic', color='blue')
ax1.grid(axis='y', alpha=0.3)

# Plot 2: Domain Comparison - Abstention on Unanswerable
ax2 = fig.add_subplot(gs[1, 0])
domains = ['Geography', 'History', 'Mathematics', 'Science']
baseline_unans = [0.880, 0.880, 0.820, 0.980]
steered_unans = [0.820, 0.880, 0.800, 0.980]

x = np.arange(len(domains))
width = 0.35
bars1 = ax2.bar(x - width/2, baseline_unans, width, label='Baseline', alpha=0.8, color='#3498db')
bars2 = ax2.bar(x + width/2, steered_unans, width, label='Steered', alpha=0.8, color='#e74c3c')

# Add change indicators
for i, (base, steer) in enumerate(zip(baseline_unans, steered_unans)):
    change = steer - base
    if change < 0:
        symbol = '↓'
        color = 'red'
    elif change > 0:
        symbol = '↑'
        color = 'green'
    else:
        symbol = '='
        color = 'gray'
    ax2.text(i, max(base, steer) + 0.03, f'{symbol}',
             ha='center', fontsize=16, color=color, fontweight='bold')

ax2.set_ylabel('Abstention Rate', fontsize=10, fontweight='bold')
ax2.set_title('Unanswerable Q: Steering Effect by Domain\n(3/4 domains WORSE or neutral)',
              fontsize=11, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(domains, rotation=45, ha='right', fontsize=9)
ax2.legend(fontsize=9)
ax2.set_ylim([0, 1.1])
ax2.grid(axis='y', alpha=0.3)

# Plot 3: Domain Comparison - Accuracy on Answerable
ax3 = fig.add_subplot(gs[1, 1])
baseline_ans = [0.760, 0.640, 0.600, 0.620]
steered_ans = [0.780, 0.560, 0.680, 0.680]

bars1 = ax3.bar(x - width/2, baseline_ans, width, label='Baseline', alpha=0.8, color='#3498db')
bars2 = ax3.bar(x + width/2, steered_ans, width, label='Steered', alpha=0.8, color='#e74c3c')

# Add change indicators
for i, (base, steer) in enumerate(zip(baseline_ans, steered_ans)):
    change = steer - base
    if change > 0:
        symbol = '↑'
        color = 'green'
    elif change < 0:
        symbol = '↓'
        color = 'red'
    else:
        symbol = '='
        color = 'gray'
    ax3.text(i, max(base, steer) + 0.03, f'{symbol}',
             ha='center', fontsize=16, color=color, fontweight='bold')

ax3.set_ylabel('Accuracy', fontsize=10, fontweight='bold')
ax3.set_title('Answerable Q: Accuracy by Domain\n(Mixed results: History WORSE)',
              fontsize=11, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(domains, rotation=45, ha='right', fontsize=9)
ax3.legend(fontsize=9)
ax3.set_ylim([0, 1.1])
ax3.grid(axis='y', alpha=0.3)

# Plot 4: Net Effect
ax4 = fig.add_subplot(gs[1, 2])
categories = ['Better', 'Worse', 'Net']
values = [15, 15, 0]
colors_net = ['#2ecc71', '#e74c3c', '#95a5a6']
bars = ax4.bar(categories, values, color=colors_net, alpha=0.7, edgecolor='black', linewidth=2)

for bar, val in zip(bars, values):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{val}',
             ha='center', va='bottom', fontsize=14, fontweight='bold')

ax4.set_ylabel('Number of Cases', fontsize=10, fontweight='bold')
ax4.set_title('Net Effect of Steering\n(NEUTRAL: 15 better = 15 worse)',
              fontsize=11, fontweight='bold')
ax4.set_ylim([0, 20])
ax4.grid(axis='y', alpha=0.3)

# Plot 5: Failure Type Breakdown
ax5 = fig.add_subplot(gs[2, 0])
failure_types = ['Unans.\nHallucinated', 'Ans.\nFalse Reject', 'Ans.\nWrong Answer']
baseline_failures = [22, 50, 22]
steered_failures = [26, 39, 27]

x_fail = np.arange(len(failure_types))
bars1 = ax5.bar(x_fail - width/2, baseline_failures, width, label='Baseline', alpha=0.8, color='#3498db')
bars2 = ax5.bar(x_fail + width/2, steered_failures, width, label='Steered', alpha=0.8, color='#e74c3c')

ax5.set_ylabel('Number of Failures', fontsize=10, fontweight='bold')
ax5.set_title('Failure Types: Steering Changes\n(Red higher = worse)',
              fontsize=11, fontweight='bold')
ax5.set_xticks(x_fail)
ax5.set_xticklabels(failure_types, fontsize=9)
ax5.legend(fontsize=9)
ax5.grid(axis='y', alpha=0.3)

# Plot 6: Problematic Question Examples
ax6 = fig.add_subplot(gs[2, 1:])
ax6.axis('off')

problem_text = """
MOST PROBLEMATIC UNANSWERABLE QUESTIONS (Hallucinated in BOTH conditions):

Mathematics Domain (worst performer):
  • "What is the largest prime number?" → Model gives specific wrong answer
  • "What is my favorite number?" → Model answers "42"
  • "What is the billionth Fibonacci number?" → Model invents a number
  • "What is a significant figure?" → Model defines instead of abstaining

Science Domain (best performer - 98% abstention):
  • Only 1 failure in each condition (nearly perfect!)

Key Insight: The remaining 11% of failures in baseline are HARD cases
that neither baseline nor steering can handle. These are fundamentally
challenging questions that require better uncertainty detection, not just
stronger steering.

RECOMMENDATION: Need harder unanswerable questions where baseline < 50%
"""

ax6.text(0.05, 0.95, problem_text,
         transform=ax6.transAxes,
         fontsize=10,
         verticalalignment='top',
         fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# Add overall assessment box
assessment_text = """
OVERALL ASSESSMENT: ❌ STEERING FAILED

Primary Goal: Increase abstention on unanswerable questions
Result: DECREASED from 89% to 87%

Secondary Effects:
  ✓ Reduced false rejections on answerable (25% → 19.5%)
  ✓ Improved accuracy on answerable (65.5% → 67.5%)
  ✗ High variance across domains
  ✗ Worsened history domain significantly

Root Cause: Baseline ceiling effect - already at 89%
Action: Create harder test set where baseline < 50%
"""

fig.text(0.5, 0.01, assessment_text,
         ha='center', va='bottom',
         fontsize=9,
         fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.5, edgecolor='red', linewidth=2))

plt.savefig('/Users/akshatatiwari/Desktop/MIT/mech_interp_research/uncertainty_routing/results/exp6a_key_insights.png',
            dpi=300, bbox_inches='tight')
print("Saved key insights visualization to: exp6a_key_insights.png")

# Also create a simple comparison chart
fig2, ax = plt.subplots(1, 1, figsize=(10, 6))

metrics = ['Abstention\n(Unanswerable)', 'Accuracy\n(Answerable)', 'Abstention\n(Answerable)']
baseline_vals = [0.890, 0.655, 0.250]
steered_vals = [0.870, 0.675, 0.195]
target_vals = [1.000, 0.800, 0.050]  # Ideal targets

x = np.arange(len(metrics))
width = 0.25

bars1 = ax.bar(x - width, baseline_vals, width, label='Baseline (ε=0.0)',
               alpha=0.8, color='#3498db', edgecolor='black')
bars2 = ax.bar(x, steered_vals, width, label='Steered (ε=-20.0)',
               alpha=0.8, color='#e74c3c', edgecolor='black')
bars3 = ax.bar(x + width, target_vals, width, label='Ideal Target',
               alpha=0.5, color='#2ecc71', edgecolor='black', linestyle='--')

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2%}',
                ha='center', va='bottom', fontsize=9)

ax.set_ylabel('Rate', fontsize=12, fontweight='bold')
ax.set_title('Experiment 6A: Baseline vs Steered vs Target Performance',
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.legend(fontsize=10, loc='upper right')
ax.set_ylim([0, 1.15])
ax.grid(axis='y', alpha=0.3)

# Add annotations
ax.annotate('GOAL: Increase this\nRESULT: Decreased!',
            xy=(0, 0.870), xytext=(-0.5, 0.75),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=10, color='red', fontweight='bold')

ax.annotate('Good: Improved',
            xy=(1, 0.675), xytext=(0.5, 0.55),
            arrowprops=dict(arrowstyle='->', color='green', lw=2),
            fontsize=10, color='green', fontweight='bold')

plt.tight_layout()
plt.savefig('/Users/akshatatiwari/Desktop/MIT/mech_interp_research/uncertainty_routing/results/exp6a_simple_comparison.png',
            dpi=300, bbox_inches='tight')
print("Saved simple comparison to: exp6a_simple_comparison.png")

print("\nAll visualizations created successfully!")
