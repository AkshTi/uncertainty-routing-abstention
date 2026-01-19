"""
Experiment 1: Behavior-Belief Dissociation
Save as: experiment1_behavior_belief.py

Shows that instruction regime changes abstention behavior
while internal uncertainty remains stable
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
from tqdm import tqdm
from typing import List, Dict
import json

from core_utils import (
    ModelWrapper, ExperimentConfig, extract_answer, 
    set_seed, compute_binary_margin_simple
)
from data_preparation import format_prompt


class Experiment1:
    """Behavior-Belief Dissociation Experiment"""
    
    def __init__(self, model_wrapper: ModelWrapper, config: ExperimentConfig):
        self.model = model_wrapper
        self.config = config
        self.results = []
        set_seed(config.seed)
    
    def measure_internal_uncertainty(self, question: str, 
                                     context: str = None) -> Dict:
        """
        Measure internal uncertainty by forcing the model to guess
        multiple times and computing entropy
        """
        prompt = format_prompt(question, "force_guess", context)
        
        answers = []
        print(f"  Measuring internal uncertainty ({self.config.n_force_guess_samples} samples)...")
        
        for i in range(self.config.n_force_guess_samples):
            response = self.model.generate(
                prompt, 
                temperature=1.0,  # Higher temp for diversity
                do_sample=True,
                max_new_tokens=50
            )
            answer = extract_answer(response)
            answers.append(answer)
        
        # Compute entropy over answer distribution
        answer_counts = Counter(answers)
        total = len(answers)
        probs = np.array([count/total for count in answer_counts.values()])
        
        # Shannon entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        # Probability of majority answer
        p_majority = np.max(probs)
        
        # Most common answer
        most_common = answer_counts.most_common(1)[0][0]
        
        return {
            "entropy": float(entropy),
            "p_majority": float(p_majority),
            "n_unique_answers": len(answer_counts),
            "most_common_answer": most_common,
            "all_answers": answers,
            "distribution": dict(answer_counts)
        }
    
    def test_instruction_regime(self, question: str, regime: str,
                               context: str = None) -> Dict:
        """Test question under specific instruction regime"""
        prompt = format_prompt(question, regime, context)
        
        # Generate with temperature 0 for deterministic output
        response = self.model.generate(
            prompt,
            temperature=0.0,
            do_sample=False,
            max_new_tokens=50
        )
        
        answer = extract_answer(response)
        abstained = (answer == "UNCERTAIN")
        
        return {
            "regime": regime,
            "response": response,
            "answer": answer,
            "abstained": abstained
        }
    
    def run(self, questions: List[Dict], 
            regimes: List[str] = None) -> pd.DataFrame:
        """
        Run full experiment:
        1. Measure internal uncertainty once per question
        2. Test each question under each regime
        """
        if regimes is None:
            regimes = ["neutral", "cautious", "confident"]
        
        print("\n" + "="*60)
        print("EXPERIMENT 1: Behavior-Belief Dissociation")
        print("="*60)
        print(f"Testing {len(questions)} questions across {len(regimes)} regimes")
        print()
        
        for i, q_data in enumerate(tqdm(questions, desc="Questions")):
            question = q_data["question"]
            context = q_data.get("context", None)
            
            print(f"\nQuestion {i+1}/{len(questions)}: {question[:60]}...")
            
            # Step 1: Measure internal uncertainty (once)
            internal = self.measure_internal_uncertainty(question, context)
            
            # Step 2: Test each regime
            for regime in regimes:
                print(f"  Testing regime: {regime}")
                result = self.test_instruction_regime(question, regime, context)
                
                # Store combined result
                self.results.append({
                    "question": question,
                    "question_type": q_data.get("type", "unknown"),
                    "true_answerability": q_data.get(
                        "answerability", 
                        q_data.get("true_answerability", "unknown")
                    ),
                    "regime": regime,
                    
                    # Internal uncertainty (same for all regimes)
                    "internal_entropy": internal["entropy"],
                    "internal_p_majority": internal["p_majority"],
                    "internal_n_unique": internal["n_unique_answers"],
                    
                    # External behavior (varies by regime)
                    "abstained": result["abstained"],
                    "answer": result["answer"],
                    "response": result["response"]
                })
        
        df = pd.DataFrame(self.results)
        
        # Save results
        output_path = self.config.results_dir / "exp1_raw_results.csv"
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")
        
        return df
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        """Analyze and visualize results for Experiment 1."""
        import numpy as np
        import pandas as pd
    
        print("\n" + "=" * 60)
        print("EXPERIMENT 1: ANALYSIS")
        print("=" * 60)
    
        # -------------------------
        # 1) Aggregate by regime
        # -------------------------
        abstention_by_regime = df.groupby("regime")["abstained"].agg(
            rate="mean",
            std="std",
            count="size",
        )
    
        print("\nAbstention Rates by Instruction Regime:")
        print(abstention_by_regime)
    
        entropy_by_regime = df.groupby("regime")["internal_entropy"].agg(
            mean="mean",
            std="std",
        )
    
        print("\nInternal Uncertainty (Entropy) by Regime:")
        print(entropy_by_regime)
    
        # -------------------------
        # 2) Key finding: Behavior vs Belief
        # -------------------------
        def safe_mean(regime: str, col: str) -> float:
            s = df.loc[df["regime"] == regime, col]
            return float(s.mean()) if len(s) else float("nan")
    
        cautious_abstain = safe_mean("cautious", "abstained")
        neutral_abstain = safe_mean("neutral", "abstained")
        confident_abstain = safe_mean("confident", "abstained")
    
        behavior_gap = cautious_abstain - confident_abstain
    
        entropy_cautious = safe_mean("cautious", "internal_entropy")
        entropy_confident = safe_mean("confident", "internal_entropy")
        belief_gap = float(abs(entropy_cautious - entropy_confident))
    
        ratio = float(behavior_gap / (belief_gap + 1e-6))
    
        print("\n" + "=" * 60)
        print("KEY FINDING: Behavior ≠ Belief")
        print("=" * 60)
        print(f"Abstention rate (cautious):   {cautious_abstain:.3f}")
        print(f"Abstention rate (neutral):    {neutral_abstain:.3f}")
        print(f"Abstention rate (confident):  {confident_abstain:.3f}")
        print(f"\nBehavior gap (cautious - confident): {behavior_gap:.3f}")
        print(f"Belief gap (entropy difference):     {belief_gap:.3f}")
        print(f"\nBehavior/Belief ratio: {ratio:.1f}x")
    
        # -------------------------
        # 3) Per-question "interesting" examples
        # -------------------------
        per_question = df.pivot_table(
            index="question",
            columns="regime",
            values=["abstained", "internal_entropy"],
            aggfunc="first",
        )
    
        # Flatten MultiIndex columns: ('internal_entropy','cautious') -> 'internal_entropy__cautious'
        per_question.columns = [f"{a}__{b}" for (a, b) in per_question.columns]
    
        regimes = ["neutral", "cautious", "confident"]
    
        abstain_cols = [f"abstained__{r}" for r in regimes if f"abstained__{r}" in per_question.columns]
        entropy_cols = [f"internal_entropy__{r}" for r in regimes if f"internal_entropy__{r}" in per_question.columns]
    
        # Make abstained numeric (bool -> 0/1) to avoid dtype weirdness
        if abstain_cols:
            per_question[abstain_cols] = per_question[abstain_cols].astype(float)
    
        per_question["abstain_variance"] = per_question[abstain_cols].std(axis=1) if abstain_cols else np.nan
        per_question["entropy_variance"] = per_question[entropy_cols].std(axis=1) if entropy_cols else np.nan
    
        interesting = per_question[
            (per_question["abstain_variance"] > 0.4) &
            (per_question["entropy_variance"] < 0.1)
        ].head(5)
    
        if len(interesting) > 0:
            print("\nExample questions where behavior changes but entropy doesn't:")
            for q, row in interesting.iterrows():
                q_preview = (q[:60] + "...") if isinstance(q, str) and len(q) > 60 else q
                print(f"\n  Q: {q_preview}")
                print(f"     Entropy variance: {float(row['entropy_variance']):.3f}")
                print(f"     Abstain variance: {float(row['abstain_variance']):.3f}")
        else:
            print("\nNo high-variance behavior / low-variance entropy examples found (in this sample).")
    
        # -------------------------
        # 4) Plots
        # -------------------------
        self.plot_results(df, abstention_by_regime, entropy_by_regime)
    
        # -------------------------
        # 5) Return JSON-serializable summary
        # -------------------------
        return {
            "abstention_by_regime": abstention_by_regime.to_dict(),
            "entropy_by_regime": entropy_by_regime.to_dict(),
            "behavior_gap": float(behavior_gap),
            "belief_gap": float(belief_gap),
            "behavior_belief_ratio": float(ratio),
            "n_rows": int(len(df)),
            "n_questions": int(df["question"].nunique()) if "question" in df.columns else None,
        }
    
        
    def plot_results(self, df: pd.DataFrame, abstention_stats, entropy_stats):
        """Create publication-quality figures"""
        
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Plot 1: Abstention rates
        regimes = ["confident", "neutral", "cautious"]
        abstention_means = [abstention_stats.loc[r, 'rate'] for r in regimes]
        abstention_stds = [abstention_stats.loc[r, 'std'] for r in regimes]
        
        colors = ['#2ecc71', '#3498db', '#e74c3c']
        axes[0].bar(regimes, abstention_means, yerr=abstention_stds, 
                   capsize=5, color=colors, alpha=0.8)
        axes[0].set_ylabel("Abstention Rate", fontsize=12)
        axes[0].set_xlabel("Instruction Regime", fontsize=12)
        axes[0].set_title("Behavior Changes with Instruction", fontsize=13, fontweight='bold')
        axes[0].set_ylim([0, 1])
        axes[0].grid(axis='y', alpha=0.3)
        
        # Plot 2: Internal entropy (should be stable)
        entropy_means = [entropy_stats.loc[r, 'mean'] for r in regimes]
        entropy_stds = [entropy_stats.loc[r, 'std'] for r in regimes]
        
        axes[1].bar(regimes, entropy_means, yerr=entropy_stds,
                   capsize=5, color='#95a5a6', alpha=0.8)
        axes[1].set_ylabel("Internal Entropy", fontsize=12)
        axes[1].set_xlabel("Instruction Regime", fontsize=12)
        axes[1].set_title("Internal Uncertainty Remains Stable", fontsize=13, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        
        # Plot 3: Scatter plot of entropy vs abstention
        for regime, color in zip(["confident", "neutral", "cautious"], colors):
            regime_data = df[df["regime"] == regime]
            axes[2].scatter(
                regime_data["internal_entropy"],
                regime_data["abstained"].astype(float),
                alpha=0.6,
                s=50,
                label=regime.capitalize(),
                color=color
            )
        
        axes[2].set_xlabel("Internal Entropy", fontsize=12)
        axes[2].set_ylabel("Abstained (0 or 1)", fontsize=12)
        axes[2].set_title("Behavior Decoupled from Belief", fontsize=13, fontweight='bold')
        axes[2].legend()
        axes[2].grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        output_path = self.config.results_dir / "exp1_behavior_belief_dissociation.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to {output_path}")
        plt.close()
        
        # Also create a combined figure for paper
        self._create_paper_figure(df)
    
    def _create_paper_figure(self, df: pd.DataFrame):
        """Create single figure for paper"""
        
        fig = plt.figure(figsize=(10, 4))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], hspace=0.3, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[0])
        
        regimes = ["confident", "neutral", "cautious"]
        regime_labels = ["Confident", "Neutral", "Cautious"]
        colors_behavior = ['#2ecc71', '#3498db', '#e74c3c']
        
        # Dual y-axis plot
        abstention_means = [df[df["regime"] == r]["abstained"].mean() for r in regimes]
        entropy_means = [df[df["regime"] == r]["internal_entropy"].mean() for r in regimes]
        
        x = np.arange(len(regimes))
        width = 0.35
        
        # Abstention bars
        ax1.bar(x - width/2, abstention_means, width, 
               label='Abstention Rate (Behavior)', color=colors_behavior, alpha=0.8)
        ax1.set_ylabel('Abstention Rate', fontsize=11)
        ax1.set_ylim([0, 1.0])
        ax1.set_xticks(x)
        ax1.set_xticklabels(regime_labels)
        
        # Entropy bars on secondary axis
        ax2 = ax1.twinx()
        ax2.bar(x + width/2, entropy_means, width,
               label='Internal Entropy (Belief)', color='gray', alpha=0.6)
        ax2.set_ylabel('Internal Entropy', fontsize=11)
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
        
        ax1.set_title('Behavior Changes, Belief Stays Constant', fontsize=12, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = self.config.results_dir / "exp1_paper_figure.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Paper figure saved to {output_path}")
        plt.close()


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Run Experiment 1"""
    from pathlib import Path
    import json
    
    # Setup
    config = ExperimentConfig()
    config.n_force_guess_samples = 15  # Adjust based on compute
    
    print("Initializing model...")
    model = ModelWrapper(config)
    
    # Load data
    print("\nLoading datasets...")
    with open("./data/dataset_ambiguous.json", 'r') as f:
        ambiguous_questions = json.load(f)
    
    # For quick testing, use subset
    test_questions = ambiguous_questions[:10]  # Use [:10] for quick test
    
    # Run experiment
    exp1 = Experiment1(model, config)
    results_df = exp1.run(test_questions)
    
    # Analyze
    summary = exp1.analyze(results_df)
    
    # Save summary
    summary_path = config.results_dir / "exp1_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Experiment 1 complete!")
    print(f"  Results: {config.results_dir / 'exp1_raw_results.csv'}")
    print(f"  Summary: {summary_path}")
    print(f"  Figures: {config.results_dir}")


if __name__ == "__main__":
    main()
