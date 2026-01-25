"""
Quick Diagnostic for Uncertainty Routing
==========================================

This script runs a fast diagnostic to find:
1. Optimal epsilon (direction and magnitude)
2. Optimal layer
3. Whether steering is working at all

Run this BEFORE running the full experiment.
"""

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Import your existing code
from experiment6_publication_ready import Experiment6PublicationReady
from core_utils import ModelWrapper, ExperimentConfig, set_seed


# ============================================================================
# Test Questions (Small Set for Speed)
# ============================================================================

TEST_QUESTIONS = [
    # Answerable - Clear factual questions
    {"q": "What is 2+2?", "a": "4", "is_unanswerable": False, "domain": "math"},
    {"q": "What is the capital of France?", "a": "Paris", "is_unanswerable": False, "domain": "geography"},
    {"q": "How many continents are there?", "a": "7", "is_unanswerable": False, "domain": "geography"},
    {"q": "What is 10/2?", "a": "5", "is_unanswerable": False, "domain": "math"},
    {"q": "What year did WWII end?", "a": "1945", "is_unanswerable": False, "domain": "history"},
    {"q": "What is the chemical symbol for gold?", "a": "Au", "is_unanswerable": False, "domain": "science"},
    {"q": "What is 5*5?", "a": "25", "is_unanswerable": False, "domain": "math"},
    {"q": "What is the capital of Japan?", "a": "Tokyo", "is_unanswerable": False, "domain": "geography"},
    {"q": "How many days are in a week?", "a": "7", "is_unanswerable": False, "domain": "general"},
    {"q": "What is 100/10?", "a": "10", "is_unanswerable": False, "domain": "math"},
    
    # Unanswerable - Personal information / unknowable
    {"q": "What number am I thinking of?", "is_unanswerable": True, "domain": "personal"},
    {"q": "What did I eat yesterday?", "is_unanswerable": True, "domain": "personal"},
    {"q": "What color is my shirt?", "is_unanswerable": True, "domain": "personal"},
    {"q": "What's in my pocket?", "is_unanswerable": True, "domain": "personal"},
    {"q": "What's my favorite movie?", "is_unanswerable": True, "domain": "personal"},
    {"q": "How many coins do I have?", "is_unanswerable": True, "domain": "personal"},
    {"q": "What am I looking at right now?", "is_unanswerable": True, "domain": "personal"},
    {"q": "What's my dog's name?", "is_unanswerable": True, "domain": "personal"},
    {"q": "What did my friend say to me this morning?", "is_unanswerable": True, "domain": "personal"},
    {"q": "What's my phone password?", "is_unanswerable": True, "domain": "personal"},
]


# ============================================================================
# Diagnostic Class
# ============================================================================

class QuickDiagnostic:
    """Run fast diagnostic to find optimal settings"""
    
    def __init__(self, model_wrapper, config, steering_vectors):
        self.model = model_wrapper
        self.config = config
        self.steering_vectors = steering_vectors
        set_seed(config.seed)
    
    def run_single_test(self, question_data, layer, epsilon):
        """Test single question with given layer and epsilon"""
        from unified_prompts import unified_prompt_strict as unified_prompt
        from parsing_fixed import extract_answer, is_abstention
        
        question = question_data['q']
        prompt = unified_prompt(question)
        
        # Clear hooks
        self.model.clear_hooks()
        
        # Apply steering if epsilon != 0
        if epsilon != 0.0 and layer in self.steering_vectors:
            self.model.register_steering_hook(
                layer, -1, self.steering_vectors[layer], epsilon
            )
        
        # Generate with more tokens (allow abstention)
        response = self.model.generate(
            prompt,
            max_new_tokens=30,  # INCREASED from 12
            temperature=0.0,
            do_sample=False
        )
        
        # Clear hooks
        self.model.clear_hooks()
        
        # Parse
        extracted = extract_answer(response)
        abstained = is_abstention(response)
        
        # Evaluate
        is_unans = question_data.get('is_unanswerable', 'a' not in question_data)
        
        result = {
            'question': question,
            'is_unanswerable': is_unans,
            'layer': layer,
            'epsilon': epsilon,
            'response': response[:100],  # First 100 chars
            'abstained': abstained,
            'hallucinated': False,
            'correct': None,
        }
        
        if is_unans:
            result['hallucinated'] = not abstained
            result['correct'] = abstained
        else:
            expected = question_data.get('a', '').lower()
            if abstained:
                result['correct'] = False  # False abstention
            else:
                result['correct'] = expected in response.lower()
        
        return result
    
    def run_grid_search(self, test_layers=None, test_epsilons=None):
        """
        Run grid search over layers and epsilons
        
        Args:
            test_layers: List of layers to test (default: [10, 14, 18, 22, 26])
            test_epsilons: List of epsilons to test
        
        Returns:
            DataFrame with results
        """
        if test_layers is None:
            # Test layers: early-mid, mid, late-mid, late
            available_layers = sorted(self.steering_vectors.keys())
            if len(available_layers) > 5:
                test_layers = available_layers[::len(available_layers)//5][:5]
            else:
                test_layers = available_layers
        
        if test_epsilons is None:
            # Test both directions with increasing magnitude
            test_epsilons = [-50, -30, -10, -5, 0, 5, 10, 30, 50]
        
        print(f"\nRunning Grid Search:")
        print(f"  Layers: {test_layers}")
        print(f"  Epsilons: {test_epsilons}")
        print(f"  Questions: {len(TEST_QUESTIONS)}")
        print(f"  Total tests: {len(test_layers) * len(test_epsilons) * len(TEST_QUESTIONS)}")
        
        results = []
        
        for layer in tqdm(test_layers, desc="Layers"):
            for epsilon in tqdm(test_epsilons, desc="Epsilons", leave=False):
                for q_data in TEST_QUESTIONS:
                    result = self.run_single_test(q_data, layer, epsilon)
                    results.append(result)
        
        return pd.DataFrame(results)
    
    def analyze_results(self, df):
        """Analyze grid search results and find optimal settings"""
        
        print("\n" + "="*70)
        print("DIAGNOSTIC RESULTS")
        print("="*70)
        
        # Aggregate by layer and epsilon
        agg = df.groupby(['layer', 'epsilon']).apply(
            lambda x: pd.Series({
                'accuracy_answerable': x[~x['is_unanswerable']]['correct'].mean(),
                'false_abstention_answerable': x[~x['is_unanswerable']]['abstained'].mean(),
                'abstention_unanswerable': x[x['is_unanswerable']]['correct'].mean(),
                'hallucination_unanswerable': x[x['is_unanswerable']]['hallucinated'].mean(),
                'n_answerable': (~x['is_unanswerable']).sum(),
                'n_unanswerable': x['is_unanswerable'].sum(),
            })
        ).reset_index()
        
        # Compute composite score
        # Objectives:
        # 1. Maximize accuracy on answerable (weight=2.0)
        # 2. Maximize abstention on unanswerable (weight=1.0)
        # 3. Minimize false abstention on answerable (weight=-3.0)
        # 4. Minimize hallucination on unanswerable (weight=-5.0)
        
        agg['score'] = (
            agg['accuracy_answerable'] * 2.0 +
            agg['abstention_unanswerable'] * 1.0 -
            agg['false_abstention_answerable'] * 3.0 -
            agg['hallucination_unanswerable'] * 5.0
        )
        
        # Find best configuration
        best_idx = agg['score'].idxmax()
        best = agg.loc[best_idx]
        
        # Find baseline (epsilon=0)
        baseline = agg[agg['epsilon'] == 0].iloc[0]
        
        print("\n--- BASELINE (ε=0) ---")
        print(f"Layer: {baseline['layer']}")
        print(f"Accuracy (answerable): {baseline['accuracy_answerable']:.1%}")
        print(f"False abstention (answerable): {baseline['false_abstention_answerable']:.1%}")
        print(f"Abstention (unanswerable): {baseline['abstention_unanswerable']:.1%}")
        print(f"Hallucination (unanswerable): {baseline['hallucination_unanswerable']:.1%}")
        print(f"Score: {baseline['score']:.3f}")
        
        print("\n--- BEST CONFIGURATION ---")
        print(f"Layer: {best['layer']}")
        print(f"Epsilon: {best['epsilon']}")
        print(f"Accuracy (answerable): {best['accuracy_answerable']:.1%}")
        print(f"False abstention (answerable): {best['false_abstention_answerable']:.1%}")
        print(f"Abstention (unanswerable): {best['abstention_unanswerable']:.1%}")
        print(f"Hallucination (unanswerable): {best['hallucination_unanswerable']:.1%}")
        print(f"Score: {best['score']:.3f}")
        
        print("\n--- IMPROVEMENT ---")
        print(f"Δ Accuracy: {(best['accuracy_answerable'] - baseline['accuracy_answerable']):.1%}")
        print(f"Δ False abstention: {(best['false_abstention_answerable'] - baseline['false_abstention_answerable']):.1%}")
        print(f"Δ Abstention (unans): {(best['abstention_unanswerable'] - baseline['abstention_unanswerable']):.1%}")
        print(f"Δ Hallucination: {(best['hallucination_unanswerable'] - baseline['hallucination_unanswerable']):.1%}")
        print(f"Δ Score: {(best['score'] - baseline['score']):.3f}")
        
        # Show top 5 configurations
        print("\n--- TOP 5 CONFIGURATIONS ---")
        top5 = agg.nlargest(5, 'score')[['layer', 'epsilon', 'accuracy_answerable', 
                                          'abstention_unanswerable', 'score']]
        print(top5.to_string(index=False))
        
        # Check if steering is working
        print("\n--- STEERING EFFECTIVENESS ---")
        eps_range = agg.groupby('epsilon')['score'].mean().sort_values(ascending=False)
        print("Average score by epsilon:")
        print(eps_range.to_string())
        
        if abs(eps_range.iloc[0] - baseline['score']) < 0.1:
            print("\n⚠️  WARNING: Steering appears to have MINIMAL effect!")
            print("   Consider:")
            print("   1. Check steering vector quality")
            print("   2. Try different layers")
            print("   3. Try larger epsilon magnitudes")
        
        return agg, best
    
    def plot_results(self, agg_df, output_path='diagnostic_results.png'):
        """Plot heatmaps of results"""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        metrics = [
            ('accuracy_answerable', 'Accuracy on Answerable'),
            ('abstention_unanswerable', 'Abstention on Unanswerable'),
            ('false_abstention_answerable', 'False Abstention (Bad)'),
            ('hallucination_unanswerable', 'Hallucination (Bad)'),
        ]
        
        for ax, (metric, title) in zip(axes.flat, metrics):
            pivot = agg_df.pivot(index='epsilon', columns='layer', values=metric)
            sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax,
                       vmin=0, vmax=1, cbar_kws={'label': 'Rate'})
            ax.set_title(title)
            ax.set_xlabel('Layer')
            ax.set_ylabel('Epsilon')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Plot saved to {output_path}")
        
        return fig
    
    def sample_responses(self, df, layer, epsilon, n=5):
        """Show sample responses for given configuration"""
        
        subset = df[(df['layer'] == layer) & (df['epsilon'] == epsilon)]
        
        print(f"\n--- SAMPLE RESPONSES (layer={layer}, ε={epsilon}) ---")
        
        print("\nAnswerable questions:")
        ans_subset = subset[~subset['is_unanswerable']].head(n)
        for _, row in ans_subset.iterrows():
            print(f"\nQ: {row['question']}")
            print(f"R: {row['response']}")
            print(f"   Correct: {row['correct']}, Abstained: {row['abstained']}")
        
        print("\n" + "-"*70)
        print("\nUnanswerable questions:")
        unans_subset = subset[subset['is_unanswerable']].head(n)
        for _, row in unans_subset.iterrows():
            print(f"\nQ: {row['question']}")
            print(f"R: {row['response']}")
            print(f"   Abstained: {row['abstained']}, Hallucinated: {row['hallucinated']}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("QUICK DIAGNOSTIC FOR UNCERTAINTY ROUTING")
    print("="*70)
    
    # Load model and steering vectors
    print("\nLoading model...")
    config = ExperimentConfig()
    model = ModelWrapper(config)
    
    print("Loading steering vectors...")
    possible_files = [
        "steering_vectors_calibrated.pt",
        "steering_vectors.pt",
        "steering_vectors_explicit.pt",
        "steering_vectors_fixed.pt",
    ]
    
    steering_vectors = None
    for filename in possible_files:
        filepath = config.results_dir / filename
        if filepath.exists():
            steering_vectors = torch.load(filepath)
            print(f"✓ Loaded from {filename}")
            print(f"  Available layers: {list(steering_vectors.keys())}")
            break
    
    if steering_vectors is None:
        print("✗ No steering vectors found!")
        import sys
        sys.exit(1)
    
    # Run diagnostic
    diagnostic = QuickDiagnostic(model, config, steering_vectors)
    
    # Grid search
    print("\nRunning grid search (this will take 5-10 minutes)...")
    results_df = diagnostic.run_grid_search()
    
    # Save raw results
    results_df.to_csv('diagnostic_raw_results.csv', index=False)
    print("✓ Raw results saved to diagnostic_raw_results.csv")
    
    # Analyze
    agg_df, best_config = diagnostic.analyze_results(results_df)
    
    # Save aggregated results
    agg_df.to_csv('diagnostic_aggregated_results.csv', index=False)
    print("✓ Aggregated results saved to diagnostic_aggregated_results.csv")
    
    # Plot
    diagnostic.plot_results(agg_df)
    
    # Show sample responses for best config
    diagnostic.sample_responses(
        results_df, 
        layer=int(best_config['layer']),
        epsilon=float(best_config['epsilon'])
    )
    
    # Also show baseline for comparison
    print("\n" + "="*70)
    print("BASELINE COMPARISON (ε=0)")
    print("="*70)
    diagnostic.sample_responses(
        results_df,
        layer=int(best_config['layer']),
        epsilon=0.0
    )
    
    print("\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print(f"1. Use layer={int(best_config['layer'])}, epsilon={float(best_config['epsilon'])}")
    print(f"2. Review sample responses above")
    print(f"3. If steering effect is weak, consider:")
    print(f"   - Re-creating steering vectors with better contrast")
    print(f"   - Modifying prompt format")
    print(f"   - Checking if max_tokens is sufficient")
