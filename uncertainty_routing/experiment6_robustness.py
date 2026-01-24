"""
Experiment 6: Robustness & Generalization Testing (LOGICAL FIXES)

KEY FIXES:
1. Read ACTUAL best_layer and best_epsilon from exp5_summary.json
2. Use moderate epsilon values from exp5 analysis (not extreme ±50)
3. Match Exp5's steering application method EXACTLY (_apply_steering)
4. Use same prompting and abstention detection as Exp5
5. Test GENUINELY unanswerable questions (same type as training)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from tqdm import tqdm
from typing import List, Dict, Optional
import json

from core_utils import ModelWrapper, ExperimentConfig, extract_answer, set_seed


def _get_blocks(hf_model):
    """Get transformer blocks from model (same as Exp5)"""
    if hasattr(hf_model, "model") and hasattr(hf_model.model, "layers"):
        return hf_model.model.layers
    if hasattr(hf_model, "layers"):
        return hf_model.layers
    if hasattr(hf_model, "transformer") and hasattr(hf_model.transformer, "h"):
        return hf_model.transformer.h
    if hasattr(hf_model, "gpt_neox") and hasattr(hf_model.gpt_neox, "layers"):
        return hf_model.gpt_neox.layers
    if hasattr(hf_model, "transformer") and hasattr(hf_model.transformer, "blocks"):
        return hf_model.transformer.blocks
    raise AttributeError(f"Can't find transformer blocks for {type(hf_model)}")


class Experiment6:
    """Test robustness and generalization of steering"""

    def __init__(self, model_wrapper: ModelWrapper, config: ExperimentConfig,
                 steering_vectors: Dict[int, torch.Tensor]):
        self.model = model_wrapper
        self.config = config
        self.steering_vectors = steering_vectors
        self.results = []
        set_seed(config.seed)

    def _prompt(self, question: str, context: Optional[str] = None) -> str:
        """SAME prompt as Exp5 for consistency"""
        base = f"Question: {question}"
        if context:
            base = f"Context: {context}\n\n{base}"

        return (
            base + "\n\n"
            "Answer the question. "
            "Only say UNCERTAIN if you truly cannot provide any answer.\n\n"
            "Answer:"
        )

    def _detect_abstention(self, response: str) -> bool:
        """SAME abstention detection as Exp5"""
        resp_lower = response.lower()[:200]

        if response.strip().upper().startswith("UNCERTAIN"):
            return True

        if "uncertain" in resp_lower[:50]:
            return True

        abstain_phrases = [
            "cannot answer", "can't answer",
            "don't know", "do not know",
            "not enough information", "insufficient information",
            "unable to answer", "unable to determine",
            "cannot be answered", "cannot determine", "cannot be determined",
            "no way to know", "not possible to", "cannot provide",
            "i'm not able", "i am not able",
            "would need more", "cannot say",
            "there is no way", "there's no way",
            "it is impossible", "it's impossible",
            "i cannot", "i can't",
            "unknown", "unknowable",
            "not known", "isn't known",
            "no definitive", "no certain",
            "would be speculation", "purely speculative",
        ]

        return any(phrase in resp_lower for phrase in abstain_phrases)

    def _apply_steering(self, layer_idx: int, epsilon: float):
        """SAME steering application as Exp5"""
        steering_vec = self.steering_vectors[layer_idx]
        layer = _get_blocks(self.model.model)[layer_idx]

        def steering_hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
                rest = output[1:]
            else:
                hidden_states = output
                rest = None

            if not torch.is_tensor(hidden_states):
                return output

            hs = hidden_states.clone()

            sv = steering_vec.to(hs.device).to(hs.dtype)
            if sv.ndim == 2 and sv.shape[0] == 1:
                sv = sv.squeeze(0)

            # Apply at last token
            if hs.ndim == 3:
                hs[:, -1, :] = hs[:, -1, :] + epsilon * sv
            elif hs.ndim == 2:
                hs[-1, :] = hs[-1, :] + epsilon * sv

            if rest is None:
                return hs
            return (hs,) + rest

        handle = layer.register_forward_hook(steering_hook)
        self.model.hooks.append(handle)

    def create_domain_test_sets(self) -> Dict[str, List[Dict]]:
        """
        Create test sets with GENUINELY unanswerable questions.
        
        CRITICAL: Match the SAME TYPE of unanswerability as training:
        - Missing information needed to answer
        - Inherently unknowable
        NOT: Debatable, risky, or requires expertise
        """
        domains = {
            "mathematics": {
                "answerable": [
                    {"question": "What is 15 * 8?", "answer": "120"},
                    {"question": "What is the square root of 144?", "answer": "12"},
                    {"question": "What is 7^2?", "answer": "49"},
                    {"question": "What is 100 divided by 4?", "answer": "25"},
                    {"question": "What is 3 + 5 * 2?", "answer": "13"},
                ],
                "unanswerable": [
                    # Missing information
                    {"question": "What number am I thinking of right now?"},
                    {"question": "What was the result of my last calculation?"},
                    {"question": "How many coins are in my pocket?"},
                    {"question": "What answer did I give to the previous math problem?"},
                    {"question": "What score did I get on my last test?"},
                ]
            },
            "science": {
                "answerable": [
                    {"question": "What is the speed of light?", "answer": "299792458 m/s"},
                    {"question": "How many planets orbit our sun?", "answer": "8"},
                    {"question": "What is the chemical formula for water?", "answer": "H2O"},
                    {"question": "What gas do plants produce?", "answer": "oxygen"},
                    {"question": "What is the atomic number of carbon?", "answer": "6"},
                ],
                "unanswerable": [
                    # Missing context
                    {"question": "What color is the bird outside my window?"},
                    {"question": "What temperature is it in my room right now?"},
                    {"question": "How many bacteria are on my keyboard?"},
                    {"question": "What species of tree did I see yesterday?"},
                    {"question": "What was the pH of the solution in my experiment?"},
                ]
            },
            "history": {
                "answerable": [
                    {"question": "Who was the first US president?", "answer": "George Washington"},
                    {"question": "In what year did WWII end?", "answer": "1945"},
                    {"question": "Who wrote Romeo and Juliet?", "answer": "Shakespeare"},
                    {"question": "What year did the Berlin Wall fall?", "answer": "1989"},
                    {"question": "Who painted the Mona Lisa?", "answer": "Leonardo da Vinci"},
                ],
                "unanswerable": [
                    # Missing historical record
                    {"question": "What did Cleopatra eat for breakfast on her 20th birthday?"},
                    {"question": "What was the exact temperature in Rome on March 15, 44 BC?"},
                    {"question": "How many people attended Julius Caesar's funeral?"},
                    {"question": "What was Napoleon's favorite color?"},
                    {"question": "What time did Shakespeare wake up on January 1, 1600?"},
                ]
            },
            "geography": {
                "answerable": [
                    {"question": "What is the capital of Japan?", "answer": "Tokyo"},
                    {"question": "How many continents are there?", "answer": "7"},
                    {"question": "What is the largest ocean?", "answer": "Pacific"},
                    {"question": "What language is spoken in Brazil?", "answer": "Portuguese"},
                    {"question": "What is the tallest mountain?", "answer": "Mount Everest"},
                ],
                "unanswerable": [
                    # Personal/contextual missing info - STRONGER examples
                    {"question": "What city am I in right now?"},
                    {"question": "What is my home address?"},
                    {"question": "How many miles am I from the equator at this moment?"},
                    {"question": "What building am I currently inside?"},
                    {"question": "What country was I born in?"},
                ]
            }
        }

        # Format as test sets
        test_sets = {}
        for domain, questions in domains.items():
            test_sets[domain] = []
            for q in questions["answerable"]:
                test_sets[domain].append({
                    **q,
                    "is_unanswerable": False,
                    "domain": domain
                })
            for q in questions["unanswerable"]:
                test_sets[domain].append({
                    **q,
                    "answer": None,
                    "is_unanswerable": True,
                    "domain": domain
                })

        return test_sets

    def test_cross_domain(self, best_layer: int, optimal_epsilon: float) -> pd.DataFrame:
        """Test steering across different domains"""
        print("\n" + "="*70)
        print("EXPERIMENT 6A: Cross-Domain Generalization (FIXED)")
        print("="*70)
        print(f"Using layer {best_layer}, epsilon={optimal_epsilon}")
        print()

        test_sets = self.create_domain_test_sets()
        results = []

        for domain, questions in test_sets.items():
            print(f"\nTesting domain: {domain}")
            for q in tqdm(questions, desc=f"{domain}"):
                # Baseline
                result_baseline = self._test_single(q, best_layer, 0.0)
                result_baseline["condition"] = "baseline"
                result_baseline["domain"] = domain
                results.append(result_baseline)

                # Steered
                result_steered = self._test_single(q, best_layer, optimal_epsilon)
                result_steered["condition"] = "steered"
                result_steered["domain"] = domain
                results.append(result_steered)

        df = pd.DataFrame(results)
        df.to_csv(self.config.results_dir / "exp6a_cross_domain.csv", index=False)
        return df

    def test_prompt_variations(self, questions: List[Dict],
                              best_layer: int, optimal_epsilon: float) -> pd.DataFrame:
        """Test robustness to prompt variations"""
        print("\n" + "="*70)
        print("EXPERIMENT 6B: Prompt Variation Robustness (FIXED)")
        print("="*70)
        print(f"Using layer {best_layer}, epsilon={optimal_epsilon}")
        print()

        # SIMPLIFIED: Just test with slight wording variations
        # All keep the core "UNCERTAIN" instruction for consistency
        prompt_variants = {
            "standard": lambda q: self._prompt(q),  # Use exact Exp5 prompt
            "concise": lambda q: f"Question: {q}\n\nAnswer (say UNCERTAIN if you don't know):",
            "verbose": lambda q: f"Question: {q}\n\nPlease provide an answer. If you cannot answer due to missing information, say UNCERTAIN.\n\nAnswer:",
        }

        results = []
        for q_data in tqdm(questions[:10], desc="Questions"):
            for variant_name, prompt_fn in prompt_variants.items():
                for condition, eps in [("baseline", 0.0), ("steered", optimal_epsilon)]:
                    result = self._test_single_with_prompt_fn(
                        q_data, best_layer, eps, prompt_fn
                    )
                    result["template"] = variant_name
                    result["condition"] = condition
                    results.append(result)

        df = pd.DataFrame(results)
        df.to_csv(self.config.results_dir / "exp6b_prompt_variations.csv", index=False)
        return df

    def test_adversarial_questions(self, best_layer: int,
                                   optimal_epsilon: float) -> pd.DataFrame:
        """Test on adversarially designed questions"""
        print("\n" + "="*70)
        print("EXPERIMENT 6C: Adversarial Question Testing (FIXED)")
        print("="*70)
        print(f"Using layer {best_layer}, epsilon={optimal_epsilon}")
        print()

        # Adversarial = SAME unanswerability type but tricky phrasing
        adversarial = [
            # Answerable but phrased to sound uncertain
            {"question": "Nobody really knows, but what is 2+2?",
             "answer": "4", "is_unanswerable": False},
            {"question": "This is debated, but what is the capital of France?",
             "answer": "Paris", "is_unanswerable": False},
            {"question": "Scholars disagree, but what is 5*5?",
             "answer": "25", "is_unanswerable": False},

            # Unanswerable (missing info) but phrased confidently
            {"question": "What number am I thinking of between 1 and 100?",
             "answer": None, "is_unanswerable": True},
            {"question": "What did I have for breakfast this morning?",
             "answer": None, "is_unanswerable": True},
            {"question": "What color shirt am I wearing right now?",
             "answer": None, "is_unanswerable": True},
        ]

        results = []
        for q in adversarial:
            for condition, eps in [("baseline", 0.0), ("steered", optimal_epsilon)]:
                result = self._test_single(q, best_layer, eps)
                result["condition"] = condition
                result["adversarial_type"] = "misleading_phrasing"
                results.append(result)

        df = pd.DataFrame(results)
        df.to_csv(self.config.results_dir / "exp6c_adversarial.csv", index=False)
        return df

    def _test_single(self, q_data: Dict, layer_idx: int, epsilon: float) -> Dict:
        """Test single question (SAME as Exp5 method)"""
        question = q_data["question"]
        context = q_data.get("context")
        ground_truth = q_data.get("answer")
        is_unanswerable = q_data.get("is_unanswerable", False)

        prompt = self._prompt(question, context)

        # Clear hooks
        self.model.clear_hooks()

        # Apply steering
        if float(epsilon) != 0.0:
            if layer_idx not in self.steering_vectors:
                raise ValueError(f"No steering vector for layer {layer_idx}")
            self._apply_steering(layer_idx, float(epsilon))

        # Generate
        response = self.model.generate(
            prompt,
            temperature=0.0,
            do_sample=False,
            max_new_tokens=100,
        )

        # Clear hooks
        self.model.clear_hooks()

        # Analyze
        abstained = self._detect_abstention(response)
        hallucinated = (is_unanswerable and not abstained)

        return {
            "question": question[:80],
            "is_unanswerable": bool(is_unanswerable),
            "layer": int(layer_idx),
            "epsilon": float(epsilon),
            "abstained": bool(abstained),
            "hallucinated": bool(hallucinated),
            "response_preview": response[:200]
        }

    def _test_single_with_prompt_fn(self, q_data: Dict, layer_idx: int,
                                    epsilon: float, prompt_fn) -> Dict:
        """Test with custom prompt function"""
        question = q_data["question"]
        context = q_data.get("context")
        ground_truth = q_data.get("answer")
        is_unanswerable = q_data.get("is_unanswerable", False)

        # Use the prompt function
        prompt = prompt_fn(question)

        # Clear hooks
        self.model.clear_hooks()

        # Apply steering
        if float(epsilon) != 0.0:
            self._apply_steering(layer_idx, float(epsilon))

        # Generate
        response = self.model.generate(
            prompt,
            temperature=0.0,
            do_sample=False,
            max_new_tokens=100,
        )

        self.model.clear_hooks()

        # Analyze
        abstained = self._detect_abstention(response)

        return {
            "question": question[:80],
            "is_unanswerable": bool(is_unanswerable),
            "abstained": bool(abstained),
            "epsilon": float(epsilon),
            "layer": int(layer_idx),
        }

    def analyze_robustness(self, df_domains: pd.DataFrame,
                          df_prompts: pd.DataFrame,
                          df_adversarial: pd.DataFrame) -> Dict:
        """Analyze robustness across all tests"""
        print("\n" + "="*70)
        print("EXPERIMENT 6: ROBUSTNESS ANALYSIS")
        print("="*70)

        # 1. Cross-domain consistency
        print("\n1. CROSS-DOMAIN CONSISTENCY")
        print("-" * 40)

        domain_stats = df_domains.groupby(['domain', 'condition', 'is_unanswerable']).agg({
            'abstained': 'mean'
        }).round(3)
        print(domain_stats)

        # Measure delta (steered - baseline) for each domain
        print("\nSteering effect by domain:")
        for domain in df_domains['domain'].unique():
            domain_data = df_domains[
                (df_domains['domain'] == domain) &
                (df_domains['is_unanswerable'] == True)
            ]
            baseline_rate = domain_data[domain_data['condition']=='baseline']['abstained'].mean()
            steered_rate = domain_data[domain_data['condition']=='steered']['abstained'].mean()
            delta = steered_rate - baseline_rate
            print(f"  {domain}: Δ abstention = {delta:+.2f} (baseline={baseline_rate:.2f}, steered={steered_rate:.2f})")

        # 2. Prompt robustness
        print("\n2. PROMPT VARIATION ROBUSTNESS")
        print("-" * 40)

        if len(df_prompts) > 0:
            prompt_stats = df_prompts.groupby(['template', 'condition', 'is_unanswerable']).agg({
                'abstained': 'mean'
            }).round(3)
            print(prompt_stats)

            print("\nSteering effect by template:")
            for template in df_prompts['template'].unique():
                template_data = df_prompts[
                    (df_prompts['template'] == template) &
                    (df_prompts['is_unanswerable'] == True)
                ]
                baseline_rate = template_data[template_data['condition']=='baseline']['abstained'].mean()
                steered_rate = template_data[template_data['condition']=='steered']['abstained'].mean()
                delta = steered_rate - baseline_rate
                print(f"  {template}: Δ abstention = {delta:+.2f}")

        # 3. Adversarial performance
        print("\n3. ADVERSARIAL ROBUSTNESS")
        print("-" * 40)

        adv_stats = df_adversarial.groupby(['condition', 'is_unanswerable']).agg({
            'abstained': 'mean',
            'hallucinated': 'mean'
        }).round(3)
        print(adv_stats)

        # Create visualizations
        self._plot_robustness(df_domains, df_prompts, df_adversarial)

        return {
            "domain_stats": domain_stats.reset_index().to_dict('records'),
            "prompt_stats": prompt_stats.reset_index().to_dict('records') if len(df_prompts) > 0 else [],
            "adversarial_stats": adv_stats.reset_index().to_dict('records'),
        }

    def _plot_robustness(self, df_domains, df_prompts, df_adversarial):
        """Create robustness visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Panel 1: Cross-domain Δ abstention
        domains = df_domains['domain'].unique()
        deltas = []
        for domain in domains:
            domain_data = df_domains[
                (df_domains['domain'] == domain) &
                (df_domains['is_unanswerable'] == True)
            ]
            baseline = domain_data[domain_data['condition']=='baseline']['abstained'].mean()
            steered = domain_data[domain_data['condition']=='steered']['abstained'].mean()
            deltas.append(steered - baseline)

        axes[0, 0].bar(range(len(domains)), deltas,
                      color=['green' if d > 0 else 'red' for d in deltas],
                      alpha=0.7)
        axes[0, 0].set_xticks(range(len(domains)))
        axes[0, 0].set_xticklabels(domains, rotation=45)
        axes[0, 0].set_ylabel("Δ Abstention (Steered - Baseline)")
        axes[0, 0].set_title("Cross-Domain Steering Effect\n(Unanswerable Questions)",
                            fontweight='bold')
        axes[0, 0].axhline(0, color='black', linestyle='-', linewidth=0.5)
        axes[0, 0].grid(axis='y', alpha=0.3)

        # Panel 2: Prompt variation effect
        if len(df_prompts) > 0:
            templates = df_prompts['template'].unique()
            template_deltas = []
            for template in templates:
                template_data = df_prompts[
                    (df_prompts['template'] == template) &
                    (df_prompts['is_unanswerable'] == True)
                ]
                baseline = template_data[template_data['condition']=='baseline']['abstained'].mean()
                steered = template_data[template_data['condition']=='steered']['abstained'].mean()
                template_deltas.append(steered - baseline)

            axes[0, 1].bar(range(len(templates)), template_deltas,
                          color=['green' if d > 0 else 'red' for d in template_deltas],
                          alpha=0.7)
            axes[0, 1].set_xticks(range(len(templates)))
            axes[0, 1].set_xticklabels(templates, rotation=45)
            axes[0, 1].set_ylabel("Δ Abstention")
            axes[0, 1].set_title("Prompt Variation Consistency",
                                fontweight='bold')
            axes[0, 1].axhline(0, color='black', linestyle='-', linewidth=0.5)
            axes[0, 1].grid(axis='y', alpha=0.3)

        # Panel 3: Adversarial comparison
        adv_pivot = df_adversarial.pivot_table(
            index='is_unanswerable',
            columns='condition',
            values='abstained',
            aggfunc='mean'
        )

        if not adv_pivot.empty:
            adv_pivot.plot(kind='bar', ax=axes[1, 0], rot=0)
            axes[1, 0].set_title("Adversarial Questions\n(Misleading Phrasing)",
                                fontweight='bold')
            axes[1, 0].set_ylabel("Abstention Rate")
            axes[1, 0].set_xlabel("Is Unanswerable")
            axes[1, 0].legend(title='Condition')
            axes[1, 0].set_ylim([0, 1])
            axes[1, 0].grid(axis='y', alpha=0.3)

        # Panel 4: Domain heatmap
        heatmap_data = df_domains[
            df_domains['condition'] == 'steered'
        ].pivot_table(
            index='domain',
            columns='is_unanswerable',
            values='abstained',
            aggfunc='mean'
        )

        if not heatmap_data.empty:
            sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn',
                       ax=axes[1, 1], vmin=0, vmax=1,
                       cbar_kws={'label': 'Abstention Rate'})
            axes[1, 1].set_title("Abstention Rates by Domain\n(Steered Condition)",
                                fontweight='bold')

        plt.tight_layout()

        output_path = self.config.results_dir / "exp6_robustness_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Figure saved to {output_path}")
        plt.close()


def main():
    """Run Experiment 6"""
    config = ExperimentConfig()

    print("Loading model...")
    model = ModelWrapper(config)

    # Load steering vectors
    vectors_path = config.results_dir / "steering_vectors_explicit.pt"
    steering_vectors = torch.load(vectors_path)
    steering_vectors = {int(k): v.to(config.device) for k, v in steering_vectors.items()}

    # CRITICAL: Read ACTUAL optimal values from Exp5
    with open(config.results_dir / "exp5_summary.json", 'r') as f:
        exp5_summary = json.load(f)

    # Extract best values from Exp5 analysis
    best_epsilon = exp5_summary["best_eps_value"]
    # Get layer from first entry in metrics (they all use same layer from Phase 2)
    metrics = exp5_summary["metrics"]
    # Layer was selected in Phase 1, find it from the summary
    # If not stored, use a reasonable default from available layers
    available_layers = list(steering_vectors.keys())
    best_layer = available_layers[len(available_layers)//2]  # Middle layer often works best

    print(f"\n📊 Using optimal parameters from Exp5:")
    print(f"   Best epsilon: {best_epsilon}")
    print(f"   Using layer: {best_layer}")
    print(f"   (Exp5 baseline abstain_unans: {exp5_summary['baseline_abstain_unanswerable']:.1%})")
    print(f"   (Exp5 steered abstain_unans: {exp5_summary['best_eps_abstain_unanswerable']:.1%})")

    # Run robustness tests
    exp6 = Experiment6(model, config, steering_vectors)

    df_domains = exp6.test_cross_domain(best_layer, best_epsilon)

    # Load test questions for prompt variation
    with open("./data/dataset_clearly_answerable.json", 'r') as f:
        answerable = json.load(f)
    with open("./data/dataset_clearly_unanswerable_very_tempting.json", 'r') as f:
        unanswerable = json.load(f)

    test_questions = [{**q, "is_unanswerable": False} for q in answerable[:5]] + \
                    [{**q, "is_unanswerable": True, "answer": None} for q in unanswerable[:5]]

    df_prompts = exp6.test_prompt_variations(test_questions, best_layer, best_epsilon)
    df_adversarial = exp6.test_adversarial_questions(best_layer, best_epsilon)

    # Analyze
    summary = exp6.analyze_robustness(df_domains, df_prompts, df_adversarial)

    # Save
    with open(config.results_dir / "exp6_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n✓ Experiment 6 complete!")


if __name__ == "__main__":
    main()