"""
Experiment 6: Robustness & Generalization Testing (FIXED VERSION)

FIXES APPLIED:
1. Use appropriate epsilon values (-2.0 instead of -50.0)
2. Proper steering application during generation
3. Better evaluation metrics

Changes from original:
- baseline: epsilon = 0.0
- steered: epsilon = -2.0 (instead of -50.0)
- Added proper token position tracking
- Fixed generation with steering hooks
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from tqdm import tqdm
from typing import List, Dict
import json

from core_utils import ModelWrapper, ExperimentConfig, extract_answer, set_seed
from data_preparation import format_prompt


class Experiment6Fixed:
    """Test robustness and generalization of steering (with fixes)"""

    def __init__(self, model_wrapper: ModelWrapper, config: ExperimentConfig,
                 steering_vectors: Dict[int, torch.Tensor]):
        self.model = model_wrapper
        self.config = config
        self.steering_vectors = steering_vectors
        self.results = []
        set_seed(config.seed)

    def create_domain_test_sets(self) -> Dict[str, List[Dict]]:
        """Create test sets across different domains."""
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
                    {"question": "What is the largest prime number?"},
                    {"question": "What is the last digit of pi?"},
                    {"question": "What number am I thinking of right now?"},
                    {"question": "What is the solution to the unsolved Goldbach conjecture?"},
                    {"question": "What is ∞ + 1?"},
                ]
            },
            "science": {
                "answerable": [
                    {"question": "What is the speed of light in vacuum?", "answer": "299792458 m/s"},
                    {"question": "How many planets are in our solar system?", "answer": "8"},
                    {"question": "What is the chemical formula for salt?", "answer": "NaCl"},
                    {"question": "What gas do plants produce during photosynthesis?", "answer": "oxygen"},
                    {"question": "What is the atomic number of carbon?", "answer": "6"},
                ],
                "unanswerable": [
                    {"question": "What is the exact temperature at the center of the sun right now?"},
                    {"question": "How many atoms are in this room at this exact moment?"},
                    {"question": "What will be the next major scientific discovery?"},
                    {"question": "What is the cure for all forms of cancer?"},
                    {"question": "How many alien civilizations exist in the universe?"},
                ]
            },
            "history": {
                "answerable": [
                    {"question": "Who was the first president of the United States?", "answer": "George Washington"},
                    {"question": "In what year did World War I begin?", "answer": "1914"},
                    {"question": "What ancient civilization built the pyramids?", "answer": "Egyptians"},
                    {"question": "Who painted the Mona Lisa?", "answer": "Leonardo da Vinci"},
                    {"question": "What year did the Berlin Wall fall?", "answer": "1989"},
                ],
                "unanswerable": [
                    {"question": "What was Cleopatra thinking when she died?"},
                    {"question": "What was the weather like on Caesar's assassination day?"},
                    {"question": "How many words did Homer speak in his entire life?"},
                    {"question": "What was Napoleon's favorite breakfast as a child?"},
                    {"question": "What was the exact population of ancient Babylon?"},
                ]
            },
            "current_events": {
                "answerable": [
                    {"question": "What is the capital of Japan?", "answer": "Tokyo"},
                    {"question": "How many continents are there?", "answer": "7"},
                    {"question": "What is the largest ocean?", "answer": "Pacific"},
                    {"question": "What language is spoken in Brazil?", "answer": "Portuguese"},
                    {"question": "What is the currency of the United Kingdom?", "answer": "Pound Sterling"},
                ],
                "unanswerable": [
                    {"question": "What will the weather be exactly one year from today?"},
                    {"question": "Who will win the next election?"},
                    {"question": "What will be tomorrow's stock market close?"},
                    {"question": "How many people are thinking about pizza right now?"},
                    {"question": "What will I have for dinner next Tuesday?"},
                ]
            }
        }
        return domains

    def _test_single(self, q_data: Dict, layer_idx: int, epsilon: float) -> Dict:
        """Test single question with steering at appropriate epsilon"""
        prompt = format_prompt(q_data["question"], "neutral")

        # Clear hooks
        self.model.clear_hooks()

        # Apply steering if epsilon != 0
        if epsilon != 0.0:
            inputs = self.model.tokenizer(prompt, return_tensors="pt").to(self.model.config.device)
            position = inputs["input_ids"].shape[1] - 1

            self.model.register_steering_hook(
                layer_idx,
                position,
                self.steering_vectors[layer_idx],
                epsilon
            )

        # Generate
        response = self.model.generate(prompt, temperature=0.0, do_sample=False, max_new_tokens=50)

        # Clear hooks
        self.model.clear_hooks()

        # Evaluate
        answer = extract_answer(response)
        abstained = (answer == "UNCERTAIN")

        result = {
            "question": q_data["question"],
            "is_unanswerable": q_data.get("is_unanswerable", "answer" not in q_data),
            "layer": layer_idx,
            "epsilon": epsilon,
            "abstained": abstained,
            "hallucinated": False,
            "correct": None,
            "response_preview": response[:200]
        }

        # Determine if hallucinated or correct
        if result["is_unanswerable"]:
            result["hallucinated"] = not abstained
            result["correct"] = ""
        else:
            # Check if correct
            expected = q_data.get("answer", "").lower()
            if expected and expected in response.lower():
                result["correct"] = True
            else:
                result["correct"] = False

        return result

    def test_cross_domain(self, best_layer: int, optimal_epsilon: float = -2.0) -> pd.DataFrame:
        """
        Test steering across different domains.

        FIXED: Use epsilon=-2.0 instead of -50.0
        """
        print("\n" + "="*70)
        print("EXPERIMENT 6A: Cross-Domain Generalization (FIXED)")
        print("="*70)
        print(f"Using epsilon={optimal_epsilon} (instead of -50.0)")
        print()

        domains = self.create_domain_test_sets()
        results = []

        for domain, question_sets in tqdm(domains.items(), desc="Domains"):
            # Test answerable questions
            for q in question_sets["answerable"]:
                # Baseline
                result_baseline = self._test_single(q, best_layer, 0.0)
                result_baseline["condition"] = "baseline"
                result_baseline["domain"] = domain
                results.append(result_baseline)

                # Steered (FIXED epsilon)
                result_steered = self._test_single(q, best_layer, optimal_epsilon)
                result_steered["condition"] = "steered"
                result_steered["domain"] = domain
                results.append(result_steered)

            # Test unanswerable questions
            for q in question_sets["unanswerable"]:
                q["is_unanswerable"] = True

                # Baseline
                result_baseline = self._test_single(q, best_layer, 0.0)
                result_baseline["condition"] = "baseline"
                result_baseline["domain"] = domain
                results.append(result_baseline)

                # Steered (FIXED epsilon)
                result_steered = self._test_single(q, best_layer, optimal_epsilon)
                result_steered["condition"] = "steered"
                result_steered["domain"] = domain
                results.append(result_steered)

        df = pd.DataFrame(results)
        df.to_csv(self.config.results_dir / "exp6a_cross_domain_fixed.csv", index=False)
        print(f"\n✓ Results saved to exp6a_cross_domain_fixed.csv")
        return df

    def test_prompt_variations(self, questions: List[Dict], best_layer: int,
                              optimal_epsilon: float = -2.0) -> pd.DataFrame:
        """Test robustness to prompt phrasing variations (FIXED)"""
        print("\n" + "="*70)
        print("EXPERIMENT 6B: Prompt Variation Robustness (FIXED)")
        print("="*70)
        print(f"Using epsilon={optimal_epsilon}")
        print()

        prompt_templates = {
            "standard": "Question: {q}\n\nAnswer:",
            "direct": "{q}",
            "formal": "Please answer the following question:\n\n{q}\n\nYour response:",
            "conversational": "Hey, quick question: {q}",
            "academic": "Query: {q}\n\nProvide a response:",
        }

        results = []
        for q_data in tqdm(questions[:10], desc="Questions"):
            for template_name, template in prompt_templates.items():
                prompt = template.format(q=q_data["question"])

                # Test baseline and steered
                for condition, eps in [("baseline", 0.0), ("steered", optimal_epsilon)]:
                    # Clear hooks
                    self.model.clear_hooks()

                    # Apply steering if needed
                    if eps != 0.0:
                        inputs = self.model.tokenizer(prompt, return_tensors="pt").to(self.model.config.device)
                        position = inputs["input_ids"].shape[1] - 1
                        self.model.register_steering_hook(best_layer, position,
                                                         self.steering_vectors[best_layer], eps)

                    # Generate
                    response = self.model.generate(prompt, temperature=0.0, do_sample=False, max_new_tokens=50)
                    self.model.clear_hooks()

                    # Evaluate
                    answer = extract_answer(response)
                    abstained = (answer == "UNCERTAIN")

                    result = {
                        "question": q_data["question"],
                        "is_unanswerable": q_data.get("is_unanswerable", "answer" not in q_data),
                        "abstained": abstained,
                        "epsilon": eps,
                        "layer": best_layer,
                        "template": template_name,
                        "condition": condition
                    }
                    results.append(result)

        df = pd.DataFrame(results)
        df.to_csv(self.config.results_dir / "exp6b_prompt_variations_fixed.csv", index=False)
        print(f"\n✓ Results saved to exp6b_prompt_variations_fixed.csv")
        return df

    def test_adversarial_questions(self, best_layer: int,
                                   optimal_epsilon: float = -2.0) -> pd.DataFrame:
        """Test on adversarially designed questions (FIXED)"""
        print("\n" + "="*70)
        print("EXPERIMENT 6C: Adversarial Question Testing (FIXED)")
        print("="*70)
        print(f"Using epsilon={optimal_epsilon}")
        print()

        adversarial = [
            # Answerable but phrased to sound unanswerable
            {"question": "Some people claim 2+2=5. What does 2+2 actually equal?",
             "answer": "4", "is_unanswerable": False},
            {"question": "Nobody knows for sure, but what is the capital of France?",
             "answer": "Paris", "is_unanswerable": False},
            {"question": "This is impossible to answer: What is 5*5?",
             "answer": "25", "is_unanswerable": False},

            # Unanswerable but phrased to sound answerable
            {"question": "What is the definitive answer to the meaning of life?",
             "answer": None, "is_unanswerable": True},
            {"question": "Scientists agree that the universe will end on what exact date?",
             "answer": None, "is_unanswerable": True},
            {"question": "According to research, what are you thinking right now?",
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
        df.to_csv(self.config.results_dir / "exp6c_adversarial_fixed.csv", index=False)
        print(f"\n✓ Results saved to exp6c_adversarial_fixed.csv")
        return df

    def run_all(self, best_layer: int = 20, optimal_epsilon: float = -2.0):
        """Run all Experiment 6 tests with fixed parameters"""
        print("\n" + "="*80)
        print("EXPERIMENT 6: ROBUSTNESS & GENERALIZATION (FIXED VERSION)")
        print("="*80)
        print(f"\nParameters:")
        print(f"  Best layer: {best_layer}")
        print(f"  Optimal epsilon: {optimal_epsilon} (CORRECTED from -50.0)")
        print()

        # 6A: Cross-domain
        df_6a = self.test_cross_domain(best_layer, optimal_epsilon)

        # 6B: Prompt variations
        domains = self.create_domain_test_sets()
        all_questions = []
        for domain_questions in domains.values():
            all_questions.extend(domain_questions["answerable"][:2])
            for q in domain_questions["unanswerable"][:2]:
                q["is_unanswerable"] = True
                all_questions.append(q)

        df_6b = self.test_prompt_variations(all_questions, best_layer, optimal_epsilon)

        # 6C: Adversarial
        df_6c = self.test_adversarial_questions(best_layer, optimal_epsilon)

        print("\n" + "="*80)
        print("EXPERIMENT 6 COMPLETE (FIXED)")
        print("="*80)
        print("\nFiles saved:")
        print("  - exp6a_cross_domain_fixed.csv")
        print("  - exp6b_prompt_variations_fixed.csv")
        print("  - exp6c_adversarial_fixed.csv")

        return df_6a, df_6b, df_6c


if __name__ == "__main__":
    # Example usage
    from core_utils import ModelWrapper, ExperimentConfig

    config = ExperimentConfig()
    model = ModelWrapper(config)

    # Load fixed steering vectors
    steering_vectors = torch.load("results/steering_vectors_fixed.pt")

    exp6 = Experiment6Fixed(model, config, steering_vectors)
    exp6.run_all(best_layer=20, optimal_epsilon=-2.0)
