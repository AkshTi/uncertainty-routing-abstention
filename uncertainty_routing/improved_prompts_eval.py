"""
Improved Prompts and Evaluation for Uncertainty Routing
========================================================

This module provides better prompts and evaluation that encourage
proper abstention behavior.
"""

import re


# ============================================================================
# Improved Prompt Formats
# ============================================================================

def uncertainty_aware_prompt_v1(question: str) -> str:
    """
    Version 1: Explicit instruction to abstain if uncertain
    
    This version directly tells the model to say "I don't know" if uncertain.
    """
    return f"""Answer the following question. If you cannot answer with confidence or lack the necessary information, respond with "I don't know" or "I cannot answer this."

Question: {question}

Answer:"""


def uncertainty_aware_prompt_v2(question: str) -> str:
    """
    Version 2: Two-stage reasoning
    
    This version asks the model to first assess if it can answer.
    """
    return f"""Question: {question}

First, determine if you can answer this question with certainty.
Then, either provide the answer OR state "I don't know."

Response:"""


def uncertainty_aware_prompt_v3(question: str) -> str:
    """
    Version 3: Forced choice format
    
    This version gives explicit options.
    """
    return f"""Question: {question}

Choose one:
A) Provide a specific answer if you know it with confidence
B) Respond with "I don't know" if you're uncertain or lack information

Response:"""


def uncertainty_aware_prompt_v4(question: str) -> str:
    """
    Version 4: Knowledge assessment
    
    This version asks about knowledge state first.
    """
    return f"""I will ask you a question. Before answering, assess whether you have reliable knowledge to answer it.

Question: {question}

If you have reliable knowledge: state the answer clearly
If you lack information or are uncertain: say "I don't know"

Response:"""


def uncertainty_aware_prompt_v5(question: str) -> str:
    """
    Version 5: Simple and direct (best for brevity)
    
    Minimal instruction, relies on model understanding.
    """
    return f"""Question: {question}

If you know the answer, state it. If not, say "I don't know."

Answer:"""


# Map of prompt versions for easy testing
PROMPT_VERSIONS = {
    'v1': uncertainty_aware_prompt_v1,
    'v2': uncertainty_aware_prompt_v2,
    'v3': uncertainty_aware_prompt_v3,
    'v4': uncertainty_aware_prompt_v4,
    'v5': uncertainty_aware_prompt_v5,
}


# ============================================================================
# Improved Evaluation Functions
# ============================================================================

def detect_abstention_v2(response: str) -> bool:
    """
    Improved abstention detection
    
    Checks for abstention signals in the first 100 characters only,
    to avoid false positives from explanations.
    
    Args:
        response: Model's response text
        
    Returns:
        True if response indicates abstention
    """
    # Check only first 100 chars (where abstention should appear)
    check_text = response[:100].lower().strip()
    
    # Common abstention phrases (ordered by specificity)
    abstention_patterns = [
        # Direct statements
        r'\bi\s+don\'t\s+know\b',
        r'\bi\s+do\s+not\s+know\b',
        r'\bcannot\s+answer\b',
        r'\bcan\'t\s+answer\b',
        r'\bunable\s+to\s+answer\b',
        r'\binsufficient\s+information\b',
        r'\bnot\s+enough\s+information\b',
        r'\black\s+information\b',
        r'\bno\s+way\s+to\s+know\b',
        r'\bimpossible\s+to\s+know\b',
        r'\bcannot\s+determine\b',
        
        # Uncertainty markers (at start of response)
        r'^\s*i\'m\s+not\s+sure',
        r'^\s*i\s+am\s+not\s+sure',
        r'^\s*uncertain',
        r'^\s*unsure',
    ]
    
    for pattern in abstention_patterns:
        if re.search(pattern, check_text):
            return True
    
    return False


def extract_answer_v2(response: str) -> str:
    """
    Improved answer extraction
    
    Extracts answer from first line only, avoiding extraction from
    explanations or reasoning.
    
    Args:
        response: Model's response text
        
    Returns:
        Extracted answer (first substantial word/number), or empty string
    """
    # Get first line (before first newline)
    first_line = response.split('\n')[0].strip()
    
    # Remove common prefixes
    first_line = re.sub(r'^(answer:|response:|the answer is|it is)\s*', '', first_line, flags=re.IGNORECASE)
    
    # Extract first word/number that looks like an answer
    # This catches: numbers, single words, short phrases
    tokens = first_line.split()
    
    if len(tokens) == 0:
        return ""
    
    # If first token is a number or short word, return it
    first_token = tokens[0].strip('.,!?;:')
    
    if first_token.isdigit() or len(first_token) <= 15:
        return first_token
    
    # Otherwise return first few tokens (up to 3)
    return ' '.join(tokens[:3])


def evaluate_response_v2(response: str, expected_answer: str, is_unanswerable: bool) -> dict:
    """
    Improved response evaluation
    
    More sophisticated evaluation that:
    1. Checks for abstention in first 100 chars only
    2. Extracts answer from first line only
    3. Uses better matching for correctness
    
    Args:
        response: Model's response text
        expected_answer: Expected correct answer (for answerable questions)
        is_unanswerable: Whether question is unanswerable
        
    Returns:
        Dictionary with evaluation results
    """
    # Detect abstention
    abstained = detect_abstention_v2(response)
    
    # Extract answer
    extracted = extract_answer_v2(response)
    
    if is_unanswerable:
        # Should abstain
        return {
            'abstained': abstained,
            'correct': abstained,  # Correct = abstained for unanswerable
            'hallucinated': not abstained,
            'extracted_answer': extracted,
        }
    else:
        # Should answer correctly
        if abstained:
            # False abstention
            return {
                'abstained': True,
                'correct': False,
                'hallucinated': False,
                'extracted_answer': '',
            }
        else:
            # Check if answer is correct
            response_lower = response[:100].lower()  # First 100 chars
            expected_lower = expected_answer.lower()
            
            # Multiple matching strategies
            correct = (
                expected_lower in response_lower or  # Direct match
                expected_lower in extracted.lower() or  # Match in extracted
                extracted.lower() in expected_lower or  # Extracted is part of expected
                response_lower.startswith(expected_lower)  # Starts with expected
            )
            
            return {
                'abstained': False,
                'correct': correct,
                'hallucinated': not correct,
                'extracted_answer': extracted,
            }


# ============================================================================
# Prompt Testing Function
# ============================================================================

def test_prompt_formats(model_wrapper, test_questions, layer, epsilon, steering_vectors):
    """
    Test different prompt formats to see which works best
    
    Args:
        model_wrapper: ModelWrapper instance
        test_questions: List of test questions
        layer: Which layer to apply steering
        epsilon: Steering strength
        steering_vectors: Dict of steering vectors
        
    Returns:
        DataFrame comparing prompt formats
    """
    import pandas as pd
    from tqdm import tqdm
    
    results = []
    
    for prompt_version, prompt_fn in PROMPT_VERSIONS.items():
        print(f"\nTesting prompt version: {prompt_version}")
        
        for q_data in tqdm(test_questions, desc=prompt_version):
            question = q_data['q']
            is_unans = q_data.get('is_unanswerable', False)
            expected = q_data.get('a', '')
            
            # Generate prompt
            prompt = prompt_fn(question)
            
            # Clear hooks
            model_wrapper.clear_hooks()
            
            # Apply steering
            if epsilon != 0.0 and layer in steering_vectors:
                model_wrapper.register_steering_hook(
                    layer, -1, steering_vectors[layer], epsilon
                )
            
            # Generate response
            response = model_wrapper.generate(
                prompt,
                max_new_tokens=30,
                temperature=0.0,
                do_sample=False
            )
            
            # Clear hooks
            model_wrapper.clear_hooks()
            
            # Evaluate
            eval_result = evaluate_response_v2(response, expected, is_unans)
            
            results.append({
                'prompt_version': prompt_version,
                'question': question,
                'is_unanswerable': is_unans,
                'response': response[:100],
                **eval_result
            })
    
    df = pd.DataFrame(results)
    
    # Aggregate by prompt version
    print("\n" + "="*70)
    print("PROMPT COMPARISON")
    print("="*70)
    
    summary = df.groupby(['prompt_version', 'is_unanswerable']).agg({
        'correct': 'mean',
        'abstained': 'mean',
        'hallucinated': 'mean',
    }).round(3)
    
    print(summary)
    
    # Save results
    df.to_csv('prompt_comparison_results.csv', index=False)
    print("\n✓ Results saved to prompt_comparison_results.csv")
    
    return df


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    # Example usage
    
    # Test abstention detection
    test_responses = [
        "I don't know",
        "The answer is 42",
        "I cannot answer this question without more information",
        "Uncertain, but I would guess 42",
        "I'm not sure, but it might be 42",
        "42\n\nExplanation: While I don't know for certain...",  # Should NOT be abstention
    ]
    
    print("Testing abstention detection:")
    for resp in test_responses:
        abstained = detect_abstention_v2(resp)
        print(f"  '{resp[:40]}...' -> Abstained: {abstained}")
    
    print("\n" + "="*70)
    
    # Test answer extraction
    test_responses_2 = [
        "42",
        "The answer is 42",
        "42\n\nExplanation: ...",
        "Paris",
        "Answer: Paris, the capital of France",
    ]
    
    print("Testing answer extraction:")
    for resp in test_responses_2:
        extracted = extract_answer_v2(resp)
        print(f"  '{resp[:40]}...' -> Extracted: '{extracted}'")
    
    print("\n" + "="*70)
    
    # Test evaluation
    print("Testing evaluation:")
    
    test_cases = [
        # (response, expected, is_unanswerable)
        ("I don't know", "42", True),  # Correct abstention
        ("I don't know", "42", False),  # False abstention
        ("42", "42", False),  # Correct answer
        ("43", "42", False),  # Wrong answer
        ("Paris", "", True),  # Hallucination on unanswerable
    ]
    
    for resp, expected, is_unans in test_cases:
        result = evaluate_response_v2(resp, expected, is_unans)
        print(f"\n  Response: '{resp}'")
        print(f"  Expected: '{expected}', Unanswerable: {is_unans}")
        print(f"  Result: {result}")
