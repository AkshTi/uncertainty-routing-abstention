"""
Debug Utilities - PUBLICATION READY
Addresses ChatGPT feedback: "Debug artifact: dump ~20 JSONL examples/condition with {q,label,resp,extracted,abstained}"

CRITICAL: These tools catch parsing bugs before analysis.
Reviewers often request these artifacts to verify methodology.
"""

import json
from pathlib import Path
from typing import List, Dict
import pandas as pd
from parsing_fixed import extract_answer, is_abstention


# ============================================================================
# Debug Export Functions
# ============================================================================

def export_debug_samples(results: List[Dict],
                        output_file: str = "debug_samples.jsonl",
                        samples_per_condition: int = 20):
    """
    Export sample results to JSONL for manual inspection.

    This is CRITICAL for catching parsing bugs. Always run this before
    analyzing results and manually inspect the output file.

    Args:
        results: List of result dictionaries from experiments
        output_file: Path to output JSONL file
        samples_per_condition: How many samples per condition to export

    Output format (one JSON per line):
    {
        "question": "What is 2+2?",
        "is_unanswerable": false,
        "condition": "baseline",
        "response_full": "4",
        "response_preview": "4",
        "extracted_answer": "4",
        "abstained": false,
        "parsing_correct": true  # manual verification field
    }
    """
    # Group by condition
    by_condition = {}
    for r in results:
        cond = r.get('condition', 'unknown')
        if cond not in by_condition:
            by_condition[cond] = []
        by_condition[cond].append(r)

    samples = []

    # Sample from each condition
    for condition, cond_results in by_condition.items():
        # Stratified sampling: mix of answerable/unanswerable
        answerable = [r for r in cond_results if not r.get('is_unanswerable', False)]
        unanswerable = [r for r in cond_results if r.get('is_unanswerable', False)]

        # Take samples
        n_per_type = samples_per_condition // 2

        for r in answerable[:n_per_type] + unanswerable[:n_per_type]:
            response = r.get('response_preview', r.get('response', ''))
            extracted = extract_answer(response)

            sample = {
                "question": r.get('question', ''),
                "is_unanswerable": r.get('is_unanswerable', False),
                "condition": condition,
                "domain": r.get('domain', ''),
                "response_full": response,
                "response_preview": response[:100] + "..." if len(response) > 100 else response,
                "extracted_answer": extracted,
                "abstained": extracted == "UNCERTAIN",
                "abstained_recorded": r.get('abstained', False),
                "parsing_matches": (extracted == "UNCERTAIN") == r.get('abstained', False),
                # Fields for manual verification
                "parsing_correct": None,  # Fill in manually: true/false
                "notes": ""  # Add notes during manual review
            }

            samples.append(sample)

    # Write to JSONL
    output_path = Path(output_file)
    with open(output_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"✓ Exported {len(samples)} debug samples to {output_file}")
    print(f"  Conditions: {list(by_condition.keys())}")
    print(f"  Samples per condition: ~{samples_per_condition}")
    print(f"\nNEXT STEP: Manually review {output_file} and verify parsing accuracy")

    return output_path


def analyze_debug_samples(debug_file: str = "debug_samples.jsonl"):
    """
    Analyze manually-reviewed debug samples to identify parsing issues.

    After manually reviewing debug_samples.jsonl and filling in the
    'parsing_correct' field, run this to get summary statistics.

    Returns:
        dict with parsing accuracy statistics
    """
    samples = []
    with open(debug_file, 'r') as f:
        for line in f:
            samples.append(json.loads(line))

    # Count manually verified samples
    verified = [s for s in samples if s.get('parsing_correct') is not None]

    if not verified:
        print("⚠ No samples have been manually verified yet!")
        print("   Open debug_samples.jsonl and set 'parsing_correct' to true/false")
        return None

    correct = sum(1 for s in verified if s['parsing_correct'])
    incorrect = len(verified) - correct

    accuracy = correct / len(verified) if verified else 0

    print("=" * 70)
    print("DEBUG SAMPLE ANALYSIS")
    print("=" * 70)
    print(f"Total samples: {len(samples)}")
    print(f"Manually verified: {len(verified)}")
    print(f"Parsing correct: {correct}")
    print(f"Parsing incorrect: {incorrect}")
    print(f"Accuracy: {accuracy:.1%}")
    print()

    if incorrect > 0:
        print("⚠ PARSING ERRORS FOUND:")
        print("-" * 70)
        for s in verified:
            if not s['parsing_correct']:
                print(f"Question: {s['question']}")
                print(f"Response: {s['response_preview']}")
                print(f"Extracted: {s['extracted_answer']}")
                print(f"Notes: {s.get('notes', 'No notes')}")
                print()

    if accuracy < 0.95:
        print("❌ CRITICAL: Parsing accuracy < 95%")
        print("   DO NOT proceed with analysis until parsing is fixed!")
    else:
        print("✓ Parsing accuracy acceptable (≥95%)")

    return {
        "total": len(samples),
        "verified": len(verified),
        "correct": correct,
        "incorrect": incorrect,
        "accuracy": accuracy,
        "errors": [s for s in verified if not s['parsing_correct']]
    }


# ============================================================================
# Response Format Validation
# ============================================================================

def validate_response_formats(results: List[Dict]) -> Dict:
    """
    Validate that responses follow the expected one-line format.

    Analyzes all responses to check:
    - How many are single line
    - How many are multi-line
    - How many mix answer + uncertainty

    Returns:
        dict with validation statistics
    """
    single_line = 0
    multi_line = 0
    mixed_uncertain = 0
    empty = 0

    format_issues = []

    for r in results:
        response = r.get('response_preview', r.get('response', ''))
        lines = [l.strip() for l in response.split('\n') if l.strip()]

        if len(lines) == 0:
            empty += 1
            format_issues.append({
                "question": r.get('question', ''),
                "issue": "empty_response",
                "response": response
            })
        elif len(lines) == 1:
            single_line += 1
        else:
            multi_line += 1

            # Check if it's mixed answer + uncertain
            first_upper = lines[0].upper()
            rest = ' '.join(lines[1:]).upper()

            if first_upper != "UNCERTAIN" and "UNCERTAIN" in rest:
                mixed_uncertain += 1
                format_issues.append({
                    "question": r.get('question', ''),
                    "issue": "mixed_answer_uncertain",
                    "response": response[:100]
                })
            elif first_upper == "UNCERTAIN" and len(lines) > 1:
                format_issues.append({
                    "question": r.get('question', ''),
                    "issue": "uncertain_with_explanation",
                    "response": response[:100]
                })

    total = len(results)

    print("=" * 70)
    print("RESPONSE FORMAT VALIDATION")
    print("=" * 70)
    print(f"Total responses: {total}")
    print(f"Single line (correct): {single_line} ({single_line/total:.1%})")
    print(f"Multi-line: {multi_line} ({multi_line/total:.1%})")
    print(f"  - Mixed answer+uncertain: {mixed_uncertain}")
    print(f"Empty: {empty}")
    print()

    if multi_line > total * 0.1:  # More than 10% multi-line
        print("⚠ WARNING: >10% of responses are multi-line")
        print("   Consider enforcing max_new_tokens=12 for stricter output")
        print()

    if format_issues:
        print(f"Found {len(format_issues)} format issues:")
        for issue in format_issues[:5]:  # Show first 5
            print(f"  - {issue['issue']}: {issue['question'][:50]}...")

    return {
        "total": total,
        "single_line": single_line,
        "multi_line": multi_line,
        "mixed_uncertain": mixed_uncertain,
        "empty": empty,
        "format_issues": format_issues
    }


# ============================================================================
# Parsing Comparison (Old vs New)
# ============================================================================

def compare_parsing_methods_on_results(results: List[Dict]) -> Dict:
    """
    Compare old (buggy) vs new (fixed) parsing on actual results.

    Shows how many results would change with the fix.
    """
    from parsing_fixed import extract_answer_old

    changes = []
    disagreement_count = 0

    for r in results:
        response = r.get('response_preview', r.get('response', ''))

        old = extract_answer_old(response)
        new = extract_answer(response)

        if old != new:
            disagreement_count += 1
            changes.append({
                "question": r.get('question', ''),
                "response": response[:80],
                "old_result": old,
                "new_result": new,
                "old_abstained": old == "UNCERTAIN",
                "new_abstained": new == "UNCERTAIN"
            })

    total = len(results)
    rate = disagreement_count / total if total > 0 else 0

    print("=" * 70)
    print("PARSING METHOD COMPARISON (Old vs New)")
    print("=" * 70)
    print(f"Total responses: {total}")
    print(f"Disagreements: {disagreement_count} ({rate:.1%})")
    print()

    if disagreement_count > 0:
        print(f"⚠ {disagreement_count} results would CHANGE with new parsing:")
        print()
        for change in changes[:10]:  # Show first 10
            print(f"Q: {change['question'][:50]}...")
            print(f"   Response: {change['response']}")
            print(f"   Old: {change['old_result']}")
            print(f"   New: {change['new_result']}")
            print()

        # Show impact on abstention rate
        old_abstentions = sum(1 for c in changes if c['old_abstained'])
        new_abstentions = sum(1 for c in changes if c['new_abstained'])
        print(f"Impact on abstention count in changed samples:")
        print(f"  Old: {old_abstentions} abstentions")
        print(f"  New: {new_abstentions} abstentions")
        print(f"  Δ = {new_abstentions - old_abstentions}")

    return {
        "total": total,
        "disagreements": disagreement_count,
        "disagreement_rate": rate,
        "changes": changes
    }


# ============================================================================
# Integration with Experiments
# ============================================================================

def add_debug_export_to_experiment(results: List[Dict],
                                   experiment_name: str,
                                   export_dir: Path = Path("debug_outputs")):
    """
    Convenience function to add to experiment code.

    Usage in experiment code:
        results = run_experiment()
        add_debug_export_to_experiment(results, "exp6a")
        # This creates debug_outputs/exp6a_debug_samples.jsonl
    """
    export_dir.mkdir(exist_ok=True)
    output_file = export_dir / f"{experiment_name}_debug_samples.jsonl"

    export_debug_samples(results, str(output_file))

    # Also validate format
    print()
    validate_response_formats(results)

    # And compare parsing methods
    print()
    compare_parsing_methods_on_results(results)

    return output_file


# ============================================================================
# Manual Review Template
# ============================================================================

def create_review_template(debug_file: str = "debug_samples.jsonl",
                          output_file: str = "review_template.md"):
    """
    Create a markdown template for manual review of debug samples.

    Makes it easier to review samples systematically.
    """
    samples = []
    with open(debug_file, 'r') as f:
        for line in f:
            samples.append(json.loads(line))

    template = "# Debug Samples Manual Review\n\n"
    template += "Instructions: For each sample, verify if the parsing is correct.\n"
    template += "- If correct: set `parsing_correct` to `true`\n"
    template += "- If incorrect: set `parsing_correct` to `false` and add notes\n\n"
    template += "---\n\n"

    for i, sample in enumerate(samples, 1):
        template += f"## Sample {i}\n\n"
        template += f"**Question:** {sample['question']}\n\n"
        template += f"**Is Unanswerable:** {sample['is_unanswerable']}\n\n"
        template += f"**Condition:** {sample['condition']}\n\n"
        template += f"**Response:**\n```\n{sample['response_full']}\n```\n\n"
        template += f"**Extracted:** `{sample['extracted_answer']}`\n\n"
        template += f"**Abstained:** `{sample['abstained']}`\n\n"
        template += f"**Parsing Correct?** [ ] Yes [ ] No\n\n"
        template += f"**Notes:** \n\n"
        template += "---\n\n"

    with open(output_file, 'w') as f:
        f.write(template)

    print(f"✓ Created review template: {output_file}")
    print(f"  Review {len(samples)} samples and update the JSONL file")

    return output_file


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("DEBUG UTILITIES - Testing")
    print("=" * 70)

    # Example usage
    example_results = [
        {
            "question": "What is 2+2?",
            "is_unanswerable": False,
            "condition": "baseline",
            "response": "4",
            "abstained": False
        },
        {
            "question": "What number am I thinking?",
            "is_unanswerable": True,
            "condition": "baseline",
            "response": "UNCERTAIN\n\nI cannot determine what number you are thinking.",
            "abstained": True
        },
        {
            "question": "What is the capital of France?",
            "is_unanswerable": False,
            "condition": "steered",
            "response": "Paris\n\nI'm not entirely certain about this answer.",
            "abstained": False  # OLD parsing would mark as abstained!
        }
    ]

    # Test export
    print("\nTest 1: Export debug samples")
    export_debug_samples(example_results, "test_debug.jsonl", samples_per_condition=5)

    # Test format validation
    print("\nTest 2: Validate response formats")
    validate_response_formats(example_results)

    # Test parsing comparison
    print("\nTest 3: Compare parsing methods")
    compare_parsing_methods_on_results(example_results)
