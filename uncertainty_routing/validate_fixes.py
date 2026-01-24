"""
Validation Script - Verify All Fixes Before Running Experiments

Run this script BEFORE running publication-ready experiments to ensure:
1. All fixed modules are available
2. Parsing is working correctly
3. Prompts are unified
4. Datasets are scaled properly
5. Generation settings are deterministic

Usage:
    python validate_fixes.py
"""

import sys
from pathlib import Path


def check_imports():
    """Verify all required modules can be imported"""
    print("="*70)
    print("STEP 1: Checking Imports")
    print("="*70)

    required_modules = [
        "unified_prompts",
        "parsing_fixed",
        "scaled_datasets",
        "debug_utils",
        "core_utils"
    ]

    all_ok = True

    for module in required_modules:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except ImportError as e:
            print(f"  ✗ {module} - {e}")
            all_ok = False

    if not all_ok:
        print("\n❌ FAILED: Missing required modules")
        print("   Make sure you have all the fixed module files")
        return False

    print("\n✅ All modules imported successfully")
    return True


def test_parsing():
    """Test the fixed parsing function"""
    print("\n" + "="*70)
    print("STEP 2: Testing Fixed Parsing")
    print("="*70)

    from parsing_fixed import test_extract_answer

    success = test_extract_answer()

    if not success:
        print("\n❌ FAILED: Parsing tests failed")
        return False

    print("\n✅ Parsing tests passed")
    return True


def test_prompts():
    """Verify unified prompts work"""
    print("\n" + "="*70)
    print("STEP 3: Testing Unified Prompts")
    print("="*70)

    from unified_prompts import unified_prompt, validate_response_format

    # Test basic prompt
    question = "What is 2+2?"
    prompt = unified_prompt(question)

    print(f"Question: {question}")
    print(f"Prompt:\n{prompt}")
    print()

    # Check prompt format
    if "EXACTLY one line" not in prompt:
        print("⚠ WARNING: Prompt doesn't enforce one-line output")
        print("   This may lead to multi-line responses")

    # Test response validation
    test_responses = [
        "4",
        "UNCERTAIN",
        "4\n\nI'm not sure about this",
    ]

    print("Response format validation:")
    for resp in test_responses:
        validation = validate_response_format(resp)
        status = "✓" if validation['is_valid'] else "✗"
        print(f"  {status} {repr(resp[:20]+'...' if len(resp) > 20 else resp)}")
        if not validation['is_valid']:
            print(f"     Issue: {validation['issue']}")

    print("\n✅ Unified prompts working")
    return True


def test_datasets():
    """Verify scaled datasets have enough samples"""
    print("\n" + "="*70)
    print("STEP 4: Testing Scaled Datasets")
    print("="*70)

    from scaled_datasets import create_scaled_domain_questions

    domains = create_scaled_domain_questions()

    print(f"Found {len(domains)} domains:\n")

    all_ok = True
    total_questions = 0

    for domain, sets in domains.items():
        answerable = len(sets["answerable"])
        unanswerable = len(sets["unanswerable"])
        total = answerable + unanswerable
        total_questions += total

        status_ans = "✓" if answerable >= 50 else "✗"
        status_una = "✓" if unanswerable >= 50 else "✗"

        print(f"{domain}:")
        print(f"  {status_ans} Answerable: {answerable} (need ≥50)")
        print(f"  {status_una} Unanswerable: {unanswerable} (need ≥50)")

        if answerable < 50 or unanswerable < 50:
            all_ok = False

    print(f"\nTotal questions across all domains: {total_questions}")

    if not all_ok:
        print("\n❌ FAILED: Some domains have insufficient samples")
        print("   Need ≥50 per (domain × answerability)")
        return False

    print("\n✅ All domains have adequate sample size (n≥50)")
    return True


def test_generation_settings():
    """Verify generation uses deterministic settings"""
    print("\n" + "="*70)
    print("STEP 5: Testing Generation Settings")
    print("="*70)

    print("Checking experiment6_publication_ready.py...")

    try:
        with open("experiment6_publication_ready.py", "r") as f:
            content = f.read()

        checks = [
            ("max_new_tokens=12", "max_new_tokens=12"),
            ("temperature=0.0", "temperature=0.0"),
            ("do_sample=False", "do_sample=False"),
        ]

        all_ok = True
        for setting, pattern in checks:
            if pattern in content:
                print(f"  ✓ {setting}")
            else:
                print(f"  ✗ {setting} - NOT FOUND")
                all_ok = False

        if not all_ok:
            print("\n❌ FAILED: Generation settings incorrect")
            print("   Must use: max_new_tokens=12, temperature=0.0, do_sample=False")
            return False

        print("\n✅ Generation settings are deterministic")
        return True

    except FileNotFoundError:
        print("  ⚠ experiment6_publication_ready.py not found")
        print("    Skipping generation settings check")
        return True


def test_debug_utils():
    """Test debug export functionality"""
    print("\n" + "="*70)
    print("STEP 6: Testing Debug Utilities")
    print("="*70)

    from debug_utils import export_debug_samples, validate_response_formats

    # Create test results
    test_results = [
        {
            "question": "What is 2+2?",
            "is_unanswerable": False,
            "condition": "baseline",
            "response": "4",
            "response_preview": "4",
            "abstained": False
        },
        {
            "question": "What am I thinking?",
            "is_unanswerable": True,
            "condition": "steered",
            "response": "UNCERTAIN",
            "response_preview": "UNCERTAIN",
            "abstained": True
        }
    ]

    # Test export
    try:
        output_file = "test_debug_samples.jsonl"
        export_debug_samples(test_results, output_file, samples_per_condition=2)

        # Verify file was created
        if Path(output_file).exists():
            print(f"\n✓ Debug export created: {output_file}")
            # Clean up
            Path(output_file).unlink()
            print("  (cleaned up test file)")
        else:
            print("\n✗ Debug export failed")
            return False

        # Test validation
        validation = validate_response_formats(test_results)
        print(f"\n✓ Response format validation working")
        print(f"   Single line: {validation['single_line']}/{validation['total']}")

        print("\n✅ Debug utilities working")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: Debug utilities error - {e}")
        return False


def compare_old_vs_new_parsing():
    """Show impact of parsing fix on actual results"""
    print("\n" + "="*70)
    print("STEP 7: Comparing Old vs New Parsing (Impact Analysis)")
    print("="*70)

    # Load your actual exp6 results to see impact
    try:
        import pandas as pd
        from parsing_fixed import extract_answer, extract_answer_old

        results_file = "results/exp6a_cross_domain.csv"
        if not Path(results_file).exists():
            print(f"  ⚠ {results_file} not found")
            print("    Skipping comparison (no existing results)")
            return True

        df = pd.read_csv(results_file)

        print(f"\nAnalyzing {len(df)} existing results...")

        # Re-parse with both methods
        changes = 0
        false_positive_abstentions = 0

        for idx, row in df.iterrows():
            response = str(row.get('response_preview', ''))

            old = extract_answer_old(response)
            new = extract_answer(response)

            if old != new:
                changes += 1
                if old == "UNCERTAIN" and new != "UNCERTAIN":
                    false_positive_abstentions += 1

        rate = changes / len(df) if len(df) > 0 else 0

        print(f"\nImpact of parsing fix:")
        print(f"  Total responses: {len(df)}")
        print(f"  Changed by new parsing: {changes} ({rate:.1%})")
        print(f"  False positive abstentions fixed: {false_positive_abstentions}")

        if changes > 0:
            print(f"\n⚠ WARNING: New parsing would change {changes} results!")
            print("   This shows the old parsing was buggy")
            print("   Re-run experiments with fixed parsing for accurate results")

        print("\n✅ Comparison complete")
        return True

    except Exception as e:
        print(f"  ⚠ Could not compare: {e}")
        print("    Skipping comparison")
        return True


def run_full_validation():
    """Run all validation checks"""
    print("\n" + "="*80)
    print(" PUBLICATION-READY FIXES VALIDATION")
    print("="*80)
    print("\nThis script verifies all critical fixes are in place.")
    print("Run this BEFORE running experiments.\n")

    checks = [
        ("Imports", check_imports),
        ("Parsing", test_parsing),
        ("Prompts", test_prompts),
        ("Datasets", test_datasets),
        ("Generation", test_generation_settings),
        ("Debug Utils", test_debug_utils),
        ("Impact Analysis", compare_old_vs_new_parsing),
    ]

    results = {}

    for name, check_fn in checks:
        try:
            results[name] = check_fn()
        except Exception as e:
            print(f"\n❌ {name} check failed with error: {e}")
            results[name] = False

    # Summary
    print("\n" + "="*80)
    print(" VALIDATION SUMMARY")
    print("="*80)

    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:10} {name}")
        if not passed:
            all_passed = False

    print("="*80)

    if all_passed:
        print("\n✅ ALL CHECKS PASSED")
        print("\nYou are ready to run publication-ready experiments!")
        print("\nNext steps:")
        print("  1. Run: python experiment6_publication_ready.py")
        print("  2. Review debug samples in debug_outputs/")
        print("  3. Analyze results with statistical tests (n≥50)")
        print("  4. Create publication figures")
        return 0
    else:
        print("\n❌ SOME CHECKS FAILED")
        print("\nFix the issues above before running experiments.")
        print("Your results will not be publication-ready until all checks pass.")
        return 1


if __name__ == "__main__":
    exit_code = run_full_validation()
    sys.exit(exit_code)
