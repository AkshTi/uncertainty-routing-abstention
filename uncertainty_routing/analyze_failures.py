import pandas as pd

# Read the CSV file
csv_path = "/Users/akshatatiwari/Desktop/MIT/mech_interp_research/uncertainty_routing/results/exp6a_cross_domain_publication_ready.csv"
df = pd.read_csv(csv_path)

print("=" * 80)
print("FAILURE CASE ANALYSIS")
print("=" * 80)

# Separate answerable and unanswerable
answerable = df[df['is_unanswerable'] == False]
unanswerable = df[df['is_unanswerable'] == True]

# Type 1: Unanswerable questions that were NOT abstained (hallucinations)
print("\n1. UNANSWERABLE QUESTIONS NOT ABSTAINED (Hallucinations)")
print("=" * 80)

unans_not_abstained = unanswerable[unanswerable['abstained'] == False]
print(f"\nTotal: {len(unans_not_abstained)} cases")

for condition in ['baseline', 'steered']:
    cond_failures = unans_not_abstained[unans_not_abstained['condition'] == condition]
    print(f"\n{condition.upper()}: {len(cond_failures)} cases")

    # Show breakdown by domain
    print("\nBy domain:")
    for domain in sorted(df['domain'].unique()):
        domain_failures = cond_failures[cond_failures['domain'] == domain]
        if len(domain_failures) > 0:
            print(f"  {domain}: {len(domain_failures)} failures")

    # Show a few examples
    if len(cond_failures) > 0:
        print(f"\nExample failures (first 3):")
        for idx, row in cond_failures.head(3).iterrows():
            print(f"\n  Question: {row['question']}")
            print(f"  Domain: {row['domain']}")
            print(f"  Extracted Answer: {row['extracted_answer'][:100] if pd.notna(row['extracted_answer']) else 'N/A'}")

# Type 2: Answerable questions that WERE abstained (false rejections)
print("\n\n2. ANSWERABLE QUESTIONS INCORRECTLY ABSTAINED (False Rejections)")
print("=" * 80)

ans_abstained = answerable[answerable['abstained'] == True]
print(f"\nTotal: {len(ans_abstained)} cases")

for condition in ['baseline', 'steered']:
    cond_failures = ans_abstained[ans_abstained['condition'] == condition]
    print(f"\n{condition.upper()}: {len(cond_failures)} cases")

    # Show breakdown by domain
    print("\nBy domain:")
    for domain in sorted(df['domain'].unique()):
        domain_failures = cond_failures[cond_failures['domain'] == domain]
        if len(domain_failures) > 0:
            print(f"  {domain}: {len(domain_failures)} failures")

    # Show a few examples
    if len(cond_failures) > 0:
        print(f"\nExample failures (first 3):")
        for idx, row in cond_failures.head(3).iterrows():
            print(f"\n  Question: {row['question']}")
            print(f"  Domain: {row['domain']}")
            print(f"  Response Preview: {row['response_preview'][:100] if pd.notna(row['response_preview']) else 'N/A'}")

# Type 3: Answerable questions answered INCORRECTLY (not abstained, just wrong)
print("\n\n3. ANSWERABLE QUESTIONS ANSWERED INCORRECTLY")
print("=" * 80)

ans_wrong = answerable[(answerable['abstained'] == False) & (answerable['correct'] == False)]
print(f"\nTotal: {len(ans_wrong)} cases")

for condition in ['baseline', 'steered']:
    cond_failures = ans_wrong[ans_wrong['condition'] == condition]
    print(f"\n{condition.upper()}: {len(cond_failures)} cases")

    # Show breakdown by domain
    print("\nBy domain:")
    for domain in sorted(df['domain'].unique()):
        domain_failures = cond_failures[cond_failures['domain'] == domain]
        if len(domain_failures) > 0:
            print(f"  {domain}: {len(domain_failures)} failures")

# Type 4: Domain-specific patterns
print("\n\n4. DOMAIN-SPECIFIC PATTERNS")
print("=" * 80)

for domain in sorted(df['domain'].unique()):
    print(f"\n{domain.upper()}:")
    print("-" * 60)

    domain_df = df[df['domain'] == domain]

    # For unanswerable in this domain
    domain_unans = domain_df[domain_df['is_unanswerable'] == True]

    for condition in ['baseline', 'steered']:
        cond_unans = domain_unans[domain_unans['condition'] == condition]
        abst_rate = cond_unans['abstained'].mean()
        hall_rate = cond_unans['hallucinated'].mean()

        print(f"  {condition.capitalize()} unanswerable: Abst={abst_rate:.3f}, Hall={hall_rate:.3f}")

    # For answerable in this domain
    domain_ans = domain_df[domain_df['is_unanswerable'] == False]

    for condition in ['baseline', 'steered']:
        cond_ans = domain_ans[domain_ans['condition'] == condition]
        acc = cond_ans['correct'].mean()
        abst_rate = cond_ans['abstained'].mean()

        print(f"  {condition.capitalize()} answerable: Acc={acc:.3f}, Abst={abst_rate:.3f}")

# Type 5: Questions where steering made things WORSE
print("\n\n5. CASES WHERE STEERING MADE THINGS WORSE")
print("=" * 80)

# Get unique questions (each appears twice - baseline and steered)
unique_questions = df[df['condition'] == 'baseline'][['question', 'is_unanswerable', 'domain']].reset_index(drop=True)

worse_cases = []

for idx, row in unique_questions.iterrows():
    question = row['question']
    is_unans = row['is_unanswerable']
    domain = row['domain']

    baseline_row = df[(df['question'] == question) & (df['condition'] == 'baseline')].iloc[0]
    steered_row = df[(df['question'] == question) & (df['condition'] == 'steered')].iloc[0]

    if is_unans:
        # For unanswerable, worse = steered abstained less than baseline
        if baseline_row['abstained'] and not steered_row['abstained']:
            worse_cases.append({
                'question': question,
                'domain': domain,
                'type': 'unanswerable',
                'baseline': 'abstained',
                'steered': 'hallucinated'
            })
    else:
        # For answerable, worse = correct in baseline but not in steered
        if baseline_row['correct'] and not steered_row['correct']:
            worse_cases.append({
                'question': question,
                'domain': domain,
                'type': 'answerable',
                'baseline': 'correct',
                'steered': 'incorrect or abstained'
            })

print(f"\nTotal cases where steering made things worse: {len(worse_cases)}")

if len(worse_cases) > 0:
    worse_df = pd.DataFrame(worse_cases)
    print("\nBreakdown by type:")
    print(worse_df['type'].value_counts())

    print("\nBreakdown by domain:")
    print(worse_df['domain'].value_counts())

    print("\nExamples (first 5):")
    for i, case in enumerate(worse_df.head(5).to_dict('records'), 1):
        print(f"\n{i}. [{case['type']}] {case['question']}")
        print(f"   Domain: {case['domain']}")
        print(f"   Baseline: {case['baseline']}, Steered: {case['steered']}")

# Type 6: Questions where steering HELPED
print("\n\n6. CASES WHERE STEERING HELPED")
print("=" * 80)

better_cases = []

for idx, row in unique_questions.iterrows():
    question = row['question']
    is_unans = row['is_unanswerable']
    domain = row['domain']

    baseline_row = df[(df['question'] == question) & (df['condition'] == 'baseline')].iloc[0]
    steered_row = df[(df['question'] == question) & (df['condition'] == 'steered')].iloc[0]

    if is_unans:
        # For unanswerable, better = steered abstained but baseline didn't
        if not baseline_row['abstained'] and steered_row['abstained']:
            better_cases.append({
                'question': question,
                'domain': domain,
                'type': 'unanswerable',
                'baseline': 'hallucinated',
                'steered': 'abstained'
            })
    else:
        # For answerable, better = incorrect in baseline but correct in steered
        if not baseline_row['correct'] and steered_row['correct']:
            better_cases.append({
                'question': question,
                'domain': domain,
                'type': 'answerable',
                'baseline': 'incorrect',
                'steered': 'correct'
            })

print(f"\nTotal cases where steering helped: {len(better_cases)}")

if len(better_cases) > 0:
    better_df = pd.DataFrame(better_cases)
    print("\nBreakdown by type:")
    print(better_df['type'].value_counts())

    print("\nBreakdown by domain:")
    print(better_df['domain'].value_counts())

print("\n\n" + "=" * 80)
print("NET EFFECT SUMMARY")
print("=" * 80)

print(f"\nCases where steering made things WORSE: {len(worse_cases)}")
print(f"Cases where steering made things BETTER: {len(better_cases)}")
print(f"Net benefit: {len(better_cases) - len(worse_cases)} cases")

if len(better_cases) > len(worse_cases):
    print("\n✓ Overall, steering has a POSITIVE effect")
elif len(worse_cases) > len(better_cases):
    print("\n✗ Overall, steering has a NEGATIVE effect")
else:
    print("\n= Steering has a NEUTRAL effect")
