import pandas as pd

# Read the CSV file
csv_path = "/Users/akshatatiwari/Desktop/MIT/mech_interp_research/uncertainty_routing/results/exp6a_cross_domain_publication_ready.csv"
df = pd.read_csv(csv_path)

print("=" * 80)
print("DETAILED EXAMPLE ANALYSIS")
print("=" * 80)

# Find specific interesting cases

# 1. Science domain - perfect performance on unanswerable
print("\n1. SCIENCE DOMAIN: Why perfect performance on unanswerable?")
print("=" * 80)

science_unans = df[(df['domain'] == 'science') & (df['is_unanswerable'] == True)]
print(f"\nScience unanswerable questions: {len(science_unans) // 2} unique questions")
print(f"Baseline abstention: {science_unans[science_unans['condition']=='baseline']['abstained'].mean():.3f}")
print(f"Steered abstention: {science_unans[science_unans['condition']=='steered']['abstained'].mean():.3f}")

# Show the one that failed (hallucinated)
science_hall = science_unans[science_unans['abstained'] == False]
if len(science_hall) > 0:
    print(f"\nThe {len(science_hall)} hallucination cases in science:")
    for idx, row in science_hall.iterrows():
        print(f"\n  Condition: {row['condition']}")
        print(f"  Question: {row['question']}")
        print(f"  Answer: {row['extracted_answer']}")

# 2. Mathematics - worst performance on unanswerable
print("\n\n2. MATHEMATICS DOMAIN: Why worst performance on unanswerable?")
print("=" * 80)

math_unans = df[(df['domain'] == 'mathematics') & (df['is_unanswerable'] == True)]
print(f"\nMathematics unanswerable questions: {len(math_unans) // 2} unique questions")
print(f"Baseline abstention: {math_unans[math_unans['condition']=='baseline']['abstained'].mean():.3f}")
print(f"Steered abstention: {math_unans[math_unans['condition']=='steered']['abstained'].mean():.3f}")

# Show examples of hallucinations
math_hall = math_unans[math_unans['abstained'] == False]
print(f"\nMathematics hallucination examples ({len(math_hall)} total):")

# Group by question to see which questions are problematic
math_hall_questions = math_hall.groupby('question').size().sort_values(ascending=False)
print(f"\nMost commonly hallucinated questions (both baseline and steered):")
for question, count in math_hall_questions.head(5).items():
    print(f"  ({count}/2 conditions) {question}")

# 3. History - accuracy dropped with steering
print("\n\n3. HISTORY DOMAIN: Why did accuracy DROP with steering?")
print("=" * 80)

history_ans = df[(df['domain'] == 'history') & (df['is_unanswerable'] == False)]
print(f"\nHistory answerable questions: {len(history_ans) // 2} unique questions")
print(f"Baseline accuracy: {history_ans[history_ans['condition']=='baseline']['correct'].mean():.3f}")
print(f"Steered accuracy: {history_ans[history_ans['condition']=='steered']['correct'].mean():.3f}")
print(f"Baseline abstention: {history_ans[history_ans['condition']=='baseline']['abstained'].mean():.3f}")
print(f"Steered abstention: {history_ans[history_ans['condition']=='steered']['abstained'].mean():.3f}")

# Questions that went from correct to incorrect
unique_questions = df[(df['condition'] == 'baseline') & (df['domain'] == 'history') &
                      (df['is_unanswerable'] == False)]['question'].tolist()

history_degraded = []
for question in unique_questions:
    baseline_row = df[(df['question'] == question) & (df['condition'] == 'baseline')].iloc[0]
    steered_row = df[(df['question'] == question) & (df['condition'] == 'steered')].iloc[0]

    if baseline_row['correct'] and not steered_row['correct']:
        history_degraded.append({
            'question': question,
            'baseline_abstained': baseline_row['abstained'],
            'steered_abstained': steered_row['abstained'],
            'steered_correct': steered_row['correct']
        })

print(f"\nQuestions that degraded with steering: {len(history_degraded)}")
if len(history_degraded) > 0:
    for case in history_degraded[:3]:
        print(f"\n  Q: {case['question']}")
        print(f"  Baseline: Correct")
        if case['steered_abstained']:
            print(f"  Steered: Abstained (false rejection)")
        else:
            print(f"  Steered: Incorrect answer")

# 4. Mathematics - accuracy IMPROVED with steering
print("\n\n4. MATHEMATICS DOMAIN: Why did accuracy IMPROVE with steering?")
print("=" * 80)

math_ans = df[(df['domain'] == 'mathematics') & (df['is_unanswerable'] == False)]
print(f"\nMathematics answerable questions: {len(math_ans) // 2} unique questions")
print(f"Baseline accuracy: {math_ans[math_ans['condition']=='baseline']['correct'].mean():.3f}")
print(f"Steered accuracy: {math_ans[math_ans['condition']=='steered']['correct'].mean():.3f}")
print(f"Baseline abstention: {math_ans[math_ans['condition']=='baseline']['abstained'].mean():.3f}")
print(f"Steered abstention: {math_ans[math_ans['condition']=='steered']['abstained'].mean():.3f}")

# 5. Overall patterns
print("\n\n5. OVERALL PATTERN ANALYSIS")
print("=" * 80)

print("\nSummary of steering effects by metric:")
print("-" * 60)

for domain in sorted(df['domain'].unique()):
    domain_df = df[df['domain'] == domain]

    # Unanswerable abstention
    base_unans_abst = domain_df[(domain_df['is_unanswerable']==True) &
                                 (domain_df['condition']=='baseline')]['abstained'].mean()
    steer_unans_abst = domain_df[(domain_df['is_unanswerable']==True) &
                                  (domain_df['condition']=='steered')]['abstained'].mean()

    # Answerable accuracy
    base_ans_acc = domain_df[(domain_df['is_unanswerable']==False) &
                              (domain_df['condition']=='baseline')]['correct'].mean()
    steer_ans_acc = domain_df[(domain_df['is_unanswerable']==False) &
                               (domain_df['condition']=='steered')]['correct'].mean()

    # Answerable abstention
    base_ans_abst = domain_df[(domain_df['is_unanswerable']==False) &
                               (domain_df['condition']=='baseline')]['abstained'].mean()
    steer_ans_abst = domain_df[(domain_df['is_unanswerable']==False) &
                                (domain_df['condition']=='steered')]['abstained'].mean()

    print(f"\n{domain.upper()}:")
    print(f"  Unanswerable abstention: {base_unans_abst:.3f} → {steer_unans_abst:.3f} " +
          f"(Δ={steer_unans_abst - base_unans_abst:+.3f})" +
          (" ✓" if steer_unans_abst > base_unans_abst else " ✗" if steer_unans_abst < base_unans_abst else " ="))
    print(f"  Answerable accuracy:     {base_ans_acc:.3f} → {steer_ans_acc:.3f} " +
          f"(Δ={steer_ans_acc - base_ans_acc:+.3f})" +
          (" ✓" if steer_ans_acc > base_ans_acc else " ✗" if steer_ans_acc < base_ans_acc else " ="))
    print(f"  Answerable abstention:   {base_ans_abst:.3f} → {steer_ans_abst:.3f} " +
          f"(Δ={steer_ans_abst - base_ans_abst:+.3f})" +
          (" ✓" if steer_ans_abst < base_ans_abst else " ✗" if steer_ans_abst > base_ans_abst else " ="))

# 6. Check baseline performance - is it already too good?
print("\n\n6. BASELINE PERFORMANCE ANALYSIS")
print("=" * 80)
print("\nIs the baseline already performing well enough that steering can't help much?")
print("-" * 60)

baseline_df = df[df['condition'] == 'baseline']
baseline_unans = baseline_df[baseline_df['is_unanswerable'] == True]
baseline_ans = baseline_df[baseline_df['is_unanswerable'] == False]

print(f"\nBaseline (ε=0.0) already achieves:")
print(f"  Unanswerable abstention: {baseline_unans['abstained'].mean():.3f} (89%!)")
print(f"  Answerable accuracy: {baseline_ans['correct'].mean():.3f}")
print(f"  Answerable abstention: {baseline_ans['abstained'].mean():.3f}")

print("\nThis suggests the baseline is already quite good at abstaining on")
print("unanswerable questions, leaving little room for improvement from steering.")

print("\n\nRECOMMENDATION:")
print("Consider testing with more challenging unanswerable questions where")
print("the baseline doesn't already achieve 89% abstention rate.")
