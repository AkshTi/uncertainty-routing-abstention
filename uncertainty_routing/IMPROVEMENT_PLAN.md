# Improvement Plan for Experiments 6 & 7

## Root Cause
Steering vectors from Exp5 were trained on only 5 answerable + 5 unanswerable questions, with minimal domain diversity. This caused:
- Mathematics: 0% abstention (only trained on "2+2")
- Overall: 55% abstention (expected ~100%)
- High variance across domains

## Priority 1: Retrain Steering Vectors (CRITICAL)

### Expand Training Data

**Current training set:** 5 + 5 questions
**Target training set:** 20 + 20 questions minimum

#### Add to `dataset_clearly_answerable.json`:
```json
// Mathematics (add 5 more):
{"question": "What is 15 * 8?", "answer": "120"},
{"question": "What is the square root of 144?", "answer": "12"},
{"question": "What is 7^2?", "answer": "49"},
{"question": "What is 100 divided by 4?", "answer": "25"},
{"question": "What is 3 + 5 * 2?", "answer": "13"}

// Science (add 5):
{"question": "What is the speed of light in vacuum?", "answer": "299792458 m/s"},
{"question": "How many planets are in our solar system?", "answer": "8"},
{"question": "What is the chemical formula for salt?", "answer": "NaCl"},
{"question": "What gas do plants produce?", "answer": "oxygen"},
{"question": "What is the atomic number of carbon?", "answer": "6"}

// History (add 3):
{"question": "Who was the first US president?", "answer": "George Washington"},
{"question": "In what year did World War I begin?", "answer": "1914"},
{"question": "Who painted the Mona Lisa?", "answer": "Leonardo da Vinci"}

// Geography (add 2):
{"question": "What is the capital of Japan?", "answer": "Tokyo"},
{"question": "What is the largest ocean?", "answer": "Pacific"}
```

#### Add to `dataset_clearly_unanswerable.json`:
```json
// Mathematics (add 5):
{"question": "What is the largest prime number?", "answer": null},
{"question": "What is the last digit of pi?", "answer": null},
{"question": "What number am I thinking of right now?", "answer": null},
{"question": "What is the solution to the unsolved Goldbach conjecture?", "answer": null},
{"question": "What is infinity + 1?", "answer": null}

// Science (add 5):
{"question": "What is the exact temperature at the center of the sun right now?", "answer": null},
{"question": "How many atoms are in this room at this exact moment?", "answer": null},
{"question": "What will be the next major scientific discovery?", "answer": null},
{"question": "What is the cure for all forms of cancer?", "answer": null},
{"question": "How many alien civilizations exist?", "answer": null}

// History (add 3):
{"question": "What was Cleopatra thinking when she died?", "answer": null},
{"question": "What was the weather on Caesar's assassination day?", "answer": null},
{"question": "How many words did Homer speak in his life?", "answer": null}

// Future predictions (add 2):
{"question": "What will the weather be exactly one year from today?", "answer": null},
{"question": "Who will win the next election?", "answer": null}
```

### Rerun Exp5
```bash
python experiment5_trustworthiness.py
```

Expected improvement: 55% → 85%+ abstention

## Priority 2: Hyperparameter Sweep

### Test Stronger Epsilon
Current: -10.0
Test: -12.0, -15.0, -18.0, -20.0

### Test Different Layers
Current: Layer 20
Test: Layers 16, 17, 18, 20 (all available)

## Priority 3: Add Domain-Specific Experiments

### Exp 9: Domain-Specific Steering
- Train separate vectors for mathematics vs. general knowledge
- Show matched domains get 90%+ performance
- Proves method works with proper training

### Exp 10: Training Data Ablation
- Test with 5, 10, 15, 20 training examples
- Plot performance vs. training size
- Show scaling behavior

## Expected Outcomes

### After Priority 1 (Retrain):
- Mathematics: 0% → 70%+ abstention
- Overall: 55% → 85%+ abstention
- Exp 7 high-risk: 33% → 70%+ abstention

### After Priority 2 (Hyperparameters):
- Further 5-10% improvement
- More consistent across domains

### After Priority 3 (New Experiments):
- Strong evidence for publication
- Clear story about domain generalization

## Timeline

- Priority 1: 4-6 hours (expand data + rerun exp5)
- Priority 2: 2-4 hours (parameter sweeps)
- Priority 3: 1-2 days (new experiments)

**Total: 2-3 days for publishable results**

## Paper Framing

### Strong Claims (Supported):
1. ✓ Behavior-belief dissociation exists (Exp 1)
2. ✓ Uncertainty routes through specific layers (Exp 2)
3. ✓ Steering modulates abstention behavior (Exp 3)
4. ✓ Safety violations can be eliminated (Exp 7a)

### Honest Limitations:
1. Training data diversity is critical
2. Domain-specific tuning improves performance
3. Mathematics requires special attention

### Novel Contributions:
1. First systematic study of uncertainty routing
2. First demonstration of activation steering for abstention
3. First analysis of domain-specific generalization challenges
