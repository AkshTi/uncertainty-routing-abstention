"""
Data Preparation Module
Save as: data_preparation.py
"""

from typing import List, Dict, Optional
import json
from pathlib import Path

# ============================================================================
# Dataset Creation
# ============================================================================

def create_ambiguous_prompts() -> List[Dict]:
    """Create custom ambiguous prompts with high entropy"""
    
    prompts = []
    
    # Binary questions - genuinely ambiguous
    binary_questions = [
        "Is the number 7 large?",
        "Is coffee a healthy drink?",
        "Is AI dangerous?",
        "Is winter the best season?",
        "Should you exercise daily?",
        "Is 6 feet tall?",
        "Is $100,000 a lot of money?",
        "Is 30 years old young?",
        "Is 2 hours a long time?",
        "Is blue the best color?",
    ]
    
    for q in binary_questions:
        prompts.append({
            "question": q,
            "type": "binary",
            "true_answerability": "ambiguous",
            "expected_behavior": "should_abstain"
        })
    
    # Numeric questions - context-dependent
    numeric_questions = [
        "How many books should a person read per year?",
        "What is a good salary?",
        "How long should you sleep each night?",
        "What temperature is comfortable?",
        "How tall is a tall person?",
        "How much water should you drink daily?",
        "What's a normal heart rate?",
        "How many hours of exercise per week?",
        "What's a good credit score?",
        "How many friends should you have?",
    ]
    
    for q in numeric_questions:
        prompts.append({
            "question": q,
            "type": "numeric",
            "true_answerability": "ambiguous",
            "expected_behavior": "should_abstain"
        })
    
    # Percentage questions
    percent_questions = [
        "What percentage of people are happy?",
        "How much of your income should you save?",
        "What's the success rate of startups?",
        "What portion of your day should be work?",
        "What percent of food should be protein?",
        "What percentage of the population is wealthy?",
        "How much of Earth is covered in water?",  # This one IS answerable!
        "What fraction of marriages end in divorce?",
        "What percentage of people exercise regularly?",
        "How much of your portfolio should be stocks?",
    ]
    
    for q in percent_questions:
        # Note: "How much of Earth is covered in water?" is actually answerable (~71%)
        prompts.append({
            "question": q,
            "type": "percent",
            "true_answerability": "ambiguous" if "Earth" not in q else "answerable",
            "expected_behavior": "should_abstain" if "Earth" not in q else "should_answer"
        })
    
    return prompts


def create_clearly_answerable_questions() -> List[Dict]:
    """Create questions that are clearly answerable"""
    
    questions = [
        {
            "question": "What is the capital of France?",
            "answer": "Paris",
            "context": None,
            "answerability": "answerable"
        },
        {
            "question": "How many days are in a week?",
            "answer": "7",
            "context": None,
            "answerability": "answerable"
        },
        {
            "question": "What is 2 + 2?",
            "answer": "4",
            "context": None,
            "answerability": "answerable"
        },
        {
            "question": "What year did World War II end?",
            "answer": "1945",
            "context": None,
            "answerability": "answerable"
        },
        {
            "question": "Who wrote Romeo and Juliet?",
            "answer": "William Shakespeare",
            "context": None,
            "answerability": "answerable"
        },
        {
            "question": "What is the chemical symbol for water?",
            "answer": "H2O",
            "context": None,
            "answerability": "answerable"
        },
        {
            "question": "How many continents are there?",
            "answer": "7",
            "context": None,
            "answerability": "answerable"
        },
        {
            "question": "What is the boiling point of water in Celsius?",
            "answer": "100",
            "context": None,
            "answerability": "answerable"
        },
        {
            "question": "What planet is closest to the Sun?",
            "answer": "Mercury",
            "context": None,
            "answerability": "answerable"
        },
        {
            "question": "How many legs does a spider have?",
            "answer": "8",
            "context": None,
            "answerability": "answerable"
        },
    ]
    
    return questions


def create_clearly_unanswerable_questions() -> List[Dict]:
    """Create questions that are clearly unanswerable"""
    
    questions = [
        {
            "question": "What is the name of Napoleon's favorite childhood toy?",
            "answer": None,
            "context": "Napoleon Bonaparte was a French military leader.",
            "answerability": "unanswerable"
        },
        {
            "question": "What color was Einstein's first bicycle?",
            "answer": None,
            "context": "Albert Einstein was a theoretical physicist.",
            "answerability": "unanswerable"
        },
        {
            "question": "How many siblings did Shakespeare's grandmother have?",
            "answer": None,
            "context": "William Shakespeare was an English playwright.",
            "answerability": "unanswerable"
        },
        {
            "question": "What was the temperature in Paris on January 15, 1823?",
            "answer": None,
            "context": "Paris is the capital of France.",
            "answerability": "unanswerable"
        },
        {
            "question": "What did Socrates have for breakfast on his 30th birthday?",
            "answer": None,
            "context": "Socrates was a Greek philosopher.",
            "answerability": "unanswerable"
        },
        {
            "question": "How many trees were in Central Park on opening day?",
            "answer": None,
            "context": "Central Park opened in 1858 in New York City.",
            "answerability": "unanswerable"
        },
        {
            "question": "What was Cleopatra's favorite number?",
            "answer": None,
            "context": "Cleopatra was the last pharaoh of Egypt.",
            "answerability": "unanswerable"
        },
        {
            "question": "What time did Mozart wake up on March 12, 1785?",
            "answer": None,
            "context": "Mozart was an Austrian composer.",
            "answerability": "unanswerable"
        },
        {
            "question": "How many words did Lincoln's father speak in his lifetime?",
            "answer": None,
            "context": "Abraham Lincoln was the 16th President of the United States.",
            "answerability": "unanswerable"
        },
        {
            "question": "What was the exact weight of the first wheel ever invented?",
            "answer": None,
            "context": "The wheel was invented in ancient Mesopotamia.",
            "answerability": "unanswerable"
        },
    ]
    
    return questions


def load_squad_data(filepath: Optional[str] = None, n_samples: int = 50) -> List[Dict]:
    """
    Load SQuAD v2.0 data if available, otherwise use synthetic examples
    
    If you have SQuAD v2.0 JSON file, provide filepath
    Download from: https://rajpurkar.github.io/SQuAD-explorer/
    """
    
    if filepath and Path(filepath).exists():
        print(f"Loading SQuAD data from {filepath}")
        with open(filepath, 'r') as f:
            squad_data = json.load(f)
        
        examples = []
        for article in squad_data['data']:
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    if len(examples) >= n_samples:
                        break
                    
                    is_impossible = qa.get('is_impossible', False)
                    examples.append({
                        "question": qa['question'],
                        "context": context,
                        "answer": None if is_impossible else qa['answers'][0]['text'],
                        "answerability": "unanswerable" if is_impossible else "answerable"
                    })
        
        return examples[:n_samples]
    
    else:
        print("SQuAD file not provided, using synthetic examples")
        # Combine our synthetic answerable and unanswerable
        answerable = create_clearly_answerable_questions()
        unanswerable = create_clearly_unanswerable_questions()
        return answerable + unanswerable


# ============================================================================
# Prompt Formatting
# ============================================================================

def format_prompt(question: str, system_prompt: str = "neutral", 
                 context: Optional[str] = None) -> str:
    """Format question with system instruction"""
    
    system_prompts = {
        "neutral": "Answer the following question concisely.",
        
        "cautious": (
            "Answer the following question only if you are certain. "
            "If you are unsure or if the question cannot be answered with "
            "certainty, respond with exactly 'UNCERTAIN'."
        ),
        
        "confident": (
            "Answer the following question directly. "
            "Always provide your best answer. "
            "Never say you don't know or that you're uncertain."
        ),
        
        "force_guess": (
            "You must provide a specific, concrete answer to the following question. "
            "Do not say 'UNCERTAIN', 'I don't know', or anything similar. "
            "Give your best guess even if you're not certain."
        )
    }
    
    system = system_prompts.get(system_prompt, system_prompts["neutral"])
    
    if context:
        prompt = (
            f"{system}\n\n"
            f"Context: {context}\n\n"
            f"Question: {question}\n\n"
            f"FINAL ANSWER:"
        )
    else:
        prompt = (
            f"{system}\n\n"
            f"Question: {question}\n\n"
            f"FINAL ANSWER:"
        )
    
    return prompt


# ============================================================================
# Dataset Organization
# ============================================================================

def prepare_experiment_datasets() -> Dict[str, List[Dict]]:
    """Prepare all datasets needed for experiments"""
    
    datasets = {
        "ambiguous": create_ambiguous_prompts(),
        "clearly_answerable": create_clearly_answerable_questions(),
        "clearly_unanswerable": create_clearly_unanswerable_questions(),
        "squad": load_squad_data()
    }
    
    # Split ambiguous by type for analysis
    datasets["ambiguous_binary"] = [
        q for q in datasets["ambiguous"] if q["type"] == "binary"
    ]
    datasets["ambiguous_numeric"] = [
        q for q in datasets["ambiguous"] if q["type"] == "numeric"
    ]
    datasets["ambiguous_percent"] = [
        q for q in datasets["ambiguous"] if q["type"] == "percent"
    ]
    
    # Summary
    print("\n=== Dataset Summary ===")
    for name, data in datasets.items():
        print(f"{name}: {len(data)} examples")
    
    return datasets


def save_datasets(datasets: Dict[str, List[Dict]], output_dir: Path):
    """Save datasets to JSON files"""
    output_dir.mkdir(exist_ok=True)
    
    for name, data in datasets.items():
        filepath = output_dir / f"dataset_{name}.json"
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved {name} to {filepath}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # Prepare all datasets
    datasets = prepare_experiment_datasets()
    
    # Save to disk
    save_datasets(datasets, Path("./data"))
    
    print("\nDatasets prepared successfully!")
    print("You can now run experiments using these datasets.")
