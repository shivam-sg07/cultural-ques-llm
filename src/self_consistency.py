from collections import Counter
from typing import List, Tuple
from .model import CulturalQAModel, extract_answer


def sample_multiple_answers(model: CulturalQAModel, prompt: str, num_samples: int = 5,
                            temperature: float = 0.7, max_new_tokens: int = 50) -> List[str]:
    
    answers = []
    
    for _ in range(num_samples):
        response = model.generate(
            prompt, max_new_tokens=max_new_tokens,
            temperature=temperature, do_sample=True, top_p=0.9
        )
        answer = extract_answer(response)
        answers.append(answer)
    
    return answers


def majority_vote(answers: List[str]) -> Tuple[str, float]:
    if not answers:
        return "A", 0.0
    
    counter = Counter(answers)
    most_common = counter.most_common(1)[0]
    winner = most_common[0]
    confidence = most_common[1] / len(answers)
    
    return winner, confidence


def self_consistent_answer(model: CulturalQAModel, prompt: str, num_samples: int = 5,
                           temperature: float = 0.7, max_new_tokens: int = 50,
                           return_details: bool = False) -> str | Tuple[str, float, List[str]]:
    
    answers = sample_multiple_answers(
        model=model, prompt=prompt, num_samples=num_samples,
        temperature=temperature, max_new_tokens=max_new_tokens
    )
    
    winner, confidence = majority_vote(answers)
    
    if return_details:
        return winner, confidence, answers
    return winner


def analyze_consistency(answers: List[str]) -> dict:
    counter = Counter(answers)
    total = len(answers)
    
    analysis = {
        "total_samples": total,
        "unique_answers": len(counter),
        "distribution": {k: v/total for k, v in counter.items()},
        "entropy": -sum((v/total) * (v/total).bit_length() for v in counter.values() if v > 0),
        "is_unanimous": len(counter) == 1,
        "winner": counter.most_common(1)[0][0],
        "confidence": counter.most_common(1)[0][1] / total
    }
    
    return analysis
