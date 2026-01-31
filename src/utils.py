import yaml
import os
from typing import Dict, Any


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], config_path: str = "config.yaml"):
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def ensure_dirs(paths: list):
    for path in paths:
        os.makedirs(os.path.dirname(path) if '.' in os.path.basename(path) else path, exist_ok=True)


def print_config(config: Dict[str, Any], indent: int = 0):
    for key, value in config.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key}:")
            print_config(value, indent + 1)
        else:
            print("  " * indent + f"{key}: {value}")


class ProgressTracker:
    
    def __init__(self, total: int, print_every: int = 100):
        self.total = total
        self.current = 0
        self.print_every = print_every
        self.correct = 0
    
    def update(self, n: int = 1, correct: bool = None):
        self.current += n
        if correct is not None:
            self.correct += int(correct)
        
        if self.current % self.print_every == 0 or self.current == self.total:
            self._print_progress()
    
    def _print_progress(self):
        pct = 100 * self.current / self.total
        msg = f"Progress: {self.current}/{self.total} ({pct:.1f}%)"
        if self.correct > 0:
            acc = 100 * self.correct / self.current
            msg += f" | Accuracy: {acc:.1f}%"
        print(msg)
    
    def get_accuracy(self) -> float:
        if self.current == 0:
            return 0.0
        return self.correct / self.current


def calculate_accuracy(predictions: list, ground_truth: list) -> float:
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length")
    
    correct = sum(p == g for p, g in zip(predictions, ground_truth))
    return correct / len(predictions)


def analyze_errors(predictions: list, ground_truth: list, 
                   questions: list, countries: list) -> Dict[str, Any]:
    
    errors = []
    correct_by_country = {}
    total_by_country = {}
    
    for pred, gt, q, c in zip(predictions, ground_truth, questions, countries):
        if c not in total_by_country:
            total_by_country[c] = 0
            correct_by_country[c] = 0
        total_by_country[c] += 1
        
        if pred == gt:
            correct_by_country[c] += 1
        else:
            errors.append({
                'question': q[:100] + '...' if len(q) > 100 else q,
                'predicted': pred,
                'actual': gt,
                'country': c
            })
    
    accuracy_by_country = {
        c: correct_by_country[c] / total_by_country[c]
        for c in total_by_country
    }
    
    return {
        'total_errors': len(errors),
        'accuracy_by_country': accuracy_by_country,
        'sample_errors': errors[:10]
    }
