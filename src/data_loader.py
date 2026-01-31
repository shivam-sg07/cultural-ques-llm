import pandas as pd
from typing import List, Dict, Optional
import json


def load_mcq_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    return df


def parse_choices(choices_str: str) -> Dict[str, str]:
    try:
        return json.loads(choices_str)
    except json.JSONDecodeError:
        return {}


def get_few_shot_examples(
    train_df: pd.DataFrame,
    target_country: Optional[str] = None,
    num_examples: int = 3,
    random_state: int = 42
) -> List[Dict]:
    
    if target_country:
        filtered_df = train_df[train_df['country'] == target_country]
        if len(filtered_df) < num_examples:
            filtered_df = train_df
    else:
        filtered_df = train_df
    
    sampled = filtered_df.sample(n=num_examples, random_state=random_state)
    
    examples = []
    for _, row in sampled.iterrows():
        examples.append({
            'prompt': row['prompt'],
            'answer': row['answer_idx'],
            'country': row['country']
        })
    
    return examples


def get_stratified_few_shot_examples(
    train_df: pd.DataFrame,
    num_per_country: int = 1,
    random_state: int = 42
) -> List[Dict]:
    
    examples = []
    countries = train_df['country'].unique()
    
    for country in countries:
        country_df = train_df[train_df['country'] == country]
        n_samples = min(num_per_country, len(country_df))
        if n_samples == 0:
            continue
        sampled = country_df.sample(n=n_samples, random_state=random_state)
        
        for _, row in sampled.iterrows():
            examples.append({
                'prompt': row['prompt'],
                'answer': row['answer_idx'],
                'country': row['country']
            })
    
    return examples


def prepare_submission(mcq_ids: List[str], predictions: List[str], output_path: str) -> pd.DataFrame:
    submission_data = {
        'MCQID': mcq_ids,
        'A': [pred == 'A' for pred in predictions],
        'B': [pred == 'B' for pred in predictions],
        'C': [pred == 'C' for pred in predictions],
        'D': [pred == 'D' for pred in predictions],
    }
    
    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv(output_path, sep='\t', index=False)
    print(f"Submission saved to {output_path}")
    
    return submission_df
