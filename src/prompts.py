from typing import List, Dict, Optional


SYSTEM_PROMPT_BASIC = """You are a helpful assistant with knowledge about different cultures around the world. 
Answer cultural questions accurately based on your knowledge."""

SYSTEM_PROMPT_COT = """You are a helpful assistant with knowledge about different cultures around the world.
When answering questions, think step by step about the cultural context before providing your answer."""


def build_zero_shot_prompt(question_prompt: str) -> str:
    prompt = f"""You are an expert on world cultures. Answer this cultural question by selecting the option that best reflects authentic cultural knowledge.

{question_prompt}

Provide ONLY the letter of your answer (A, B, C, or D). Do not include any explanation."""
    
    return prompt


def build_few_shot_prompt(question_prompt: str, examples: List[Dict], 
                          include_country_hint: bool = False) -> str:
    
    examples_text = ""
    for i, ex in enumerate(examples, 1):
        ex_prompt = ex['prompt']
        ex_answer = ex['answer']
        
        if include_country_hint:
            examples_text += f"Example {i} ({ex['country']}):\n"
        else:
            examples_text += f"Example {i}:\n"
        
        examples_text += f"{ex_prompt}\n{ex_answer}\n\n"
    
    prompt = f"""Answer cultural multiple choice questions. Respond with ONLY the letter (A, B, C, or D).

Here are some examples:

{examples_text}Now answer this question:

{question_prompt}"""
    
    return prompt


def build_chain_of_thought_prompt(question_prompt: str) -> str:
    prompt = f"""Answer the following cultural multiple choice question.

First, think about:
1. Which country/culture is this question about?
2. What cultural knowledge is relevant here?
3. Why might each option be correct or incorrect?

Then provide your final answer as a single letter (A, B, C, or D).

{question_prompt}

Let me think through this step by step:"""
    
    return prompt


def build_self_consistency_prompt(question_prompt: str) -> str:
    prompt = f"""Consider this cultural question carefully and select the best answer.

{question_prompt}

The answer is:"""
    
    return prompt


def format_for_llama3(user_prompt: str, system_prompt: str = SYSTEM_PROMPT_BASIC) -> str:
    formatted = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    return formatted


def format_with_tokenizer(tokenizer, user_prompt: str, 
                          system_prompt: str = SYSTEM_PROMPT_BASIC) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    return formatted


def build_prompt(question_prompt: str, strategy: str = "zero_shot",
                 examples: Optional[List[Dict]] = None, tokenizer=None,
                 use_tokenizer_template: bool = True) -> str:
    
    if strategy == "zero_shot":
        if examples is not None and "Iran" in question_prompt:
            user_prompt = build_few_shot_prompt(
                question_prompt, examples, include_country_hint=True
            )
        else:
            user_prompt = build_zero_shot_prompt(question_prompt)
        system_prompt = SYSTEM_PROMPT_BASIC
    
    elif strategy == "few_shot":
        if examples is None:
            raise ValueError("Examples required for few-shot prompting")
        user_prompt = build_few_shot_prompt(question_prompt, examples)
        system_prompt = SYSTEM_PROMPT_BASIC
    
    elif strategy == "chain_of_thought":
        user_prompt = build_chain_of_thought_prompt(question_prompt)
        system_prompt = SYSTEM_PROMPT_COT
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    if use_tokenizer_template and tokenizer is not None:
        return format_with_tokenizer(tokenizer, user_prompt, system_prompt)
    else:
        return format_for_llama3(user_prompt, system_prompt)


def build_mcq_prompt(country: str, question: str, options: dict):
    system_prompt = (
        "You are a culturally knowledgeable assistant.\n"
        "You answer multiple-choice questions using culturally specific knowledge, "
        "traditions, customs, and everyday common sense from the given country.\n"
        "Do not use knowledge from other cultures.\n"
        "Do not explain your reasoning.\n"
        "You must choose exactly one option."
    )

    user_prompt = (
        f"The following question originates from the cultural context of: {country}.\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n"
        f"A. {options['A']}\n"
        f"B. {options['B']}\n"
        f"C. {options['C']}\n"
        f"D. {options['D']}\n\n"
        f"Answer with exactly one letter: A, B, C, or D."
    )

    return system_prompt, user_prompt
