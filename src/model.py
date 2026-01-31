import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, List
import re


class CulturalQAModel:
    
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3-8B",
                 device: str = "cuda", dtype: str = "float16", token: Optional[str] = None):
        
        self.model_name = model_name
        self.device = device
        
        if token is None:
            token = os.environ.get("HF_TOKEN", None)
        
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32
        }
        torch_dtype = dtype_map.get(dtype, torch.float16)
        
        print(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Loading model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto" if device == "cuda" else None,
            token=token
        )
        
        if device == "cuda" and self.model.device.type != "cuda":
            self.model = self.model.to(device)
        
        self.model.eval()
        print(f"Model loaded on {self.model.device}")
    
    def generate(self, prompt: str, max_new_tokens: int = 50, temperature: float = 1.0,
                 do_sample: bool = False, top_p: float = 1.0) -> str:
        
        inputs = self.tokenizer(
            prompt, return_tensors="pt", padding=True,
            truncation=True, max_length=2048
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                do_sample=do_sample,
                top_p=top_p if do_sample else 1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        generated = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )
        
        return generated.strip()
    
    def generate_batch(self, prompts: List[str], max_new_tokens: int = 50,
                       temperature: float = 1.0, do_sample: bool = False) -> List[str]:
        
        self.tokenizer.padding_side = "left"
        
        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=2048
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        generated = []
        for i, output in enumerate(outputs):
            text = self.tokenizer.decode(
                output[inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True
            )
            generated.append(text.strip())
        
        return generated


def extract_answer(text: str):
    if not text:
        return "A"
    
    # XML format for chain of thought
    match = re.search(r"<answer>\s*([ABCD])\s*</answer>", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # "Final Answer: X" pattern
    match = re.search(r"Final Answer:\s*([ABCD])", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Standalone letter on its own line
    match = re.search(r"^([ABCD])\.?\s*$", text.strip(), re.MULTILINE)
    if match:
        return match.group(1).upper()
    
    # Last mentioned letter
    match = re.search(r"\b([ABCD])\b(?!.*\b[ABCD]\b)", text)
    if match:
        return match.group(1).upper()
    
    # First mentioned letter
    match = re.search(r"\b([ABCD])\b", text)
    if match:
        return match.group(1).upper()
    
    # JSON format
    match = re.search(r'"answer[^"]*"\s*:\s*"([ABCD])"', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    print(f"Warning: Could not extract answer from: {text[:100]}")
    return "A"
