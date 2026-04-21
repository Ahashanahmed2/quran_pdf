#!/usr/bin/env python3
# ai/train_model.py
"""
Local Model Training Script
Trains on locally generated prompts and saves to HF Hub
"""

import os
import json
import argparse
from pathlib import Path
from typing import Optional
import logging

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DPOTrainer
from huggingface_hub import login

# ============ Configuration ============

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_CONFIGS = {
    "tiny": "Qwen/Qwen2.5-0.5B-Instruct",
    "small": "Qwen/Qwen2.5-1.5B-Instruct",
    "base": "Qwen/Qwen2.5-3B-Instruct",
    "large": "Qwen/Qwen2.5-7B-Instruct",
}

# ============ Training Functions ============

def setup_model_and_tokenizer(model_name: str, use_4bit: bool = True):
    """Setup model and tokenizer"""
    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def setup_lora(model):
    """Setup LoRA"""
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model


def train_sft(data_path: str, model_name: str, output_dir: str, hf_repo: Optional[str] = None):
    """SFT Training"""
    logger.info(f"Starting SFT Training with {model_name}")
    
    dataset = load_dataset("json", data_files=data_path, split="train")
    logger.info(f"Loaded {len(dataset)} examples")
    
    model, tokenizer = setup_model_and_tokenizer(model_name)
    model = setup_lora(model)
    
    def format_sft(example):
        return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}
    
    dataset = dataset.map(format_sft)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        push_to_hub=True if hf_repo else False,
        hub_model_id=hf_repo,
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_seq_length=2048,
    )
    
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    if hf_repo:
        trainer.push_to_hub()
        logger.info(f"Model pushed to HF: {hf_repo}")
    
    logger.info(f"SFT Training complete!")


def train_dpo(data_path: str, model_name: str, output_dir: str, hf_repo: Optional[str] = None):
    """DPO Training"""
    logger.info(f"Starting DPO Training with {model_name}")
    
    dataset = load_dataset("json", data_files=data_path, split="train")
    logger.info(f"Loaded {len(dataset)} examples")
    
    model, tokenizer = setup_model_and_tokenizer(model_name)
    model = setup_lora(model)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        learning_rate=5e-5,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        push_to_hub=True if hf_repo else False,
        hub_model_id=hf_repo,
    )
    
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        beta=0.1,
    )
    
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    if hf_repo:
        trainer.push_to_hub()
        logger.info(f"Model pushed to HF: {hf_repo}")
    
    logger.info(f"DPO Training complete!")


# ============ Main ============

def main():
    parser = argparse.ArgumentParser(description="Train models on local prompts")
    
    parser.add_argument("--paradigm", type=str, required=True,
                       choices=["sft", "dpo", "orpo", "simpo", "cpo"],
                       help="Training paradigm")
    parser.add_argument("--data", type=str, required=True,
                       help="Path to prompts JSONL file")
    parser.add_argument("--model", type=str, default="base",
                       choices=["tiny", "small", "base", "large"],
                       help="Base model")
    parser.add_argument("--output", type=str, default="./models/tafsir-model",
                       help="Output directory")
    parser.add_argument("--hf_repo", type=str, default=None,
                       help="HuggingFace repository")
    parser.add_argument("--hf_token", type=str, default=None,
                       help="HuggingFace API token")
    
    args = parser.parse_args()
    
    if args.hf_token:
        login(token=args.hf_token)
    
    model_name = MODEL_CONFIGS[args.model]
    os.makedirs(args.output, exist_ok=True)
    
    if args.paradigm == "sft":
        train_sft(args.data, model_name, args.output, args.hf_repo)
    elif args.paradigm in ["dpo", "orpo", "simpo", "cpo"]:
        train_dpo(args.data, model_name, args.output, args.hf_repo)


if __name__ == "__main__":
    main()