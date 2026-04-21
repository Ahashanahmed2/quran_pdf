#!/usr/bin/env python3
# train_model.py
"""
Model Training Script - Trains on Generated Prompts
Saves fine-tuned models to HuggingFace Hub
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Optional
import logging

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    HfArgumentParser,
)
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DPOTrainer, PPOTrainer, PPOConfig
from huggingface_hub import HfApi, create_repo

# ============ Configuration ============

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============ Model Configs ============

MODEL_CONFIGS = {
    "tiny": "Qwen/Qwen2.5-0.5B-Instruct",
    "small": "Qwen/Qwen2.5-1.5B-Instruct",
    "base": "Qwen/Qwen2.5-3B-Instruct",
    "large": "Qwen/Qwen2.5-7B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama": "meta-llama/Llama-3.2-3B-Instruct",
}

# ============ Training Functions ============

def setup_model_and_tokenizer(model_name: str, use_4bit: bool = True):
    """Setup model and tokenizer with quantization"""
    
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
    """Setup LoRA for efficient fine-tuning"""
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
    logger.info(f"Data: {data_path}")
    
    # Load dataset
    dataset = load_dataset("json", data_files=data_path, split="train")
    logger.info(f"Loaded {len(dataset)} examples")
    
    # Setup model
    model, tokenizer = setup_model_and_tokenizer(model_name)
    model = setup_lora(model)
    
    # Format for SFT
    def format_sft(example):
        return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}
    
    dataset = dataset.map(format_sft)
    
    # Training arguments
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
        hub_strategy="every_save",
    )
    
    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_seq_length=2048,
    )
    
    # Train
    trainer.train()
    
    # Save
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    if hf_repo:
        trainer.push_to_hub()
        logger.info(f"Model pushed to HF: {hf_repo}")
    
    logger.info(f"SFT Training complete! Model saved to {output_dir}")


def train_dpo(data_path: str, model_name: str, output_dir: str, hf_repo: Optional[str] = None):
    """DPO Training"""
    logger.info(f"Starting DPO Training with {model_name}")
    logger.info(f"Data: {data_path}")
    
    # Load dataset
    dataset = load_dataset("json", data_files=data_path, split="train")
    logger.info(f"Loaded {len(dataset)} examples")
    
    # Setup model
    model, tokenizer = setup_model_and_tokenizer(model_name)
    model = setup_lora(model)
    
    # Training arguments
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
    
    # DPO Trainer
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        beta=0.1,
    )
    
    # Train
    trainer.train()
    
    # Save
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    if hf_repo:
        trainer.push_to_hub()
        logger.info(f"Model pushed to HF: {hf_repo}")
    
    logger.info(f"DPO Training complete! Model saved to {output_dir}")


def train_orpo(data_path: str, model_name: str, output_dir: str, hf_repo: Optional[str] = None):
    """ORPO Training - Similar to DPO"""
    # ORPO uses same data format as DPO
    train_dpo(data_path, model_name, output_dir, hf_repo)


def train_kto(data_path: str, model_name: str, output_dir: str, hf_repo: Optional[str] = None):
    """KTO Training"""
    logger.info(f"Starting KTO Training with {model_name}")
    
    dataset = load_dataset("json", data_files=data_path, split="train")
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
    )
    
    from trl import KTOTrainer
    
    trainer = KTOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        beta=0.1,
    )
    
    trainer.train()
    trainer.save_model(output_dir)
    
    if hf_repo:
        trainer.push_to_hub()
    
    logger.info(f"KTO Training complete!")


def train_rlhf(data_path: str, model_name: str, output_dir: str, hf_repo: Optional[str] = None):
    """RLHF Training (Reward Modeling)"""
    logger.info(f"Starting Reward Model Training with {model_name}")
    
    dataset = load_dataset("json", data_files=data_path, split="train")
    
    model, tokenizer = setup_model_and_tokenizer(model_name)
    
    # Add reward head
    from transformers import AutoModelForSequenceClassification
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        device_map="auto",
    )
    reward_model = setup_lora(reward_model)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        learning_rate=1e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )
    
    # Format for reward modeling
    def format_reward(example):
        return {
            "input_ids": tokenizer(example["prompt"] + "\n" + example["response"], truncation=True, max_length=1024)["input_ids"],
            "labels": torch.tensor(example["rating"], dtype=torch.float),
        }
    
    dataset = dataset.map(format_reward)
    
    trainer = Trainer(
        model=reward_model,
        args=training_args,
        train_dataset=dataset,
    )
    
    trainer.train()
    trainer.save_model(output_dir)
    
    if hf_repo:
        trainer.push_to_hub()
    
    logger.info(f"Reward Model Training complete!")


def train_curriculum(data_dir: str, model_name: str, output_dir: str, hf_repo: Optional[str] = None):
    """Curriculum Learning - Train on stages sequentially"""
    logger.info(f"Starting Curriculum Learning with {model_name}")
    
    model, tokenizer = setup_model_and_tokenizer(model_name)
    model = setup_lora(model)
    
    for stage in range(1, 6):
        stage_file = Path(data_dir) / f"curriculum_stage_{stage}.jsonl"
        if not stage_file.exists():
            continue
        
        logger.info(f"Training stage {stage}...")
        
        dataset = load_dataset("json", data_files=str(stage_file), split="train")
        
        def format_sft(example):
            return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}
        
        dataset = dataset.map(format_sft)
        
        training_args = TrainingArguments(
            output_dir=f"{output_dir}/stage_{stage}",
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            num_train_epochs=1,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            save_strategy="epoch",
            report_to="none",
        )
        
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
            max_seq_length=2048,
        )
        
        trainer.train()
    
    # Save final model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    if hf_repo:
        model.push_to_hub(hf_repo)
        tokenizer.push_to_hub(hf_repo)
    
    logger.info(f"Curriculum Learning complete!")


# ============ Main ============

def main():
    parser = argparse.ArgumentParser(description="Train models on generated prompts")
    
    parser.add_argument("--paradigm", type=str, required=True,
                       choices=["sft", "dpo", "ppo", "rlhf", "kto", "orpo", "simpo", "cpo", "agentic", "curriculum"],
                       help="Training paradigm")
    parser.add_argument("--data", type=str, required=True,
                       help="Path to data file or directory")
    parser.add_argument("--model", type=str, default="base",
                       choices=["tiny", "small", "base", "large", "mistral", "llama"],
                       help="Base model to use")
    parser.add_argument("--output", type=str, default="./models/tafsir-model",
                       help="Output directory for model")
    parser.add_argument("--hf_repo", type=str, default=None,
                       help="HuggingFace repository (e.g., username/model-name)")
    parser.add_argument("--hf_token", type=str, default=None,
                       help="HuggingFace API token")
    
    args = parser.parse_args()
    
    # Set HF token
    if args.hf_token:
        from huggingface_hub import login
        login(token=args.hf_token)
    
    model_name = MODEL_CONFIGS[args.model]
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Route to appropriate training function
    if args.paradigm == "sft":
        train_sft(args.data, model_name, args.output, args.hf_repo)
    elif args.paradigm == "dpo":
        train_dpo(args.data, model_name, args.output, args.hf_repo)
    elif args.paradigm == "orpo":
        train_orpo(args.data, model_name, args.output, args.hf_repo)
    elif args.paradigm == "simpo":
        train_orpo(args.data, model_name, args.output, args.hf_repo)
    elif args.paradigm == "cpo":
        train_dpo(args.data, model_name, args.output, args.hf_repo)
    elif args.paradigm == "kto":
        train_kto(args.data, model_name, args.output, args.hf_repo)
    elif args.paradigm == "rlhf":
        train_rlhf(args.data, model_name, args.output, args.hf_repo)
    elif args.paradigm == "curriculum":
        train_curriculum(args.data, model_name, args.output, args.hf_repo)
    else:
        logger.error(f"Unknown paradigm: {args.paradigm}")


if __name__ == "__main__":
    main()