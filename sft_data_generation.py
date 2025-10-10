# sft_training.py
# Fine-tune model on format learning using GPT-4o-mini generated data

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import wandb
from typing import Dict, List

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_NAME = "huyxdang/countdown-format-gpt4"
OUTPUT_DIR = "./sft_checkpoint"
WANDB_PROJECT = "countdown-sft"

# LoRA Configuration
LORA_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Training Configuration
TRAINING_ARGS = {
    "output_dir": OUTPUT_DIR,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine",
    "logging_steps": 10,
    "eval_strategy": "steps",
    "eval_steps": 50,
    "save_strategy": "steps",
    "save_steps": 100,
    "save_total_limit": 3,
    "fp16": torch.cuda.is_available(),
    "gradient_checkpointing": True,
    "report_to": "wandb",
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
}


def format_prompt(target: int, nums: List[int]) -> str:
    """Format the input prompt for the model."""
    return f"""You are solving a math puzzle called "Countdown". 

Task: Using the numbers {nums}, create an equation that equals {target}.
Rules:
- You can use basic arithmetic operations (+, -, *, /)
- Each number must be used exactly once
- You can use parentheses

Format your response EXACTLY like this:
<think>
[Your step-by-step reasoning here - keep it concise, under 100 words]
</think>

<answer>[equation]</answer>

Now solve this:
Numbers: {nums}, Target: {target}"""


def preprocess_function(examples: Dict, tokenizer) -> Dict:
    """
    Preprocess dataset examples for SFT training.
    
    The dataset has: target, nums, solution, completion, result, is_exact
    We want to train on: prompt + completion
    """
    prompts = []
    completions = []
    
    for i in range(len(examples["target"])):
        target = examples["target"][i]
        nums = examples["nums"][i]
        completion = examples["completion"][i]
        
        # Format prompt
        prompt = format_prompt(target, nums)
        
        prompts.append(prompt)
        completions.append(completion)
    
    # Create full training texts (prompt + completion)
    full_texts = [f"{p}\n\n{c}" for p, c in zip(prompts, completions)]
    
    # Tokenize
    tokenized = tokenizer(
        full_texts,
        truncation=True,
        max_length=512,
        padding="max_length",
    )
    
    # Set labels (for causal LM, labels = input_ids)
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized


def main():
    print("="*60)
    print("SFT Training with GPT-4o-mini Generated Data")
    print("="*60)
    
    # Initialize wandb
    wandb.init(project=WANDB_PROJECT, name="sft-format-learning")
    
    # Load tokenizer
    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model
    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    
    # Prepare for LoRA training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LORA_CONFIG)
    
    print("\nModel architecture:")
    model.print_trainable_parameters()
    
    # Load dataset from HuggingFace
    print(f"\nLoading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME)
    
    print(f"Train examples: {len(dataset['train'])}")
    print(f"Validation examples: {len(dataset['validation'])}")
    
    # Show sample
    print("\nSample training example:")
    sample = dataset["train"][0]
    print(f"Target: {sample['target']}")
    print(f"Numbers: {sample['nums']}")
    print(f"Solution: {sample['solution']}")
    print(f"Is exact: {sample['is_exact']}")
    print(f"\nCompletion preview:")
    print(sample['completion'][:200] + "...")
    
    # Preprocess datasets
    print("\nPreprocessing datasets...")
    tokenized_train = dataset["train"].map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    
    tokenized_val = dataset["validation"].map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=dataset["validation"].column_names,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )
    
    # Training arguments
    training_args = TrainingArguments(**TRAINING_ARGS)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
    )
    
    # Train
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60 + "\n")
    
    trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nModel saved to: {OUTPUT_DIR}")
    print("\nNext steps:")
    print("1. Test the SFT model with: python test_sft_model.py")
    print("2. Use this checkpoint as the base for GRPO training")
    print("3. Update your GRPO config to load from this checkpoint")
    
    wandb.finish()


if __name__ == "__main__":
    main()