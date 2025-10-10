# sft_training.py
# Supervised fine-tuning for Countdown task format learning

import os
import torch
from datasets import load_dataset, Dataset as HFDataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import datetime

# ✅ Prompt template (same as used in data generation)
TEMPLATE = """Using the numbers {numbers}, create an equation that equals {target}. 
You can use basic arithmetic operations (+, -, *, /) and each number can only be used once.
Show your reasoning in <think> </think> tags. And return the final equation in <answer> </answer> tags. Keep your reasoning under 256 tokens.
For example, numbers = [1, 2, 3, 4] and target = 5, the answer is <answer>(1 + 2) * 3 - 4</answer>."""


def prepare_dataset(dataset_name: str, tokenizer, max_length: int = 512, split: str = "train"):
    """
    Prepare dataset by tokenizing and creating proper format.
    """
    # Load from HuggingFace
    dataset = load_dataset(dataset_name, split=split)
    
    def preprocess_function(examples):
        """Process a batch of examples with fixed-width padding."""
        prompts = []
        completions = []
        
        # Build prompts and get completions
        for i in range(len(examples["target"])):
            prompt = TEMPLATE.format(
                numbers=examples["nums"][i], 
                target=examples["target"][i]
            )
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
            prompt_formatted = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
            
            prompts.append(prompt_formatted)
            completions.append(examples["completion"][i].strip())
        
        # Combine prompt + completion
        full_texts = [p + c for p, c in zip(prompts, completions)]
        
        # Tokenize full texts with FIXED-WIDTH padding
        model_inputs = tokenizer(
            full_texts,
            max_length=max_length,
            truncation=True,
            padding="max_length",  # ✅ Pad to max_length (fixed-width)
        )
        
        # Create labels by masking the prompt part
        labels = []
        for i, (prompt, full_text) in enumerate(zip(prompts, full_texts)):
            # Tokenize just the prompt to find its length
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            prompt_len = len(prompt_ids)
            
            # Create label sequence: -100 for prompt, actual tokens for completion
            label = model_inputs["input_ids"][i][:]
            # Mask prompt tokens
            label[:prompt_len] = [-100] * prompt_len
            # Mask padding tokens
            for j in range(len(label)):
                if model_inputs["input_ids"][i][j] == tokenizer.pad_token_id:
                    label[j] = -100
            labels.append(label)
        
        model_inputs["labels"] = labels
        
        return model_inputs
    
    # Process dataset
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc=f"Tokenizing {split} dataset"
    )
    
    return tokenized_dataset


def train_sft(
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    dataset_name: str = "huyxdang/countdown-format-gpt4",
    output_dir: str = "./sft_checkpoint",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    max_length: int = 512,
):
    """Train the model on Countdown SFT dataset from HuggingFace."""
    
    print("=" * 60)
    print("🔹 Countdown SFT Training")
    print("=" * 60)
    
    print(f"\n🔹 Loading model and tokenizer: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
        use_cache=False
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"\n🔹 Loading and preparing datasets from: {dataset_name}")
    train_dataset = prepare_dataset(dataset_name, tokenizer, max_length, split="train")
    val_dataset = prepare_dataset(dataset_name, tokenizer, max_length, split="validation")
    
    print(f"  ✅ Train examples: {len(train_dataset)}")
    print(f"  ✅ Val examples: {len(val_dataset)}")
    
    # Show a sample
    print("\n📝 Sample training example (first item):")
    sample = train_dataset[0]
    print(f"  Input IDs length: {len(sample['input_ids'])}")
    print(f"  Labels length: {len(sample['labels'])}")
    print(f"  Number of -100s (masked): {sum(1 for x in sample['labels'] if x == -100)}")

    # Data collator for fixed-width tensors (no dynamic padding needed)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    timestamp = int(datetime.datetime.now(datetime.timezone.utc).timestamp())
    run_name = f"sft_countdown_{timestamp}"
    
    training_args = TrainingArguments(
        output_dir=f"{output_dir}/{run_name}",
        run_name=run_name,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=100,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        fp16=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="tensorboard",
        dataloader_pin_memory=True,
        dataloader_num_workers=0,  # Set to 0 to avoid multiprocessing issues
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    print("\n" + "=" * 60)
    print("🚀 Starting training...")
    print("=" * 60)
    trainer.train()

    final_output_dir = f"{output_dir}/final_{timestamp}"
    print(f"\n💾 Saving final model to {final_output_dir}...")
    trainer.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
    print(f"Model saved to: {final_output_dir}")
    print("\nNext steps:")
    print("1. Test the model with inference")
    print("2. Use this checkpoint for GRPO training")

    return final_output_dir


if __name__ == "__main__":
    final_dir = train_sft(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        dataset_name="huyxdang/countdown-format-gpt4",
        output_dir="./sft_checkpoint",
        num_epochs=3,
        batch_size=4,
        learning_rate=2e-5,
        max_length=512,
    )
    
    print(f"\n🎉 All done! Model saved at: {final_dir}")