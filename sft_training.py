# sft_training.py
# Supervised fine-tuning for Countdown task format learning

import os
import json
import torch
from torch.utils.data import Dataset
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

class CountdownSFTDataset(Dataset):
    """
    Custom dataset class for Countdown SFT.
    Uses only 'nums', 'target', and 'completion' fields from your JSONL dataset.
    """
    def __init__(self, jsonl_file: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load JSONL file
        self.examples = []
        with open(jsonl_file, 'r') as f:
            for line in f:
                ex = json.loads(line)
                # ✅ Only keep valid entries that have completion field
                if "completion" in ex and "target" in ex and "nums" in ex:
                    self.examples.append(ex)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # ✅ Build input prompt using target + nums
        prompt = TEMPLATE.format(
            numbers=example["nums"],
            target=example["target"]
        )

        # Create chat-style format for Qwen tokenizer
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        prompt_formatted = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )

        # ✅ Combine input + desired output
        completion = example["completion"].strip()
        full_text = prompt_formatted + completion

        # Tokenize combined text
        tokenized = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None
        )

        # ✅ Mask out the prompt portion (so model only learns to predict completion)
        prompt_length = len(self.tokenizer(prompt_formatted, add_special_tokens=False)["input_ids"])
        labels = tokenized["input_ids"].copy()
        labels[:prompt_length] = [-100] * prompt_length  # mask prompt

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels
        }

def train_sft(
    model_id: str = "Qwen/Qwen3-1.7B",
    train_file: str = "sft_train_gpt4.jsonl",
    val_file: str = "sft_val_gpt4.jsonl",
    output_dir: str = "./sft_checkpoint",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    max_length: int = 512,
):
    """Train the model on Countdown SFT dataset."""
    
    print("🔹 Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=False
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("🔹 Loading datasets...")
    train_dataset = CountdownSFTDataset(train_file, tokenizer, max_length)
    val_dataset = CountdownSFTDataset(val_file, tokenizer, max_length)
    
    print(f"  Train examples: {len(train_dataset)}")
    print(f"  Val examples: {len(val_dataset)}")

    # ✅ Data collator handles padding dynamically
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )

    timestamp = int(datetime.datetime.now(datetime.timezone.utc).timestamp())
    training_args = TrainingArguments(
        output_dir=f"{output_dir}_{timestamp}",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=10,
        eval_steps=100,
        save_steps=100,
        save_total_limit=2,
        bf16=True,
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="tensorboard",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    print("🚀 Starting training...")
    trainer.train()

    final_output_dir = f"{output_dir}_final_{timestamp}"
    print(f"💾 Saving final model to {final_output_dir}...")
    model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)

    return final_output_dir
