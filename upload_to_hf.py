# upload_to_hf.py
# Upload your SFT dataset to HuggingFace Hub

import json
import os
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, login
from dotenv import load_dotenv

load_dotenv()

def load_jsonl(file_path: str) -> list:
    """Load data from JSONL file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def create_hf_dataset(
    train_file: str = "data/sft_train_gpt4.jsonl",
    val_file: str = "data/sft_val_gpt4.jsonl",
) -> DatasetDict:
    """
    Create HuggingFace DatasetDict from JSONL files.
    """
    print("Loading data files...")
    train_data = load_jsonl(train_file)
    val_data = load_jsonl(val_file)
    
    print(f"Train examples: {len(train_data)}")
    print(f"Val examples: {len(val_data)}")
    
    # Create datasets
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    # Combine into DatasetDict
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
    })
    
    return dataset_dict

def upload_to_hub(
    dataset: DatasetDict,
    repo_name: str,
    private: bool = False,
    token: str = None,
):
    """
    Upload dataset to HuggingFace Hub.
    
    Args:
        dataset: DatasetDict to upload
        repo_name: Name for the dataset repo (e.g., "username/countdown-sft")
        private: Whether to make the dataset private
        token: HuggingFace token (if not set in environment)
    """
    # Login to HuggingFace
    if token:
        login(token=token)
    else:
        # Will use token from huggingface-cli login or HF_TOKEN env var
        login()
    
    print(f"\nUploading dataset to: {repo_name}")
    print(f"Privacy: {'Private' if private else 'Public'}")
    
    # Push to hub
    dataset.push_to_hub(
        repo_name,
        private=private,
    )
    
    print(f"\n✅ Successfully uploaded to https://huggingface.co/datasets/{repo_name}")

def create_readme(
    repo_name: str,
    num_train: int,
    num_val: int,
    description: str = None,
) -> str:
    """Generate a README for the dataset."""
    
    default_description = """This dataset contains supervised fine-tuning (SFT) data for the Countdown math puzzle task.

## Task Description
Given a set of numbers and a target number, generate an equation using basic arithmetic operations (+, -, *, /) where each number is used exactly once.

## Format
Each example contains:
- `target`: The target number to reach
- `nums`: List of available numbers
- `solution`: The equation that solves the puzzle
- `completion`: Full formatted response with <think> reasoning and <answer> tags
- `result`: The evaluated result of the equation
- `is_exact`: Whether the solution exactly matches the target

## Usage
```python
from datasets import load_dataset

dataset = load_dataset("YOUR_USERNAME/countdown-sft")
train_data = dataset["train"]
val_data = dataset["validation"]
```
"""
    
    readme_content = f"""---
language:
- en
task_categories:
- text-generation
tags:
- math
- reasoning
- reinforcement-learning
- sft
pretty_name: Countdown SFT Dataset
size_categories:
- n<1K
---

# Countdown SFT Dataset

{description if description else default_description}

## Dataset Statistics
- **Train examples**: {num_train}
- **Validation examples**: {num_val}
- **Total**: {num_train + num_val}

## Example

```python
{{
  "target": 36,
  "nums": [79, 17, 60],
  "solution": "79 - (60 - 17)",
  "completion": "<think>\\nI need to reach 36 using 79, 17, and 60...\\n</think>\\n\\n<answer>79 - (60 - 17)</answer>",
  "result": 36.0,
  "is_exact": true
}}
```

## Citation
If you use this dataset, please cite the original Countdown task dataset:
```
@dataset{{countdown_tasks,
  author = {{Justin Phan}},
  title = {{Countdown Tasks}},
  year = {{2024}},
  publisher = {{HuggingFace}},
  url = {{https://huggingface.co/datasets/justinphan3110/Countdown-Tasks-3to4}}
}}
```
"""
    
    return readme_content

def save_readme_locally(readme_content: str, output_path: str = "README.md"):
    """Save README to file."""
    with open(output_path, 'w') as f:
        f.write(readme_content)
    print(f"README saved to {output_path}")

def main():
    """Main upload workflow."""
    import sys
    
    print("="*60)
    print("HuggingFace Dataset Upload Tool")
    print("="*60)
    
    # Step 1: Get user input
    print("\n📝 Dataset Information:")
    repo_name = input("Enter dataset name (e.g., 'username/countdown-sft'): ").strip()
    
    if not repo_name or '/' not in repo_name:
        print("❌ Error: Please provide a valid repo name in format 'username/dataset-name'")
        sys.exit(1)
    
    private = input("Make dataset private? (y/n, default: n): ").strip().lower() == 'y'
    
    # Step 2: Check for HF token
    print("\n🔑 Authentication:")
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    
    if not token:
        print("HuggingFace token not found in environment.")
        print("Options:")
        print("  1. Run: huggingface-cli login")
        print("  2. Set HF_TOKEN in your .env file")
        print("  3. Enter token now (not recommended for security)")
        
        choice = input("\nEnter token now? (y/n): ").strip().lower()
        if choice == 'y':
            token = input("Paste your HF token: ").strip()
        else:
            print("Please authenticate first. Exiting.")
            sys.exit(1)
    
    # Step 3: Load data
    print("\n📂 Loading datasets...")
    train_file = input("Train file path (default: sft_train_gpt4.jsonl): ").strip() or "sft_train_gpt4.jsonl"
    val_file = input("Val file path (default: sft_val_gpt4.jsonl): ").strip() or "sft_val_gpt4.jsonl"
    
    if not os.path.exists(train_file):
        print(f"❌ Error: {train_file} not found!")
        sys.exit(1)
    if not os.path.exists(val_file):
        print(f"❌ Error: {val_file} not found!")
        sys.exit(1)
    
    dataset = create_hf_dataset(train_file, val_file)
    
    # Step 4: Create README
    print("\n📄 Creating README...")
    readme = create_readme(
        repo_name=repo_name,
        num_train=len(dataset["train"]),
        num_val=len(dataset["validation"]),
    )
    save_readme_locally(readme)
    
    # Step 5: Confirm and upload
    print("\n" + "="*60)
    print("Ready to upload:")
    print(f"  Repository: {repo_name}")
    print(f"  Privacy: {'Private' if private else 'Public'}")
    print(f"  Train examples: {len(dataset['train'])}")
    print(f"  Val examples: {len(dataset['validation'])}")
    print("="*60)
    
    confirm = input("\nProceed with upload? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Upload cancelled.")
        sys.exit(0)
    
    # Step 6: Upload
    print("\n🚀 Uploading to HuggingFace Hub...")
    try:
        upload_to_hub(
            dataset=dataset,
            repo_name=repo_name,
            private=private,
            token=token,
        )
        
        print("\n✅ Upload complete!")
        print(f"\n🔗 View your dataset at:")
        print(f"   https://huggingface.co/datasets/{repo_name}")
        
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        print("\nTroubleshooting:")
        print("  - Check your token has write permissions")
        print("  - Verify the repo name is available")
        print("  - Try: huggingface-cli login")
        sys.exit(1)

# Alternative: Quick upload without prompts
def quick_upload(
    train_file: str = "sft_train_gpt4.jsonl",
    val_file: str = "sft_val_gpt4.jsonl",
    repo_name: str = None,
    private: bool = False,
):
    """Quick upload without interactive prompts."""
    if not repo_name:
        print("Error: repo_name is required")
        return
    
    print(f"Loading datasets from {train_file} and {val_file}...")
    dataset = create_hf_dataset(train_file, val_file)
    
    print(f"Uploading to {repo_name}...")
    upload_to_hub(dataset, repo_name, private)
    
    print("Done!")

if __name__ == "__main__":
    # Interactive mode
    main()
    
    # Or use quick_upload for scripting:
    # quick_upload(
    #     train_file="sft_train_gpt4.jsonl",
    #     val_file="sft_val_gpt4.jsonl",
    #     repo_name="your-username/countdown-sft",
    #     private=False,
    # )