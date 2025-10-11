"""
Generate "easy" dataset containing only 3-number countdown tasks.
This filtered dataset is used for curriculum learning, starting with simpler tasks.
"""

import json
import os
from datasets import load_dataset

def main():
    print("=" * 60)
    print("Generating Easy Dataset (3-number examples only)")
    print("=" * 60)
    
    # Load original dataset
    print("\n📥 Loading original dataset from HuggingFace...")
    train_data = load_dataset("justinphan3110/Countdown-Tasks-3to4", split="train")
    test_data = load_dataset("justinphan3110/Countdown-Tasks-3to4", split="test")
    
    print(f"Original train size: {len(train_data):,}")
    print(f"Original test size: {len(test_data):,}")
    
    # Filter to only 3-number examples
    print("\n🔍 Filtering to 3-number examples...")
    train_easy = [ex for ex in train_data if len(ex["nums"]) == 3]
    test_easy = [ex for ex in test_data if len(ex["nums"]) == 3]
    
    print(f"Filtered train size: {len(train_easy):,} ({len(train_easy)/len(train_data)*100:.1f}%)")
    print(f"Filtered test size: {len(test_easy):,} ({len(test_easy)/len(test_data)*100:.1f}%)")
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Save as JSONL files
    print("\n💾 Saving filtered datasets...")
    
    train_file = "data/easy_train.jsonl"
    with open(train_file, "w") as f:
        for ex in train_easy:
            json.dump({"target": ex["target"], "nums": ex["nums"]}, f)
            f.write("\n")
    print(f"✅ Saved train split to: {train_file}")
    
    test_file = "data/easy_test.jsonl"
    with open(test_file, "w") as f:
        for ex in test_easy:
            json.dump({"target": ex["target"], "nums": ex["nums"]}, f)
            f.write("\n")
    print(f"✅ Saved test split to: {test_file}")
    
    # Show some examples
    print("\n📊 Sample examples from easy dataset:")
    print("-" * 60)
    for i, ex in enumerate(train_easy[:5]):
        print(f"Example {i+1}: target={ex['target']}, nums={ex['nums']}")
    
    print("\n" + "=" * 60)
    print("✨ Easy dataset generation complete!")
    print("=" * 60)
    print("\n📖 Usage in starter_kl.py:")
    print('train_data = load_dataset("json", data_files="data/easy_train.jsonl", split="train")')
    print('test_data = load_dataset("json", data_files="data/easy_test.jsonl", split="train")')
    print()

if __name__ == "__main__":
    main()


