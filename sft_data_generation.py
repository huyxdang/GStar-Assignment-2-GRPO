# sft_data_generation.py (with GPT-4o-mini)
# Generate SFT data using GPT-4o-mini API for format learning

import os
import json
from typing import List, Dict, Optional
from datasets import load_dataset
from tqdm import tqdm
import time
import re
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

GENERATION_PROMPT = """You are solving a math puzzle called "Countdown". 

Task: Using the numbers {numbers}, create an equation that equals {target}.
Rules:
- You can use basic arithmetic operations (+, -, *, /)
- Each number must be used exactly once
- You can use parentheses

Format your response EXACTLY like this:
<think>
[Your step-by-step reasoning here - keep it concise, under 100 words]
</think>

<answer>[equation]</answer>

Example:
Numbers: [1, 2, 3, 4], Target: 5
<think>
I need to make 5 from 1, 2, 3, 4. Let me try: (1 + 2) = 3, then 3 * 3 = 9, then 9 - 4 = 5. So the equation is (1 + 2) * 3 - 4.
</think>

<answer>(1 + 2) * 3 - 4</answer>

Now solve this:
Numbers: {numbers}, Target: {target}"""

def _extract_answer(text: str) -> Optional[str]:
    """Extract answer from <answer> tags."""
    matches = list(re.finditer(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL | re.IGNORECASE))
    return matches[-1].group(1).strip() if matches else None

def _validate_numbers(equation_str: str, available_numbers: List[int]) -> bool:
    """Check if equation uses correct numbers."""
    try:
        found_numbers = re.findall(r"\d+", equation_str)
        str_found_numbers = list(map(int, found_numbers))
        return sorted(str_found_numbers) == sorted(available_numbers)
    except:
        return False

def _evaluate_equation(equation_str: str) -> Optional[float]:
    """Safely evaluate equation."""
    try:
        if not re.fullmatch(r"[0-9+\-*/()\s]+", equation_str):
            return None
        result = eval(equation_str, {"__builtins__": None}, {})
        return float(result)
    except:
        return None

def generate_solution_with_gpt4(
    target: int, 
    nums: List[int],
    model: str = "gpt-4o-mini",  
    max_retries: int = 3
) -> Optional[Dict]:
    """
    Generate a solution using GPT-4 API.
    
    Args:
        target: Target number
        nums: List of available numbers
        model: OpenAI model to use (gpt-4o-mini is recommended for cost)
        max_retries: Number of retries if generation fails
    
    Returns:
        Dict with solution info or None if failed
    """
    prompt = GENERATION_PROMPT.format(numbers=nums, target=target)
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful math assistant that solves Countdown puzzles."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=300,
            )
            
            completion = response.choices[0].message.content
            
            # Validate the response
            equation = _extract_answer(completion)
            if equation is None:
                if attempt < max_retries - 1:
                    continue  # Retry
                return None
            
            # Check if numbers are valid
            if not _validate_numbers(equation, nums):
                if attempt < max_retries - 1:
                    continue
                return None
            
            # Check if equation evaluates correctly
            result = _evaluate_equation(equation)
            if result is None:
                if attempt < max_retries - 1:
                    continue
                return None
            
            # For SFT, we accept close answers (within 5% or exact)
            # This gives more training data and model will learn the format
            is_correct = abs(result - target) < 1e-6
            is_close = abs(result - target) < max(1, abs(target) * 0.05)
            
            if is_correct or is_close:
                return {
                    "target": target,
                    "nums": nums,
                    "solution": equation,
                    "completion": completion,
                    "result": result,
                    "is_exact": is_correct
                }
            
            # If not close enough, retry
            if attempt < max_retries - 1:
                continue
            return None
            
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)  # Brief pause before retry
                continue
            return None
    
    return None

def create_sft_dataset_with_gpt4(
    dataset_name: str = "justinphan3110/Countdown-Tasks-3to4",
    split: str = "train",
    output_file: str = "sft_data_gpt4.jsonl",
    num_examples: int = 500,
    model: str = "gpt-4o-mini",
    batch_delay: float = 0.1,  # Delay between API calls to avoid rate limits
):
    """
    Create SFT dataset using GPT-4 API.
    
    Args:
        dataset_name: HuggingFace dataset name
        split: Dataset split to use
        output_file: Output JSONL file
        num_examples: Number of examples to generate
        model: OpenAI model (gpt-4o-mini recommended for cost/performance)
        batch_delay: Seconds to wait between API calls
    """
    print(f"Loading dataset {dataset_name}...")
    data = load_dataset(dataset_name, split=split)
    
    # Shuffle and take num_examples
    import random
    indices = random.sample(range(len(data)), min(num_examples, len(data)))
    
    sft_examples = []
    failed_count = 0
    exact_count = 0
    close_count = 0
    
    print(f"Generating {num_examples} solutions with {model}...")
    print(f"Estimated cost: ${num_examples * 0.001:.2f} (approximate)")
    
    for idx in tqdm(indices):
        example = data[idx]
        target = example["target"]
        nums = example["nums"]
        
        # Generate solution
        result = generate_solution_with_gpt4(target, nums, model=model)
        
        if result is not None:
            sft_examples.append(result)
            if result["is_exact"]:
                exact_count += 1
            else:
                close_count += 1
        else:
            failed_count += 1
        
        # Rate limiting
        time.sleep(batch_delay)
    
    print(f"\n{'='*60}")
    print(f"Generation Complete!")
    print(f"{'='*60}")
    print(f"Total generated: {len(sft_examples)}")
    print(f"  - Exact solutions: {exact_count}")
    print(f"  - Close solutions: {close_count}")
    print(f"  - Failed: {failed_count}")
    print(f"Success rate: {len(sft_examples) / num_examples * 100:.1f}%")
    
    # Save to JSONL
    with open(output_file, 'w') as f:
        for example in sft_examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"\nSaved to {output_file}")
    
    return sft_examples

def show_sample_outputs(jsonl_file: str, num_samples: int = 3):
    """Show sample outputs from generated dataset."""
    print(f"\n{'='*60}")
    print(f"Sample Outputs from {jsonl_file}")
    print(f"{'='*60}\n")
    
    with open(jsonl_file, 'r') as f:
        examples = [json.loads(line) for line in f]
    
    for i, example in enumerate(examples[:num_samples], 1):
        print(f"Example {i}:")
        print(f"Target: {example['target']}, Numbers: {example['nums']}")
        print(f"Solution: {example['solution']} = {example['result']}")
        print(f"Exact match: {example['is_exact']}")
        print(f"\nFull completion:")
        print(example['completion'])
        print(f"\n{'-'*60}\n")

# Cost estimation helper
def estimate_cost(num_examples: int, model: str = "gpt-4o-mini"):
    """Estimate API cost."""
    costs = {
        "gpt-4o-mini": 0.00015,  # per request (very rough estimate)
        "gpt-4o": 0.0005,
        "gpt-4-turbo": 0.001,
    }
    
    cost_per_request = costs.get(model, 0.001)
    total_cost = num_examples * cost_per_request
    
    print(f"\nCost Estimation for {model}:")
    print(f"  - {num_examples} examples")
    print(f"  - ~${cost_per_request:.5f} per request")
    print(f"  - Total: ~${total_cost:.2f}")
    print(f"  (This is a rough estimate; actual cost may vary)\n")

if __name__ == "__main__":
    import sys
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not found in environment!")
        print("Please set it in your .env file or environment:")
        print("  export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)
    
    # Show cost estimate
    estimate_cost(num_examples=500, model="gpt-4o-mini")
    
    # Ask for confirmation
    response = input("Continue with generation? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        sys.exit(0)
    
    # Generate training data (500 examples)
    print("\n" + "="*60)
    print("Generating Training Data")
    print("="*60 + "\n")
    
    train_data = create_sft_dataset_with_gpt4(
        dataset_name="justinphan3110/Countdown-Tasks-3to4",
        split="train",
        output_file="sft_train_gpt4.jsonl",
        num_examples=500,
        model="gpt-4o-mini",  # Fast and cheap
        batch_delay=0.1,
    )
    
    # Show samples
    show_sample_outputs("sft_train_gpt4.jsonl", num_samples=3)
    
    # Generate validation data (100 examples)
    print("\n" + "="*60)
    print("Generating Validation Data")
    print("="*60 + "\n")
    
    val_data = create_sft_dataset_with_gpt4(
        dataset_name="justinphan3110/Countdown-Tasks-3to4",
        split="test",  # Use test split for validation
        output_file="sft_val_gpt4.jsonl",
        num_examples=100,
        model="gpt-4o-mini",
        batch_delay=0.1,
    )
    
    print("\n" + "="*60)
    print("ALL DONE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review the generated data in sft_train_gpt4.jsonl")
    print("2. Run: python sft_training.py")
    print("3. Use the SFT checkpoint in your GRPO training")