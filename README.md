---
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

This dataset contains supervised fine-tuning (SFT) data for the Countdown math puzzle task.

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


## Dataset Statistics
- **Train examples**: 260
- **Validation examples**: 50
- **Total**: 310

## Example

```python
{
  "target": 36,
  "nums": [79, 17, 60],
  "solution": "79 - (60 - 17)",
  "completion": "<think>\nI need to reach 36 using 79, 17, and 60...\n</think>\n\n<answer>79 - (60 - 17)</answer>",
  "result": 36.0,
  "is_exact": true
}
```

## Citation
If you use this dataset, please cite the original Countdown task dataset:
```
@dataset{countdown_tasks,
  author = {Justin Phan},
  title = {Countdown Tasks},
  year = {2024},
  publisher = {HuggingFace},
  url = {https://huggingface.co/datasets/justinphan3110/Countdown-Tasks-3to4}
}
```
