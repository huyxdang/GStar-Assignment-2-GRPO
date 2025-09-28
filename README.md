# GRPO for Countdown Math Problems

This project implements the **Group-Reward Policy Optimization (GRPO)** algorithm to fine-tune a language model on the "Countdown" math task. The goal is to train an LLM to generate correct mathematical equations that reach a target number using a given set of integers.

## Setup

```
pip install --upgrade uv
uv venv
source .venv/bin/activate
uv pip install vllm==0.7.2 triton==3.1.0 datasets transformers==4.51.3 tensorboard torch gpustat datasets python-dotenv
uv pip install flash-attn==2.7.4.post1 --no-build-isolation
```

## Assignment Structure

This repository is also structured as a hands-on assignment where students implement the core components of the GRPO algorithm. The key tasks are located in `starter.py`:

## Problem 1
### Part 1: GRPO Pipeline Implementation
-   **Task 1**: Implement reward helper functions (`_extract_answer`, `_validate_numbers`, `_evaluate_equation`).
-   **Task 2**: Implement the main `reward_fn`.
-   **Task 3**: Implement `compute_group_normalized_advantages`.
-   **Task 5**: Implement `masked mean` for GRPO and `masked_mean_drgrpo` for DR.GRPO.

### Part 2: Main Experiments and Report (please refer to section 1.6 of [gstar_assignment2.pdf](./gstar_assignment2.pdf) for the Main Report and Experiments

## Problem 2: (Optional) Open-ended Investigation. Please refer to Section 2 of [gstar_assignment2.pdf](./gstar_assignment2.pdf)
