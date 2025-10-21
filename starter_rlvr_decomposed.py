# NSR (Negative Sample Reinforcement) Implementation
# Based on: "The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning"
# Paper: https://arxiv.org/pdf/2506.01347
#
# KEY CHANGES FROM starter.py (GRPO):
#
# 1. BINARY REWARD FUNCTION (line ~196)
#    - Changed from {0, 0.1, 1.0} to {-1, +1}
#    - +1 if correct (has <answer>, valid numbers, equals target)
#    - -1 otherwise (matches paper's verifiable reward setup)
#
# 2. NEW: RLObjective ENUM (line ~334)
#    - RLVR: Standard baseline (all samples, ±1 rewards)
#    - PSR: Positive Sample Reinforcement (only correct samples)
#    - NSR: Negative Sample Reinforcement (only incorrect samples)
#    - W_REINFORCE: Weighted-REINFORCE (correct=+λ, incorrect=-1)
#
# 3. NEW: make_weighted_rewards() (line ~340)
#    - Applies objective-specific reward weighting
#    - Filters samples for PSR/NSR
#    - Returns: weighted_rewards, keep_mask
#
# 4. MODIFIED: compute_group_normalized_advantages() (line ~299)
#    - Added precomputed_rewards parameter
#    - Can now accept pre-weighted rewards
#    - Enables NSR/PSR/W-REINFORCE advantage computation
#
# 5. MODIFIED: train() function (line ~515)
#    - Added objective and lambda_psr parameters
#    - Sample filtering logic after rollout (lines ~560-587)
#    - Ensures batch size is multiple of group_size
#    - Uses effective batch size for loss averaging
#
# 6. NEW: NSR METRICS LOGGING (lines ~547-558)
#    - samples/correct_ratio: Learning progress (should increase)
#    - filtering/samples_kept: Sample efficiency (varies by objective)
#
# 7. MODIFIED: main() (line ~639)
#    - Added objective selection (line ~635)
#    - Added lambda_psr hyperparameter (line ~636)
#    - Passes these to train() function
#
# USAGE:
#   Change line ~635 to switch objectives:
#   - objective = RLObjective.W_REINFORCE  (paper's best, λ=0.1)
#   - objective = RLObjective.NSR          (only train on mistakes)
#   - objective = RLObjective.PSR          (only train on correct)
#   - objective = RLObjective.RLVR         (standard baseline)

import os
import datetime
import random
from typing import Callable, Dict, List, Tuple, Any
import logging
import warnings
import math

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from unittest.mock import patch
from datasets import load_dataset
from torch.utils.tensorboard import SummaryWriter
import re
from dotenv import load_dotenv
from enum import Enum

load_dotenv()

logging.getLogger("vllm.engine.scheduler").setLevel(logging.ERROR)
os.environ["VLLM_USE_V1"] = "0"

from torch.optim.lr_scheduler import LambdaLR

def get_constant_schedule_with_warmup(optimizer: torch.optim.Optimizer, num_warmup_steps: int, last_epoch: int = -1):
    def lr_lambda(current_step: int):
        return min(1.0, float(current_step) / float(max(1, num_warmup_steps)))
    return LambdaLR(optimizer, lr_lambda, last_epoch)

# -------------------------
# Prompting helpers (Countdown-style)
# -------------------------
TEMPLATE = """Using the numbers {numbers}, create an equation that equals {target}. 
You can use basic arithmetic operations (+, -, *, /) and each number can only be used once.
Show your reasoning in <think> </think> tags. And return the final equation in <answer> </answer> tags. Keep your reasoning under {max_tokens} tokens.
For example, numbers = [1, 2, 3, 4] and target = 5, the answer is <answer>(1 + 2) * 3 - 4</answer>."""


# vLLM utilities
def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85) -> LLM:
    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=2048
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM) -> None:
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def init_sampling_params(temperature: float, min_tokens: int, max_tokens: int) -> SamplingParams:
    sp = SamplingParams(
        temperature=temperature,
        top_p=1.0,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        logprobs=0,
    )
    sp.stop = ["</answer>"]
    sp.include_stop_str_in_output = True
    return sp


# Tokenization utilities
def tokenize_prompt_and_output(prompt_strs: List[str], output_strs: List[str], tokenizer: AutoTokenizer) -> Dict[str, torch.Tensor]:
    batch_data = []
    max_len = 0
    for prompt, output in zip(prompt_strs, output_strs):
        prompt_tokens = tokenizer(prompt)["input_ids"]
        output_tokens = tokenizer(output)["input_ids"]
        combined_tokens = prompt_tokens + output_tokens
        max_len = max(max_len, len(combined_tokens))
        batch_data.append({
            "tokens": combined_tokens, "prompt_len": len(prompt_tokens), "total_len": len(combined_tokens)
        })
    batch_size = len(batch_data)
    input_ids = torch.full((batch_size, max_len - 1), tokenizer.eos_token_id, dtype=torch.long)
    labels = torch.full((batch_size, max_len - 1), tokenizer.eos_token_id, dtype=torch.long)
    response_mask = torch.zeros((batch_size, max_len - 1), dtype=torch.bool)
    for i, data in enumerate(batch_data):
        tokens, seq_len = torch.tensor(data["tokens"]), len(data["tokens"])
        input_ids[i, :seq_len-1], labels[i, :seq_len-1] = tokens[:-1], tokens[1:]
        response_start, response_end = data["prompt_len"] - 1, seq_len - 1
        if response_end > response_start:
            response_mask[i, response_start:response_end] = True
    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}


def _extract_answer(solution_str: str) -> str | None:
    """
    Extract the content from the last <answer>...</answer> tag in the given string.

    Hint: Use the `re` module. `re.finditer` can find all occurrences of a pattern.
    Args:
        solution_str: The full text generated by the model.

    Returns:
        The stripped string content of the last answer tag, or None if no tag is found.
    """
    if not solution_str:
        return None
    matches = list(re.finditer(r"<answer>\s*(.*?)\s*</answer>", solution_str, flags=re.DOTALL | re.IGNORECASE))
    return matches[-1].group(1).strip() if matches else None


def _validate_numbers(equation_str: str, available_numbers: List[int]) -> bool:
    """
    Check if the numbers used in the equation are exactly the ones available.

    Hint:
    1. Use `re.findall(r"\d+", equation_str)` to get all number strings from the equation.
    2. Remember to handle potential errors and return False.

    Args:
        equation_str: The equation string to validate.
        available_numbers: A list of integers that are allowed to be used.

    Returns:
        True if the equation uses the correct numbers, False otherwise.
    """
    ### YOUR CODE HERE ###
    try: 
        # Extract all numbers from the equation 
        found_numbers = re.findall(r"\d+", equation_str)
        
        # Convert them into string
        str_found_numbers = list(map(int, found_numbers))
        
        # Compare (check for number & frequency - since each number must only appear once)
        return sorted(str_found_numbers) == sorted(available_numbers)
        
    except Exception:
        return False 


def _evaluate_equation(equation_str: str) -> float | None:
    """
    Safely evaluate a mathematical equation string.
    Args:
        equation_str: The equation string to evaluate.
    Returns:
        The result of the equation as a float, or None if it's invalid or unsafe.
    """
    ### YOUR CODE HERE ###
    try: 
        # Allows only valid characters
        if not re.fullmatch(r"[0-9+\-*/()\s]+", equation_str): 
            return None
        
        # Evaluate
        result = eval(equation_str, {"__builtins__": None}, {})

        # Return to float
        return float(result)
        
    except Exception:
        return None
    ### END YOUR CODE ###



def reward_fn(generated_text: str, ground_truth: Dict, scale_factor: float = 1.0) -> float:
    """
    Binary verifiable reward for Countdown:
    +1.0 if equation valid + uses exactly the given numbers + equals target,
    -1.0 otherwise. (Paper uses ±1; we'll keep that and do weighting later.)
    """
    target = ground_truth.get("target")
    numbers = ground_truth.get("numbers", []) or ground_truth.get("nums", [])

    eq = _extract_answer(generated_text)
    if eq is None:
        return -1.0

    if not _validate_numbers(eq, numbers):
        return -1.0

    val = _evaluate_equation(eq)
    if val is None:
        return -1.0

    return 1.0 if abs(val - target) < 1e-6 else -1.0



def evaluate_model(llm: LLM, sampling_params: SamplingParams, eval_prompts: List[str], eval_answers: List[Dict]) -> Dict[str, Any]:
    rollouts = llm.generate(eval_prompts, sampling_params)
    examples, rewards, output_token_lengths = [], [], []
    for rollout, gt in zip(rollouts, eval_answers):
        response_text = rollout.outputs[0].text
        reward_value = reward_fn(response_text, gt)
        equation = _extract_answer(response_text)
        result = _evaluate_equation(equation) if equation is not None else None
        output_tokens = len(llm.llm_engine.tokenizer.encode(response_text))
        output_token_lengths.append(output_tokens)
        examples.append({
            "prompt": rollout.prompt, "response": response_text, "answer": gt, "equation": equation,
            "result": result, "reward": reward_value, "output_tokens": output_tokens,
        })
        rewards.append(reward_value)
    rewards_tensor = torch.tensor(rewards) if rewards else torch.tensor([0.0])
    tol = 1e-8
    count_correct = sum(1 for r in rewards if abs(r - 1.0) < tol)
    count_incorrect = sum(1 for r in rewards if abs(r - (-1.0)) < tol)
    accuracy = (count_correct / len(rewards)) * 100 if rewards else 0.0
    avg_output_tokens = sum(output_token_lengths) / len(output_token_lengths) if output_token_lengths else 0.0
    return {
        "mean_reward": float(rewards_tensor.mean().item()),
        "std_reward": float(rewards_tensor.std().item()) if rewards_tensor.numel() > 1 else 0.0,
        "num_examples": len(rewards), "examples": examples, "count_correct": count_correct,
        "count_incorrect": count_incorrect, "accuracy": accuracy,
        "avg_output_tokens": avg_output_tokens,
    }


def _format_eval_example(example: Dict[str, Any]) -> str:
    target = example["answer"]["target"] if isinstance(example.get("answer"), dict) and "target" in example["answer"] else "?"
    numbers = example["answer"].get("numbers") if isinstance(example.get("answer"), dict) else None
    return (
        f"Prompt: {example.get('prompt', '')}\n"
        f"Response: {example.get('response', '')}\n"
        f"Equation: {example.get('equation', None)} | Result: {example.get('result', None)} | Target: {target} | Numbers: {numbers}\n"
        f"Reward: {example.get('reward', 0.0):.3f}\n"
    )


def log_train(rollout_batch_loss: float, grad_norm: float, reward_metadata: Dict[str, Any], avg_output_tokens: float, writer: SummaryWriter | None, step: int) -> None:
    writer.add_scalar("train/loss", float(rollout_batch_loss), global_step=step)
    writer.add_scalar("train/grad_norm", float(grad_norm), global_step=step)
    writer.add_scalar("train/reward_mean", float(reward_metadata["mean"]), global_step=step)
    writer.add_scalar("train/reward_std", float(reward_metadata["std"]), global_step=step)
    writer.add_scalar("train/avg_output_tokens", float(avg_output_tokens), global_step=step)
    print(f"Step {step} | Loss: {rollout_batch_loss:.4f} | Grad norm: {grad_norm:.4f} | Reward mean: {float(reward_metadata['mean']):.4f} | Reward std: {float(reward_metadata['std']):.4f} | Avg output tokens: {avg_output_tokens:.1f}")


def log_eval(metrics: Dict[str, Any], writer: SummaryWriter | None, step: int) -> None:
    examples = metrics.get("examples", []) or []
    if not examples: return
    tol = 1e-8
    correct_examples = [ex for ex in examples if abs(float(ex.get("reward", 0.0)) - 1.0) < tol][:10]
    incorrect_examples = [ex for ex in examples if abs(float(ex.get("reward", 0.0)) - (-1.0)) < tol][:10]
    if correct_examples:
        print(f"\n=== Eval examples (CORRECT, reward=+1) @ step {step} ===")
        for idx, ex in enumerate(correct_examples[:2], 1): print(f"[CORRECT #{idx}]\n" + _format_eval_example(ex))
    if incorrect_examples:
        print(f"\n=== Eval examples (INCORRECT, reward=-1) @ step {step} ===")
        for idx, ex in enumerate(incorrect_examples[:2], 1): print(f"[INCORRECT #{idx}]\n" + _format_eval_example(ex))
    if writer:
        correct_text = "\n\n".join([_format_eval_example(ex) for ex in correct_examples]) or ""
        incorrect_text = "\n\n".join([_format_eval_example(ex) for ex in incorrect_examples]) or ""
        if correct_text: writer.add_text("eval/examples_correct", correct_text, global_step=step)
        if incorrect_text: writer.add_text("eval/examples_incorrect", incorrect_text, global_step=step)
    print(f"Eval @ step {step}: accuracy={metrics['accuracy']:.1f}% mean_reward={metrics['mean_reward']:.4f} "
          f"avg_tokens={metrics['avg_output_tokens']:.1f} | correct:{metrics['count_correct']} "
          f"incorrect:{metrics['count_incorrect']}")


def compute_group_normalized_advantages(
    rollout_responses: List[str],
    repeated_ground_truths: List[Dict],
    reward_fn: Callable[[str, Dict], float],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
    precomputed_rewards: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    # 1) compute or use provided rewards
    if precomputed_rewards is None:
        rewards = [reward_fn(resp, gt) for resp, gt in zip(rollout_responses, repeated_ground_truths)]
        raw_rewards = torch.tensor(rewards, dtype=torch.float32)
    else:
        raw_rewards = precomputed_rewards.float()

    # 2) reshape into groups (assumes length multiple of group_size)
    rewards_grouped = raw_rewards.view(-1, group_size)
    group_mean = rewards_grouped.mean(dim=1, keepdim=True)
    group_std = rewards_grouped.std(dim=1, keepdim=True, unbiased=False)

    advantages = rewards_grouped - group_mean
    if normalize_by_std:
        advantages = advantages / (group_std + advantage_eps)

    advantages = advantages.flatten()
    metadata = {
        "mean": torch.mean(raw_rewards),
        "std": torch.std(raw_rewards, unbiased=False),
        "max": torch.max(raw_rewards),
        "min": torch.min(raw_rewards),
    }
    return advantages, raw_rewards, metadata


class RLObjective(str, Enum):
    RLVR = "rlvr"            # use all samples, rewards = {+1, -1}
    PSR = "psr"              # keep only correct samples
    NSR = "nsr"              # keep only incorrect samples
    W_REINFORCE = "w_reinforce"  # rewards = {+λ, -1}

def make_weighted_rewards(rollout_responses, repeated_ground_truths, base_reward_fn, objective: RLObjective, lambda_psr: float = 0.1):
    """
    Returns:
      rewards: torch.FloatTensor [N]
      keep_mask: torch.BoolTensor [N] -> which samples to keep in this objective
    """
    raw = [base_reward_fn(r, gt) for r, gt in zip(rollout_responses, repeated_ground_truths)]
    rewards = []
    keep = []
    for rv in raw:
        if objective == RLObjective.RLVR:
            rewards.append(rv)  # +1 or -1
            keep.append(True)
        elif objective == RLObjective.PSR:
            keep.append(rv > 0)
            if rv > 0:
                rewards.append(1.0)
        elif objective == RLObjective.NSR:
            keep.append(rv < 0)
            if rv < 0:
                rewards.append(-1.0)
        elif objective == RLObjective.W_REINFORCE:
            # +λ for correct, -1 for incorrect
            if rv > 0:
                rewards.append(lambda_psr)
                keep.append(True)
            else:
                rewards.append(-1.0)
                keep.append(True)
        else:
            raise ValueError(objective)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    keep_mask = torch.tensor(keep, dtype=torch.bool)
    return rewards, keep_mask



# ==============================================================================
# TASK 4: Implement per-token Loss
# ==============================================================================
def compute_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    clip_range: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Computes the per-token PPO clipped surrogate loss.

    Why omit the KL divergence term?
    For simplicity and following trends in recent RLVR (e.g Dr.GRPO),
    we omit the KL penalty term often found in PPO. This simplifies the implementation
    and has been shown to work well in practice.

    Steps:
    1. Calculate the probability ratio `pi_ratio = exp(policy_log_probs - old_log_probs)`.
    2. Calculate the unclipped term: `advantages * pi_ratio`.
    3. Calculate the clipped term by clipping `pi_ratio` to `[1-clip_range, 1+clip_range]`
       and then multiplying by `advantages`.
    4. The final loss is `-torch.minimum(unclipped_term, clipped_term)`.
    """
    loss = 0.0
    ### YOUR CODE HERE ###

    # 1. Ratio between new and old policies
    pi_ratio = torch.exp(policy_log_probs - old_log_probs)

    # 2. Unclipped term
    unclipped = advantages * pi_ratio

    # 3. Clipped term
    clipped_ratio = torch.clamp(pi_ratio, 1 - clip_range, 1 + clip_range)
    clipped = advantages * clipped_ratio

    # 4. Take elementwise minimum and negate (PPO-style objective)
    loss = -torch.minimum(unclipped, clipped)

    # Optional metadata for logging/debugging
    stats = {
        "ratio_mean": pi_ratio.mean(),
        "ratio_std": pi_ratio.std(),
        "ratio_min": pi_ratio.min(),
        "ratio_max": pi_ratio.max(),
    }
    
    ### END YOUR CODE ###
    return loss, stats


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute the mean of tensor values where mask=True for each row, then average across the batch.
    """
    ### YOUR CODE HERE ###
    # Convert mask to float for proper multiplication
    mask = mask.float()
    
    # Sum over tokens for each sequence, considering only response tokens
    masked_sum = (tensor * mask).sum(dim=1)
    
    # Count valid tokens per sequence
    token_count = mask.sum(dim=1).clamp(min=1)
    
    # Mean over valid tokens in each sequence, then average across batch
    mean_per_seq = masked_sum / token_count
    loss = mean_per_seq.mean()
    ### END YOUR CODE ###
    return loss

def masked_mean_drgrpo(tensor: torch.Tensor, mask: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """
    Compute the sum of tensor values where mask=True, divided by num_tokens, then average across the batch.
    This is used for the DR-GRPO loss
    """
    ### YOUR CODE HERE ###
    # Convert mask to float for proper multiplication
    mask = mask.float()
    
    masked_sum = (tensor * mask).sum(dim=1)
    
    # Divide by fixed constant num_tokens (same for all sequences)
    mean_per_seq = masked_sum / float(num_tokens)
    loss = mean_per_seq.mean()
    ### END YOUR CODE ###
    return loss

def get_response_log_probs(model: PreTrainedModel, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits = model(input_ids).logits
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return log_probs


# Training helpers and loop
def duplicate_data(arr: List, group_size: int) -> List:
    return [x for x in arr for _ in range(group_size)]


def rollout_with_vllm(policy: PreTrainedModel, llm: LLM, sampling_params: SamplingParams, prompts_batch: List[str], group_size: int) -> Tuple[List[str], List[str], List[int]]:
    load_policy_into_vllm_instance(policy, llm)
    prompts_dup = duplicate_data(prompts_batch, group_size)
    vllm_rollouts = llm.generate(prompts_dup, sampling_params, use_tqdm=False)
    rollout_input_text, rollout_response_text, rollout_output_tokens = [], [], []
    for rollout in vllm_rollouts:
        for r in rollout.outputs:
            rollout_input_text.append(rollout.prompt)
            rollout_response_text.append(r.text)
            rollout_output_tokens.append(len(llm.llm_engine.tokenizer.encode(r.text)))
    return rollout_input_text, rollout_response_text, rollout_output_tokens


def tokenize_rollouts(rollout_input_text: List[str], rollout_response_text: List[str], tokenizer: AutoTokenizer) -> Dict[str, torch.Tensor]:
    return tokenize_prompt_and_output(rollout_input_text, rollout_response_text, tokenizer)


def grpo_microbatch_step(
    policy: PreTrainedModel, input_ids: torch.Tensor, labels: torch.Tensor, response_mask: torch.Tensor,
    advantages_per_seq: torch.Tensor, gradient_accumulation_steps: int, clip_range: float,
    loss_type: str = "grpo", max_completion_length: int = 512,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    policy_log_probs = get_response_log_probs(policy, input_ids, labels)
    old_log_probs = policy_log_probs.detach()
    advantages = advantages_per_seq.unsqueeze(-1)
    loss_per_token, metadata = compute_loss(advantages, policy_log_probs, old_log_probs, clip_range)
    if loss_type == "grpo":
        loss = masked_mean(loss_per_token, response_mask)
    elif loss_type == "dr_grpo":
        loss = masked_mean_drgrpo(loss_per_token, response_mask, max_completion_length)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    loss = loss / gradient_accumulation_steps
    loss.backward()
    return loss.detach(), metadata


def train(
    policy: PreTrainedModel, tokenizer: AutoTokenizer, llm: LLM, sampling_params: SamplingParams, *,
    train_prompts: List[str], train_answers: List[Dict], eval_prompts: List[str], eval_answers: List[Dict],
    optimizer: torch.optim.Optimizer, scheduler, n_grpo_steps: int, rollout_batch_size: int,
    group_size: int, gradient_accumulation_steps: int, clip_range: float, use_std_normalization: bool,
    advantage_eps: float, device: str, eval_every: int = 5, writer: SummaryWriter = None, seed: int,
    loss_type: str = "grpo", max_completion_length: int = 256,
    objective: RLObjective = RLObjective.RLVR, lambda_psr: float = 0.1,
) -> None:
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    micro_train_batch_size = rollout_batch_size // gradient_accumulation_steps
    random.seed(seed)
    train_step = 0

    metrics = evaluate_model(llm, sampling_params, eval_prompts, eval_answers)
    if writer:
        for k in ["accuracy", "mean_reward", "std_reward", "avg_output_tokens", "count_correct", "count_incorrect"]:
            writer.add_scalar(f"eval/{k}", metrics[k], global_step=train_step)
        log_eval(metrics, writer, train_step)

    for _ in range(n_grpo_steps):
        sampled = random.sample(list(zip(train_prompts, train_answers)), n_prompts_per_rollout_batch)
        prompts_batch, answers_batch = [p for p, _ in sampled], [a for _, a in sampled]
        rollout_input, rollout_response, rollout_tokens = rollout_with_vllm(policy, llm, sampling_params, prompts_batch, group_size)
        answers_dup = duplicate_data(answers_batch, group_size)
        avg_output_tokens = sum(rollout_tokens) / len(rollout_tokens) if rollout_tokens else 0.0
                
        # NSR/PSR filtering: Build weighted rewards + keep mask for objective
        weighted_rewards, keep_mask = make_weighted_rewards(
            rollout_response, answers_dup, reward_fn, objective=objective, lambda_psr=lambda_psr
        )

        # Calculate and log critical NSR metrics
        raw_rewards_for_metrics = torch.tensor(
            [reward_fn(r, gt) for r, gt in zip(rollout_response, answers_dup)],
            dtype=torch.float32
        )
        num_correct = (raw_rewards_for_metrics > 0).sum().item()
        correct_ratio = num_correct / len(raw_rewards_for_metrics)
        samples_kept = keep_mask.sum().item()
        
        if writer:
            writer.add_scalar("samples/correct_ratio", correct_ratio, train_step)
            writer.add_scalar("filtering/samples_kept", samples_kept, train_step)

        # If PSR/NSR filtered some samples, we must filter everything consistently:
        if keep_mask.sum().item() != keep_mask.numel():
            keep_list = keep_mask.tolist()
            rollout_input   = [t for t, k in zip(rollout_input,   keep_list) if k]
            rollout_response= [t for t, k in zip(rollout_response,keep_list) if k]
            answers_dup     = [t for t, k in zip(answers_dup,     keep_list) if k]
            weighted_rewards = weighted_rewards[keep_mask]
            rollout_batch_size_effective = len(rollout_response)
        else:
            rollout_batch_size_effective = rollout_batch_size

        # Ensure length is a multiple of group_size (needed for .view(-1, group_size))
        rem = rollout_batch_size_effective % group_size
        if rem != 0:
            cut = rollout_batch_size_effective - rem
            rollout_input        = rollout_input[:cut]
            rollout_response     = rollout_response[:cut]
            answers_dup          = answers_dup[:cut]
            weighted_rewards     = weighted_rewards[:cut]
            rollout_batch_size_effective = cut
            if rollout_batch_size_effective == 0:
                # skip this iteration if nothing remains
                continue

        # Compute advantages using precomputed (possibly weighted) rewards
        advantages, _, reward_meta = compute_group_normalized_advantages(
            rollout_response, answers_dup, reward_fn, group_size, advantage_eps, use_std_normalization,
            precomputed_rewards=weighted_rewards
        )

        # Tokenize the (possibly filtered) rollouts
        tokenized = tokenize_rollouts(rollout_input, rollout_response, tokenizer)

        # Microbatch loop should iterate over rollout_batch_size_effective instead of rollout_batch_size
        optimizer.zero_grad()
        rollout_loss = 0.0
        for micro_idx in range(0, rollout_batch_size_effective, micro_train_batch_size):
            s = slice(micro_idx, micro_idx + micro_train_batch_size)
            loss, _ = grpo_microbatch_step(
                policy,
                tokenized["input_ids"][s].to(device),
                tokenized["labels"][s].to(device),
                tokenized["response_mask"][s].to(device),
                advantages[s].to(device),
                gradient_accumulation_steps, clip_range, loss_type=loss_type,
                max_completion_length=max_completion_length
            )
            rollout_loss += float(loss.item())



            
        grad_norm = torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.grad is not None], 1.0)
        optimizer.step()
        scheduler.step()
        # Note: rollout_loss is already scaled by gradient_accumulation_steps in grpo_microbatch_step
        train_step += 1
        print(f"Step {train_step} | Loss: {rollout_loss:.4f} | Grad: {grad_norm:.4f} | "
              f"Reward mean: {reward_meta['mean']:.4f} | Reward std: {reward_meta['std']:.4f} | "
              f"Correct: {correct_ratio:.2%} | Samples kept: {samples_kept}/{rollout_batch_size}")
        log_train(rollout_loss, grad_norm, reward_meta, avg_output_tokens, writer, train_step)
        if train_step % eval_every == 0:
            metrics = evaluate_model(llm, sampling_params, eval_prompts, eval_answers)
            log_eval(metrics, writer, train_step)


def init_policy(model_id: str, device: str) -> Tuple[PreTrainedModel, AutoTokenizer]:
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", use_cache=False
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.to(device).train()
    return model, tokenizer


def main() -> None:
    objective = RLObjective.W_REINFORCE  # choices: RLVR, PSR, NSR, W_REINFORCE
    lambda_psr = 0.1                     # paper recommended 
    # Hyperparameters
    model_id = "Qwen/Qwen3-1.7B"
    device = "cuda"
    seed, gpu_mem_util = 42, 0.4
    n_grpo_steps, rollout_batch_size, group_size, grad_acc_steps = 200, 128, 8, 16
    lr, clip_range, adv_eps = 3e-6, 0.2, 1e-4
    temperature, min_tokens = 0.7, 4
    eval_every = 10

    # CHANGING HYPERPARAMETERS for main assignment
    loss_type = "grpo" # or "dr_grpo"
    max_tokens = 512 # or 512, 1024
    
    # Initialization
    use_std_norm = loss_type == "grpo"
    policy, tokenizer = init_policy(model_id=model_id, device=device)
    llm = init_vllm(model_id=model_id, device=device, seed=seed, gpu_memory_utilization=gpu_mem_util)
    sampling_params = init_sampling_params(temperature=temperature, min_tokens=min_tokens, max_tokens=max_tokens)
    
    # Dataset
    def build_dataset(split):
        data = []
        for ex in split:
            prompt = TEMPLATE.format(numbers=ex["nums"], target=ex["target"], max_tokens=max_tokens)
            prompt = tokenizer.apply_chat_template(
                [dict(role="system", content="You are a helpful assistant."),
                dict(role="user", content=prompt)],
                add_generation_prompt=True, tokenize=False)
            data.append({"prompt": prompt,"answer": {"target": ex["target"], "numbers": ex["nums"]},})
        return data

    # Load properly split dataset
    train_data = load_dataset("justinphan3110/Countdown-Tasks-3to4", split="train")
    eval_data = load_dataset("justinphan3110/Countdown-Tasks-3to4", split="test")
    
    train_examples = build_dataset(train_data)
    eval_examples = build_dataset(eval_data)
    
    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-2, betas=(0.9, 0.95))
    scheduler = get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0)
    
    # Logging
    timestamp = int(datetime.datetime.now(datetime.timezone.utc).timestamp())
    log_dir = os.path.join("./output", "tb", f"hw_a2_{loss_type}", str(timestamp))
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    # Training
    train(
        policy=policy, tokenizer=tokenizer, llm=llm, sampling_params=sampling_params,
        train_prompts=[ex["prompt"] for ex in train_examples], train_answers=[ex["answer"] for ex in train_examples],
        eval_prompts=[ex["prompt"] for ex in eval_examples], eval_answers=[ex["answer"] for ex in eval_examples],
        optimizer=optimizer, scheduler=scheduler, n_grpo_steps=n_grpo_steps,
        rollout_batch_size=rollout_batch_size, group_size=group_size,
        gradient_accumulation_steps=grad_acc_steps, clip_range=clip_range,
        use_std_normalization=use_std_norm, advantage_eps=adv_eps, device=device,
        eval_every=eval_every, writer=writer, seed=seed, loss_type=loss_type,
        max_completion_length=max_tokens,
        objective=objective, lambda_psr=lambda_psr
    )
    
    # Save model
    out_dir = os.path.join("./output", f"hw_a2_solution_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    policy.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Saved model and tokenizer to {out_dir}")
    writer.close()

if __name__ == "__main__":
    main()