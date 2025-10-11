# KL Divergence Implementation

This document explains the KL divergence feature added to `starter.py`.

## Overview

KL (Kullback-Leibler) divergence has been integrated into the GRPO training pipeline. KL divergence measures the difference between two probability distributions and is commonly used in reinforcement learning from human feedback (RLHF) to prevent the policy from deviating too much from a reference model during training.

## What Was Added

### 1. `compute_kl_divergence()` Function (Line 388-408)

```python
def compute_kl_divergence(
    policy_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
) -> torch.Tensor
```

**Purpose:** Computes the per-token KL divergence between the current policy and a reference policy.

**Formula:** KL(π || π_ref) = log π - log π_ref (in log space)

**Returns:** Per-token KL divergence tensor with shape `[batch_size, seq_len]`

### 2. Enhanced `compute_loss()` Function (Line 489-547)

The loss function now accepts two additional parameters:
- `ref_log_probs`: Log probabilities from the reference model (optional)
- `kl_coef`: Coefficient for the KL penalty term (default: 0.0)

**Modified Loss Formula:**
```
Total Loss = PPO_clipped_loss + kl_coef * KL(policy || reference)
```

**New Stats:** The function now also returns `kl_penalty_mean` in the metadata dictionary.

### 3. Updated `grpo_microbatch_step()` Function (Line 616-642)

Now accepts:
- `ref_model`: Reference model for computing KL divergence (optional)
- `kl_coef`: KL penalty coefficient

**Behavior:** 
- If `ref_model` is provided and `kl_coef > 0`, computes reference log probabilities
- Passes them to `compute_loss()` for KL penalty calculation
- Reference model is used in `torch.no_grad()` mode for efficiency

### 4. Enhanced `train()` Function (Line 645-701)

Accepts two new parameters:
- `ref_model`: Reference model (optional)
- `kl_coef`: KL penalty coefficient (default: 0.0)

**Features:**
- Tracks KL penalty across microbatches
- Computes average KL penalty for logging

### 5. Updated `log_train()` Function (Line 342-350)

Now logs KL penalty to TensorBoard and console:
- Adds `train/kl_penalty` scalar to TensorBoard
- Includes KL penalty in console output when enabled

### 6. Updated `main()` Function (Line 708-812)

**New Hyperparameters:**
```python
use_kl_penalty = False  # Enable/disable KL divergence
kl_coef = 0.01         # KL penalty coefficient
```

**Reference Model Initialization:**
- When `use_kl_penalty = True`, creates a frozen copy of the initial policy
- Reference model is set to eval mode
- All parameters are frozen (requires_grad = False)

### 7. New Tests (Line 818-871)

Added comprehensive tests:
1. `test_compute_kl_divergence`: Verifies KL calculation
2. `test_compute_loss_with_kl`: Verifies KL integration into loss

## How to Use

### Basic Usage

To enable KL divergence, modify the hyperparameters in `main()`:

```python
# Enable KL divergence penalty
use_kl_penalty = True   # Set to True
kl_coef = 0.01         # Adjust coefficient (0.001 - 0.1)
```

### Tuning the KL Coefficient

The `kl_coef` parameter controls how strongly the policy is constrained:

| kl_coef | Effect |
|---------|--------|
| 0.0     | No KL penalty (original GRPO) |
| 0.001-0.01 | Light constraint (recommended starting point) |
| 0.01-0.05 | Moderate constraint |
| 0.05-0.1 | Strong constraint |
| > 0.1   | Very strong constraint (may limit learning) |

**Guidelines:**
- **Too low:** Policy may deviate significantly, potential instability
- **Too high:** Policy updates become too conservative, slow learning
- **Recommended:** Start with 0.01 and adjust based on training behavior

### Monitoring KL Divergence

During training, KL penalty is logged:

1. **Console Output:**
```
Step 1 | Loss: 0.1234 | Grad norm: 0.5678 | ... | KL penalty: 0.0012
```

2. **TensorBoard:**
- `train/kl_penalty`: Per-step KL penalty values
- Compare with `train/loss` to see relative contribution

## Benefits of KL Divergence

1. **Prevents Catastrophic Forgetting:** Keeps the policy close to its initialization
2. **Training Stability:** Reduces risk of policy collapse
3. **Controlled Exploration:** Balances exploration with constraint
4. **Better Convergence:** Often leads to more stable training curves

## Performance Considerations

- **Memory:** Reference model requires additional GPU memory (~same as policy)
- **Computation:** Additional forward pass through reference model per microbatch
- **Recommendation:** If GPU memory is limited, reduce batch size or use gradient checkpointing

## Example Training Run

```bash
# Default (no KL penalty)
python starter.py

# With KL penalty
# 1. Edit main() to set use_kl_penalty = True
# 2. Adjust kl_coef as needed
python starter.py
```

## Mathematical Background

The KL divergence between two distributions P (policy) and Q (reference) is:

```
KL(P || Q) = Σ P(x) * log(P(x) / Q(x))
           = Σ P(x) * (log P(x) - log Q(x))
           = E_P[log P(x) - log Q(x)]
```

In our implementation:
- We compute log probabilities for each token
- KL divergence is calculated per token: `log π(token) - log π_ref(token)`
- The KL penalty is masked to response tokens only
- Final loss adds: `kl_coef * mean(KL_divergence)`

## Testing

Run the test suite to verify the implementation:

```bash
python starter.py  # Runs test_functions() before main()
```

Expected test output:
```
=== Running Quick Tests ===
Testing compute_kl_divergence...
✅ compute_kl_divergence works! Mean KL: 0.XXXX
Testing masked_mean...
✅ masked_mean works! Result: 0.XXXX
...
Testing compute_loss with KL penalty...
✅ compute_loss with KL penalty works! KL penalty mean: 0.XXXX
...
=== All Tests Passed! ===
```

## Troubleshooting

**Issue:** GPU out of memory when enabling KL divergence

**Solution:** 
- Reduce `rollout_batch_size` or `gradient_accumulation_steps`
- Reduce `gpu_mem_util` parameter
- Use a smaller model

**Issue:** Training not improving with KL penalty

**Solution:**
- Reduce `kl_coef` (try 0.001 or 0.005)
- Check TensorBoard to see if KL penalty dominates the loss

**Issue:** Policy diverges despite KL penalty

**Solution:**
- Increase `kl_coef` 
- Check if reference model is properly frozen
- Verify learning rate isn't too high

## References

1. Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
2. Ouyang et al., "Training language models to follow instructions with human feedback" (2022)
3. Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models" (2024)

## Summary

KL divergence is now fully integrated into the GRPO training pipeline:
- ✅ Core KL computation implemented
- ✅ Integrated into loss function
- ✅ Reference model support
- ✅ Logging and monitoring
- ✅ Comprehensive tests
- ✅ Configurable via hyperparameters
- ✅ Backward compatible (disabled by default)

