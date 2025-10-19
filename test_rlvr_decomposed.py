"""
Unit tests for starter_rlvr_decomposed.py
Tests all NSR/PSR/W-REINFORCE functionality
"""

import sys
import torch
from starter_rlvr_decomposed import (
    _extract_answer,
    _validate_numbers,
    _evaluate_equation,
    reward_fn,
    RLObjective,
    make_weighted_rewards,
    compute_group_normalized_advantages,
)

def test_extract_answer():
    """Test answer extraction from <answer> tags"""
    print("Testing _extract_answer...")
    
    # Basic test
    text = "Here is my solution: <answer>(79 - 60) + 17</answer>"
    assert _extract_answer(text) == "(79 - 60) + 17"
    
    # Multiple tags (should get last one)
    text = "<answer>wrong</answer> Actually: <answer>correct</answer>"
    assert _extract_answer(text) == "correct"
    
    # No tag
    assert _extract_answer("no answer here") is None
    
    # Empty
    assert _extract_answer("") is None
    
    print("✅ _extract_answer tests passed")


def test_validate_numbers():
    """Test number validation"""
    print("Testing _validate_numbers...")
    
    # Correct usage
    assert _validate_numbers("(79 - 60) + 17", [79, 60, 17]) == True
    assert _validate_numbers("79 + 60 + 17", [17, 79, 60]) == True  # order doesn't matter
    
    # Wrong count
    assert _validate_numbers("79 + 60", [79, 60, 17]) == False
    
    # Repeated number
    assert _validate_numbers("17 + 17", [79, 60, 17]) == False
    
    # Wrong numbers
    assert _validate_numbers("1 + 2 + 3", [79, 60, 17]) == False
    
    # Empty
    assert _validate_numbers("", [1, 2, 3]) == False
    
    print("✅ _validate_numbers tests passed")


def test_evaluate_equation():
    """Test safe equation evaluation"""
    print("Testing _evaluate_equation...")
    
    # Valid equations
    assert _evaluate_equation("(79 - 60) + 17") == 36.0
    assert _evaluate_equation("2 + 2") == 4.0
    assert _evaluate_equation("10 / 2") == 5.0
    
    # Invalid/unsafe
    assert _evaluate_equation("import os") is None
    assert _evaluate_equation("__import__('os')") is None
    assert _evaluate_equation("1/0") is None  # Division by zero
    
    print("✅ _evaluate_equation tests passed")


def test_reward_fn_binary():
    """Test binary reward function (±1)"""
    print("Testing reward_fn (binary ±1)...")
    
    # Correct answer: +1
    text_correct = "<answer>(79 - 60) + 17</answer>"
    gt = {"target": 36, "numbers": [79, 60, 17]}
    reward = reward_fn(text_correct, gt)
    assert reward == 1.0, f"Expected +1 for correct, got {reward}"
    
    # Wrong result: -1
    text_wrong = "<answer>79 + 60 + 17</answer>"
    reward = reward_fn(text_wrong, gt)
    assert reward == -1.0, f"Expected -1 for wrong result, got {reward}"
    
    # No answer tag: -1
    text_no_answer = "I don't know"
    reward = reward_fn(text_no_answer, gt)
    assert reward == -1.0, f"Expected -1 for no answer, got {reward}"
    
    # Wrong numbers: -1
    text_wrong_nums = "<answer>1 + 2 + 3</answer>"
    reward = reward_fn(text_wrong_nums, gt)
    assert reward == -1.0, f"Expected -1 for wrong numbers, got {reward}"
    
    # Invalid equation: -1
    text_invalid = "<answer>invalid equation!</answer>"
    reward = reward_fn(text_invalid, gt)
    assert reward == -1.0, f"Expected -1 for invalid equation, got {reward}"
    
    print("✅ reward_fn tests passed")


def test_rlobjective_enum():
    """Test RLObjective enum"""
    print("Testing RLObjective enum...")
    
    assert RLObjective.RLVR == "rlvr"
    assert RLObjective.PSR == "psr"
    assert RLObjective.NSR == "nsr"
    assert RLObjective.W_REINFORCE == "w_reinforce"
    
    print("✅ RLObjective enum tests passed")


def test_make_weighted_rewards_rlvr():
    """Test make_weighted_rewards with RLVR objective"""
    print("Testing make_weighted_rewards (RLVR)...")
    
    responses = [
        "<answer>(79 - 60) + 17</answer>",  # correct: +1
        "<answer>wrong</answer>",            # incorrect: -1
        "<answer>17 + (79 - 60)</answer>",  # correct: +1
        "<answer>bad</answer>",              # incorrect: -1
    ]
    ground_truths = [{"target": 36, "numbers": [79, 60, 17]}] * 4
    
    def mock_reward_fn(text, gt):
        return reward_fn(text, gt)
    
    weighted_rewards, keep_mask = make_weighted_rewards(
        responses, ground_truths, mock_reward_fn, 
        RLObjective.RLVR, lambda_psr=0.1
    )
    
    # RLVR: all samples kept, rewards unchanged
    assert keep_mask.sum() == 4, f"RLVR should keep all 4 samples, kept {keep_mask.sum()}"
    assert torch.all(keep_mask == True), "RLVR should keep all samples"
    
    # Check rewards are ±1
    expected = torch.tensor([1.0, -1.0, 1.0, -1.0])
    assert torch.allclose(weighted_rewards, expected), \
        f"Expected {expected}, got {weighted_rewards}"
    
    print("✅ make_weighted_rewards (RLVR) tests passed")


def test_make_weighted_rewards_nsr():
    """Test make_weighted_rewards with NSR objective"""
    print("Testing make_weighted_rewards (NSR)...")
    
    responses = [
        "<answer>(79 - 60) + 17</answer>",  # correct: should be filtered out
        "<answer>wrong</answer>",            # incorrect: kept
        "<answer>17 + (79 - 60)</answer>",  # correct: should be filtered out
        "<answer>bad</answer>",              # incorrect: kept
    ]
    ground_truths = [{"target": 36, "numbers": [79, 60, 17]}] * 4
    
    weighted_rewards, keep_mask = make_weighted_rewards(
        responses, ground_truths, reward_fn,
        RLObjective.NSR, lambda_psr=0.1
    )
    
    # NSR: only incorrect samples kept
    assert keep_mask.sum() == 2, f"NSR should keep 2 incorrect samples, kept {keep_mask.sum()}"
    expected_mask = torch.tensor([False, True, False, True])
    assert torch.all(keep_mask == expected_mask), \
        f"Expected mask {expected_mask}, got {keep_mask}"
    
    # Check rewards are all -1 for kept samples
    expected = torch.tensor([-1.0, -1.0])
    assert torch.allclose(weighted_rewards, expected), \
        f"NSR rewards should all be -1, got {weighted_rewards}"
    
    print("✅ make_weighted_rewards (NSR) tests passed")


def test_make_weighted_rewards_psr():
    """Test make_weighted_rewards with PSR objective"""
    print("Testing make_weighted_rewards (PSR)...")
    
    responses = [
        "<answer>(79 - 60) + 17</answer>",  # correct: kept
        "<answer>wrong</answer>",            # incorrect: filtered out
        "<answer>17 + (79 - 60)</answer>",  # correct: kept
        "<answer>bad</answer>",              # incorrect: filtered out
    ]
    ground_truths = [{"target": 36, "numbers": [79, 60, 17]}] * 4
    
    weighted_rewards, keep_mask = make_weighted_rewards(
        responses, ground_truths, reward_fn,
        RLObjective.PSR, lambda_psr=0.1
    )
    
    # PSR: only correct samples kept
    assert keep_mask.sum() == 2, f"PSR should keep 2 correct samples, kept {keep_mask.sum()}"
    expected_mask = torch.tensor([True, False, True, False])
    assert torch.all(keep_mask == expected_mask), \
        f"Expected mask {expected_mask}, got {keep_mask}"
    
    # Check rewards are all +1 for kept samples
    expected = torch.tensor([1.0, 1.0])
    assert torch.allclose(weighted_rewards, expected), \
        f"PSR rewards should all be +1, got {weighted_rewards}"
    
    print("✅ make_weighted_rewards (PSR) tests passed")


def test_make_weighted_rewards_w_reinforce():
    """Test make_weighted_rewards with W_REINFORCE objective"""
    print("Testing make_weighted_rewards (W-REINFORCE)...")
    
    responses = [
        "<answer>(79 - 60) + 17</answer>",  # correct: +λ
        "<answer>wrong</answer>",            # incorrect: -1
        "<answer>17 + (79 - 60)</answer>",  # correct: +λ
        "<answer>bad</answer>",              # incorrect: -1
    ]
    ground_truths = [{"target": 36, "numbers": [79, 60, 17]}] * 4
    
    lambda_psr = 0.1
    weighted_rewards, keep_mask = make_weighted_rewards(
        responses, ground_truths, reward_fn,
        RLObjective.W_REINFORCE, lambda_psr=lambda_psr
    )
    
    # W-REINFORCE: all samples kept
    assert keep_mask.sum() == 4, f"W-REINFORCE should keep all 4 samples, kept {keep_mask.sum()}"
    
    # Check rewards: +λ for correct, -1 for incorrect
    expected = torch.tensor([lambda_psr, -1.0, lambda_psr, -1.0])
    assert torch.allclose(weighted_rewards, expected), \
        f"Expected {expected}, got {weighted_rewards}"
    
    print("✅ make_weighted_rewards (W-REINFORCE) tests passed")


def test_compute_group_normalized_advantages_precomputed():
    """Test compute_group_normalized_advantages with precomputed_rewards"""
    print("Testing compute_group_normalized_advantages with precomputed_rewards...")
    
    responses = ["dummy"] * 8  # Not used when precomputed_rewards provided
    ground_truths = [{"target": 36, "numbers": [79, 60, 17]}] * 8
    
    # Precomputed rewards
    precomputed = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0])
    
    advantages, raw_rewards, metadata = compute_group_normalized_advantages(
        responses, ground_truths, reward_fn,
        group_size=4, advantage_eps=1e-6, normalize_by_std=True,
        precomputed_rewards=precomputed
    )
    
    # Check that precomputed rewards were used
    assert torch.allclose(raw_rewards, precomputed), \
        "Should use precomputed rewards"
    
    # Check advantages shape
    assert advantages.shape == (8,), f"Expected shape (8,), got {advantages.shape}"
    
    # Check metadata
    assert "mean" in metadata
    assert "std" in metadata
    assert "max" in metadata
    assert "min" in metadata
    
    print("✅ compute_group_normalized_advantages with precomputed_rewards tests passed")


def test_edge_cases():
    """Test edge cases and error handling"""
    print("Testing edge cases...")
    
    # Empty responses
    try:
        weighted_rewards, keep_mask = make_weighted_rewards(
            [], [], reward_fn, RLObjective.RLVR, lambda_psr=0.1
        )
        assert len(weighted_rewards) == 0
        assert len(keep_mask) == 0
        print("  ✓ Empty responses handled")
    except Exception as e:
        print(f"  ⚠ Empty responses error: {e}")
    
    # All correct samples with NSR (should filter all)
    responses = ["<answer>(79 - 60) + 17</answer>"] * 4
    ground_truths = [{"target": 36, "numbers": [79, 60, 17]}] * 4
    weighted_rewards, keep_mask = make_weighted_rewards(
        responses, ground_truths, reward_fn,
        RLObjective.NSR, lambda_psr=0.1
    )
    assert keep_mask.sum() == 0, "NSR with all correct should keep 0 samples"
    print("  ✓ NSR with all correct handled")
    
    # All incorrect samples with PSR (should filter all)
    responses = ["<answer>wrong</answer>"] * 4
    weighted_rewards, keep_mask = make_weighted_rewards(
        responses, ground_truths, reward_fn,
        RLObjective.PSR, lambda_psr=0.1
    )
    assert keep_mask.sum() == 0, "PSR with all incorrect should keep 0 samples"
    print("  ✓ PSR with all incorrect handled")
    
    print("✅ Edge cases tests passed")


def run_all_tests():
    """Run all test suites"""
    print("\n" + "="*70)
    print("RUNNING TESTS FOR starter_rlvr_decomposed.py")
    print("="*70 + "\n")
    
    tests = [
        test_extract_answer,
        test_validate_numbers,
        test_evaluate_equation,
        test_reward_fn_binary,
        test_rlobjective_enum,
        test_make_weighted_rewards_rlvr,
        test_make_weighted_rewards_nsr,
        test_make_weighted_rewards_psr,
        test_make_weighted_rewards_w_reinforce,
        test_compute_group_normalized_advantages_precomputed,
        test_edge_cases,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
            print()
        except AssertionError as e:
            print(f"❌ FAILED: {e}\n")
            failed += 1
        except Exception as e:
            print(f"❌ ERROR: {e}\n")
            failed += 1
    
    print("="*70)
    print(f"TEST SUMMARY: {passed}/{len(tests)} tests passed")
    if failed == 0:
        print("✅ ALL TESTS PASSED!")
    else:
        print(f"⚠️  {failed} tests failed")
    print("="*70 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

