"""REINFORCE++ baseline: per-group mean baseline for groups of size > 1, baseline 0 for size-1 groups."""

import numpy as np

from rllm.trainer.algorithms.advantage import calculate_reinforce_plus_plus_baseline_advantages
from rllm.trainer.algorithms.config import AlgorithmConfig


def _advantages(rewards):
    advantages, returns = calculate_reinforce_plus_plus_baseline_advantages([np.asarray(r, dtype=float) for r in rewards], AlgorithmConfig())
    assert [a.shape for a in advantages] == [np.asarray(r).shape for r in rewards]
    assert all(np.array_equal(a, r) for a, r in zip(advantages, returns, strict=True))
    return advantages


def test_size_one_groups_keep_a_learning_signal():
    # With one trajectory per group the baseline is 0, so rewards must not be centered to zero.
    advantages = _advantages([[1.0], [0.0], [1.0]])
    flat = np.concatenate(advantages)

    assert np.any(np.abs(flat) > 1e-6)
    assert flat[0] > 0 > flat[1]
    assert flat[0] == flat[2]
    np.testing.assert_allclose(flat.mean(), 0.0, atol=1e-6)
    np.testing.assert_allclose(flat.std(), 1.0, atol=1e-4)


def test_size_one_groups_with_equal_rewards_do_not_blow_up():
    flat = np.concatenate(_advantages([[1.0], [1.0], [1.0]]))
    np.testing.assert_allclose(flat, 0.0, atol=1e-6)


def test_mixed_group_sizes_whiten_over_the_role_batch():
    # Size-1 group: centered value 1.0 (baseline 0). Size-2 group: centered by its own mean.
    flat = np.concatenate(_advantages([[1.0], [0.0, 1.0]]))
    centered = np.array([1.0, -0.5, 0.5])
    expected = (centered - centered.mean()) / (centered.std() + 1e-6)
    np.testing.assert_allclose(flat, expected, atol=1e-6)


def test_groups_larger_than_one_are_unchanged():
    rewards = [[1.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 0.0]]
    centered = np.concatenate([np.asarray(r) - np.mean(r) for r in rewards])
    expected = centered / (centered.std() + 1e-6)
    np.testing.assert_allclose(np.concatenate(_advantages(rewards)), expected, atol=1e-9)
