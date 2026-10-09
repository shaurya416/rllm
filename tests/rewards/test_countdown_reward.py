import pytest

from rllm.rewards.countdown_reward import compute_score, countdown_reward_fn, evaluate_equation


class TestCountdownReward:
    """Countdown answers may only combine the given numbers with +, -, *, /."""

    def test_correct_equation_gets_full_score(self):
        assert compute_score("<answer>(25 + 3) * 2</answer>", {"target": 56, "numbers": [25, 3, 2]}) == 1.0

    def test_correct_equation_with_division(self):
        assert compute_score("<answer>(8 - 2) / 3</answer>", {"target": 2, "numbers": [8, 2, 3]}) == 1.0

    @pytest.mark.parametrize(
        "equation, numbers, target",
        [
            # Exponentiation is not one of the allowed operations.
            ("2**3", [2, 3], 8),
            # Floor division is not one of the allowed operations.
            ("7//2", [7, 2], 3),
            # Joining two given digits into a decimal literal invents a number that was never provided.
            ("1.5*2", [1, 5, 2], 3),
        ],
    )
    def test_disallowed_construct_is_not_scored_correct(self, equation, numbers, target):
        score = compute_score(f"<answer>{equation}</answer>", {"target": target, "numbers": numbers})
        assert score < 1.0

        output = countdown_reward_fn({"target": target, "nums": numbers}, f"<answer>{equation}</answer>")
        assert output.is_correct is False
        assert output.reward == 0.0

    @pytest.mark.parametrize("equation", ["2**10", "7//2", "1.5*2"])
    def test_evaluate_equation_rejects_disallowed_constructs(self, equation):
        # These never reach eval(); an exponent tower such as 9**9**9 would otherwise never finish.
        assert evaluate_equation(equation) is None

    def test_evaluate_equation_allows_basic_arithmetic(self):
        assert evaluate_equation("(25 + 3) * 2") == 56
        assert evaluate_equation("-3 + 10 / 2") == 2.0
