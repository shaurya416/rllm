"""Tests for agentic benchmarks: BFCL and LLM judge reward functions.

After the inlining refactor, the per-task scorers live as plain
``evaluate(task, episode)`` functions in ``rllm.eval.reward_fns.*``.
``load_evaluator(name)`` returns a thin wrapper that auto-adapts the
legacy dict-call form used by these tests.
"""

from __future__ import annotations

import json

from rllm.eval.evaluator_loader import load_evaluator
from rllm.eval.reward_fns.bfcl import _compare_function_calls
from rllm.types import Episode, Evaluator

# ---------------------------------------------------------------------------
# bfcl_reward_fn
# ---------------------------------------------------------------------------


class TestBFCLEvaluator:
    def test_correct_function_call(self):
        evaluator = load_evaluator("bfcl_reward_fn")
        task = {
            "ground_truth": [json.dumps({"name": "get_weather", "arguments": {"city": "Paris"}})],
        }
        ep = Episode(
            artifacts={
                "answer": "",
                "tool_calls": [{"name": "get_weather", "arguments": '{"city": "Paris"}'}],
            }
        )
        result = evaluator.evaluate(task, ep)
        assert result.is_correct is True
        assert result.reward == 1.0

    def test_wrong_function_name(self):
        evaluator = load_evaluator("bfcl_reward_fn")
        task = {
            "ground_truth": [json.dumps({"name": "get_weather", "arguments": {"city": "Paris"}})],
        }
        ep = Episode(
            artifacts={
                "answer": "",
                "tool_calls": [{"name": "get_time", "arguments": '{"city": "Paris"}'}],
            }
        )
        result = evaluator.evaluate(task, ep)
        assert result.is_correct is False

    def test_no_model_calls(self):
        evaluator = load_evaluator("bfcl_reward_fn")
        task = {
            "ground_truth": [json.dumps({"name": "func", "arguments": {}})],
        }
        ep = Episode(artifacts={"answer": "", "tool_calls": []})
        result = evaluator.evaluate(task, ep)
        assert result.is_correct is False

    def test_no_ground_truth(self):
        evaluator = load_evaluator("bfcl_reward_fn")
        task = {"ground_truth": []}
        # No call is expected, so making one is wrong and making none is right.
        ep = Episode(artifacts={"answer": "", "tool_calls": [{"name": "f", "arguments": "{}"}]})
        assert evaluator.evaluate(task, ep).is_correct is False
        ep = Episode(artifacts={"answer": "", "tool_calls": []})
        assert evaluator.evaluate(task, ep).is_correct is True

    def test_enumerating_argument_guesses_is_not_rewarded(self):
        evaluator = load_evaluator("bfcl_reward_fn")
        task = {"ground_truth": [json.dumps({"name": "calc", "arguments": {"n": 20, "k": 5}})]}
        guesses = [{"name": "calc", "arguments": {"n": 20, "k": k}} for k in range(21)]
        result = evaluator.evaluate(task, Episode(artifacts={"answer": json.dumps(guesses)}))
        assert result.is_correct is False
        assert result.reward == 0.0
        assert result.metadata["reason"] == "wrong_number_of_calls"
        assert next(s.value for s in result.signals if s.name == "ast_accuracy") < 0.1

    def test_non_object_call_in_answer_is_scored_not_raised(self):
        evaluator = load_evaluator("bfcl_reward_fn")
        task = {"ground_truth": [json.dumps({"name": "f", "arguments": {"a": 1}})]}
        result = evaluator.evaluate(task, Episode(artifacts={"answer": "[1]"}))
        assert result.is_correct is False

    def test_is_evaluator(self):
        evaluator = load_evaluator("bfcl_reward_fn")
        assert isinstance(evaluator, Evaluator)

    def test_signals_present(self):
        evaluator = load_evaluator("bfcl_reward_fn")
        task = {"ground_truth": []}
        ep = Episode(artifacts={"answer": "", "tool_calls": []})
        result = evaluator.evaluate(task, ep)
        assert any(s.name == "ast_accuracy" for s in result.signals)


# ---------------------------------------------------------------------------
# _compare_function_calls
# ---------------------------------------------------------------------------


class TestCompareFunctionCalls:
    def test_exact_match(self):
        model = [{"name": "f", "arguments": {"a": 1}}]
        gt = [json.dumps({"name": "f", "arguments": {"a": 1}})]
        is_correct, _ = _compare_function_calls(model, gt)
        assert is_correct is True

    def test_no_match(self):
        model = [{"name": "f", "arguments": {"a": 1}}]
        gt = [json.dumps({"name": "g", "arguments": {"a": 1}})]
        is_correct, _ = _compare_function_calls(model, gt)
        assert is_correct is False

    def test_empty_model(self):
        gt = [json.dumps({"name": "f", "arguments": {}})]
        is_correct, _ = _compare_function_calls([], gt)
        assert is_correct is False

    def test_empty_gt(self):
        is_correct, _ = _compare_function_calls([{"name": "f"}], [])
        assert is_correct is False
        is_correct, _ = _compare_function_calls([], [])
        assert is_correct is True

    def test_one_call_cannot_satisfy_two_expected_calls(self):
        call = {"name": "f", "arguments": {"a": 1}}
        gt = [json.dumps(call), json.dumps(call)]
        is_correct, details = _compare_function_calls([call], gt)
        assert is_correct is False
        assert details["matched"] == 1
        is_correct, _ = _compare_function_calls([call, call], gt)
        assert is_correct is True

    def test_parallel_calls_in_any_order(self):
        gt = [json.dumps({"name": "f", "arguments": {}}), json.dumps({"name": "f", "arguments": {"a": 1}})]
        model = [{"name": "f", "arguments": {"a": 1}}, {"name": "f", "arguments": {"b": 2}}]
        is_correct, _ = _compare_function_calls(model, gt)
        assert is_correct is True


# ---------------------------------------------------------------------------
# llm_judge_reward_fn
# ---------------------------------------------------------------------------


class TestLLMJudgeEvaluator:
    def test_no_rubric_fallback(self):
        evaluator = load_evaluator("llm_judge_reward_fn")
        task = {"question": "Hello"}
        ep = Episode(artifacts={"answer": "Hi there!"})
        result = evaluator.evaluate(task, ep)
        # No rubric → nothing to judge against, so the answer is not graded correct
        assert result.is_correct is False
        assert result.reward == 0.0
        assert result.metadata.get("reason") == "no_rubric_available"
        assert result.metadata.get("ungraded") is True

    def test_empty_answer_no_rubric(self):
        evaluator = load_evaluator("llm_judge_reward_fn")
        task = {"question": "Hello"}
        ep = Episode(artifacts={"answer": ""})
        result = evaluator.evaluate(task, ep)
        assert result.is_correct is False

    def test_is_evaluator(self):
        evaluator = load_evaluator("llm_judge_reward_fn")
        assert isinstance(evaluator, Evaluator)

    def test_signals_present(self):
        evaluator = load_evaluator("llm_judge_reward_fn")
        task = {"question": "Hello"}
        ep = Episode(artifacts={"answer": "Hi"})
        result = evaluator.evaluate(task, ep)
        assert any(s.name == "judge_score" for s in result.signals)

    def test_with_rubric_no_judge(self):
        evaluator = load_evaluator("llm_judge_reward_fn")  # No judge_base_url
        task = {"question": "Hello", "rubric": "Should greet politely"}
        ep = Episode(artifacts={"answer": "Hi there!"})
        result = evaluator.evaluate(task, ep)
        # No judge available → ungraded, not correct
        assert result.metadata.get("reason") == "judge_unavailable_fallback"
        assert result.metadata.get("ungraded") is True
        assert result.is_correct is False
        assert result.reward == 0.0

    def test_failed_judge_call_does_not_pass_answer(self, monkeypatch):
        from rllm.eval.reward_fns import llm_judge

        # The judge is configured but the call fails (or its reply has no verdict).
        monkeypatch.setattr(llm_judge, "_call_judge", lambda *args, **kwargs: None)
        evaluator = load_evaluator("llm_judge_reward_fn")
        task = {"question": "Hello", "rubric": "Should greet politely", "judge_base_url": "http://judge.invalid/v1"}
        ep = Episode(artifacts={"answer": "Go away."})
        result = evaluator.evaluate(task, ep)
        assert result.is_correct is False
        assert result.reward == 0.0
        assert result.metadata.get("ungraded") is True
