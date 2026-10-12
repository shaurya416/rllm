"""Tests for the IFEval reward function and instruction verifiers.

The class-based ``IFEvalEvaluator`` was inlined into
``rllm.eval.reward_fns.ifeval`` as a plain ``evaluate(task, episode)``
function. ``load_evaluator`` returns an adapter that accepts the
legacy dict-call form.
"""

from __future__ import annotations

import pytest

from rllm.eval.evaluator_loader import load_evaluator
from rllm.eval.reward_fns.ifeval import _verify_instruction as verify_instruction
from rllm.types import Episode, Evaluator

# ---------------------------------------------------------------------------
# Instruction verification dispatcher
# ---------------------------------------------------------------------------


class TestVerifyInstruction:
    def test_keywords_existence_pass(self):
        assert (
            verify_instruction(
                "keywords:existence",
                "Hello world, this is a test",
                {"keywords": ["hello", "world"]},
            )
            is True
        )

    def test_keywords_existence_fail(self):
        assert (
            verify_instruction(
                "keywords:existence",
                "Hello world",
                {"keywords": ["hello", "missing"]},
            )
            is False
        )

    def test_keywords_forbidden_pass(self):
        assert (
            verify_instruction(
                "keywords:forbidden_words",
                "Hello world",
                {"forbidden_words": ["bad", "evil"]},
            )
            is True
        )

    def test_keywords_forbidden_fail(self):
        assert (
            verify_instruction(
                "keywords:forbidden_words",
                "Hello bad world",
                {"forbidden_words": ["bad"]},
            )
            is False
        )

    def test_length_number_words_at_least(self):
        response = " ".join(["word"] * 50)
        assert (
            verify_instruction(
                "length_constraints:number_words",
                response,
                {"num_words": 50, "relation": "at least"},
            )
            is True
        )

    def test_length_number_words_at_most_fail(self):
        response = " ".join(["word"] * 50)
        assert (
            verify_instruction(
                "length_constraints:number_words",
                response,
                {"num_words": 10, "relation": "at most"},
            )
            is False
        )

    def test_change_case_lowercase(self):
        assert (
            verify_instruction(
                "change_case:english_lowercase",
                "hello world 123",
                {},
            )
            is True
        )

    def test_change_case_lowercase_fail(self):
        assert (
            verify_instruction(
                "change_case:english_lowercase",
                "Hello World",
                {},
            )
            is False
        )

    def test_change_case_uppercase(self):
        assert (
            verify_instruction(
                "change_case:english_uppercase",
                "HELLO WORLD 123",
                {},
            )
            is True
        )

    def test_json_format(self):
        assert (
            verify_instruction(
                "detectable_format:json_format",
                '{"key": "value"}',
                {},
            )
            is True
        )

    def test_json_format_in_code_block(self):
        assert (
            verify_instruction(
                "detectable_format:json_format",
                'Here is the JSON:\n```json\n{"key": "value"}\n```',
                {},
            )
            is True
        )

    def test_no_comma(self):
        assert (
            verify_instruction(
                "punctuation:no_comma",
                "Hello world. This is great.",
                {},
            )
            is True
        )

    def test_no_comma_fail(self):
        assert (
            verify_instruction(
                "punctuation:no_comma",
                "Hello, world",
                {},
            )
            is False
        )

    def test_title(self):
        assert (
            verify_instruction(
                "detectable_format:title",
                "<<My Title>>\nContent here",
                {},
            )
            is True
        )

    def test_postscript(self):
        assert (
            verify_instruction(
                "detectable_content:postscript",
                "Main content\n\nP.S. Don't forget!",
                {},
            )
            is True
        )

    def test_unknown_instruction(self):
        # Unknown instructions pass (lenient mode)
        assert verify_instruction("unknown:type", "response", {}) is True

    def test_end_checker(self):
        assert (
            verify_instruction(
                "startend:end_checker",
                "Some content ending with goodbye",
                {"end_phrase": "goodbye"},
            )
            is True
        )

    def test_keywords_frequency(self):
        assert (
            verify_instruction(
                "keywords:frequency",
                "cat cat cat dog",
                {"keyword": "cat", "frequency": 3, "relation": "at least"},
            )
            is True
        )

    # IFEval's own comparison relations are "less than" and "at least".
    @pytest.mark.parametrize(
        ("instruction_id", "response", "kwargs"),
        [
            ("length_constraints:number_words", "one two three", {"num_words": 50, "relation": "less than"}),
            ("length_constraints:number_sentences", "One. Two.", {"num_sentences": 3, "relation": "less than"}),
            ("keywords:frequency", "no felines here", {"keyword": "cat", "frequency": 2, "relation": "less than"}),
            ("keywords:letter_frequency", "abc", {"letter": "z", "let_frequency": 5, "let_relation": "less than"}),
        ],
    )
    def test_less_than_relation_can_be_satisfied(self, instruction_id, response, kwargs):
        assert verify_instruction(instruction_id, response, kwargs) is True

    def test_less_than_relation_is_strict(self):
        assert verify_instruction("length_constraints:number_words", "one two three", {"num_words": 3, "relation": "less than"}) is False

    @pytest.mark.parametrize("response", ["", "  \n"])
    def test_empty_response_follows_no_instruction(self, response):
        assert verify_instruction("punctuation:no_comma", response, {}) is False
        assert verify_instruction("keywords:forbidden_words", response, {"forbidden_words": ["bad"]}) is False

    # "Your entire response should be in English, and in all capital letters."
    def test_english_capital_requires_all_capital_letters(self):
        assert verify_instruction("change_case:english_capital", "Hello World", {}) is False
        assert verify_instruction("change_case:english_capital", "HELLO WORLD 123", {}) is True

    # Official IFEval instruction IDs must be checked, not passed as unknown.
    @pytest.mark.parametrize(
        ("instruction_id", "failing", "passing", "kwargs"),
        [
            ("startend:quotation", "Hello there", '"Hello there"', {}),
            (
                "change_case:capital_word_frequency",
                "This is VERY important",
                "THIS is VERY important",
                {"capital_frequency": 2, "capital_relation": "at least"},
            ),
            (
                "change_case:capital_word_frequency",
                "THIS is VERY IMPORTANT",
                "This is VERY important",
                {"capital_frequency": 2, "capital_relation": "less than"},
            ),
            (
                "length_constraints:nth_paragraph_first_word",
                "Alpha one.\n\nGamma two.",
                "Alpha one.\n\nBeta, two.",
                {"num_paragraphs": 2, "nth_paragraph": 2, "first_word": "beta"},
            ),
            (
                "length_constraints:nth_paragraph_first_word",
                "Alpha one.\n\nBeta two.\n\nGamma three.",
                "Alpha one.\n\nBeta two.",
                {"num_paragraphs": 2, "nth_paragraph": 2, "first_word": "beta"},
            ),
        ],
    )
    def test_official_instruction_is_verified(self, instruction_id, failing, passing, kwargs):
        assert verify_instruction(instruction_id, failing, kwargs) is False
        assert verify_instruction(instruction_id, passing, kwargs) is True

    # Checked the way the official IFEval checkers check them.
    @pytest.mark.parametrize(
        ("instruction_id", "failing", "passing", "kwargs"),
        [
            # "Answer with one of the following options: My answer is yes./no./maybe."
            ("detectable_format:constrained_response", "Banana", "My answer is yes.", {}),
            # A plain P.S. is not the P.P.S the instruction asks for.
            (
                "detectable_content:postscript",
                "Hi.\n\nP.S. Bye.",
                "Hi.\n\nP.P.S. Bye.",
                {"postscript_marker": "P.P.S"},
            ),
            # "Highlight ... with markdown, i.e. *highlighted section*"; headings are not highlights.
            (
                "detectable_format:number_highlighted_sections",
                "# Intro\n## Body\ntext",
                "*Part one* and **part two**",
                {"num_highlights": 2},
            ),
            # "placeholders represented by square brackets, such as [address]"
            (
                "detectable_content:number_placeholders",
                "Meet at [ADDRESS].",
                "Meet at [address] on [date].",
                {"num_placeholders": 2},
            ),
        ],
    )
    def test_matches_official_checker(self, instruction_id, failing, passing, kwargs):
        assert verify_instruction(instruction_id, failing, kwargs) is False
        assert verify_instruction(instruction_id, passing, kwargs) is True


# ---------------------------------------------------------------------------
# ifeval_reward_fn (loaded via the registry adapter)
# ---------------------------------------------------------------------------


class TestIFEvalEvaluator:
    def test_all_pass(self):
        evaluator = load_evaluator("ifeval_reward_fn")
        task = {
            "instruction_id_list": ["keywords:existence", "punctuation:no_comma"],
            "kwargs": [{"keywords": ["hello"]}, {}],
        }
        ep = Episode(artifacts={"answer": "hello world"})
        result = evaluator.evaluate(task, ep)
        assert result.is_correct is True
        assert result.reward == 1.0
        assert any(s.name == "strict_accuracy" and s.value == 1.0 for s in result.signals)

    def test_partial_pass(self):
        evaluator = load_evaluator("ifeval_reward_fn")
        task = {
            "instruction_id_list": ["keywords:existence", "punctuation:no_comma"],
            "kwargs": [{"keywords": ["hello"]}, {}],
        }
        ep = Episode(artifacts={"answer": "hello, world"})
        result = evaluator.evaluate(task, ep)
        assert result.is_correct is False
        assert result.reward == 0.0
        # Loose accuracy should be 0.5 (1 of 2 passed)
        loose = next(s for s in result.signals if s.name == "loose_accuracy")
        assert loose.value == pytest.approx(0.5)

    @pytest.mark.parametrize("episode", [Episode(artifacts={"answer": ""}), Episode()])
    def test_empty_response_is_not_correct(self, episode):
        evaluator = load_evaluator("ifeval_reward_fn")
        task = {
            "instruction_id_list": ["punctuation:no_comma", "keywords:forbidden_words", "change_case:english_lowercase"],
            "kwargs": [{}, {"forbidden_words": ["bad"]}, {}],
        }
        result = evaluator.evaluate(task, episode)
        assert result.is_correct is False
        assert result.reward == 0.0
        loose = next(s for s in result.signals if s.name == "loose_accuracy")
        assert loose.value == 0.0

    def test_no_instructions(self):
        evaluator = load_evaluator("ifeval_reward_fn")
        task = {"instruction_id_list": [], "kwargs": []}
        ep = Episode(artifacts={"answer": "anything"})
        result = evaluator.evaluate(task, ep)
        assert result.is_correct is True

    def test_is_evaluator(self):
        evaluator = load_evaluator("ifeval_reward_fn")
        assert isinstance(evaluator, Evaluator)

    def test_metadata_has_results(self):
        evaluator = load_evaluator("ifeval_reward_fn")
        task = {
            "instruction_id_list": ["keywords:existence"],
            "kwargs": [{"keywords": ["test"]}],
        }
        ep = Episode(artifacts={"answer": "test response"})
        result = evaluator.evaluate(task, ep)
        assert "instruction_results" in result.metadata
        assert result.metadata["instruction_results"]["keywords:existence"] is True
