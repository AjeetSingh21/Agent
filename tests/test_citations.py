"""Regression tests for citation handling.

A real eval run scored three correctly-cited answers as having zero citations,
because GPT-OSS emits 【1】 / 【15†L15-L19】 rather than [1] and both the agent's
output and the scorer's regex assumed ASCII brackets. These tests pin down both
halves of that fix.
"""

from eval.run_eval import find_citations, score_run
from src.agent.nodes import _normalize_citations


class TestNormalisation:
    def test_plain_lenticular_becomes_ascii(self):
        assert _normalize_citations("Toyota leads 【1】.") == "Toyota leads [1]."

    def test_dagger_line_range_form_is_stripped(self):
        assert _normalize_citations("RAG wins【15†L15-L19】.") == "RAG wins[15]."

    def test_consecutive_citations_all_convert(self):
        assert _normalize_citations("【5】【6】【7】") == "[5][6][7]"

    def test_mixed_forms_in_one_answer(self):
        text = "A [1] and B 【2】 and C 【16†L1-L4】"
        assert _normalize_citations(text) == "A [1] and B [2] and C [16]"

    def test_ascii_citations_pass_through_untouched(self):
        assert _normalize_citations("Already fine [3][4].") == "Already fine [3][4]."

    def test_three_digit_refs_supported(self):
        assert _normalize_citations("【123】") == "[123]"

    def test_text_without_citations_is_unchanged(self):
        assert _normalize_citations("No citations here.") == "No citations here."


class TestFindCitations:
    def test_finds_ascii_form(self):
        assert find_citations("a [1] b [2]") == {1, 2}

    def test_finds_lenticular_form(self):
        assert find_citations("a 【1】 b 【2】") == {1, 2}

    def test_finds_dagger_form(self):
        assert find_citations("a 【15†L15-L19】") == {15}

    def test_finds_mixed_forms_together(self):
        assert find_citations("[1] 【2】 【33†L1-L4】") == {1, 2, 33}

    def test_deduplicates_repeated_refs(self):
        assert find_citations("[1] [1] 【1】") == {1}

    def test_empty_when_no_citations(self):
        assert find_citations("plain prose") == set()


class TestScoring:
    @staticmethod
    def _result(answer, n_sources=6):
        return {
            "answer": answer,
            "sources": [{"ref": i} for i in range(1, n_sources + 1)],
            "error": None,
        }

    def test_lenticular_citations_now_pass_the_cited_check(self):
        """The exact failure mode from the first full eval run."""
        answer = "Toyota leads 【1】. Density is rising 【2】【3】. Costs remain high 【4】."
        score = score_run(self._result(answer), {"must_mention": []}, {})

        assert score["checks"]["cited"]
        assert score["citations"] == 4
        assert score["success"]

    def test_invented_reference_still_fails(self):
        """Normalisation must not weaken hallucination detection."""
        answer = "Claim 【1】, claim 【2】, claim 【3】, invented 【99】."
        score = score_run(self._result(answer, n_sources=6), {"must_mention": []}, {})

        assert not score["checks"]["cited"]
        assert score["hallucinated_refs"] == [99]

    def test_too_few_citations_fails(self):
        score = score_run(self._result("Only one 【1】."), {"must_mention": []}, {})
        assert not score["checks"]["cited"]

    def test_missing_required_term_fails_relevance(self):
        answer = "Claim [1], claim [2], claim [3]."
        score = score_run(self._result(answer), {"must_mention": ["batteries"]}, {})

        assert not score["checks"]["relevant"]
        assert score["missing_terms"] == ["batteries"]

    def test_empty_answer_fails_completion(self):
        score = score_run(self._result(""), {"must_mention": []}, {})
        assert not score["checks"]["completed"]

    def test_too_few_sources_fails(self):
        answer = "Claim [1], claim [2], claim [3]."
        score = score_run(self._result(answer, n_sources=2), {"must_mention": []}, {})
        assert not score["checks"]["enough_sources"]
