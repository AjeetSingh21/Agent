"""Tests for quota handling.

A full eval run exhausted the free tier's 200k tokens/day partway through. The
remaining goals all returned empty answers and were scored as agent failures,
dropping a genuine 10/10 to a meaningless 5/10. These tests pin down the
distinction between "out of quota" and "did badly".
"""

import pytest

from src.agent.llm import RateLimitedError, _as_rate_limit

TPD_MESSAGE = (
    "Error code: 429 - {'error': {'message': 'Rate limit reached for model "
    "`openai/gpt-oss-120b` in organization `org_abc` service tier `on_demand` on "
    "tokens per day (TPD): Limit 200000, Used 194831, Requested 6205. Please try "
    "again in 7m27.551999999s.', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}"
)

RPM_MESSAGE = (
    "Error code: 429 - {'error': {'code': 'rate_limit_exceeded', 'message': "
    "'Rate limit reached on requests per minute. Please try again in 12.5s'}}"
)


class TestDetection:
    def test_daily_token_limit_is_recognised(self):
        err = _as_rate_limit(Exception(TPD_MESSAGE))
        assert isinstance(err, RateLimitedError)
        assert "daily token budget" in str(err)

    def test_per_minute_limit_is_recognised_but_not_labelled_daily(self):
        err = _as_rate_limit(Exception(RPM_MESSAGE))
        assert isinstance(err, RateLimitedError)
        assert "daily token budget" not in str(err)

    def test_unrelated_errors_are_not_treated_as_rate_limits(self):
        assert _as_rate_limit(ValueError("model does not exist")) is None
        assert _as_rate_limit(ConnectionError("network unreachable")) is None

    def test_a_404_is_not_a_rate_limit(self):
        assert _as_rate_limit(Exception("Error code: 404 - model_not_found")) is None


class TestRetryAfterParsing:
    def test_minutes_and_seconds_are_combined(self):
        err = _as_rate_limit(Exception(TPD_MESSAGE))
        assert err.retry_after_s == pytest.approx(447.55, abs=0.1)

    def test_bare_seconds_are_parsed(self):
        err = _as_rate_limit(Exception(RPM_MESSAGE))
        assert err.retry_after_s == pytest.approx(12.5, abs=0.1)

    def test_missing_retry_hint_is_tolerated(self):
        err = _as_rate_limit(Exception("Error code: 429 rate_limit_exceeded"))
        assert err is not None
        assert err.retry_after_s is None


class TestPropagation:
    """A rate limit must escape the nodes, not be swallowed into a trail entry."""

    def test_planner_reraises_rate_limit(self, monkeypatch):
        import src.agent.nodes as nodes

        def boom(*args, **kwargs):
            raise RateLimitedError("quota gone", 60.0)

        monkeypatch.setattr(nodes, "call_structured", boom)
        with pytest.raises(RateLimitedError):
            nodes.plan_node({"goal": "anything", "trail": []})

    def test_planner_still_swallows_ordinary_errors(self, monkeypatch):
        """Non-quota failures should degrade gracefully, as before."""
        import src.agent.nodes as nodes

        def boom(*args, **kwargs):
            raise ValueError("bad json")

        monkeypatch.setattr(nodes, "call_structured", boom)
        out = nodes.plan_node({"goal": "anything", "trail": []})

        assert "Planning failed" in out["error"]
        assert out["trail"][0]["ok"] is False

    def test_synthesis_reraises_rate_limit(self, monkeypatch):
        import src.agent.nodes as nodes

        def boom(*args, **kwargs):
            raise RateLimitedError("quota gone")

        monkeypatch.setattr(nodes, "call_text", boom)
        state = {
            "goal": "g",
            "trail": [],
            "sources": [{"ref": 1, "title": "t", "url": "u", "snippet": "s"}],
        }
        with pytest.raises(RateLimitedError):
            nodes.synthesize_node(state)


class TestUpstreamErrorIsPreserved:
    """Synthesis must not relabel an upstream failure as a search problem."""

    def test_upstream_cause_is_kept(self):
        import src.agent.nodes as nodes

        out = nodes.synthesize_node(
            {"goal": "g", "trail": [], "sources": [], "error": "Planning failed: boom"}
        )
        assert out["error"] == "Planning failed: boom"
        assert "boom" in out["answer"]

    def test_genuine_search_failure_still_blames_search(self):
        import src.agent.nodes as nodes

        out = nodes.synthesize_node({"goal": "g", "trail": [], "sources": []})
        assert "search engine" in out["answer"]
