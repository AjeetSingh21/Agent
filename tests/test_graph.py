"""Graph-level tests with the LLM and search backend stubbed out."""

import pytest

from src.agent import graph as graph_module
from src.agent.state import Critique, Plan
from src.agent.tools import SearchHit, ToolResult


@pytest.fixture
def fake_backend(monkeypatch):
    """Replace the LLM and web search so the graph runs offline and deterministically."""
    calls = {"structured": 0, "text": 0, "search": 0}

    def fake_structured(schema, system, user, temperature=None):
        calls["structured"] += 1
        if schema is Plan:
            return Plan(
                interpretation="Test interpretation.",
                sub_questions=[
                    {"id": 1, "question": "Q1", "search_query": "q1", "tool": "web_search"},
                    {"id": 2, "question": "Q2", "search_query": "q2", "tool": "web_search"},
                    {"id": 3, "question": "Q3", "search_query": "2+2", "tool": "calculator"},
                ],
            )
        return Critique(sufficient=True, gaps=[], follow_up_queries=[])

    def fake_text(system, user, temperature=None):
        calls["text"] += 1
        return "Findings [1][2].\n\n## Sources\n[1] a\n[2] b"

    def fake_search(query, max_results=None):
        calls["search"] += 1
        return ToolResult(
            tool="web_search:stub",
            ok=True,
            latency_s=0.01,
            hits=[
                SearchHit(f"{query} result A", f"https://example.com/{query}/a", "snippet A"),
                SearchHit(f"{query} result B", f"https://example.com/{query}/b", "snippet B"),
            ],
        )

    import src.agent.nodes as nodes

    monkeypatch.setattr(nodes, "call_structured", fake_structured)
    monkeypatch.setattr(nodes, "call_text", fake_text)
    monkeypatch.setattr(nodes, "web_search", fake_search)
    graph_module._GRAPH = None
    return calls


def test_agent_completes_and_produces_cited_answer(fake_backend):
    result = graph_module.run_agent("Test goal")

    assert result["answer"]
    assert result["error"] is None
    assert result["interpretation"] == "Test interpretation."
    assert len(result["sub_questions"]) == 3


def test_trail_is_ordered_and_covers_every_node(fake_backend):
    result = graph_module.run_agent("Test goal")
    trail = result["trail"]

    assert [s["n"] for s in trail] == list(range(1, len(trail) + 1))
    assert {s["node"] for s in trail} == {"plan", "act", "reflect", "synthesize"}


def test_calculator_sub_question_is_dispatched_to_the_calculator(fake_backend):
    result = graph_module.run_agent("Test goal")
    tools_used = {s["tool"] for s in result["trail"] if s["tool"]}

    assert "calculator" in tools_used
    assert any(s["snippet"] == "2+2 = 4" for s in result["sources"])


def test_duplicate_urls_are_deduplicated(fake_backend):
    result = graph_module.run_agent("Test goal")
    urls = [s["url"] for s in result["sources"] if s["url"] != "(calculator)"]

    assert len(urls) == len(set(urls))


def test_metrics_are_populated(fake_backend):
    metrics = graph_module.run_agent("Test goal")["metrics"]

    assert metrics["steps"] > 0
    assert metrics["tool_calls"] >= 3
    assert metrics["sources"] >= 4
    assert metrics["latency_s"] >= 0


def test_reflect_triggers_a_follow_up_round_when_gaps_exist(monkeypatch):
    """When the critique reports gaps, the graph loops back through act."""
    state = {"critique": {"sufficient": False}, "pending_queries": ["more"], "iteration": 0}
    assert graph_module._route_after_reflect(state) == "act"


def test_reflect_stops_once_the_iteration_budget_is_spent():
    from src.agent.config import CONFIG

    state = {
        "critique": {"sufficient": False},
        "pending_queries": ["more"],
        "iteration": CONFIG.max_iterations,
    }
    assert graph_module._route_after_reflect(state) == "synthesize"


def test_planner_failure_routes_straight_to_synthesis():
    assert graph_module._route_after_plan({"error": "boom"}) == "synthesize"
