"""LangGraph wiring.

    plan -> act -> reflect -> (act again | synthesize) -> END

Control flow lives in the graph edges rather than in model tool calls, which is
what makes the loop predictable on a small open-weights model.
"""

from __future__ import annotations

import time
from typing import Any, Literal

from langgraph.graph import END, StateGraph

from src.agent.config import CONFIG
from src.agent.nodes import act_node, plan_node, reflect_node, synthesize_node
from src.agent.state import AgentState


def _route_after_reflect(state: AgentState) -> Literal["act", "synthesize"]:
    """Loop back for another search round only if there is a real gap and budget left."""
    if state.get("error"):
        return "synthesize"
    critique = state.get("critique") or {}
    has_follow_ups = bool(state.get("pending_queries"))
    under_budget = state.get("iteration", 0) < CONFIG.max_iterations
    if not critique.get("sufficient", True) and has_follow_ups and under_budget:
        return "act"
    return "synthesize"


def _route_after_plan(state: AgentState) -> Literal["act", "synthesize"]:
    return "synthesize" if state.get("error") else "act"


def build_graph():
    """Compile the agent graph."""
    builder = StateGraph(AgentState)

    builder.add_node("plan", plan_node)
    builder.add_node("act", act_node)
    builder.add_node("reflect", reflect_node)
    builder.add_node("synthesize", synthesize_node)

    builder.set_entry_point("plan")
    builder.add_conditional_edges("plan", _route_after_plan, {"act": "act", "synthesize": "synthesize"})
    builder.add_edge("act", "reflect")
    builder.add_conditional_edges("reflect", _route_after_reflect, {"act": "act", "synthesize": "synthesize"})
    builder.add_edge("synthesize", END)

    return builder.compile()


# Compiling is cheap but not free; reuse one instance per process.
_GRAPH = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def run_agent(goal: str) -> dict[str, Any]:
    """Run the agent to completion and return the final state plus run metrics."""
    started = time.perf_counter()
    final: AgentState = get_graph().invoke({"goal": goal, "trail": [], "iteration": 0})
    elapsed = time.perf_counter() - started

    trail = final.get("trail", [])
    return {
        "goal": goal,
        "interpretation": final.get("interpretation", ""),
        "sub_questions": final.get("sub_questions", []),
        "sources": final.get("sources", []),
        "trail": trail,
        "answer": final.get("answer", ""),
        "error": final.get("error"),
        "metrics": {
            "steps": len(trail),
            "tool_calls": sum(1 for s in trail if s.get("tool")),
            "failed_steps": sum(1 for s in trail if not s.get("ok")),
            "sources": len(final.get("sources", [])),
            "search_rounds": final.get("iteration", 0),
            "latency_s": round(elapsed, 2),
        },
    }


def stream_agent(goal: str):
    """Yield (node_name, partial_state) as each node completes, for live UI updates."""
    for chunk in get_graph().stream({"goal": goal, "trail": [], "iteration": 0}):
        for node_name, partial in chunk.items():
            yield node_name, partial
