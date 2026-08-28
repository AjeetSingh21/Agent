"""Tools the agent can invoke. Each returns a ToolResult, never raises."""

from __future__ import annotations

import ast
import operator
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from src.agent.config import CONFIG


@dataclass
class SearchHit:
    title: str
    url: str
    snippet: str

    def as_context(self, index: int) -> str:
        return f"[{index}] {self.title}\nURL: {self.url}\n{self.snippet}"


@dataclass
class ToolResult:
    tool: str
    ok: bool
    latency_s: float
    hits: list[SearchHit] = field(default_factory=list)
    value: str | None = None
    error: str | None = None


# --------------------------------------------------------------------------
# web search
# --------------------------------------------------------------------------

def _search_tavily(query: str, max_results: int) -> list[SearchHit]:
    from tavily import TavilyClient  # imported lazily; optional dependency

    client = TavilyClient(api_key=CONFIG.tavily_api_key)
    payload = client.search(query=query, max_results=max_results)
    return [
        SearchHit(
            title=item.get("title", "").strip(),
            url=item.get("url", "").strip(),
            snippet=item.get("content", "").strip()[:600],
        )
        for item in payload.get("results", [])
    ]


def _search_ddgs(query: str, max_results: int, backend: str) -> list[SearchHit]:
    from ddgs import DDGS

    with DDGS() as ddgs:
        raw = list(ddgs.text(query, max_results=max_results, backend=backend))

    return [
        SearchHit(
            title=(item.get("title") or "").strip(),
            url=(item.get("href") or item.get("url") or "").strip(),
            snippet=(item.get("body") or "").strip()[:600],
        )
        for item in raw
    ]


def web_search(query: str, max_results: int | None = None) -> ToolResult:
    """Search the web, failing over between engines.

    Free search engines rate-limit aggressively when queried back-to-back, and
    retrying the *same* engine just burns latency waiting on a limit that has not
    reset. So a failure moves to the next engine instead; only if every engine
    fails do we back off and make one more pass.
    """
    limit = max_results or CONFIG.results_per_query
    started = time.perf_counter()

    if CONFIG.tavily_api_key:
        try:
            hits = [h for h in _search_tavily(query, limit) if h.url]
            return ToolResult("web_search:tavily", bool(hits), time.perf_counter() - started, hits)
        except Exception as exc:
            # Fall through to the free engines rather than failing the step.
            errors = [f"tavily: {exc}"]
    else:
        errors = []

    for attempt in range(2):
        for backend in CONFIG.search_backends:
            try:
                hits = [h for h in _search_ddgs(query, limit, backend) if h.url]
                if hits:
                    return ToolResult(
                        tool=f"web_search:{backend}",
                        ok=True,
                        latency_s=time.perf_counter() - started,
                        hits=hits,
                    )
                errors.append(f"{backend}: no results")
            except Exception as exc:  # rate limit, network flake, engine change
                errors.append(f"{backend}: {type(exc).__name__}")
        if attempt == 0:
            time.sleep(2.0)

    return ToolResult(
        tool="web_search",
        ok=False,
        latency_s=time.perf_counter() - started,
        error=f"all search engines failed ({'; '.join(errors[:4])})",
    )


# --------------------------------------------------------------------------
# calculator
# --------------------------------------------------------------------------

_OPS: dict[type, Callable[..., Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(f"unsupported constant: {node.value!r}")
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if type(node.op) is ast.Pow and (abs(right) > 64 or abs(left) > 1e6):
            raise ValueError("exponent out of supported range")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"unsupported expression element: {type(node).__name__}")


def calculator(expression: str) -> ToolResult:
    """Evaluate an arithmetic expression via AST walk. No eval(), no builtins."""
    started = time.perf_counter()
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _eval_node(tree.body)
        return ToolResult(
            tool="calculator",
            ok=True,
            latency_s=time.perf_counter() - started,
            value=f"{result:g}" if isinstance(result, float) else str(result),
        )
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError, TypeError) as exc:
        return ToolResult(
            tool="calculator",
            ok=False,
            latency_s=time.perf_counter() - started,
            error=str(exc),
        )


TOOLS = {"web_search": web_search, "calculator": calculator}
