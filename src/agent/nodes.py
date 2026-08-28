"""The four graph nodes: plan -> act -> reflect -> synthesize."""

from __future__ import annotations

import re
import time

from src.agent.config import CONFIG
from src.agent.llm import RateLimitedError, call_structured, call_text
from src.agent.state import AgentState, Critique, Plan, Source, Step
from src.agent.tools import calculator, web_search

PLANNER_SYSTEM = (
    "You are the planning stage of a research agent. Break the user's goal into "
    f"{CONFIG.min_sub_questions}-{CONFIG.max_sub_questions} sub-questions that together "
    "fully answer it. Each must be specific, non-overlapping, and answerable from public "
    "web sources. Write search_query as keywords a search engine handles well, not as a "
    "sentence. Use tool='calculator' only when the sub-question is pure arithmetic."
)

REFLECT_SYSTEM = (
    "You are the quality-control stage of a research agent. Given the goal and the "
    "evidence gathered so far, judge whether the evidence is enough to write a solid, "
    "well-supported answer. Be pragmatic: if the major aspects are covered, say "
    "sufficient=true. Only request follow-ups for genuinely material gaps, and never "
    "more than 2."
)

SYNTHESIS_SYSTEM = (
    "You are the writing stage of a research agent. Write a clear, well-organised "
    "answer to the user's goal using ONLY the numbered sources provided.\n\n"
    "CITATION RULES - these are not optional:\n"
    "- Every factual claim carries an inline citation like [1] or [2][5].\n"
    "- This applies inside markdown tables too: put the citation in the cell, e.g. "
    "'| LangGraph | Graph-based orchestration [3] |'. A table is never an excuse to "
    "drop citations.\n"
    "- Only cite numbers that appear in the source list below. Never invent a number.\n"
    "- An answer with no inline citations is a failed answer.\n\n"
    "Use short markdown sections with headings and bullets. If the sources genuinely "
    "do not cover something, say so rather than guessing. End with a '## Sources' "
    "section listing each cited number with its title and URL."
)


def _step(node: str, detail: str, ok: bool, latency: float, tool: str | None = None) -> Step:
    """Build a trail entry. `n` is fixed up by the caller once appended."""
    return Step(n=0, node=node, detail=detail, tool=tool, ok=ok, latency_s=round(latency, 2))


def _append(trail: list[Step], step: Step) -> list[Step]:
    step["n"] = len(trail) + 1
    trail.append(step)
    return trail


def plan_node(state: AgentState) -> AgentState:
    """Decompose the goal into sub-questions."""
    started = time.perf_counter()
    trail = list(state.get("trail", []))

    try:
        plan = call_structured(
            schema=Plan,
            system=PLANNER_SYSTEM,
            user=f"Goal: {state['goal']}",
        )
    except RateLimitedError:
        raise  # out of quota is not an agent failure; let the caller distinguish it
    except Exception as exc:
        _append(trail, _step("plan", f"planning failed: {exc}", False, time.perf_counter() - started))
        return {"trail": trail, "error": f"Planning failed: {exc}"}

    # Clamp to the configured bound so one bad plan cannot balloon latency.
    sub_questions = plan.sub_questions[: CONFIG.max_sub_questions]
    detail = f"decomposed goal into {len(sub_questions)} sub-questions: " + "; ".join(
        q.question for q in sub_questions
    )
    _append(trail, _step("plan", detail, True, time.perf_counter() - started))

    return {
        "interpretation": plan.interpretation,
        "sub_questions": [q.model_dump() for q in sub_questions],
        "pending_queries": [],
        "sources": [],
        "trail": trail,
        "iteration": 0,
    }


def act_node(state: AgentState) -> AgentState:
    """Run the tool for each pending sub-question or follow-up query."""
    trail = list(state.get("trail", []))
    sources: list[Source] = list(state.get("sources", []))
    seen_urls = {s["url"] for s in sources}

    pending = state.get("pending_queries") or []
    if pending:
        # Follow-up round: reflect asked for these specific queries.
        jobs = [{"question": q, "search_query": q, "tool": "web_search"} for q in pending]
    else:
        jobs = state.get("sub_questions", [])

    for position, job in enumerate(jobs):
        query = job["search_query"]

        if job.get("tool") == "calculator":
            result = calculator(query)
            if result.ok:
                sources.append(
                    Source(
                        ref=len(sources) + 1,
                        title=f"Computed: {query}",
                        url="(calculator)",
                        snippet=f"{query} = {result.value}",
                    )
                )
                detail = f"calculator({query}) = {result.value}"
            else:
                detail = f"calculator({query}) failed: {result.error}"
            _append(trail, _step("act", detail, result.ok, result.latency_s, "calculator"))
            continue

        # Space out searches; back-to-back queries are what trips the rate limiter.
        if position and CONFIG.search_delay_s:
            time.sleep(CONFIG.search_delay_s)

        result = web_search(query)
        added = 0
        for hit in result.hits:
            if hit.url in seen_urls:
                continue
            seen_urls.add(hit.url)
            sources.append(
                Source(ref=len(sources) + 1, title=hit.title, url=hit.url, snippet=hit.snippet)
            )
            added += 1

        if result.ok:
            detail = f"searched {query!r} -> {added} new sources"
        else:
            detail = f"search {query!r} failed: {result.error}"
        _append(trail, _step("act", detail, result.ok, result.latency_s, result.tool))

    return {"sources": sources, "trail": trail, "pending_queries": []}


# GPT-OSS models emit citations as 【1】 or 【15†L15-L19】 (lenticular brackets, with an
# optional dagger-delimited line range) rather than the [1] the prompt asks for. Both
# forms mean the same thing, so normalise instead of fighting the model for it.
_LENTICULAR_CITE = re.compile(r"【(\d{1,3})(?:†[^】]*)?】")


def _normalize_citations(text: str) -> str:
    """Rewrite non-ASCII citation markers to the [n] form used everywhere else."""
    return _LENTICULAR_CITE.sub(r"[\1]", text)


def _evidence_block(sources: list[Source], limit: int, snippet_chars: int) -> str:
    """Render sources for a prompt, trimmed to a token budget.

    The free Groq tier allows 200k tokens/day, and a full evaluation run was
    spending most of it re-sending the same evidence to both reflect and
    synthesise. Reflect only needs enough to spot a gap, so it gets a far
    smaller view than the node that actually writes the answer.
    """
    return "\n\n".join(
        f"[{s['ref']}] {s['title']}\nURL: {s['url']}\n{s['snippet'][:snippet_chars]}"
        for s in sources[:limit]
    )


def reflect_node(state: AgentState) -> AgentState:
    """Decide whether the evidence is good enough or another round is needed."""
    started = time.perf_counter()
    trail = list(state.get("trail", []))
    iteration = state.get("iteration", 0) + 1
    sources = state.get("sources", [])

    if not sources:
        _append(trail, _step("reflect", "no sources gathered; stopping", False, time.perf_counter() - started))
        return {"trail": trail, "iteration": iteration, "critique": {"sufficient": True, "gaps": ["no sources"]}}

    user = (
        f"Goal: {state['goal']}\n\nSub-questions:\n"
        + "\n".join(f"- {q['question']}" for q in state.get("sub_questions", []))
        + f"\n\nEvidence gathered ({len(sources)} sources):\n"
        + _evidence_block(sources, limit=16, snippet_chars=180)
    )

    try:
        critique = call_structured(schema=Critique, system=REFLECT_SYSTEM, user=user)
    except RateLimitedError:
        raise
    except Exception as exc:
        # A failed critique should not sink the run; proceed to synthesis.
        _append(
            trail,
            _step("reflect", f"critique failed ({exc}); proceeding with what we have", False, time.perf_counter() - started),
        )
        return {"trail": trail, "iteration": iteration, "critique": {"sufficient": True, "gaps": []}}

    follow_ups = [] if critique.sufficient else critique.follow_up_queries[:2]
    if critique.sufficient:
        detail = f"evidence sufficient ({len(sources)} sources)"
    else:
        gaps = "; ".join(critique.gaps) or "unspecified"
        detail = f"gaps found: {gaps} -> {len(follow_ups)} follow-up searches"
    _append(trail, _step("reflect", detail, True, time.perf_counter() - started))

    return {
        "trail": trail,
        "iteration": iteration,
        "critique": critique.model_dump(),
        "pending_queries": follow_ups,
    }


def synthesize_node(state: AgentState) -> AgentState:
    """Write the final cited answer."""
    started = time.perf_counter()
    trail = list(state.get("trail", []))
    sources = state.get("sources", [])

    if not sources:
        # An upstream failure (a 429 in the planner, say) is the real cause here.
        # Blaming search would send the reader to the wrong subsystem, so keep it.
        upstream = state.get("error")
        _append(trail, _step("synthesize", "no sources to synthesize from", False, time.perf_counter() - started))
        return {
            "answer": (
                f"The run could not gather any sources. Cause: {upstream}"
                if upstream
                else "The run could not gather any sources — every search engine failed. "
                "They rate-limit aggressively; please try again in a moment."
            ),
            "trail": trail,
            "error": upstream or "no sources gathered",
        }

    user = (
        f"Goal: {state['goal']}\n\n"
        f"Interpretation: {state.get('interpretation', '')}\n\n"
        f"Numbered sources:\n{_evidence_block(sources, limit=20, snippet_chars=500)}\n\n"
        "Write the final answer now."
    )

    try:
        answer = call_text(system=SYNTHESIS_SYSTEM, user=user, temperature=0.3)
    except RateLimitedError:
        raise
    except Exception as exc:
        _append(trail, _step("synthesize", f"synthesis failed: {exc}", False, time.perf_counter() - started))
        return {"trail": trail, "error": f"Synthesis failed: {exc}", "answer": ""}

    _append(
        trail,
        _step("synthesize", f"wrote answer from {len(sources)} sources", True, time.perf_counter() - started),
    )
    return {"answer": _normalize_citations(answer.strip()), "trail": trail}
