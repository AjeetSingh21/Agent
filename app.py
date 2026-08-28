"""Streamlit front-end for the research agent. Also the Hugging Face Spaces entrypoint."""

from __future__ import annotations

import time

import streamlit as st

from src.agent.config import CONFIG
from src.agent.graph import stream_agent
from src.agent.llm import LLMUnavailableError

st.set_page_config(page_title="Multi-Step Research Agent", page_icon="🔎", layout="wide")

EXAMPLE_GOALS = {
    "Batteries": "Research the current state of solid-state batteries for EVs and summarise the key barriers to mass production.",
    "Agent frameworks": "Compare LangGraph, CrewAI and AutoGen for building production agents, and recommend one for a small team.",
    "EU AI Act": "Summarise what the EU AI Act requires of small software companies and when the main obligations take effect.",
    "Heat pumps": "Research whether air-source heat pumps are cost-effective in cold climates and summarise the evidence on both sides.",
}

NODE_LABELS = {
    "plan": "🧭 Plan",
    "act": "🔧 Act",
    "reflect": "🔍 Reflect",
    "synthesize": "✍️ Synthesise",
}


def render_trail(trail: list[dict]) -> None:
    for step in trail:
        icon = "✅" if step["ok"] else "⚠️"
        label = NODE_LABELS.get(step["node"], step["node"])
        tool = f" · `{step['tool']}`" if step.get("tool") else ""
        st.markdown(f"{icon} **{step['n']}. {label}**{tool} — {step['detail']}  \n"
                    f"<span style='color:#888;font-size:0.85em'>{step['latency_s']}s</span>",
                    unsafe_allow_html=True)


def main() -> None:
    st.title("🔎 Multi-Step Research Agent")
    st.caption(
        "Give it a research goal. It plans sub-questions, searches the web, critiques its "
        "own evidence, then writes a cited summary — with the full step trail visible."
    )

    with st.sidebar:
        st.subheader("Configuration")
        if CONFIG.groq_api_key:
            st.success("Groq API key detected")
        else:
            st.error("No `GROQ_API_KEY` set")
            st.caption(
                "Get a free key at [console.groq.com](https://console.groq.com/keys), then add "
                "it to `.env` locally or to Space secrets on Hugging Face."
            )
        st.markdown(
            f"**Model** `{CONFIG.model}`  \n"
            f"**Search** `{'Tavily' if CONFIG.tavily_api_key else 'DuckDuckGo'}`  \n"
            f"**Max search rounds** `{CONFIG.max_iterations}`  \n"
            f"**Results per query** `{CONFIG.results_per_query}`"
        )
        st.divider()
        st.subheader("How it works")
        st.markdown(
            "```\nplan → act → reflect ─┬→ act (if gaps)\n"
            "                      └→ synthesise → done\n```\n"
            "Control flow lives in the graph edges, not in model tool-calls — which is what "
            "keeps the loop predictable on a small open-weights model."
        )

    if "goal" not in st.session_state:
        st.session_state.goal = next(iter(EXAMPLE_GOALS.values()))

    st.markdown("**Try an example**")
    for col, (label, example) in zip(st.columns(len(EXAMPLE_GOALS)), EXAMPLE_GOALS.items()):
        with col:
            if st.button(label, use_container_width=True, help=example):
                st.session_state.goal = example

    goal = st.text_area("Research goal", key="goal", height=100)
    run = st.button("Run agent", type="primary", disabled=not goal.strip())

    if not run:
        return

    if not CONFIG.groq_api_key:
        st.error("Set `GROQ_API_KEY` before running the agent.")
        return

    trail_box = st.container()
    started = time.perf_counter()
    final: dict = {}

    try:
        with st.status("Agent running…", expanded=True) as status:
            seen = 0
            for node_name, partial in stream_agent(goal):
                final.update(partial)
                trail = final.get("trail", [])
                with trail_box:
                    render_trail(trail[seen:])
                seen = len(trail)
                status.update(label=f"{NODE_LABELS.get(node_name, node_name)} complete…")
            status.update(label="Done", state="complete", expanded=False)
    except LLMUnavailableError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.error(f"The run failed: {exc}")
        return

    elapsed = time.perf_counter() - started
    trail = final.get("trail", [])
    sources = final.get("sources", [])

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Steps", len(trail))
    m2.metric("Tool calls", sum(1 for s in trail if s.get("tool")))
    m3.metric("Sources", len(sources))
    m4.metric("Latency", f"{elapsed:.1f}s")

    if final.get("error"):
        st.warning(f"Completed with an issue: {final['error']}")

    answer = final.get("answer", "")
    if answer:
        st.markdown("## Answer")
        st.markdown(answer)
        st.download_button(
            "Download as Markdown",
            data=f"# {goal}\n\n{answer}\n",
            file_name="research-summary.md",
            mime="text/markdown",
        )

    if sources:
        with st.expander(f"All {len(sources)} sources gathered"):
            for s in sources:
                st.markdown(f"**[{s['ref']}]** [{s['title'] or s['url']}]({s['url']})  \n{s['snippet'][:300]}")


if __name__ == "__main__":
    main()
