# 🔎 Multi-Step Research Agent

A goal-driven AI agent that decomposes a research question into sub-questions, searches the web, critiques its own evidence, and writes a cited summary — with the full reasoning trail exposed in the UI.

Built with **LangGraph** on **GPT-OSS 120B** (Groq), with an objective evaluation harness that scores task success rate without an LLM judge.

**🚀 [Live demo](https://YOUR_APP.streamlit.app)** · _(replace with your Streamlit Cloud URL after deploying)_

![Python](https://img.shields.io/badge/python-3.11%2B-blue) ![LangGraph](https://img.shields.io/badge/LangGraph-1.x-orange) ![Tests](https://img.shields.io/badge/tests-55%20passing-brightgreen)

---

## What it does

Give it a goal like _"Research the current state of solid-state batteries for EVs and summarise the key barriers to mass production."_ The agent:

1. **Plans** — decomposes the goal into 3–5 specific, non-overlapping sub-questions, each with a targeted search query
2. **Acts** — runs the appropriate tool per sub-question (web search or calculator), deduplicating sources as it goes
3. **Reflects** — critiques whether the gathered evidence actually answers the goal; if there are material gaps it issues up to 2 follow-up searches and loops back
4. **Synthesises** — writes a structured markdown answer where every factual claim carries an inline `[n]` citation to a source it actually retrieved

Every step is streamed to the UI with its tool, outcome and latency.

## Architecture

```
        ┌──────┐
        │ plan │  decompose goal → sub-questions (structured JSON)
        └───┬──┘
            ▼
        ┌──────┐
   ┌───▶│ act  │  web_search / calculator per sub-question
   │    └───┬──┘
   │        ▼
   │   ┌─────────┐
   └───┤ reflect │  gaps found & budget left? → loop back to act
       └────┬────┘
            ▼
     ┌─────────────┐
     │ synthesise  │  cited markdown answer
     └──────┬──────┘
            ▼
           END
```

**The design decision that matters:** control flow lives in the **graph edges**, not in model tool-calls. Open-weights models on free tiers are noticeably less reliable at native function calling than frontier models, so instead of a free-form ReAct loop, the agent asks the model for structured JSON at each decision point and lets LangGraph route deterministically. Bad JSON is caught by a tolerant extractor, validated against a pydantic schema, and retried once with the parse error fed back in.

This makes the loop predictable enough to run on a free-tier model, and produces a cleaner step trail as a side effect.

### Failure handling

Each node degrades rather than crashes: a failed critique falls through to synthesis with whatever evidence exists, and a failed plan routes straight to a graceful error message. Every failure still appears in the trail, marked ⚠️.

Search gets special treatment. Free engines rate-limit aggressively when queried back-to-back, and retrying the *same* engine just burns latency waiting on a limit that has not reset — in testing that cost ~20s per failed search. So a failure **fails over to the next engine** instead, and only if every engine fails does the agent back off for one more pass. Adding that failover cut a representative run from 60s to 28s and raised sources gathered from 12 to 16.

### Staying inside the free tier

The free tier allows 200,000 tokens/day, and an early full evaluation exhausted it partway through — reflect and synthesise were each being sent the same 24 sources at 600 characters apiece. Reflect only needs enough context to spot a gap, so it now receives 16 sources at 180 characters while synthesis keeps 20 at 500. That cuts the evidence payload from ~4,165 to ~1,095 tokens for reflect and ~4,165 to ~2,970 for synthesis: roughly a 21% reduction per goal, which is the difference between one full evaluation per day and not quite finishing one.

## Stack

| Layer | Choice | Why |
| --- | --- | --- |
| Orchestration | LangGraph 1.x | Explicit state machine, conditional edges, streaming |
| LLM | GPT-OSS 120B via Groq | Free tier, fast, follows JSON schemas reliably |
| Search | DuckDuckGo -> Yahoo failover (`ddgs`), optional Tavily | No API key required by default |
| Calculator | AST-walk evaluator | No `eval()`, no builtins reachable |
| UI | Streamlit | Streams the step trail live; deploys to Streamlit Community Cloud as-is |
| Eval | Custom harness | Objective scoring, no LLM judge |

## Results

Measured over **10 research goals** in [`eval/goals.yaml`](eval/goals.yaml), scored on four objective checks — run completed, ≥5 distinct sources gathered, ≥3 inline citations with **zero invented references**, and required key terms present. A goal counts as a success only if all four pass.

| Metric | Value |
| --- | --- |
| **Task success rate** | **10/10** |
| Avg steps per goal | 9.2 |
| Avg tool calls per goal | 5.6 |
| Avg sources gathered | 16.2 |
| Avg latency | 64.3s |
| Median / p90 latency | 68.9s / 90.0s |

Model `openai/gpt-oss-120b`, DuckDuckGo -> Yahoo search. Full per-goal table in [`eval/results/latest.md`](eval/results/latest.md); the raw run backing it is committed as [`eval/results/baseline-2026-08-26.json`](eval/results/baseline-2026-08-26.json), so the number can be re-scored and audited.

**What this does and does not prove.** The checks are structural: the run finished, it gathered real sources, it cited them without inventing references, and it stayed on topic. Citation integrity is verified programmatically — every `[n]` is cross-referenced against the sources actually retrieved, so a hallucinated reference fails the run. What the suite does *not* verify is whether the cited claims are factually true; that would need human review or a judge model, and both were deliberately avoided to keep the number reproducible.

Ten goals is also a small sample. Treat this as evidence the agent completes its loop reliably, not as a precision measurement.

## Running locally

```bash
git clone https://github.com/AjeetSingh21/Agent.git
```

```bash
cd Agent && python -m venv .venv && .venv/Scripts/activate
```

```bash
pip install -r requirements.txt
```

Get a free API key at [console.groq.com/keys](https://console.groq.com/keys), then:

```bash
cp .env.example .env
```

Add your key to `.env`, and launch:

```bash
streamlit run app.py
```

### Running the evaluation

```bash
python -m eval.run_eval
```

Smoke-test three goals first — a full run is most of a day's free quota:

```bash
python -m eval.run_eval --limit 3
```

Re-apply the current scoring rules to a previous run's stored answers, without spending any API quota:

```bash
python -m eval.run_eval --rescore eval/results/baseline-2026-08-26.json
```

Results are written to `eval/results/` as JSON plus a markdown table.

**Budget note.** Groq's free tier allows 200,000 tokens/day. A full 10-goal run costs roughly 160k, so you get one full evaluation per day plus a little headroom. If the quota runs out mid-run the harness stops and scores only the goals that completed, rather than recording the remaining ones as agent failures — an early version did the latter and turned a genuine 10/10 into a meaningless 5/10.

### Running the tests

```bash
pytest
```

The suite stubs out the LLM and search backend, so it needs no API key and no network.

## Configuration

All optional, set in `.env`:

| Variable | Default | Purpose |
| --- | --- | --- |
| `GROQ_API_KEY` | — | **Required.** Free at console.groq.com |
| `AGENT_MODEL` | `openai/gpt-oss-120b` | Any model your key can reach — run `python -m scripts.list_models` to check |
| `AGENT_MAX_ITERATIONS` | `2` | Max follow-up search rounds |
| `AGENT_RESULTS_PER_QUERY` | `4` | Search results per sub-question |
| `AGENT_SEARCH_BACKENDS` | `duckduckgo,yahoo` | Search engines, tried in order |
| `AGENT_SEARCH_DELAY` | `1.0` | Seconds between searches, to stay under rate limits |
| `TAVILY_API_KEY` | — | Optional: better search quality, tried first when set |

## Deployment

Deployed on Streamlit Community Cloud, which builds straight from `main` — no Dockerfile,
no deploy script. See [`deploy/DEPLOY.md`](deploy/DEPLOY.md) for step-by-step instructions.

## Project layout

```
app.py                 Streamlit UI / deployment entrypoint
scripts/list_models.py Shows which Groq models your key can reach
src/agent/
  config.py            Environment-driven settings
  llm.py               Groq client + tolerant structured-JSON layer
  tools.py             web_search, calculator (both fail-safe)
  state.py             Typed graph state and pydantic schemas
  nodes.py             plan / act / reflect / synthesise
  graph.py             LangGraph wiring and routing
eval/
  goals.yaml           10 evaluation goals with objective criteria
  run_eval.py          Scoring harness → JSON + markdown (--rescore re-grades offline)
tests/                 55 tests, no network required
  test_citations.py    Citation parsing + hallucination detection
  test_rate_limits.py  Quota detection and propagation
```

## Limitations

- Search quality is bounded by the free engines' results; Tavily improves it noticeably
- Groq model availability varies by account tier — `python -m scripts.list_models` shows what yours can reach
- The agent reads search snippets, not full page content — it trades depth for latency and cost
- Groq's free tier rate-limits, so the eval harness sleeps between goals

## License

MIT
