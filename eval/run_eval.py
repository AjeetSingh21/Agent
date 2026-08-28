"""Run the agent over the eval set and report task success rate, steps and latency.

Scoring is deliberately objective — no LLM judge — so the number in the README is
reproducible:

    python -m eval.run_eval                  # full set
    python -m eval.run_eval --limit 3        # smoke test
    python -m eval.run_eval --goal heat-pumps

Writes JSON + a markdown table to eval/results/.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.agent.config import CONFIG
from src.agent.graph import run_agent
from src.agent.llm import RateLimitedError

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
# The agent normalises citations to [n], but the scorer accepts the lenticular forms
# GPT-OSS emits natively (【1】, 【15†L15-L19】) too. Counting only ASCII brackets here
# once scored three correctly-cited answers as having zero citations.
CITATION_RE = re.compile(r"\[(\d{1,3})\]|【(\d{1,3})(?:†[^】]*)?】")


def find_citations(text: str) -> set[int]:
    """Every distinct source number cited in `text`, in either bracket style."""
    return {int(ascii_ref or lenticular_ref) for ascii_ref, lenticular_ref in CITATION_RE.findall(text)}

# Model output is full of en-dashes and non-breaking hyphens; the default Windows
# console codepage (cp1252) raises UnicodeEncodeError on them mid-run.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def load_goals(path: Path) -> tuple[dict, list[dict]]:
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    return spec.get("defaults", {}), spec["goals"]


def score_run(result: dict, case: dict, defaults: dict) -> dict:
    """Apply the four objective checks. All must pass for a success."""
    answer = result.get("answer") or ""
    sources = result.get("sources", [])
    lowered = answer.lower()

    min_sources = case.get("min_sources", defaults.get("min_sources", 5))
    min_citations = case.get("min_citations", defaults.get("min_citations", 3))

    cited_refs = find_citations(answer)
    valid_refs = {s["ref"] for s in sources}
    hallucinated = sorted(cited_refs - valid_refs)

    checks = {
        "completed": bool(answer.strip()) and not result.get("error"),
        "enough_sources": len(sources) >= min_sources,
        "cited": len(cited_refs) >= min_citations and not hallucinated,
        "relevant": all(term.lower() in lowered for term in case.get("must_mention", [])),
    }

    return {
        "checks": checks,
        "success": all(checks.values()),
        "citations": len(cited_refs),
        "hallucinated_refs": hallucinated,
        "missing_terms": [t for t in case.get("must_mention", []) if t.lower() not in lowered],
    }


def summarise(rows: list[dict]) -> dict:
    total = len(rows)
    successes = sum(1 for r in rows if r["score"]["success"])
    steps = [r["metrics"]["steps"] for r in rows]
    latencies = [r["metrics"]["latency_s"] for r in rows]
    sources = [r["metrics"]["sources"] for r in rows]

    def avg(xs: list[float]) -> float:
        return round(statistics.fmean(xs), 2) if xs else 0.0

    return {
        "goals": total,
        "successes": successes,
        "success_rate_pct": round(100 * successes / total, 1) if total else 0.0,
        "avg_steps": avg(steps),
        "avg_tool_calls": avg([r["metrics"]["tool_calls"] for r in rows]),
        "avg_sources": avg(sources),
        "avg_latency_s": avg(latencies),
        "median_latency_s": round(statistics.median(latencies), 2) if latencies else 0.0,
        "p90_latency_s": round(sorted(latencies)[int(0.9 * (len(latencies) - 1))], 2) if latencies else 0.0,
        "failed_steps_total": sum(r["metrics"]["failed_steps"] for r in rows),
    }


def to_markdown(summary: dict, rows: list[dict], meta: dict) -> str:
    lines = [
        "# Evaluation results",
        "",
        f"- **Run at:** {meta['run_at']}",
        f"- **Model:** `{meta['model']}`",
        f"- **Search backend:** `{meta['search_backend']}`",
        f"- **Max search rounds:** `{meta['max_iterations']}`",
        *(
            [f"", f"> **Partial run.** The daily token quota ran out after "
             f"{meta['goals_scored']} of {meta['goals_planned']} goals. Only completed "
             f"goals are scored below."]
            if meta.get("aborted_on_quota") else []
        ),
        "",
        "## Headline",
        "",
        f"| Metric | Value |",
        f"| --- | --- |",
        f"| Task success rate | **{summary['success_rate_pct']}%** ({summary['successes']}/{summary['goals']}) |",
        f"| Avg steps per goal | {summary['avg_steps']} |",
        f"| Avg tool calls per goal | {summary['avg_tool_calls']} |",
        f"| Avg sources gathered | {summary['avg_sources']} |",
        f"| Avg latency | {summary['avg_latency_s']}s |",
        f"| Median / p90 latency | {summary['median_latency_s']}s / {summary['p90_latency_s']}s |",
        "",
        "## Per-goal",
        "",
        "| Goal | Success | Steps | Sources | Citations | Latency | Failed checks |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        failed = [k for k, v in r["score"]["checks"].items() if not v] or ["—"]
        lines.append(
            f"| `{r['id']}` | {'✅' if r['score']['success'] else '❌'} | {r['metrics']['steps']} | "
            f"{r['metrics']['sources']} | {r['score']['citations']} | {r['metrics']['latency_s']}s | "
            f"{', '.join(failed)} |"
        )
    return "\n".join(lines) + "\n"


def rescore(results_path: Path, goals_path: Path) -> int:
    """Re-apply the current scoring rules to a stored run.

    Scoring bugs are separate from agent bugs: the first full run's answers were
    correctly cited, but the scorer's regex missed the citation style the model
    used and reported 70% instead of 100%. Re-scoring stored answers costs no API
    quota, so a scorer fix never means paying to re-run the agent.
    """
    stored = json.loads(results_path.read_text(encoding="utf-8"))
    defaults, cases = load_goals(goals_path)
    by_id = {c["id"]: c for c in cases}

    rows = []
    changed = 0
    for row in stored["rows"]:
        case = by_id.get(row["id"])
        if case is None:
            print(f"  skipping {row['id']!r}: no longer in the goal set", file=sys.stderr)
            continue
        was = row["score"]["success"]
        # Reconstruct the ref set from the recorded source count; only the numbers
        # matter for citation validation.
        reconstructed = {
            "answer": row["answer"],
            "sources": [{"ref": i} for i in range(1, row["metrics"]["sources"] + 1)],
            "error": row.get("error"),
        }
        row["score"] = score_run(reconstructed, case, defaults)
        changed += was != row["score"]["success"]
        rows.append(row)

    if not rows:
        print("Nothing to re-score.", file=sys.stderr)
        return 1

    meta = dict(stored["meta"])
    meta["rescored_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    summary = summarise(rows)

    markdown = to_markdown(summary, rows, meta)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "latest.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Re-scored {len(rows)} goals from {results_path.name}; "
          f"{changed} verdict(s) changed. Wrote eval/results/latest.md")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the research agent.")
    parser.add_argument("--goals", type=Path, default=ROOT / "goals.yaml")
    parser.add_argument("--limit", type=int, default=None, help="only run the first N goals")
    parser.add_argument("--goal", type=str, default=None, help="run a single goal by id")
    parser.add_argument("--sleep", type=float, default=3.0,
                        help="seconds between goals, to stay under free-tier rate limits")
    parser.add_argument("--rescore", type=Path, default=None, metavar="RESULTS.JSON",
                        help="re-score a previous run's stored answers without calling the API")
    args = parser.parse_args()

    if args.rescore:
        return rescore(args.rescore, args.goals)

    if not CONFIG.groq_api_key:
        print("GROQ_API_KEY is not set. Add it to .env before running the eval.", file=sys.stderr)
        return 1

    defaults, cases = load_goals(args.goals)
    if args.goal:
        cases = [c for c in cases if c["id"] == args.goal]
        if not cases:
            print(f"No goal with id {args.goal!r}", file=sys.stderr)
            return 1
    if args.limit:
        cases = cases[: args.limit]

    rows: list[dict] = []
    aborted = False
    for i, case in enumerate(cases, start=1):
        goal_text = " ".join(case["goal"].split())
        print(f"[{i}/{len(cases)}] {case['id']}: {goal_text[:70]}…", flush=True)
        try:
            result = run_agent(goal_text)
        except RateLimitedError as exc:
            # Every remaining goal would produce an empty answer and be scored as an
            # agent failure, quietly destroying the metric. Stop and report honestly
            # on what actually ran instead.
            print(f"\n  STOPPED: {exc}", file=sys.stderr)
            print(f"  Scored {len(rows)}/{len(cases)} goals before quota ran out; "
                  f"{len(cases) - len(rows)} not run.", file=sys.stderr)
            if exc.retry_after_s:
                print(f"  Retry in ~{exc.retry_after_s / 60:.0f} min.", file=sys.stderr)
            aborted = True
            break
        except Exception as exc:
            print(f"    run raised: {exc}", flush=True)
            result = {"answer": "", "sources": [], "error": str(exc),
                      "metrics": {"steps": 0, "tool_calls": 0, "failed_steps": 1,
                                  "sources": 0, "search_rounds": 0, "latency_s": 0.0}}

        score = score_run(result, case, defaults)
        rows.append({
            "id": case["id"],
            "goal": goal_text,
            "metrics": result["metrics"],
            "score": score,
            "error": result.get("error"),
            "answer": result.get("answer", ""),
        })
        mark = "PASS" if score["success"] else "FAIL"
        print(f"    {mark} · {result['metrics']['steps']} steps · "
              f"{result['metrics']['sources']} sources · {result['metrics']['latency_s']}s", flush=True)

        if i < len(cases) and args.sleep:
            time.sleep(args.sleep)

    if not rows:
        print("No goals completed - nothing to score.", file=sys.stderr)
        return 1

    summary = summarise(rows)
    meta = {
        "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "model": CONFIG.model,
        "search_backend": "tavily" if CONFIG.tavily_api_key else " -> ".join(CONFIG.search_backends),
        "max_iterations": CONFIG.max_iterations,
        "goals_planned": len(cases),
        "goals_scored": len(rows),
        "aborted_on_quota": aborted,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    (RESULTS_DIR / f"eval-{stamp}.json").write_text(
        json.dumps({"meta": meta, "summary": summary, "rows": rows}, indent=2), encoding="utf-8"
    )
    markdown = to_markdown(summary, rows, meta)
    (RESULTS_DIR / "latest.md").write_text(markdown, encoding="utf-8")

    print("\n" + markdown)
    print(f"Wrote eval/results/eval-{stamp}.json and eval/results/latest.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
