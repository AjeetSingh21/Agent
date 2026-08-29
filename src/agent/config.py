"""Runtime configuration, read once from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Config:
    """Everything the agent needs to know about its environment."""

    model: str = os.getenv("AGENT_MODEL", "openai/gpt-oss-120b")
    temperature: float = _float_env("AGENT_TEMPERATURE", 0.2)

    # How many extra search rounds the reflect node may trigger after the
    # initial pass. Each round costs latency, so keep this small.
    max_iterations: int = _int_env("AGENT_MAX_ITERATIONS", 2)
    results_per_query: int = _int_env("AGENT_RESULTS_PER_QUERY", 4)

    # Bounds on the planner so a runaway plan cannot blow up latency.
    min_sub_questions: int = 3
    max_sub_questions: int = 5

    # Free engines rate-limit hard, so searches fail over down this list rather
    # than retrying one engine that is already limiting us.
    search_backends: tuple[str, ...] = tuple(
        b.strip() for b in os.getenv("AGENT_SEARCH_BACKENDS", "duckduckgo,yahoo").split(",") if b.strip()
    )

    # Pause between tool calls within a run, to stay under those rate limits.
    search_delay_s: float = _float_env("AGENT_SEARCH_DELAY", 1.0)

    @property
    def groq_api_key(self) -> str | None:
        return os.getenv("GROQ_API_KEY")

    @property
    def tavily_api_key(self) -> str | None:
        return os.getenv("TAVILY_API_KEY")


CONFIG = Config()
