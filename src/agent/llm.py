"""LLM access plus the tolerant JSON layer the whole graph relies on.

Free-tier open-weights models on Groq are noticeably less reliable at native tool
calling than frontier models, so the graph never asks the model to call a tool. It
asks for JSON, extracts it defensively, validates it with pydantic, and retries once
with the parse error fed back in. That keeps step-level failures recoverable.
"""

from __future__ import annotations

import json
import re
from typing import Type, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, ValidationError

from src.agent.config import CONFIG

T = TypeVar("T", bound=BaseModel)

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class LLMUnavailableError(RuntimeError):
    """Raised when no API key is configured, so the UI can say so plainly."""


class RateLimitedError(RuntimeError):
    """Raised when Groq refuses on quota.

    Worth its own type: once the daily token budget is gone, every remaining run
    produces an empty answer. Scoring those as agent failures would understate the
    agent badly, so callers need to tell 'out of quota' apart from 'did badly'.
    """

    def __init__(self, message: str, retry_after_s: float | None = None):
        super().__init__(message)
        self.retry_after_s = retry_after_s


_RETRY_AFTER = re.compile(r"try again in ([\d.]+)m([\d.]+)s|try again in ([\d.]+)s")


def _as_rate_limit(exc: Exception) -> RateLimitedError | None:
    """Recognise a Groq 429 regardless of which SDK exception class carries it."""
    text = str(exc)
    if "rate_limit_exceeded" not in text and "429" not in text:
        return None

    retry_after = None
    match = _RETRY_AFTER.search(text)
    if match:
        if match.group(1) is not None:
            retry_after = float(match.group(1)) * 60 + float(match.group(2))
        elif match.group(3) is not None:
            retry_after = float(match.group(3))

    daily = "tokens per day" in text or "TPD" in text
    scope = "daily token budget (TPD) exhausted" if daily else "rate limit hit"
    return RateLimitedError(f"Groq {scope}: {text[:300]}", retry_after)


def get_llm(temperature: float | None = None) -> ChatGroq:
    if not CONFIG.groq_api_key:
        raise LLMUnavailableError(
            "GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys "
            "and put it in .env (locally) or in Space secrets (on Hugging Face)."
        )
    return ChatGroq(
        model=CONFIG.model,
        temperature=CONFIG.temperature if temperature is None else temperature,
        api_key=CONFIG.groq_api_key,
        max_retries=2,
        timeout=60,
    )


def extract_json(text: str) -> str:
    """Pull the most plausible JSON object out of a chatty model response."""
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1)

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found in model output")
    return text[start : end + 1]


def call_structured(
    schema: Type[T],
    system: str,
    user: str,
    temperature: float | None = None,
) -> T:
    """Ask the model for JSON matching `schema`, retrying once on a bad parse."""
    llm = get_llm(temperature)
    schema_hint = json.dumps(schema.model_json_schema(), indent=2)
    system_prompt = (
        f"{system}\n\n"
        "Respond with a single JSON object and nothing else. No prose, no code "
        f"fences, no explanation. It must validate against this JSON schema:\n{schema_hint}"
    )

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user)]
    last_error: Exception | None = None

    for attempt in range(2):
        try:
            response = llm.invoke(messages)
        except Exception as exc:
            if rate_limited := _as_rate_limit(exc):
                raise rate_limited from exc
            raise
        raw = response.content if isinstance(response.content, str) else str(response.content)
        try:
            return schema.model_validate_json(extract_json(raw))
        except (ValueError, ValidationError) as exc:
            last_error = exc
            if attempt == 0:
                messages.append(HumanMessage(content=raw))
                messages.append(
                    HumanMessage(
                        content=(
                            f"That was not valid JSON for the schema. Error: {exc}\n"
                            "Return only the corrected JSON object."
                        )
                    )
                )

    raise ValueError(f"model did not return valid JSON after 2 attempts: {last_error}")


def call_text(system: str, user: str, temperature: float | None = None) -> str:
    """Plain text completion, for the final prose synthesis."""
    llm = get_llm(temperature)
    try:
        response = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    except Exception as exc:
        if rate_limited := _as_rate_limit(exc):
            raise rate_limited from exc
        raise
    return response.content if isinstance(response.content, str) else str(response.content)
