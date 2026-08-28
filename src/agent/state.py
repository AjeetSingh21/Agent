"""Typed state carried through the graph, plus the schemas the planner emits."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field


class SubQuestion(BaseModel):
    """One decomposed research step."""

    id: int = Field(description="1-based index of this sub-question")
    question: str = Field(description="A specific, self-contained research question")
    search_query: str = Field(description="The web search query that answers it best")
    tool: Literal["web_search", "calculator"] = Field(
        default="web_search",
        description="Use 'calculator' only for pure arithmetic, otherwise 'web_search'",
    )


class Plan(BaseModel):
    """The planner node's output."""

    interpretation: str = Field(description="One sentence restating the user's goal")
    sub_questions: list[SubQuestion]


class Critique(BaseModel):
    """The reflect node's output."""

    sufficient: bool = Field(description="True if evidence answers the goal well enough")
    gaps: list[str] = Field(default_factory=list, description="What is still missing")
    follow_up_queries: list[str] = Field(
        default_factory=list, description="At most 2 further search queries"
    )


class Source(TypedDict):
    ref: int
    title: str
    url: str
    snippet: str


class Step(TypedDict):
    """One entry in the visible reasoning trail."""

    n: int
    node: str
    detail: str
    tool: str | None
    ok: bool
    latency_s: float


class AgentState(TypedDict, total=False):
    goal: str
    interpretation: str
    sub_questions: list[dict[str, Any]]
    pending_queries: list[str]
    sources: list[Source]
    trail: list[Step]
    iteration: int
    critique: dict[str, Any]
    answer: str
    error: str
