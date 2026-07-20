"""Typed state that flows through the LangGraph audit pipeline.

A LangGraph node receives the current state and returns a dict of the fields it
wants to change; the graph merges that dict back into the state. So this model
is the single shared contract every node reads from and writes to.

Only the Phase 1 fields exist right now. Later phases append their own fields
(queries, probe_results, the analyses, scores, recommendations) as the models
those fields need get defined. Adding them prematurely would force us to import
types that do not exist yet.
"""

from pydantic import BaseModel


class PageMeta(BaseModel):
    """One crawled page after boilerplate has been stripped."""

    url: str
    title: str | None = None
    content: str  # extracted main text (nav/footer/script removed)
    word_count: int  # lets task 1.2 later threshold which pages count as "usable"


class AuditState(BaseModel):
    run_id: str
    url: str
    brand_name: str | None = None  # detected during crawl
    pages: list[PageMeta] = []
    chunk_count: int = 0
    errors: list[str] = []  # nodes append recoverable failures here instead of raising
    status: str = "running"
    tokens_used: dict[str, int] = {}  # per provider, filled once probes run (Phase 2)
    # NOTE: queries / probe_results / citation_analysis / gap_analysis / scores /
    # recommendations are added in Phases 2-4 as their models are defined.
