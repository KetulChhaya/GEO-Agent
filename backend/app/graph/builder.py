"""Graph builder v1 (Phase 1, task 1.4).

Assembles the two Phase-1 nodes into a LangGraph StateGraph:

    START -> crawl_site --(usable pages >= 3)--> chunk_and_embed -> END
                        \\--(insufficient content)--------------> END

The checkpointer (PostgresSaver) is what makes a run resumable: after each node
LangGraph writes the state to Postgres keyed by `thread_id`. On a retry with the
same thread_id (= run_id), execution resumes from the last completed node instead
of re-crawling. Temporal drives that retry in Phase 2; here we just wire it.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.config import get_settings
from app.graph.nodes.chunk_and_embed import chunk_and_embed
from app.graph.nodes.crawl_site import crawl_site
from app.graph.state import AuditState


def route_after_crawl(state: AuditState) -> str:
    """Conditional edge (task 1.2). crawl_site sets a `failed_*` status when there
    is not enough usable content; route to END in that case, else continue.
    Routers only read state -- the status was set inside crawl_site."""
    return "end" if state.status.startswith("failed") else "continue"


def build_graph(
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> CompiledStateGraph[AuditState, Any, AuditState, AuditState]:
    """Compile the Phase-1 graph. Pass a checkpointer to make runs resumable;
    omit it (None) for a stateless smoke run."""
    builder = StateGraph(AuditState)
    builder.add_node("crawl_site", crawl_site)
    builder.add_node("chunk_and_embed", chunk_and_embed)

    builder.add_edge(START, "crawl_site")
    builder.add_conditional_edges(
        "crawl_site",
        route_after_crawl,
        {"continue": "chunk_and_embed", "end": END},
    )
    builder.add_edge("chunk_and_embed", END)

    return builder.compile(checkpointer=checkpointer)


def _postgres_dsn() -> str:
    """AsyncPostgresSaver wants a raw psycopg DSN, not a SQLAlchemy URL, so strip
    the `+asyncpg` / `+psycopg` driver suffix off DATABASE_URL."""
    return get_settings().database_url.replace("+asyncpg", "").replace("+psycopg", "")


@asynccontextmanager
async def open_checkpointer() -> AsyncIterator[AsyncPostgresSaver]:
    """Open a Postgres checkpointer and ensure its tables exist. `.setup()` is
    idempotent, so it is safe to call on every startup."""
    async with AsyncPostgresSaver.from_conn_string(_postgres_dsn()) as saver:
        await saver.setup()
        yield saver
