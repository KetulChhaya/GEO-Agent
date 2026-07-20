"""chunk_and_embed node (Phase 1, task 1.3).

Splits each crawled page into overlapping text chunks, embeds them in batches, and
stores them in the `chunks` table under namespace `site:{run_id}` so later phases
can do similarity search over the site's own content.
"""

from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.config import get_settings
from app.db.models import Audit, Chunk
from app.db.session import async_session
from app.graph.state import AuditState, PageMeta
from app.llm.base import TokenBudgetTracker
from app.llm.embeddings import OpenAIEmbeddingsClient

# 800-char chunks with 100-char overlap: small enough for precise similarity
# matches, overlapped so a claim spanning a boundary is not split in half.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def _chunk_pages(pages: list[PageMeta]) -> tuple[list[str], list[dict[str, Any]]]:
    """Split every page into overlapping chunks. Pure and deterministic (same
    pages in -> same chunks out), so it can be unit-tested without DB/embeddings.
    Returns parallel lists of chunk texts and their metadata."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    texts: list[str] = []
    metadatas: list[dict[str, Any]] = []
    for page in pages:
        for index, chunk_text in enumerate(splitter.split_text(page.content)):
            texts.append(chunk_text)
            metadatas.append({"url": page.url, "title": page.title, "chunk_index": index})
    return texts, metadatas


async def chunk_and_embed(state: AuditState) -> dict[str, Any]:
    """Chunk + embed every page in `state.pages`, persist to pgvector.

    Returns `{"chunk_count": n}`. On an empty page set returns 0 chunks without
    touching the DB or the embeddings API.
    """
    settings = get_settings()

    texts, metadatas = _chunk_pages(state.pages)

    if not texts:
        return {"chunk_count": 0}

    # One tracker per node for now; Phase 2 (task 2.6) shares a single tracker
    # across every node in a run so the budget cap spans the whole audit.
    tracker = TokenBudgetTracker(cap=settings.token_budget)
    embedder = OpenAIEmbeddingsClient(api_key=settings.openai_api_key, tracker=tracker)
    vectors = await embedder.embed_all(texts)

    namespace = f"site:{state.run_id}"
    async with async_session() as session:
        # chunks.run_id references audits.run_id, so the audit row must exist
        # first. The API creates it in Phase 5; upsert here keeps the node
        # runnable standalone (idempotent -- does nothing if the row exists).
        await session.execute(
            pg_insert(Audit)
            .values(run_id=state.run_id, url=state.url)
            .on_conflict_do_nothing(index_elements=["run_id"])
        )
        session.add_all(
            Chunk(
                run_id=state.run_id,
                namespace=namespace,
                content=text,
                embedding=vector,
                metadata_=metadata,
            )
            for text, vector, metadata in zip(texts, vectors, metadatas, strict=True)
        )
        await session.commit()

    return {"chunk_count": len(texts)}
