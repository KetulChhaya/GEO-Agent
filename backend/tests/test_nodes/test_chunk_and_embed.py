"""chunk_and_embed tests (task 1.6).

Two levels:
* `_chunk_pages` is pure -- test it directly for determinism, no DB/embeddings.
* the full node insert path -- mock embeddings (no OpenAI key, no network) but use
  the REAL Postgres so we exercise the FK upsert + pgvector insert. Skips cleanly
  when Postgres is unreachable (same policy as tests/test_db.py). CI provides the
  DB and runs migrations, so this runs for real there.
"""

import os
import uuid

# Set before importing app modules: app.db.session builds the engine from
# settings at import time, and OpenAIEmbeddingsClient needs a non-empty key at
# construction (embeddings are mocked below, so the key is never used).
os.environ.setdefault(
    "DATABASE_URL", "postgresql+asyncpg://citesight:citesight@localhost:5433/citesight"
)
os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy")

import pytest  # noqa: E402
from sqlalchemy import delete, select  # noqa: E402

import app.graph.nodes.chunk_and_embed as ce_mod  # noqa: E402
from app.db.models import Audit, Chunk  # noqa: E402
from app.db.session import async_session  # noqa: E402
from app.graph.nodes.chunk_and_embed import _chunk_pages, chunk_and_embed  # noqa: E402
from app.graph.state import AuditState, PageMeta  # noqa: E402
from app.llm.embeddings import EMBED_DIMENSIONS  # noqa: E402

# A page long enough to split into more than one 800-char chunk.
_LONG = ("Reliable industrial widgets built to last for a decade. " * 40).strip()


def _pages() -> list[PageMeta]:
    return [
        PageMeta(
            url="http://acme.test/", title="Home", content=_LONG, word_count=len(_LONG.split())
        ),
        PageMeta(
            url="http://acme.test/about",
            title="About",
            content="Short about page with a handful of words only.",
            word_count=9,
        ),
    ]


def test_chunk_pages_is_deterministic() -> None:
    texts_a, metas_a = _chunk_pages(_pages())
    texts_b, metas_b = _chunk_pages(_pages())
    assert texts_a == texts_b
    assert metas_a == metas_b


def test_long_page_splits_with_overlap() -> None:
    texts, metas = _chunk_pages(_pages())
    home_chunks = [t for t, m in zip(texts, metas, strict=True) if m["url"] == "http://acme.test/"]
    # a > 800-char page must yield more than one chunk...
    assert len(home_chunks) > 1
    # ...and consecutive chunks overlap (tail of one appears at head of the next)
    assert home_chunks[0][-50:] in home_chunks[1]
    # chunk_index metadata increments from 0
    home_meta = [m for m in metas if m["url"] == "http://acme.test/"]
    assert [m["chunk_index"] for m in home_meta] == list(range(len(home_meta)))


class _FakeEmbedder:
    """Stand-in for OpenAIEmbeddingsClient: no network, deterministic vectors."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    async def embed_all(self, texts: list[str]) -> list[list[float]]:
        return [[0.001 * (i % 7)] * EMBED_DIMENSIONS for i, _ in enumerate(texts)]


async def test_embedding_insert_with_mocked_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ce_mod, "OpenAIEmbeddingsClient", _FakeEmbedder)

    run_id = str(uuid.uuid4())  # exactly 36 chars, matching the run_id column width
    state = AuditState(run_id=run_id, url="http://acme.test/", pages=_pages())
    namespace = f"site:{run_id}"

    try:
        try:
            result = await chunk_and_embed(state)
        except OSError:
            pytest.skip("Postgres not reachable -- run `docker compose up -d postgres migrate`.")

        assert result["chunk_count"] > 0

        async with async_session() as session:
            rows = (
                await session.execute(select(Chunk).where(Chunk.namespace == namespace))
            ).scalars().all()
        assert len(rows) == result["chunk_count"]
        assert all(r.embedding is not None and len(r.embedding) == EMBED_DIMENSIONS for r in rows)
        assert all(r.run_id == run_id for r in rows)
    finally:
        # leave the dev DB clean: remove this run's chunks + the upserted audit row
        async with async_session() as session:
            await session.execute(delete(Chunk).where(Chunk.namespace == namespace))
            await session.execute(delete(Audit).where(Audit.run_id == run_id))
            await session.commit()


async def test_empty_pages_no_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ce_mod, "OpenAIEmbeddingsClient", _FakeEmbedder)
    result = await chunk_and_embed(AuditState(run_id="empty", url="http://acme.test/", pages=[]))
    assert result == {"chunk_count": 0}
