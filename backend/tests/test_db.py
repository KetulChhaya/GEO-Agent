import os

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

# Deliberately not routed through app.config.get_settings(): this test only
# needs a DB URL, and must not hard-fail via a pydantic ValidationError when
# the rest of the env (API keys, Temporal vars) isn't configured on a bare
# host run. Falls back to the same local default as .env.example.
DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql+asyncpg://citesight:citesight@localhost:5433/citesight"
)


async def test_pgvector_extension_installed() -> None:
    engine = create_async_engine(DATABASE_URL)
    try:
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'vector'"))
            assert result.scalar() == 1
    except OSError:
        pytest.skip("Postgres not reachable -- run `docker compose up -d postgres migrate` first.")
    finally:
        await engine.dispose()
