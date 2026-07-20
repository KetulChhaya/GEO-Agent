from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    google_api_key: str = ""

    max_pages: int = 30
    max_queries: int = 15
    token_budget: int = 150_000

    temporal_address: str = "localhost:7233"
    temporal_namespace: str = "default"
    temporal_task_queue: str = "audits"

    @property
    def sync_database_url(self) -> str:
        """Alembic migrations run over a sync driver; swap asyncpg for psycopg."""
        return self.database_url.replace("+asyncpg", "+psycopg")


@lru_cache
def get_settings() -> Settings:
    return Settings()
