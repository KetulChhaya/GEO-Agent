"""OpenAI embeddings client.

Text goes through this wrapper (not the raw SDK) for the same reason completions
do: every call records its token usage on the shared TokenBudgetTracker, so the
per-run budget cap covers embeddings too. Mirrors the shape of OpenAIClient in
`openai_client.py`.
"""

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.llm.base import BudgetExceeded, TokenBudgetTracker

# text-embedding-3-small returns 1536-d vectors [0.1, -0.23, 0.45, ..., 0.12] (1536 dims)
# irrespective of the length of the text (higher the length, possibly loss of information),
# matching the chunks.embedding column (Vector(1536)). Batch cap comes from the plan's cost
# guardrails.
DEFAULT_EMBED_MODEL = "text-embedding-3-small"
EMBED_DIMENSIONS = 1536
MAX_BATCH = 100


class OpenAIEmbeddingsClient:
    def __init__(
        self,
        api_key: str,
        tracker: TokenBudgetTracker,
        model: str = DEFAULT_EMBED_MODEL,
        timeout: float = 60.0,
    ) -> None:
        self._tracker = tracker
        self._model = model
        self._client = AsyncOpenAI(api_key=api_key, timeout=timeout, max_retries=0)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        reraise=True,
    )
    async def _embed_batch(self, texts: list[str]) -> tuple[list[list[float]], int]:
        """One API call for up to MAX_BATCH texts. Returns (vectors, tokens_used)."""
        response = await self._client.embeddings.create(model=self._model, input=texts)
        # The API preserves input order via the `index` field; sort to be safe.
        ordered = sorted(response.data, key=lambda d: d.index)
        vectors = [item.embedding for item in ordered]
        return vectors, response.usage.total_tokens

    async def embed_all(self, texts: list[str]) -> list[list[float]]:
        """Embed every text, batching at MAX_BATCH per request. Vectors come back
        in the same order as `texts`. Raises BudgetExceeded (via tracker.record)
        if a batch pushes the run over its token cap."""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), MAX_BATCH):
            batch = texts[start : start + MAX_BATCH]
            if self._tracker.remaining <= 0:
                raise BudgetExceeded(used=self._tracker.used, requested=0, cap=self._tracker.cap)
            batch_vectors, tokens = await self._embed_batch(batch)
            await self._tracker.record(tokens)
            vectors.extend(batch_vectors)
        return vectors
