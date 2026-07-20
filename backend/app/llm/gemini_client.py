from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

from app.llm.base import (
    BudgetExceeded,
    CompletionResult,
    ProviderClient,
    TokenBudgetTracker,
    Usage,
)


class GeminiClient(ProviderClient):
    provider_name = "gemini"

    def __init__(
        self, api_key: str, tracker: TokenBudgetTracker, model: str, timeout: float = 60.0
    ) -> None:
        super().__init__(tracker, timeout)
        self._model = model
        self._client = genai.Client(
            api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout * 1000))
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        reraise=True,
    )
    async def _call(  # type: ignore[no-untyped-def]
        self, *, system: str | None, contents: str, max_output_tokens: int
    ):
        return await self._client.aio.models.generate_content(
            model=self._model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=max_output_tokens,
            ),
        )

    async def complete(
        self,
        *,
        messages: list[dict[str, str]],
        system: str | None = None,
        max_tokens: int = 1024,
    ) -> CompletionResult:
        if self._tracker.remaining <= 0:
            raise BudgetExceeded(used=self._tracker.used, requested=0, cap=self._tracker.cap)

        contents = "\n".join(m["content"] for m in messages)
        response = await self._call(system=system, contents=contents, max_output_tokens=max_tokens)

        assert response.usage_metadata is not None
        usage = Usage(
            input_tokens=response.usage_metadata.prompt_token_count or 0,
            output_tokens=response.usage_metadata.candidates_token_count or 0,
        )
        await self._tracker.record(usage.total_tokens)

        return CompletionResult(
            text=response.text or "", usage=usage, model=self._model, provider=self.provider_name
        )
