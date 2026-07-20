from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from tenacity import retry, stop_after_attempt, wait_exponential

from app.llm.base import (
    BudgetExceeded,
    CompletionResult,
    ProviderClient,
    TokenBudgetTracker,
    Usage,
)


class OpenAIClient(ProviderClient):
    provider_name = "openai"

    def __init__(
        self, api_key: str, tracker: TokenBudgetTracker, model: str, timeout: float = 60.0
    ) -> None:
        super().__init__(tracker, timeout)
        self._model = model
        self._client = AsyncOpenAI(api_key=api_key, timeout=timeout, max_retries=0)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        reraise=True,
    )
    async def _call(  # type: ignore[no-untyped-def]
        self, *, messages: list[ChatCompletionMessageParam], max_tokens: int
    ):
        return await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_completion_tokens=max_tokens,
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

        chat_messages: list[ChatCompletionMessageParam] = []
        if system:
            chat_messages.append({"role": "system", "content": system})
        chat_messages.extend({"role": m["role"], "content": m["content"]} for m in messages)  # type: ignore[misc]

        response = await self._call(messages=chat_messages, max_tokens=max_tokens)

        assert response.usage is not None
        usage = Usage(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )
        await self._tracker.record(usage.total_tokens)

        text = response.choices[0].message.content or ""
        return CompletionResult(
            text=text, usage=usage, model=self._model, provider=self.provider_name
        )
