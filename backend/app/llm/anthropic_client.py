from typing import Literal, cast

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam
from tenacity import retry, stop_after_attempt, wait_exponential

from app.llm.base import (
    BudgetExceeded,
    CompletionResult,
    ProviderClient,
    TokenBudgetTracker,
    Usage,
)


class AnthropicClient(ProviderClient):
    provider_name = "anthropic"

    def __init__(
        self, api_key: str, tracker: TokenBudgetTracker, model: str, timeout: float = 60.0
    ) -> None:
        super().__init__(tracker, timeout)
        self._model = model
        # SDK-internal retries disabled: tenacity below is the single retry source.
        self._client = AsyncAnthropic(api_key=api_key, timeout=timeout, max_retries=0)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        reraise=True,
    )
    async def _call(  # type: ignore[no-untyped-def]
        self, *, system: str, messages: list[MessageParam], max_tokens: int
    ):
        return await self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
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

        anthropic_messages: list[MessageParam] = [
            {"role": cast(Literal["user", "assistant"], m["role"]), "content": m["content"]}
            for m in messages
        ]
        response = await self._call(
            system=system or "", messages=anthropic_messages, max_tokens=max_tokens
        )

        usage = Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        await self._tracker.record(usage.total_tokens)

        text = "".join(block.text for block in response.content if block.type == "text")
        return CompletionResult(
            text=text, usage=usage, model=self._model, provider=self.provider_name
        )
