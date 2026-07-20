import pytest

from app.llm.base import BudgetExceeded, CompletionResult, ProviderClient, TokenBudgetTracker, Usage


class CannedResponse:
    def __init__(self, text: str, input_tokens: int, output_tokens: int) -> None:
        self.text = text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class MockProviderClient(ProviderClient):
    """Returns canned responses instead of calling a real LLM API -- used to
    test budget/node logic without network calls or cost."""

    provider_name = "mock"

    def __init__(self, tracker: TokenBudgetTracker, canned_responses: list[CannedResponse]) -> None:
        super().__init__(tracker)
        self._responses = list(canned_responses)

    async def complete(
        self,
        *,
        messages: list[dict[str, str]],
        system: str | None = None,
        max_tokens: int = 1024,
    ) -> CompletionResult:
        if self._tracker.remaining <= 0:
            raise BudgetExceeded(used=self._tracker.used, requested=0, cap=self._tracker.cap)

        canned = self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        usage = Usage(input_tokens=canned.input_tokens, output_tokens=canned.output_tokens)
        await self._tracker.record(usage.total_tokens)
        return CompletionResult(
            text=canned.text, usage=usage, model="mock-model", provider=self.provider_name
        )


@pytest.fixture
def token_tracker() -> TokenBudgetTracker:
    return TokenBudgetTracker(cap=150_000)


@pytest.fixture
def mock_provider_client(token_tracker: TokenBudgetTracker) -> MockProviderClient:
    return MockProviderClient(
        tracker=token_tracker,
        canned_responses=[CannedResponse(text="canned answer", input_tokens=50, output_tokens=50)],
    )
