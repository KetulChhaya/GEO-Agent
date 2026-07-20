import pytest

from app.llm.base import BudgetExceeded, TokenBudgetTracker
from tests.conftest import CannedResponse, MockProviderClient


async def test_budget_exceeded_raises_when_cap_exceeded() -> None:
    tracker = TokenBudgetTracker(cap=100)
    client = MockProviderClient(
        tracker=tracker,
        canned_responses=[CannedResponse(text="a", input_tokens=60, output_tokens=60)],  # 120 > 100
    )
    with pytest.raises(BudgetExceeded):
        await client.complete(messages=[{"role": "user", "content": "hi"}])


async def test_budget_not_exceeded_under_cap() -> None:
    tracker = TokenBudgetTracker(cap=1000)
    client = MockProviderClient(
        tracker=tracker,
        canned_responses=[CannedResponse(text="a", input_tokens=10, output_tokens=10)],
    )
    result = await client.complete(messages=[{"role": "user", "content": "hi"}])
    assert result.usage.total_tokens == 20
    assert tracker.used == 20


async def test_budget_exceeded_blocks_further_calls_once_cap_hit() -> None:
    tracker = TokenBudgetTracker(cap=50)
    client = MockProviderClient(
        tracker=tracker,
        canned_responses=[CannedResponse(text="a", input_tokens=50, output_tokens=0)],
    )
    await client.complete(messages=[{"role": "user", "content": "first"}])
    assert tracker.remaining == 0

    with pytest.raises(BudgetExceeded):
        await client.complete(messages=[{"role": "user", "content": "second"}])
