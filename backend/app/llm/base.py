import asyncio
from abc import ABC, abstractmethod

from pydantic import BaseModel


class BudgetExceeded(Exception):
    def __init__(self, used: int, requested: int, cap: int) -> None:
        self.used = used
        self.requested = requested
        self.cap = cap
        super().__init__(
            f"Token budget exceeded: {used} already used, {requested} more requested, cap is {cap}"
        )


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class CompletionResult(BaseModel):
    text: str
    usage: Usage
    model: str
    provider: str


class TokenBudgetTracker:
    """Shared across every ProviderClient for a single audit run, so the
    150k-token cap applies to the run as a whole, not per provider."""

    def __init__(self, cap: int) -> None:
        self.cap = cap
        self._used = 0
        self._lock = asyncio.Lock()

    @property
    def used(self) -> int:
        return self._used

    @property
    def remaining(self) -> int:
        return max(self.cap - self._used, 0)

    async def record(self, tokens: int) -> None:
        """Add real usage after a call returns. Raises if this pushes the
        run over cap -- the call that tips it over is allowed to complete
        (we can't know its exact output tokens beforehand), but the usage
        still counts and the exception tells the caller to stop probing."""
        async with self._lock:
            used_before = self._used
            self._used += tokens
            if self._used > self.cap:
                raise BudgetExceeded(used=used_before, requested=tokens, cap=self.cap)


class ProviderClient(ABC):
    provider_name: str

    def __init__(self, tracker: TokenBudgetTracker, timeout: float = 60.0) -> None:
        self._tracker = tracker
        self._timeout = timeout

    @abstractmethod
    async def complete(
        self,
        *,
        messages: list[dict[str, str]],
        system: str | None = None,
        max_tokens: int = 1024,
    ) -> CompletionResult: ...
