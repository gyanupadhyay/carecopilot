"""LLM failure modes, as types the caller can actually branch on.

Provider SDKs raise their own exception hierarchies. Letting those escape
would couple every call site to one vendor — the thing PRD §4 explicitly
asks us to avoid — so each provider translates into these instead.

The distinction that matters is *retryable* versus not. A timeout or a 529
is worth another attempt; a refusal or a schema violation will produce the
same result every time, and retrying it only spends money.
"""

from __future__ import annotations


class LLMError(Exception):
    """Base class. Carries whether a retry could plausibly succeed."""

    retryable: bool = False

    def __init__(self, message: str, *, provider: str = "", model: str = "") -> None:
        super().__init__(message)
        self.provider = provider
        self.model = model


class LLMTimeoutError(LLMError):
    retryable = True


class LLMRateLimitError(LLMError):
    retryable = True

    def __init__(
        self, message: str, *, retry_after: float | None = None, **kw: str
    ) -> None:
        super().__init__(message, **kw)
        self.retry_after = retry_after


class LLMServiceError(LLMError):
    """5xx, overloaded, or a connection failure."""

    retryable = True


class LLMRequestError(LLMError):
    """4xx: malformed request, bad model id, context overflow, bad key."""


class LLMRefusalError(LLMError):
    """The model declined on safety grounds.

    Surfaced distinctly because the correct handling is to tell the user
    plainly, not to retry and not to fall back to an ungrounded answer.
    """

    def __init__(self, message: str, *, category: str | None = None, **kw: str) -> None:
        super().__init__(message, **kw)
        self.category = category


class LLMValidationError(LLMError):
    """Output did not satisfy the schema it was constrained to.

    This is the guardrail in PRD §25 firing. It is not retryable here: the
    caller decides whether to re-ask, degrade, or refuse, because only the
    caller knows whether a second attempt is worth the latency.
    """


class LLMNotConfiguredError(LLMError):
    """No usable provider — typically a missing API key."""
