"""Provider selection.

One place decides which implementation the application talks to, so the
choice is visible in one file rather than inferred from imports scattered
across the codebase.

Falling back to the stub is a *development* affordance. In staging or
production a missing key is a startup-time failure, not a silently degraded
assistant answering patients with placeholder text.
"""

from __future__ import annotations

from functools import lru_cache

from app.config import settings
from app.llm.base import LLMProvider
from app.llm.errors import LLMNotConfiguredError
from app.llm.stub import StubProvider
from app.observability.logging import get_logger

log = get_logger(__name__)

#: Providers the factory can build. Kept here rather than inferred from a
#: successful import so that a typo in LLM_PROVIDER fails with "unknown
#: provider" instead of "no module named app.llm.gemni_provider".
#:
#: "ollama" and "vllm" are the self-hosted Qwen3 paths PRD §3 asks for; the
#: rest are hosted fallbacks. "ollama_cloud" and "huggingface" are the same
#: Qwen3 weights served by someone else — the fallback to reach for when CPU
#: inference makes a 55-case run take an hour, since they keep the model
#: identical and change only where it runs.
KNOWN_PROVIDERS = frozenset(
    {
        "anthropic",
        "groq",
        "gemini",
        "openai",
        "ollama",
        "vllm",
        "ollama_cloud",
        "huggingface",
    }
)


def build_provider(*, provider: str | None = None) -> LLMProvider:
    """Construct a provider from settings. Not cached — see :func:`get_llm`."""
    name = (provider or settings.llm_provider).lower()

    if name == "stub":
        _refuse_stub_outside_development(explicit=True)
        return StubProvider()

    if name not in KNOWN_PROVIDERS:
        raise LLMNotConfiguredError(f"Unknown LLM_PROVIDER: {name!r}")

    # A local model server has no key to configure, so a missing one says
    # nothing about whether it is reachable. Falling back to the stub here
    # would replace "Ollama is not running", which names its own fix, with
    # an assistant quietly answering in placeholder text.
    if not settings.llm_api_key and name not in settings.local_llm_providers:
        _refuse_stub_outside_development(explicit=False)
        log.warning(
            "llm.stub_fallback",
            reason="LLM_API_KEY is not set",
            environment=settings.environment,
        )
        return StubProvider()

    # Imported lazily so that a machine without the anthropic or openai
    # package installed can still run the non-LLM parts of the test suite.
    if name == "anthropic":
        from app.llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider()

    from app.llm.openai_compatible import OpenAICompatibleProvider

    return OpenAICompatibleProvider(provider=name)


def _refuse_stub_outside_development(*, explicit: bool) -> None:
    if settings.environment == "development":
        return
    detail = (
        "LLM_PROVIDER=stub is not permitted"
        if explicit
        else "LLM_API_KEY is not set and the stub provider"
    )
    raise LLMNotConfiguredError(
        f"{detail} in {settings.environment}. A deployed CareCopilot must "
        "answer from a real model or not at all."
    )


@lru_cache(maxsize=1)
def get_llm() -> LLMProvider:
    """The process-wide provider.

    Cached because the underlying HTTP client owns a connection pool;
    building one per request would open a new pool per request.
    """
    return build_provider()


async def dispose_llm() -> None:
    """Close the cached provider's connections, for application shutdown."""
    if get_llm.cache_info().currsize:
        await get_llm().aclose()
    get_llm.cache_clear()
