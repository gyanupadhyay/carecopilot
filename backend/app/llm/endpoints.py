"""Where each OpenAI-compatible provider lives, and whether it wants a key.

Split out of :mod:`app.llm.openai_compatible` so that
:mod:`app.llm.factory` can ask "does this provider need an API key?" without
importing the ``openai`` package — the factory decides that question before
it has decided to build an OpenAI-compatible provider at all.

The entries divide into two kinds.

*Hosted* providers (Groq, Google AI Studio, OpenAI, Ollama Cloud, Hugging
Face) have a fixed public URL and are useless without a key.

Two of those serve the same Qwen3 the local entries do, which is the point of
them: CPU inference runs this model at tens of seconds per answer, and an
evaluation set of 55 cases is an hour of wall clock before the judge pass.
``ollama_cloud`` and ``huggingface`` trade the self-hosting property for a
run that finishes, and they are separate provider names rather than a base
URL override so the trace can still say which one produced a number.

*Local* providers (Ollama, vLLM) are the ones PRD §3 actually asks for: they
serve Qwen3 on a host you control, so there is no key to have, and the URL
is deployment-specific — ``localhost`` from a developer's shell, a compose
service name from inside a container. Their URL therefore comes from
settings rather than from a constant here, which is what lets
``LLM_PROVIDER=gemini`` keep working in a compose file whose Ollama host is
already configured. Overriding ``LLM_BASE_URL`` for one provider must not
silently redirect another.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import settings


@dataclass(frozen=True, slots=True)
class Endpoint:
    """A provider's default wire location and credential requirement."""

    base_url: str
    default_model: str
    #: False for a model server you run yourself. The OpenAI SDK still
    #: requires *some* string in the Authorization header, and both Ollama
    #: and vLLM (started without ``--api-key``) ignore whatever it is.
    requires_key: bool = True


#: Defaults per known provider. A provider not listed here can still be used
#: by setting LLM_BASE_URL and LLM_MODEL explicitly.
#:
#: The hosted defaults are models confirmed present and schema-capable
#: against a free-tier key on 2026-09-13. Model catalogues on these endpoints
#: change without notice and differ per account, so treat them as a starting
#: point and set LLM_MODEL explicitly for anything that matters.
#:
#: Local entries carry an empty ``base_url``: theirs is resolved from
#: settings by :func:`endpoint_for`, and duplicating the default string here
#: would create a second place to change it.
ENDPOINTS: dict[str, Endpoint] = {
    "groq": Endpoint("https://api.groq.com/openai/v1", "openai/gpt-oss-120b"),
    "gemini": Endpoint(
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "gemini-flash-lite-latest",
    ),
    "openai": Endpoint("https://api.openai.com/v1", "gpt-4.1-mini"),
    # The same Qwen3 weights as the `ollama` entry, on someone else's
    # hardware. Worth having as a distinct provider rather than as an
    # OLLAMA_BASE_URL override: it needs a key, it must not inherit the
    # local entry's `requires_key=False`, and the distinction is what keeps
    # "did this measurement run locally?" answerable from the trace, which
    # records the provider beside the model for exactly that reason.
    "ollama_cloud": Endpoint("https://ollama.com/v1", "qwen3:8b"),
    # Hugging Face's Inference Providers router. Model ids are repo ids, and
    # may carry a `:provider` suffix to pin which backend serves them —
    # without one the router chooses, so two runs can silently land on
    # different hardware and the latency numbers stop being comparable.
    "huggingface": Endpoint("https://router.huggingface.co/v1", "Qwen/Qwen3-8B"),
    # PRD §2: Qwen3-8B is the default development model, and §3 names Ollama
    # as the way to run it locally.
    "ollama": Endpoint("", "qwen3:8b", requires_key=False),
    # PRD §3's production-style path. The model id is a Hugging Face repo
    # rather than an Ollama tag, because that is what vLLM is started with.
    "vllm": Endpoint("", "Qwen/Qwen3-8B", requires_key=False),
}

#: Providers that serve a model on hardware you control. Kept as a name set
#: rather than derived from ``requires_key`` so that :mod:`app.config` — which
#: must not import this module, since this module imports it — can express
#: the same policy.
LOCAL_PROVIDERS = frozenset(settings.local_llm_providers)


def endpoint_for(name: str) -> Endpoint:
    """The endpoint for ``name``, with local hosts filled in from settings.

    Returns an empty endpoint for an unknown provider rather than raising:
    an unlisted provider is legitimate as long as the caller supplies
    ``LLM_BASE_URL``, and the provider is the right place to complain when
    they have not.
    """
    endpoint = ENDPOINTS.get(name)
    if endpoint is None:
        return Endpoint("", "")
    if name == "ollama":
        return Endpoint(settings.ollama_base_url, endpoint.default_model, False)
    if name == "vllm":
        return Endpoint(settings.vllm_base_url, endpoint.default_model, False)
    return endpoint


def requires_api_key(name: str) -> bool:
    """Whether ``name`` is unusable without ``LLM_API_KEY``."""
    return name not in LOCAL_PROVIDERS
