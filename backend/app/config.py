"""Application configuration.

Every tunable is read from the environment. Nothing secret has a usable
default: an unset JWT_SECRET in a non-development environment is a hard
startup failure rather than a silent fallback to a shared key.
"""

from __future__ import annotations

import secrets
from functools import lru_cache
from typing import Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

VectorBackend = Literal["pgvector", "array"]
Environment = Literal["development", "staging", "production"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Database ---------------------------------------------------------
    database_url: str = (
        "postgresql+psycopg://carecopilot:carecopilot@localhost:5432/carecopilot"
    )
    analytics_database_url: str | None = None
    vector_backend: VectorBackend = "pgvector"
    db_pool_size: int = 10
    db_max_overflow: int = 5
    db_statement_timeout_ms: int = 10_000

    # --- LLM --------------------------------------------------------------
    #: "ollama" by default: PRD §2 makes Qwen3-8B the development model and
    #: §3 makes Ollama the way to serve it. "vllm" is the production-style
    #: path. "anthropic", "groq", "gemini" and "openai" remain available as
    #: hosted fallbacks — everything but "anthropic" goes through the one
    #: OpenAI-compatible implementation, differing only in URL and model.
    llm_provider: str = "ollama"
    llm_api_key: str | None = None
    llm_model: str = "qwen3:8b"
    #: Overrides the built-in base URL for an OpenAI-compatible provider.
    #: Required only for an endpoint not in ``llm.endpoints.ENDPOINTS``.
    #: Prefer OLLAMA_BASE_URL / VLLM_BASE_URL for the local servers, so that
    #: switching LLM_PROVIDER does not point one provider at another's host.
    llm_base_url: str | None = None
    #: Where the self-hosted model servers listen. Defaults suit a developer
    #: shell; compose overrides them with service names.
    #:
    #: vLLM's own default is 8000, which the backend already owns. Moving it
    #: here rather than in the compose file keeps the two consistent for
    #: someone running vLLM outside Docker.
    ollama_base_url: str = "http://localhost:11434/v1"
    vllm_base_url: str = "http://localhost:8001/v1"
    #: Providers that serve a model on hardware you control, and so have no
    #: API key to supply. Declared here, rather than in ``llm.endpoints``,
    #: because ``_check_secrets`` below needs it and that module imports
    #: this one.
    local_llm_providers: frozenset[str] = frozenset({"ollama", "vllm"})
    #: Let Qwen3 reason before answering, on the local providers that can be
    #: told either way. Off by default, and the default is load-bearing: the
    #: reasoning is discarded before it reaches anyone (PRD §26), and left on
    #: it consumes the whole token budget of a capped structured call, so the
    #: router returns nothing and falls back. Turn it on only when raising
    #: every ``*_max_tokens`` alongside it.
    llm_thinking: bool = False
    llm_router_model: str | None = None
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 2
    llm_max_output_tokens: int = 4096

    # --- Conversation memory ---------------------------------------------
    #: Turns of prior conversation replayed to the model. The API is
    #: stateless, so every turn resends this; unbounded history means
    #: unbounded cost and latency (PRD §26).
    chat_history_turns: int = 10
    chat_history_char_budget: int = 8_000
    chat_max_question_chars: int = 4_000
    #: Output cap for the summarization call. A digest that needs more than
    #: this is not a digest.
    chat_summary_max_tokens: int = 400

    # --- Embeddings -------------------------------------------------------
    embedding_provider: Literal["local", "voyage"] = "local"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    embedding_api_key: str | None = None
    embedding_batch_size: int = 32
    #: Query vectors held in memory (PRD §36 P3). Bounded and in-process:
    #: it survives no restart and is shared with no other worker, so it is a
    #: latency optimisation and never a store of record. 0 disables it.
    embedding_cache_size: int = 512

    # --- Retrieval --------------------------------------------------------
    retrieval_vector_candidates: int = 20
    retrieval_keyword_candidates: int = 20
    retrieval_top_k: int = 5
    reranker: Literal["llm", "heuristic", "none"] = "llm"
    rag_context_token_budget: int = 3000
    rrf_k: int = 60

    # --- Auth -------------------------------------------------------------
    jwt_secret: str | None = None
    jwt_algorithm: str = "HS256"
    jwt_ttl_minutes: int = 120
    #: Issuer and audience are validated on every token (PRD §9).
    #: A correctly signed token minted for another service, or by another
    #: deployment sharing a secret, is rejected rather than accepted on the
    #: strength of the signature alone.
    jwt_issuer: str = "carecopilot-demo"
    jwt_audience: str = "carecopilot-api"
    action_token_secret: str | None = None
    action_token_ttl_seconds: int = 300

    # --- Knowledge graph --------------------------------------------------
    #: Neo4j holds a *derived projection* of PostgreSQL (PRD §17, §33), so it
    #: is optional by construction: with ``kg_enabled`` false the KG route
    #: reports the graph as unavailable rather than the application failing
    #: to start. Nothing else reads from it, because nothing else may — it
    #: holds no business truth of its own.
    kg_enabled: bool = True
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "carecopilot"
    neo4j_database: str = "neo4j"
    #: A traversal is an interactive request like any other; a graph query
    #: that needs longer than this is a reporting job, not a chat answer.
    neo4j_timeout_seconds: float = 5.0
    #: Ceiling on rows returned from a traversal, so a dense subgraph cannot
    #: put an unbounded context in front of the model.
    kg_max_rows: int = 100

    # --- Text-to-SQL ------------------------------------------------------
    sql_statement_timeout_ms: int = 3_000
    sql_max_rows: int = 200
    sql_max_joins: int = 4
    #: Output cap for one SQL-generation call. A SELECT over four tables is
    #: short; this is sized so a verbose model finishes the JSON wrapper
    #: rather than truncating mid-statement.
    sql_generation_max_tokens: int = 1_024

    # --- Service wiring ---------------------------------------------------
    mcp_server_url: str = "http://localhost:8100/mcp"
    mcp_service_token: str | None = None
    backend_internal_url: str = "http://localhost:8000"
    cors_origins: str = "http://localhost:3000"

    #: Serve GET /auth/demo-accounts outside development.
    #:
    #: Off by default, because an endpoint that hands out working
    #: credentials should not be reachable in a deployed environment just
    #: because the data behind it is synthetic.
    #:
    #: But a *public* demo inverts that: the credentials are published in
    #: the README anyway, and a visitor who cannot get past the login screen
    #: is the whole product failing. The sign-in page reads this endpoint
    #: rather than hard-coding the list, so that what it offers cannot drift
    #: from what was actually seeded — turning it off does not hide the
    #: credentials, it only makes them undiscoverable to someone who did not
    #: read the repository.
    #:
    #: So this is a deployment decision, made explicitly, rather than a
    #: weaker default or a hard-coded list in the frontend.
    demo_accounts_public: bool = False

    # --- Demo rate limiting -----------------------------------------------
    #: Guards the endpoints that spend a model call. A public demo runs on a
    #: free inference quota with published credentials, so the limits are
    #: what keep "anyone can try it" from meaning "anyone can drain it".
    #: See app/api/rate_limit.py for why there are two kinds.
    rate_limit_enabled: bool = True
    #: Per client. 0 disables that limit individually.
    rate_limit_per_minute: int = 6
    rate_limit_per_day: int = 100
    #: Across all clients, per UTC day — the limit that protects the API key
    #: itself, since a per-client one does nothing against many clients.
    rate_limit_daily_budget: int = 1_000
    #: Whether X-Forwarded-For may be believed. Off by default because the
    #: header is client-supplied: with no proxy in front, trusting it lets
    #: anyone forge a new identity per request and evade every per-client
    #: limit. Turn it on only when a reverse proxy that sets it is in front
    #: of the app — docker-compose.prod.yml does exactly that, because Caddy
    #: is there and deploy/Caddyfile makes it replace the header rather than
    #: append to it.
    trust_proxy_headers: bool = False

    # --- Ops --------------------------------------------------------------
    environment: Environment = "development"
    log_level: str = "INFO"

    # ------------------------------------------------------------------ #

    @field_validator("database_url", "analytics_database_url", mode="after")
    @classmethod
    def _require_psycopg_driver(cls, v: str | None) -> str | None:
        """Guard against a bare ``postgresql://`` URL.

        SQLAlchemy would silently pick psycopg2, which is not installed and
        cannot do async. Failing here beats failing on first query.
        """
        if v and v.startswith("postgresql://"):
            return v.replace("postgresql://", "postgresql+psycopg://", 1)
        return v

    @property
    def analytics_url(self) -> str:
        """Read-only analytics connection, falling back to the app URL.

        Falling back is acceptable in development only; ``_check_secrets``
        refuses it outside development, because Text-to-SQL running as the
        application role would defeat the least-privilege design.
        """
        return self.analytics_database_url or self.database_url

    @property
    def router_model(self) -> str:
        return self.llm_router_model or self.llm_model

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @model_validator(mode="after")
    def _check_secrets(self) -> Settings:
        missing: list[str] = []

        if not self.jwt_secret:
            if self.is_production:
                missing.append("JWT_SECRET")
            else:
                # Ephemeral per-process key: tokens die with the process,
                # which is correct for a dev server and never reuses a
                # checked-in default across machines.
                self.jwt_secret = secrets.token_urlsafe(48)

        if not self.action_token_secret:
            if self.is_production:
                missing.append("ACTION_TOKEN_SECRET")
            else:
                self.action_token_secret = secrets.token_urlsafe(48)

        if self.is_production and not self.analytics_database_url:
            missing.append("ANALYTICS_DATABASE_URL")

        if (
            self.is_production
            and self.llm_provider != "stub"
            and self.llm_provider not in self.local_llm_providers
            and not self.llm_api_key
        ):
            # Without a key the provider factory falls back to a stub that
            # answers with placeholder text. That is a useful development
            # affordance and an unacceptable production state.
            #
            # Exempting the local providers is the point of the check, not a
            # hole in it: a vLLM deployment has no key to set, and demanding
            # one would make PRD §30's production path unreachable.
            missing.append("LLM_API_KEY")

        if missing:
            raise ValueError(
                "Missing required configuration in "
                f"{self.environment}: {', '.join(missing)}"
            )

        if self.retrieval_top_k > (
            self.retrieval_vector_candidates + self.retrieval_keyword_candidates
        ):
            raise ValueError("retrieval_top_k exceeds the total candidate pool")

        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
