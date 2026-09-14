"""Provider-agnostic access to a language model (PRD §4)."""

from app.llm.base import (
    ChatMessage,
    Effort,
    LLMProvider,
    LLMResponse,
    StructuredResponse,
    TokenUsage,
    ToolCall,
    ToolCallResponse,
    ToolDefinition,
)
from app.llm.errors import (
    LLMError,
    LLMNotConfiguredError,
    LLMRateLimitError,
    LLMRefusalError,
    LLMRequestError,
    LLMServiceError,
    LLMTimeoutError,
    LLMValidationError,
)
from app.llm.factory import build_provider, dispose_llm, get_llm
from app.llm.pricing import estimate_cost_usd

__all__ = [
    "ChatMessage",
    "Effort",
    "LLMError",
    "LLMNotConfiguredError",
    "LLMProvider",
    "LLMRateLimitError",
    "LLMRefusalError",
    "LLMRequestError",
    "LLMResponse",
    "LLMServiceError",
    "LLMTimeoutError",
    "LLMValidationError",
    "StructuredResponse",
    "TokenUsage",
    "ToolCall",
    "ToolCallResponse",
    "ToolDefinition",
    "build_provider",
    "dispose_llm",
    "estimate_cost_usd",
    "get_llm",
]
