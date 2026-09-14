"""Record which inference backend served each request.

Revision ID: 0007
Revises: 0006
Create Date: 2026-09-14

PRD §26 lists "inference provider" beside "model" among the fields every
request must carry, and §28 asks for a measured Qwen3-8B vs Qwen3-14B
comparison. Neither works from the model id alone: the same weights served
by Ollama on CPU and by vLLM on a GPU differ in latency by an order of
magnitude, so a latency column without the server that produced it compares
nothing.

Nullable, with no backfill. Rows written before this migration were served
by whatever was configured at the time, and guessing that now — from the
model id, or from today's settings — would invent a fact and make it
indistinguishable from a measured one. An empty column reads as "not
recorded", which is what it is.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "request_traces",
        sa.Column("provider", sa.String(length=32), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("request_traces", "provider")
