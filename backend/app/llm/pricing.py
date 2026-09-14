"""Token pricing, for the cost figure in every request trace (PRD §26).

Prices are USD per million tokens and are a *cached snapshot*, not a live
lookup — a request must never depend on a pricing endpoint being reachable.
The number this produces is therefore an estimate for relative comparison
between routes and models, not an invoice.

An unknown model yields a cost of ``None`` rather than zero. Zero would
quietly read as "this route is free" on a dashboard, which is the one wrong
answer that nobody investigates.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelPrice:
    input_per_mtok: float
    output_per_mtok: float
    #: Cache reads bill at roughly a tenth of the input rate.
    cache_read_per_mtok: float
    #: Cache writes carry a ~25% premium over the input rate.
    cache_write_per_mtok: float


def _standard(input_rate: float, output_rate: float) -> ModelPrice:
    return ModelPrice(
        input_per_mtok=input_rate,
        output_per_mtok=output_rate,
        cache_read_per_mtok=input_rate * 0.1,
        cache_write_per_mtok=input_rate * 1.25,
    )


#: Snapshot taken 2026-06-24. Keys are exact model ids.
PRICES: dict[str, ModelPrice] = {
    "claude-opus-5": _standard(5.00, 25.00),
    "claude-opus-4-8": _standard(5.00, 25.00),
    "claude-opus-4-7": _standard(5.00, 25.00),
    "claude-opus-4-6": _standard(5.00, 25.00),
    "claude-sonnet-5": _standard(2.00, 10.00),
    "claude-sonnet-4-6": _standard(3.00, 15.00),
    "claude-haiku-4-5": _standard(1.00, 5.00),
    "claude-fable-5": _standard(10.00, 50.00),
    "claude-fable-5-1": _standard(10.00, 50.00),
}


def estimate_cost_usd(
    model: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float | None:
    """Estimate the cost of one call, or ``None`` for an unpriced model."""
    price = PRICES.get(model)
    if price is None:
        return None

    per_token = 1_000_000
    total = (
        input_tokens * price.input_per_mtok
        + output_tokens * price.output_per_mtok
        + cache_read_tokens * price.cache_read_per_mtok
        + cache_write_tokens * price.cache_write_per_mtok
    ) / per_token
    # Sub-cent precision matters: a single chat turn costs far less than a
    # cent, and rounding to cents would report every request as free.
    return round(total, 8)
