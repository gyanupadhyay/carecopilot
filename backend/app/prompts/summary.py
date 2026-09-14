"""Prompt for condensing older conversation turns (PRD §32).

The summary replaces turns that no longer fit in the replay budget, so its
job is to preserve exactly what a later turn might need to resolve a
reference — "increase it again", "the same one as last time" — and nothing
else.

Two constraints shape the wording. It must not introduce facts, because a
summary that invents a detail launders that detail into every subsequent
turn as though the patient had said it. And it must record what the
assistant *declined* to do, so a refusal is not quietly forgotten and
re-litigated three turns later.
"""

from __future__ import annotations

SUMMARY_SYSTEM_PROMPT = """
You compress an earlier portion of a conversation between a patient and a
healthcare information assistant, so that later turns keep their context.

Write a factual digest, at most 150 words. Include:
- what the patient asked about, in their own terms;
- any specific dates, medications, test names or appointments discussed, so
  that a later "it" or "that one" can still be resolved;
- anything the assistant said was unavailable or declined to answer.

Rules:
- Record only what appears in the transcript. Add nothing, infer nothing.
- Do not answer anything, offer advice, or draw clinical conclusions.
- No preamble. Output the digest only.
- If an earlier summary is supplied, merge the new turns into it and return
  one combined digest, still within the word limit.
""".strip()


def build_summary_request(*, previous_summary: str | None, transcript: str) -> str:
    """The user-side payload for a summarization call."""
    parts: list[str] = []
    if previous_summary:
        parts.append(f"EARLIER SUMMARY:\n{previous_summary}")
    parts.append(f"NEW TURNS TO FOLD IN:\n{transcript}")
    return "\n\n".join(parts)


def render_summary_for_prompt(summary: str) -> str:
    """How a stored summary is presented back to the assistant.

    Labelled as a summary rather than pasted in as if it were transcript, so
    the model does not quote it as something the patient said verbatim.
    """
    return (
        "EARLIER IN THIS CONVERSATION (summarized, not verbatim):\n"
        f"{summary.strip()}"
    )
