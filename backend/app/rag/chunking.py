"""Section-aware chunking of clinical documents (PRD §19).

Clinical notes have structure, and that structure is the most useful signal
available for splitting them. A note is a sequence of labelled sections —
Chief Complaint, History, Examination, Assessment, Medications, Plan — and
each one is a self-contained answer to a different kind of question. Cutting
every 500 characters instead would routinely split an assessment across two
chunks, so a question about the assessment retrieves half of one.

So the primary boundary is the heading. Only a section too long to embed
usefully is split further, and then on paragraph and sentence boundaries
rather than mid-thought, with a small overlap so a fact that straddles the
cut survives in both pieces.

Each chunk carries the section it came from. That is what lets a retrieved
fragment be cited as "your 10 September note, Assessment" instead of as an
anonymous passage, and what lets Phase 4 filter by section.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Sections the generator writes, plus common real-world variants. Matching
#: is case-insensitive; this list exists so a heading is recognised even
#: when it is not upper-case.
#: Written in their display form. Section names are shown to users in
#: citations ("your 10 September note, Assessment"), so they are stored the
#: way they should be read — ``str.title()`` would render "History Of
#: Present Illness", capitalizing a preposition.
KNOWN_SECTIONS: tuple[str, ...] = (
    "Chief Complaint",
    "History of Present Illness",
    "History",
    "Examination",
    "Physical Examination",
    "Assessment",
    "Assessment and Plan",
    "Medications",
    "Plan",
    "Allergies",
    "Review of Systems",
    "Family History",
    "Social History",
    "Vitals",
    "Impression",
    "Follow-up",
)

_CANONICAL = {section.upper(): section for section in KNOWN_SECTIONS}

#: Words that stay lower-case when title-casing an unrecognised heading.
_MINOR_WORDS = frozenset({"of", "and", "the", "in", "on", "for", "to", "a", "an"})

#: A heading is a short line, on its own, with no terminal punctuation.
#: Requiring upper-case *or* membership of the known list keeps a sentence
#: such as "Plan is to review in six weeks." from being read as a heading.
_HEADING_RE = re.compile(r"^\s*([A-Z][A-Z /&'\-]{2,58})\s*:?\s*$")

#: Text before any heading — a note that opens without one still has to go
#: somewhere, and inventing a section name would be worse than saying so.
PREAMBLE_SECTION = "Preamble"

DEFAULT_MAX_CHARS = 1_200
DEFAULT_OVERLAP_CHARS = 150
#: Below this, a trailing fragment is merged back into the previous chunk
#: rather than stored as a chunk of its own: a 40-character chunk retrieves
#: on almost nothing and wastes an embedding.
MIN_CHUNK_CHARS = 80


@dataclass(frozen=True, slots=True)
class Chunk:
    text: str
    section: str
    index: int

    @property
    def token_estimate(self) -> int:
        """Four characters per token — close enough for a budget, not billing."""
        return max(1, len(self.text) // 4)


def _display_case(heading: str) -> str:
    """Title-case an unrecognised heading without capitalizing prepositions."""
    words = heading.split()
    return " ".join(
        word.capitalize()
        if index == 0 or word.lower() not in _MINOR_WORDS
        else word.lower()
        for index, word in enumerate(words)
    )


def _is_heading(line: str) -> str | None:
    """Return the display form of a heading line, or ``None``."""
    stripped = line.strip().rstrip(":").strip()
    if not stripped or len(stripped) > 60:
        return None
    canonical = _CANONICAL.get(stripped.upper())
    if canonical is not None:
        return canonical
    if _HEADING_RE.match(line) and not stripped.endswith("."):
        return _display_case(stripped)
    return None


def split_sections(content: str) -> list[tuple[str, str]]:
    """Split a document into ``(section, body)`` pairs, in order."""
    sections: list[tuple[str, list[str]]] = []
    current = PREAMBLE_SECTION
    buffer: list[str] = []

    for line in content.splitlines():
        heading = _is_heading(line)
        if heading is None:
            buffer.append(line)
            continue
        if buffer and "".join(buffer).strip():
            sections.append((current, buffer))
        buffer = []
        current = heading

    if buffer and "".join(buffer).strip():
        sections.append((current, buffer))

    return [
        (section, "\n".join(lines).strip())
        for section, lines in sections
        if "\n".join(lines).strip()
    ]


def _split_long_text(text: str, *, max_chars: int, overlap_chars: int) -> list[str]:
    """Split one oversized section on natural boundaries.

    Paragraphs first, then sentences, and only then a hard cut — a hard cut
    mid-word is a last resort, not the strategy.
    """
    if len(text) <= max_chars:
        return [text]

    units = [part for part in re.split(r"\n\s*\n", text) if part.strip()]
    if any(len(unit) > max_chars for unit in units):
        expanded: list[str] = []
        for unit in units:
            if len(unit) <= max_chars:
                expanded.append(unit)
            else:
                # Sentence boundaries: a period/question/exclamation followed
                # by whitespace and a capital or digit.
                expanded.extend(
                    part.strip()
                    for part in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", unit)
                    if part.strip()
                )
        units = expanded

    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}\n\n{unit}".strip() if current else unit
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
            # Carry the tail of the previous chunk forward so a fact split
            # across the boundary is retrievable from either side.
            tail = current[-overlap_chars:] if overlap_chars else ""
            current = f"{tail}\n\n{unit}".strip() if tail else unit
        else:
            current = unit

        while len(current) > max_chars:
            # A single unit still too long: hard cut, preferring a space.
            cut = current.rfind(" ", 0, max_chars)
            cut = cut if cut > max_chars // 2 else max_chars
            chunks.append(current[:cut].strip())
            current = current[max(0, cut - overlap_chars) :].strip()

    if current:
        chunks.append(current)

    # Fold a stub trailing chunk back into its predecessor.
    if len(chunks) > 1 and len(chunks[-1]) < MIN_CHUNK_CHARS:
        chunks[-2] = f"{chunks[-2]}\n\n{chunks[-1]}"
        chunks.pop()
    return chunks


def chunk_document(
    content: str,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[Chunk]:
    """Split a document into retrievable, section-labelled chunks.

    ``index`` is document-wide and sequential, so ``(document_id, index)``
    identifies a chunk uniquely and chunks can be re-assembled in order.
    """
    chunks: list[Chunk] = []
    for section, body in split_sections(content):
        for piece in _split_long_text(
            body, max_chars=max_chars, overlap_chars=overlap_chars
        ):
            text = piece.strip()
            if text:
                chunks.append(Chunk(text=text, section=section, index=len(chunks)))
    return chunks


def contextualize(
    chunk: Chunk, *, title: str | None = None, date: str | None = None
) -> str:
    """The string actually sent to the embedding model.

    Deliberately different from the stored text. A bare "Physical therapy
    referral placed." embeds with no indication of what it is about or when;
    prefixing the document title, date and section gives the vector that
    context and measurably improves retrieval for questions phrased around a
    condition or a time.

    The *stored* text stays clean, because that is what the LLM reads and
    what a citation points at — the prefix would otherwise be quoted back to
    the patient as though it were part of their note.
    """
    parts = [title] if title else []
    # Generated titles already end with the encounter date. Repeating it
    # would spend prefix tokens on a duplicate and weight the vector toward
    # a date string rather than the clinical content.
    if date and (not title or date not in title):
        parts.append(date)
    parts.append(chunk.section)

    header = " — ".join(part for part in parts if part)
    return f"{header}\n{chunk.text}" if header else chunk.text
