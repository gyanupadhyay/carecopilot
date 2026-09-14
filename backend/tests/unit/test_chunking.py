"""Section-aware chunking (PRD §19)."""

from __future__ import annotations

from app.rag.chunking import (
    PREAMBLE_SECTION,
    Chunk,
    chunk_document,
    contextualize,
    split_sections,
)

NOTE = """CHIEF COMPLAINT
Persistent right knee pain.

HISTORY OF PRESENT ILLNESS
Patient describes knee pain that is worse when climbing stairs.

ASSESSMENT
Osteoarthritis of the knee with mechanical pain.

PLAN
Physical therapy referral placed. Follow-up in six weeks.
"""


def test_sections_are_split_on_headings() -> None:
    sections = split_sections(NOTE)
    assert [name for name, _ in sections] == [
        "Chief Complaint",
        "History of Present Illness",
        "Assessment",
        "Plan",
    ]


def test_section_names_are_readable_not_naive_title_case() -> None:
    """Names reach users in citations, so "History Of" would be visible."""
    names = [name for name, _ in split_sections(NOTE)]
    assert "History of Present Illness" in names
    assert "History Of Present Illness" not in names


def test_each_section_becomes_a_chunk() -> None:
    chunks = chunk_document(NOTE)
    assert len(chunks) == 4
    assert [c.index for c in chunks] == [0, 1, 2, 3]
    assert chunks[0].text == "Persistent right knee pain."
    assert chunks[3].section == "Plan"


def test_heading_text_is_not_duplicated_in_the_body() -> None:
    chunks = chunk_document(NOTE)
    assert not chunks[0].text.startswith("CHIEF COMPLAINT")


def test_text_before_any_heading_is_kept() -> None:
    chunks = chunk_document("Loose narrative with no headings at all.")
    assert len(chunks) == 1
    assert chunks[0].section == PREAMBLE_SECTION


def test_sentences_are_not_mistaken_for_headings() -> None:
    """A line ending in a period is prose, however short."""
    note = "ASSESSMENT\nStable.\nPlan is to review in six weeks.\n"
    sections = split_sections(note)
    assert [name for name, _ in sections] == ["Assessment"]
    assert "Plan is to review" in sections[0][1]


def test_lowercase_known_heading_is_recognized() -> None:
    sections = split_sections("Assessment:\nStable disease.\n")
    assert sections[0][0] == "Assessment"


def test_long_section_is_split_on_boundaries_not_mid_word() -> None:
    body = " ".join(f"Sentence number {n} about the patient." for n in range(200))
    chunks = chunk_document(f"ASSESSMENT\n{body}", max_chars=400, overlap_chars=50)

    assert len(chunks) > 1
    assert all(c.section == "Assessment" for c in chunks)
    for chunk in chunks:
        assert not chunk.text.startswith(" ")
        # A mid-word cut would leave a fragment like "Senten"
        assert chunk.text.split()[0].isalpha()


def test_split_chunks_overlap_so_facts_survive_the_boundary() -> None:
    body = " ".join(f"Fact {n} recorded at the visit." for n in range(100))
    chunks = chunk_document(f"PLAN\n{body}", max_chars=300, overlap_chars=80)
    assert len(chunks) > 1
    # Some text from the end of one chunk reappears at the start of the next.
    assert any(
        chunks[i].text[-40:].strip()[:20] in chunks[i + 1].text
        for i in range(len(chunks) - 1)
    )


def test_chunk_indexes_are_unique_and_sequential_across_sections() -> None:
    body = " ".join(f"Detail {n}." for n in range(120))
    chunks = chunk_document(f"ASSESSMENT\n{body}\n\nPLAN\n{body}", max_chars=300)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_empty_document_yields_no_chunks() -> None:
    assert chunk_document("") == []
    assert chunk_document("   \n\n  ") == []


def test_contextualize_prefixes_title_date_and_section() -> None:
    chunk = Chunk(text="Physical therapy referral placed.", section="Plan", index=0)
    rendered = contextualize(chunk, title="Knee osteoarthritis", date="2026-09-10")
    assert rendered.startswith("Knee osteoarthritis — 2026-09-10 — Plan")
    assert "Physical therapy referral placed." in rendered


def test_contextualize_does_not_repeat_a_date_already_in_the_title() -> None:
    chunk = Chunk(text="Stable.", section="Assessment", index=0)
    rendered = contextualize(
        chunk, title="Hypertension — 2026-09-10", date="2026-09-10"
    )
    assert rendered.count("2026-09-10") == 1


def test_contextualize_without_metadata_returns_the_section_header() -> None:
    chunk = Chunk(text="Stable.", section="Assessment", index=0)
    assert contextualize(chunk).startswith("Assessment")


def test_token_estimate_is_positive() -> None:
    assert Chunk(text="a", section="Plan", index=0).token_estimate >= 1
