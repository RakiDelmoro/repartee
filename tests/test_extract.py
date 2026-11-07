import pytest

from app import extract

SOURCE = (
    "Okay, let's start. The main thing I want to settle today is the launch date. "
    "Let's lock that in, the launch moves to Q2, targeting the second week of April. "
    "Tom, I want that documented so nobody is planning against March."
)


def test_chunker_respects_size_and_sentence_boundaries():
    text = " ".join(f"This is sentence number {i}." for i in range(200))
    chunks = extract.chunk_transcript(text, max_chars=300)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.endswith(".")
        assert len(chunk) <= 320


def test_chunker_returns_single_chunk_for_short_text():
    assert extract.chunk_transcript("Short meeting. Nothing happened.") == [
        "Short meeting. Nothing happened."
    ]


def test_evidence_matches_despite_punctuation_and_case():
    quote = "the launch moves to Q2 targeting the second week of april"
    assert extract.evidence_is_grounded(quote, SOURCE)


def test_evidence_must_appear_in_the_source():
    assert not extract.evidence_is_grounded("we will launch in June instead", SOURCE)
    assert not extract.evidence_is_grounded("", SOURCE)


def test_evidence_tolerates_a_dropped_word():
    assert extract.evidence_is_grounded(
        "Tom I want that documented so nobody is planning March", SOURCE
    )


def test_parse_response_handles_a_markdown_fence():
    raw = '```json\n{"decisions": [], "actions": [], "questions": [], "topics": []}\n```'
    assert extract.parse_response(raw)["decisions"] == []


def test_parse_response_handles_leading_prose():
    raw = 'Here you go:\n{"decisions": [], "actions": [], "questions": [], "topics": ["x"]}'
    assert extract.parse_response(raw)["topics"] == ["x"]


def test_parse_response_rejects_non_json():
    with pytest.raises(ValueError):
        extract.parse_response("I cannot help with that.")


def test_ungrounded_events_are_dropped():
    parsed = {
        "decisions": [
            {
                "text": "Launch moves to Q2",
                "evidence": "the launch moves to Q2, targeting the second week of April",
            },
            {"text": "Cancel the project", "evidence": "we cancelled the project entirely"},
        ],
        "actions": [
            {
                "text": "Document the launch date",
                "owner": "Tom",
                "due": None,
                "evidence": "Tom, I want that documented so nobody is planning against March",
            }
        ],
        "questions": [],
        "topics": ["launch date"],
    }

    events = extract.to_events(parsed, SOURCE)
    assert [e.kind for e in events] == ["decision", "action", "topic"]
    assert events[1].owner == "Tom"
    assert all("Cancel" not in e.text for e in events)


def test_duplicate_events_are_merged():
    duplicate = [
        extract.ExtractedEvent("decision", "Launch moves to Q2", "e1"),
        extract.ExtractedEvent("decision", "launch moves to  Q2.", "e2"),
        extract.ExtractedEvent("topic", "Launch moves to Q2", "e3"),
    ]
    merged = extract._dedupe(duplicate)
    assert len(merged) == 2


def test_owner_is_kept_when_the_quote_names_them():
    parsed = {
        "actions": [
            {
                "text": "Document the launch date",
                "owner": "Tom",
                "evidence": "Tom, I want that documented so nobody is planning against March",
            }
        ]
    }
    events = extract.to_events(parsed, SOURCE)
    assert events[0].owner == "Tom"


def test_owner_is_dropped_when_the_quote_does_not_name_them():
    parsed = {
        "actions": [
            {
                "text": "Bring the cost comparison",
                "owner": "Sam",
                "evidence": "And bring the cost comparison to the next review.",
            }
        ]
    }
    source = (
        "Sam, can you own the staging plan? Yeah, I'll set up the staging plan. "
        "And bring the cost comparison to the next review."
    )
    events = extract.to_events(parsed, source)
    assert events[0].owner is None


def test_overlapping_decisions_are_merged():
    source = (
        "We are not staying on the current provider for another year. "
        "We start with staging, and we revisit production after a month of numbers."
    )
    parsed = {
        "decisions": [
            {
                "text": "Do not renew with the current provider",
                "evidence": "We are not staying on the current provider for another year. We start with staging",
            },
            {
                "text": "Start with staging, revisit production later",
                "evidence": "We start with staging, and we revisit production after a month of numbers",
            },
        ]
    }

    events = extract._merge_overlapping(extract.to_events(parsed, source), source)
    assert len(events) == 1
    assert "Do not renew" in events[0].text and "Start with staging" in events[0].text


class _TruncatingClient:
    """Test double: truncates once, then answers properly."""

    def __init__(self):
        self.calls = 0

    def complete(self, messages, max_tokens=2048):
        self.calls += 1
        if self.calls == 1:
            raise extract.TruncatedResponse("{")
        return '{"decisions": [{"text": "Launch moves to Q2", "evidence": "the launch moves to Q2, targeting the second week of April"}], "actions": [], "questions": [], "topics": []}'


def test_adjacent_decision_fragments_are_merged():
    source = (
        "Let's reverse the earlier decision. We are not staying on the current provider "
        "for another year. We start with staging, and we revisit production after a month of numbers."
    )
    parsed = {
        "decisions": [
            {
                "text": "Reverse the earlier decision",
                "evidence": "Let's reverse the earlier decision. We are not staying on the current provider for another year.",
            },
            {
                "text": "Start with staging, revisit production later",
                "evidence": "We start with staging, and we revisit production after a month of numbers.",
            },
        ]
    }

    merged = extract._merge_overlapping(extract.to_events(parsed, source), source)
    assert len(merged) == 1
    assert "Reverse the earlier decision" in merged[0].text
    assert "Start with staging" in merged[0].text


def test_adjacent_actions_stay_separate():
    source = (
        "Tom will document the date. Priya will get legal to review the clause."
    )
    parsed = {
        "actions": [
            {"text": "Document the date", "owner": None, "evidence": "Tom will document the date."},
            {"text": "Get legal review", "owner": None, "evidence": "Priya will get legal to review the clause."},
        ]
    }

    merged = extract._merge_overlapping(extract.to_events(parsed, source), source)
    assert len(merged) == 2


def test_truncated_response_is_retried():
    client = _TruncatingClient()
    events = extract.extract(SOURCE, client)
    assert client.calls == 2
    assert [e.kind for e in events] == ["decision"]
