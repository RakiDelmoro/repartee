"""Cue policy: derive the transition from the relation, not from a phrase list.

The reasoner that produces the injected text also supplies two fields:

    {"text": "...", "relation": "shift|correct|remind|answer|continue", "topic": "the Sun"}

`cue_for` decides whether any words are needed at all. A cue is emitted only when the
injected text is a shift/correction/reminder **and** its own opening does not already
name the topic. No topic, no relation, or a self-introducing opening means no cue.

This replaced a hardcoded pair ("One more thing," / "Speaking of that,"): the measured
redundancy of "About the sun, light from the sun…" is why self-introducing facts get no
cue at all.
"""

from __future__ import annotations

import re

# The connective is a frame; the words come from the reasoner's topic.
FRAMES = {
    "shift": "About {topic},",
    "correct": "Actually, about {topic},",
    "remind": "Going back to {topic},",
}
NO_CUE = {"answer", "continue"}


def first_phrase(text: str) -> str:
    """The opening the listener actually hears: first comma or sentence end."""
    parts = re.split(r"(?<=[,.;:!?])\s+", text.strip())
    return parts[0] if parts else text.strip()


def names_topic(phrase: str, topic: str, window: int = 8) -> bool:
    """True when the opening already says the topic, so a cue would repeat it."""
    heads = [w for w in re.findall(r"[a-z0-9']+", topic.lower())
             if len(w) > 2 and w not in {"the", "and", "its", "their", "his", "her"}]
    if not heads:
        return False
    words = re.findall(r"[a-z0-9']+", phrase.lower())[:window]
    return any(head in words for head in heads)


def cue_for(relation: str | None, topic: str | None, text: str) -> str | None:
    """None means: say it with no cue. That is the common, correct case."""
    if not relation or relation.lower() in NO_CUE:
        return None
    if not topic:
        return None
    if names_topic(first_phrase(text), topic):
        return None
    frame = FRAMES.get(relation.lower())
    return frame.format(topic=topic) if frame else None


def compose(first: str, cue: str | None) -> str:
    """The cue becomes the opening of the first phrase, one breath, lowercase tail.

    Acronyms (NASA) and single-letter words keep their case; ordinary capitalised
    words are lowercased so the sentence flows through the cue.
    """
    if not cue:
        return first
    word = re.match(r"[A-Za-z]+", first)
    tail = first
    if word and not (len(word.group(0)) > 1 and word.group(0)[:2].isupper()):
        tail = first[:1].lower() + first[1:]
    return f"{cue} {tail}".strip()
