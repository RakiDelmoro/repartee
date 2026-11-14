"""Cue policy: derived from relation + topic, with "no cue" as a first-class outcome."""

from __future__ import annotations

from research.cues import compose, cue_for, first_phrase, names_topic

SUN = "Light from the Sun takes eight minutes to reach Earth."
BUDGET = "Its budget moved to the satellite team."


def test_self_introducing_fact_gets_no_cue():
    # measured: "About the sun, light from the sun…" is redundant
    assert cue_for("shift", "the Sun", SUN) is None


def test_unnamed_topic_gets_a_topic_bearing_cue():
    assert cue_for("shift", "the Harrow project", BUDGET) == "About the Harrow project,"
    assert cue_for("correct", "the Harrow project", BUDGET) == "Actually, about the Harrow project,"
    assert cue_for("remind", "the Harrow project", BUDGET) == "Going back to the Harrow project,"


def test_answer_and_continue_never_get_a_cue():
    for relation in ("answer", "continue"):
        assert cue_for(relation, "the Harrow project", BUDGET) is None


def test_without_a_topic_there_is_no_cue():
    assert cue_for("shift", None, BUDGET) is None
    assert cue_for("shift", "", BUDGET) is None


def test_first_phrase_stops_at_the_first_punctuation():
    assert first_phrase("Its budget moved, according to Tom, to satellites.") == \
        "Its budget moved,"
    assert first_phrase("One sentence. Another one.") == "One sentence."


def test_the_opening_must_name_the_topic_not_merely_share_a_word():
    assert names_topic("The Harrow project was cancelled.", "the Harrow project")
    assert not names_topic("Its budget moved to the satellite team.", "the Harrow project")


def test_compose_keeps_it_in_one_breath_and_lowercases_the_tail():
    assert compose("Light from the Sun takes eight minutes.", "About the Sun,") == \
        "About the Sun, light from the Sun takes eight minutes."
    assert compose("Light from the Sun takes eight minutes.", None) == \
        "Light from the Sun takes eight minutes."
    assert compose("NASA confirmed the orbit.", "About NASA,") == "About NASA, NASA confirmed the orbit."
