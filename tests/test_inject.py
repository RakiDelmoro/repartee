"""Frame schedule for injecting a sentence into a full-duplex text stream."""

from __future__ import annotations

import pytest

from app.inject import injection_schedule, word_starts

PAD, EOP = 100, 101


class PieceTokenizer:
    """Three words where one is split (" Har" + "row"), plus attached punctuation."""

    pieces = {1: " The", 2: " Har", 3: "row", 4: " project", 5: "."}

    def decode(self, ids):
        return "".join(self.pieces[token] for token in ids)


TOKENS = [1, 2, 3, 4, 5]


def schedule(**kwargs):
    starts = word_starts(PieceTokenizer(), TOKENS)
    return injection_schedule(TOKENS, starts, PAD, EOP, **kwargs)


def test_word_start_is_a_leading_space_not_every_token():
    assert word_starts(PieceTokenizer(), TOKENS) == [True, True, False, True, False]


def test_gap_separates_words_and_never_splits_one():
    assert schedule(gap=1) == [1, PAD, 2, 3, PAD, 4, 5]


def test_silent_model_starts_with_eop_and_closes_with_pad():
    assert schedule(lead_eop=True, tail_pad=2) == [EOP, 1, 2, 3, 4, 5, PAD, PAD]


def test_zero_gap_is_one_dense_run():
    assert schedule(gap=0) == TOKENS


def test_a_start_flag_per_token_is_required():
    with pytest.raises(ValueError):
        injection_schedule(TOKENS, [True] * (len(TOKENS) - 1), PAD, EOP)


class WordTokenizer(PieceTokenizer):
    """``encode`` maps a test word to preset ids; ``decode`` marks word starts."""

    ids = {"a": [10, 11], "b": [12], "c": [13, 14]}
    pieces = {**PieceTokenizer.pieces, 10: " w", 11: "w", 12: " w", 13: " w", 14: "w"}

    def encode(self, text, add_special_tokens=False):
        from types import SimpleNamespace

        return SimpleNamespace(ids=list(self.ids[text]))


PAD, EOP = 100, 101


def injector(idle_frames=2):
    from app.inject import TextInjector

    return TextInjector(WordTokenizer(), PAD, EOP, gap=1, idle_frames=idle_frames)


def test_speak_from_idle_starts_with_eop():
    inj = injector()
    assert inj.speak("a") == [EOP, 10, 11]
    assert inj.poll() == EOP
    inj.observe(EOP)
    assert inj.poll() == 10


def test_speak_mid_speech_has_no_eop():
    inj = injector()
    inj.observe(10)  # last frame was a word: speaking
    assert inj.speak("a") == [10, 11]
    assert inj.poll() == 10


def test_wait_holds_until_a_run_of_pad_frames():
    inj = injector(idle_frames=2)
    inj.observe(10)
    inj.speak("a", wait=True)
    assert inj.poll() is None
    inj.observe(PAD)
    assert inj.poll() is None          # one PAD is a word gap, not a pause
    inj.observe(PAD)
    assert inj.poll() == EOP           # idle: auto EOP then the sentence
    inj.observe(EOP)
    assert inj.poll() == 10


def test_wait_from_idle_speaks_immediately():
    inj = injector()
    assert inj.idle
    assert inj.speak("b", wait=True) == [EOP, 12]
    assert inj.poll() == EOP


def test_interrupt_drops_the_unspoken_remainder():
    inj = injector()
    inj.observe(10)
    inj.speak("a")
    assert inj.poll() == 10
    inj.speak("b")                     # interrupt: drops 11 from "a"
    assert inj.poll() == 12
    assert inj.poll() is None


def test_no_interrupt_appends_behind_the_current_sentence():
    inj = injector()
    inj.observe(10)
    inj.speak("a", interrupt=False)
    inj.speak("b", interrupt=False)
    assert [inj.poll(), inj.poll(), inj.poll(), inj.poll()] == [10, 11, 12, None]


def test_pad_run_is_reset_by_a_word():
    inj = injector(idle_frames=2)
    inj.observe(PAD)
    inj.observe(PAD)
    assert inj.idle
    inj.observe(10)
    assert not inj.idle
