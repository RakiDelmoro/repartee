"""Frame schedule for text injection into a full-duplex speech model.

A tri-stream duplex model (Haru-4B) consumes one text token per 133 ms frame
together with its own audio and the user's audio.  Its text stream is an input:
to make it say a sentence, put that sentence's token ids on consecutive frames
in place of the sampled ones.

Two properties of the trained stream constrain the schedule:

- the measured working schedule is *dense*: one token per frame, no PADs
  between words. Haru's audio head stops after a phrase when a PAD is inserted
  between words in forced text; ``gap=0`` renders the whole sentence.
  (The tokenizer still splits words, so a gap, if used, must go at a word
  boundary, never between subword pieces.)
- a silent model marks the start of speech with an end-of-padding (EOP) token on
  the frame before speaking, so an injection into silence starts with EOP.

This module is model-free: it only builds the frame list.  ``research/haru_inject.py``
feeds it into a ``DuplexSession``.
"""

from __future__ import annotations

from collections import deque
from threading import Lock


def word_starts(tokenizer, token_ids: list[int]) -> list[bool]:
    """True for tokens that begin a word.

    A word start is a token whose own decoding begins with a space (the usual
    sentencepiece/byte-BPE convention).  Punctuation and subword continuations
    decode without a leading space and stay in the same dense run.
    """
    return [tokenizer.decode([token]).startswith(" ") for token in token_ids]


def injection_schedule(
    token_ids: list[int],
    starts: list[bool],
    pad_id: int,
    eop_id: int,
    gap: int = 0,
    lead_eop: bool = False,
    tail_pad: int = 0,
) -> list[int]:
    """Forced text-stream frames for one injected sentence.

    ``eop`` when ``lead_eop``; each word at its first token and its subword
    pieces on the following frames; ``gap`` PAD frames between words; then
    ``tail_pad`` PAD frames so the turn can close.
    """
    if len(starts) != len(token_ids):
        raise ValueError("one start flag per token")
    schedule: list[int] = []
    if lead_eop:
        schedule.append(eop_id)
    for index, token in enumerate(token_ids):
        if index and starts[index]:
            schedule.extend([pad_id] * gap)
        schedule.append(token)
    schedule.extend([pad_id] * tail_pad)
    return schedule


class TextInjector:
    """Frame-by-frame text queue for a duplex model's text stream.

    The model loop calls :meth:`poll` before choosing a word and
    :meth:`observe` after emitting a frame.  ``poll`` returns a token to force,
    or ``None`` to let the model sample.  Calls may come from a background
    thread; a lock serialises the queue.
    """

    def __init__(self, tokenizer, pad_id: int, eop_id: int, gap: int = 0, idle_frames: int = 4):
        self.tokenizer = tokenizer
        self.pad_id = pad_id
        self.eop_id = eop_id
        self.gap = gap
        self.idle_frames = idle_frames
        self._frames: deque[int] = deque()
        self._pending: list[int] | None = None
        self._pending_eop = False
        self._pad_run = idle_frames  # a fresh session is silent, hence idle
        self._lock = Lock()

    @property
    def idle(self) -> bool:
        """A pause, not a word gap: enough consecutive PAD frames."""
        with self._lock:
            return self._pad_run >= self.idle_frames

    def speak(
        self,
        text: str,
        *,
        wait: bool = False,
        interrupt: bool = True,
        gap: int | None = None,
        lead_eop: bool | None = None,
        tail_pad: int = 0,
    ) -> list[int]:
        """Queue ``text``; return the frame schedule that will be forced."""
        tokenizer = self.tokenizer
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        auto_eop = lead_eop is None
        with self._lock:
            if auto_eop:
                lead_eop = self._pad_run >= self.idle_frames
            schedule = injection_schedule(
                ids, word_starts(tokenizer, ids), self.pad_id, self.eop_id,
                self.gap if gap is None else gap, lead_eop, tail_pad,
            )
            if wait and self._pad_run < self.idle_frames:
                self._pending = schedule  # spoken at the next pause
                self._pending_eop = auto_eop
                return schedule
            if interrupt:
                self._frames.clear()
            self._pending = None
            self._pending_eop = False
            self._frames.extend(schedule)
            return schedule

    def poll(self) -> int | None:
        """The token to force on this frame, or None to sample normally."""
        with self._lock:
            if self._pending is not None and not self._frames and self._pad_run >= self.idle_frames:
                if self._pending_eop:
                    self._pending.insert(0, self.eop_id)
                self._frames.extend(self._pending)
                self._pending = None
                self._pending_eop = False
                self._pad_run = 0
            return self._frames.popleft() if self._frames else None

    def observe(self, token: int) -> None:
        """Record the token emitted on the frame that just finished."""
        with self._lock:
            self._pad_run = self._pad_run + 1 if token == self.pad_id else 0
