"""Full-Duplex-Bench turn-taking metrics over speech intervals.

This module is a pure, deterministic re-implementation of the *behavioural*
parts of Full-Duplex-Bench (FDB) v1.0 and v1.5.  It deliberately works on
``(start, end)`` speech intervals in seconds instead of on audio, so the
definitions can be unit-tested and reused by any runner.

Sources of the definitions
--------------------------
FDB v1.0, *Full-Duplex-Bench: A Benchmark to Evaluate Full-duplex Spoken
Dialogue Models on Turn-taking Capabilities* (arXiv:2503.04721):

* **Takeover (TO)** is binary: ``TO = 0`` when the model is silent or emits
  only a backchannel; ``TO = 1`` otherwise.  The **Takeover Rate (TOR)** is
  ``TOR = (1/N) * sum_i TO_i`` over the N evaluation samples.
* **Pause handling** asks whether the model takes over while the *user* is
  still holding the turn through an internal pause.  Lower TOR is better.
* **Backchanneling** asks for short acknowledgements.  A segment is a
  backchannel when it is shorter than one second and contains fewer than two
  words (the reference implementation uses ``< 1 s`` and ``<= 3`` ASR chunks
  for pause handling, and ``< 1 s`` and ``<= 2`` chunks for backchannel).
  Lower TOR and higher backchannel frequency are reported; the timing
  similarity is Jensen-Shannon divergence against the ICC ground truth.
* **Smooth turn taking** reports TOR (higher is better) and the response
  latency ``output_start - input_end``, clamped at zero.  Latency is averaged
  only over samples where ``TO = 1``.
* **User interruption** reports TOR, which is intended to be 1: the model
  should take the turn after a user barges in.  Latency is measured from the
  end of the interrupting user speech to the first model speech.

FDB v1.5, *Full-Duplex-Bench v1.5: Evaluating Overlap Handling for
Full-Duplex Speech Models* (arXiv:2507.23159), adds overlap scenarios and:

* **Stop latency** — from the onset of overlapping user speech to the moment
  the model stops speaking.
* **Response latency** — from the end of the overlapping user speech to the
  model's next utterance.
* The post-overlap behaviour classes **Respond**, **Resume**, **Uncertain**
  and **Unknown**.  ``Resume`` means the model disregards the overlap and
  continues (or completes) the pre-overlap task.  We expose that as
  :func:`backchannel_resume` for user backchannels and other non-directive
  overlap events.

The constants in the reference implementation (``turn_duration_threshold = 1``
and ``turn_num_words_threshold = 3`` in ``eval_pause_handling.py``) are exposed
as keyword arguments.  They are *not* claimed to be the paper's normative
definitions: they are what the released code does, and reproducing FDB numbers
requires the same thresholds.

Interval convention
-------------------
All intervals are half-open, ``[start, end)``, in seconds, with
``0 <= start <= end``.  Zero-length intervals contain no speech.  Unless a
function says otherwise, overlapping or touching intervals belonging to the
same speaker are treated as activity (and merged) before scoring.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

Interval = tuple[float, float]

_EPS = 1e-9

# Blueprint v1.0 Gate 5 / Gate 7 thresholds.  These are project gates, not
# values prescribed by the FDB papers.
GATE5_PAUSE_TOR_MAX = 0.60
GATE5_INTERRUPTION_SUCCESS_MIN = 0.90
GATE5_TTFA_MAX_S = 0.400
GATE5_BACKCHANNEL_RESUME_MIN = 0.70
GATE7_SELF_INTERRUPTION_MAX = 0.0


# ---------------------------------------------------------------------------
# Interval primitives
# ---------------------------------------------------------------------------


def as_interval(value: Sequence[float]) -> Interval:
    """Validate and convert a two-element sequence to ``(start, end)``."""
    if len(value) != 2:
        raise ValueError(f"an interval needs two values, got {value!r}")
    start, end = float(value[0]), float(value[1])
    if not math.isfinite(start) or not math.isfinite(end):
        raise ValueError(f"non-finite interval: {value!r}")
    if start > end + _EPS:
        raise ValueError(f"interval start precedes end: {value!r}")
    if end < start:
        end = start
    return (start, max(start, end))


def normalize_intervals(
    intervals: Iterable[Sequence[float]],
    *,
    merge: bool = True,
    merge_gap_s: float = 0.0,
) -> list[Interval]:
    """Return sorted ``[start, end)`` intervals.

    With ``merge=True`` (the default), overlapping intervals — and intervals
    separated by at most ``merge_gap_s`` — are unioned.  This is the right
    operation for "when was this speaker active".  Use ``merge=False`` when
    individual onsets matter, e.g. :func:`self_interruption_rate`.
    """
    cleaned: list[Interval] = []
    for raw in intervals:
        if raw is None:
            continue
        start, end = as_interval(raw)
        if end > start:
            cleaned.append((start, end))
    cleaned.sort()
    if not merge:
        return cleaned
    merged: list[Interval] = []
    for start, end in cleaned:
        if merged and start <= merged[-1][1] + merge_gap_s + _EPS:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def overlap_s(a: Interval, b: Interval) -> float:
    """Length of the intersection of two half-open intervals."""
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def overlaps(a: Interval, b: Interval) -> bool:
    """True when two half-open intervals share positive duration."""
    return overlap_s(a, b) > _EPS


def total_duration(intervals: Iterable[Interval]) -> float:
    """Summed duration after merging (i.e. union duration)."""
    return sum(end - start for start, end in normalize_intervals(intervals))


def speech_span(intervals: Sequence[Interval]) -> float:
    """Span from the first onset to the last offset (FDB's output duration)."""
    if not intervals:
        return 0.0
    return max(end for _, end in intervals) - min(start for start, _ in intervals)


def is_backchannel(
    interval: Interval,
    *,
    n_words: int | None = None,
    max_duration_s: float = 1.0,
    max_words: int = 3,
) -> bool:
    """FDB backchannel test: short and word-poor.

    ``max_words`` defaults to the reference implementation's pause-handling
    threshold (``<= 3`` ASR chunks).  The paper's stricter definition uses
    fewer than two words; pass ``max_words=1`` for that reading.
    """
    start, end = as_interval(interval)
    if end - start >= max_duration_s - _EPS:
        return False
    return n_words is None or n_words <= max_words


def _classify_takeover(
    intervals: Sequence[Interval],
    word_counts: Sequence[int] | None,
    *,
    min_duration_s: float = 1.0,
    max_backchannel_words: int = 3,
) -> bool:
    """FDB ``TO``: false for silence or a short, word-poor segment."""
    if not intervals:
        return False
    duration = speech_span(intervals)
    if duration >= min_duration_s - _EPS:
        return True
    if word_counts is None:
        # Without word timings the released code's duration branch is the only
        # observable signal; a sub-second utterance is treated as a candidate
        # backchannel rather than a takeover.
        return False
    return sum(int(n) for n in word_counts) > max_backchannel_words


def _event_words(
    agent: Sequence[Interval],
    word_counts: Sequence[int] | None,
    event: Interval,
) -> list[int] | None:
    if word_counts is None:
        return None
    return [
        int(count)
        for segment, count in zip(agent, word_counts, strict=False)
        if overlaps(segment, event)
    ]


def _event_segments(agent: Sequence[Interval], event: Interval) -> list[Interval]:
    return [segment for segment in agent if overlaps(segment, event)]


# ---------------------------------------------------------------------------
# Pause handling
# ---------------------------------------------------------------------------


def pause_takeovers(
    pauses: Iterable[Sequence[float]],
    agent_speech: Iterable[Sequence[float]],
    *,
    word_counts: Sequence[int] | None = None,
    min_duration_s: float = 1.0,
    max_backchannel_words: int = 3,
) -> list[bool]:
    """Per-pause FDB takeover decisions (``TO_i``).

    ``pauses`` are user-internal pauses, usually the ``[PAUSE]`` annotations
    in FDB's ``pause.json``.  Agent intervals that overlap a pause are judged
    with the FDB rule: silent or a sub-second, word-poor segment is not a
    takeover; anything longer (or wordier) is.

    ``word_counts`` is optional and aligned one-to-one with ``agent_speech``.
    """
    agent = normalize_intervals(agent_speech)
    if word_counts is not None and len(word_counts) != len(agent):
        raise ValueError(
            f"word_counts has {len(word_counts)} entries for {len(agent)} agent intervals"
        )
    decisions: list[bool] = []
    for raw_pause in normalize_intervals(pauses):
        segments = _event_segments(agent, raw_pause)
        counts = _event_words(agent, word_counts, raw_pause)
        decisions.append(
            _classify_takeover(
                segments,
                counts,
                min_duration_s=min_duration_s,
                max_backchannel_words=max_backchannel_words,
            )
        )
    return decisions


def pause_tor(
    pauses: Iterable[Sequence[float]],
    agent_speech: Iterable[Sequence[float]],
    *,
    word_counts: Sequence[int] | None = None,
    min_duration_s: float = 1.0,
    max_backchannel_words: int = 3,
) -> float | None:
    """FDB pause-handling TOR: ``mean_i TO_i`` over pauses.

    Lower is better.  Returns ``None`` when there are no pauses, rather than
    inventing a value.
    """
    decisions = pause_takeovers(
        pauses,
        agent_speech,
        word_counts=word_counts,
        min_duration_s=min_duration_s,
        max_backchannel_words=max_backchannel_words,
    )
    if not decisions:
        return None
    return sum(decisions) / len(decisions)


# ---------------------------------------------------------------------------
# Interruption management
# ---------------------------------------------------------------------------


def _response_flags_after(
    user_ends: Sequence[float],
    agent: Sequence[Interval],
    word_counts: Sequence[int] | None,
    *,
    response_window_s: float,
    min_duration_s: float,
    max_backchannel_words: int,
) -> list[bool]:
    flags: list[bool] = []
    for user_end in user_ends:
        window = (user_end - _EPS, user_end + max(0.0, response_window_s))
        selected = [
            (segment, index)
            for index, segment in enumerate(agent)
            if segment[0] >= window[0] and segment[0] <= window[1]
        ]
        segments = [segment for segment, _ in selected]
        counts = (
            None
            if word_counts is None
            else [int(word_counts[index]) for _, index in selected]
        )
        flags.append(
            _classify_takeover(
                segments,
                counts,
                min_duration_s=min_duration_s,
                max_backchannel_words=max_backchannel_words,
            )
        )
    return flags


def interruption_success_flags(
    interruptions: Iterable[Sequence[float]],
    agent_speech: Iterable[Sequence[float]],
    *,
    word_counts: Sequence[int] | None = None,
    response_window_s: float = 10.0,
    min_duration_s: float = 1.0,
    max_backchannel_words: int = 3,
) -> list[bool]:
    """Per-interruption FDB ``TO`` flags (``1`` when the model takes the turn).

    The response is the agent speech that *starts* at or after the end of the
    interrupting user speech and within ``response_window_s``.  A response
    that is only silence or a backchannel is a failure (``TO = 0``), following
    the released ``eval_user_interruption.py``.
    """
    agent = normalize_intervals(agent_speech)
    if word_counts is not None and len(word_counts) != len(agent):
        raise ValueError(
            f"word_counts has {len(word_counts)} entries for {len(agent)} agent intervals"
        )
    ends = [end for _, end in normalize_intervals(interruptions)]
    return _response_flags_after(
        ends,
        agent,
        word_counts,
        response_window_s=response_window_s,
        min_duration_s=min_duration_s,
        max_backchannel_words=max_backchannel_words,
    )


def interruption_success(
    interruptions: Iterable[Sequence[float]],
    agent_speech: Iterable[Sequence[float]],
    *,
    word_counts: Sequence[int] | None = None,
    response_window_s: float = 10.0,
    min_duration_s: float = 1.0,
    max_backchannel_words: int = 3,
) -> float | None:
    """FDB interruption success rate (``TOR`` after a barge-in).

    Higher is better.  Returns ``None`` when there are no interruptions.
    """
    flags = interruption_success_flags(
        interruptions,
        agent_speech,
        word_counts=word_counts,
        response_window_s=response_window_s,
        min_duration_s=min_duration_s,
        max_backchannel_words=max_backchannel_words,
    )
    if not flags:
        return None
    return sum(flags) / len(flags)


def turn_taking_success_flags(
    user_ends: Iterable[float],
    agent_speech: Iterable[Sequence[float]],
    *,
    word_counts: Sequence[int] | None = None,
    response_window_s: float = 10.0,
    min_duration_s: float = 1.0,
    max_backchannel_words: int = 3,
) -> list[bool]:
    """Per-turn FDB smooth-turn-taking ``TO`` flags.

    A turn counts as taken when non-backchannel agent speech starts at or
    after the user turn end and within ``response_window_s``.  This is the
    same decision rule as :func:`interruption_success_flags`, applied to
    ordinary turn boundaries rather than barge-ins.
    """
    agent = normalize_intervals(agent_speech)
    if word_counts is not None and len(word_counts) != len(agent):
        raise ValueError(
            f"word_counts has {len(word_counts)} entries for {len(agent)} agent intervals"
        )
    return _response_flags_after(
        [float(end) for end in user_ends],
        agent,
        word_counts,
        response_window_s=response_window_s,
        min_duration_s=min_duration_s,
        max_backchannel_words=max_backchannel_words,
    )


def turn_taking_success(
    user_ends: Iterable[float],
    agent_speech: Iterable[Sequence[float]],
    *,
    word_counts: Sequence[int] | None = None,
    response_window_s: float = 10.0,
    min_duration_s: float = 1.0,
    max_backchannel_words: int = 3,
) -> float | None:
    """FDB smooth-turn-taking TOR (higher is better; ``None`` when empty)."""
    flags = turn_taking_success_flags(
        user_ends,
        agent_speech,
        word_counts=word_counts,
        response_window_s=response_window_s,
        min_duration_s=min_duration_s,
        max_backchannel_words=max_backchannel_words,
    )
    if not flags:
        return None
    return sum(flags) / len(flags)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def time_to_first_audio(
    user_end_s: float,
    agent_speech: Iterable[Sequence[float]],
    *,
    window_s: float | None = None,
    clamp_negative: bool = True,
) -> float | None:
    """Latency from a user turn end to the next agent onset, in seconds.

    This is FDB v1.0 smooth-turn-taking latency (``output_start -
    input_end_time``) and FDB v1.5 response latency.  When the agent is
    already speaking at ``user_end_s`` the official code clamps the value to
    zero.  Returns ``None`` when no onset occurs (optionally within
    ``window_s``), because a missing response must not be averaged in as a
    zero-latency success.
    """
    user_end = float(user_end_s)
    agent = normalize_intervals(agent_speech)
    for start, end in agent:
        if start <= user_end + _EPS and end > user_end + _EPS:
            return 0.0 if clamp_negative else max(0.0, start - user_end)
    for start, _ in agent:
        if start >= user_end - _EPS:
            latency = start - user_end
            if latency < 0 and clamp_negative:
                latency = 0.0
            if window_s is not None and latency > window_s + _EPS:
                return None
            return max(0.0, latency) if clamp_negative else latency
    return None


def stop_latency(
    user_onset_s: float,
    agent_speech: Iterable[Sequence[float]],
) -> float | None:
    """FDB v1.5 stop latency: overlap onset -> the model's next speech offset.

    Returns ``0.0`` when the agent is silent at the onset.  The return type is
    ``float | None`` to leave room for callers that model unmeasured stops,
    but this interval implementation always returns a finite value.
    """
    onset = float(user_onset_s)
    for start, end in normalize_intervals(agent_speech):
        if start <= onset + _EPS < end - _EPS:
            return max(0.0, end - onset)
    return 0.0


def stop_latencies(
    user_speech: Iterable[Sequence[float]],
    agent_speech: Iterable[Sequence[float]],
) -> list[float]:
    """Per-overlap stop latencies for FDB v1.5 timing tables.

    Only user intervals that overlap agent speech contribute; a user turn
    during agent silence has no stop event to measure (it is not recorded as
    a zero-latency stop).
    """
    agent = normalize_intervals(agent_speech)
    values: list[float] = []
    for start, end in normalize_intervals(user_speech):
        if not any(overlaps((start, end), segment) for segment in agent):
            continue
        value = stop_latency(start, agent)
        if value is None:  # pragma: no cover - stop_latency always returns a number
            continue
        values.append(value)
    return values


def ttfa_stats(
    user_ends: Iterable[float],
    agent_speech: Iterable[Sequence[float]],
    *,
    window_s: float | None = None,
) -> dict[str, Any]:
    """Aggregate TTFA over user-turn ends, counting misses explicitly."""
    agent = normalize_intervals(agent_speech)
    values: list[float] = []
    missing = 0
    for user_end in user_ends:
        value = time_to_first_audio(user_end, agent, window_s=window_s)
        if value is None:
            missing += 1
        else:
            values.append(value)
    summary = aggregate_latencies(values)
    summary["missing"] = missing
    return summary


# ---------------------------------------------------------------------------
# Backchannels and self-interruption
# ---------------------------------------------------------------------------


def backchannel_resume_flags(
    backchannels: Iterable[Sequence[float]],
    agent_speech: Iterable[Sequence[float]],
    *,
    resume_gap_s: float = 0.5,
    require_active: bool = True,
) -> list[bool]:
    """Per-backchannel FDB v1.5 ``Resume`` decisions.

    A resume means the model disregards the brief user cue and keeps its turn:
    either its speech spans the cue, or speech resumes within
    ``resume_gap_s`` of the cue's end without first yielding a long silence.
    A stop followed by a fresh answer later is *not* a resume (it would be a
    ``Respond`` or ``Unknown`` class).

    With ``require_active=True`` a cue that begins while the model is silent
    cannot be a resume.  FDB v1.5 injects these cues during model speech, so
    this also catches a cue that arrived after the model had already stopped.
    """
    agent = normalize_intervals(agent_speech)
    flags: list[bool] = []
    for cue in normalize_intervals(backchannels):
        cue_start, cue_end = cue
        active_before = any(
            start <= cue_start + _EPS and end > cue_start + _EPS
            for start, end in agent
        )
        if require_active and not active_before:
            flags.append(False)
            continue

        spans_cue = any(
            start < cue_end - _EPS and end >= cue_end - _EPS for start, end in agent
        )
        if spans_cue:
            flags.append(True)
            continue

        # A segment that was already playing at cue onset may end inside the
        # cue.  The turn is resumed if the next onset follows promptly (it may
        # begin during the cue, e.g. the model briefly dipped for 100 ms).
        pre_end = max(
            (
                end
                for start, end in agent
                if start < cue_end - _EPS and end > cue_start + _EPS
            ),
            default=None,
        )
        if pre_end is None:
            flags.append(False)
            continue
        next_onset = min(
            (start for start, _ in agent if start >= pre_end - _EPS),
            default=None,
        )
        resumed = (
            next_onset is not None
            and next_onset - max(pre_end, cue_end) <= resume_gap_s + _EPS
        )
        flags.append(bool(resumed))
    return flags


def backchannel_resume(
    backchannels: Iterable[Sequence[float]],
    agent_speech: Iterable[Sequence[float]],
    *,
    resume_gap_s: float = 0.5,
    require_active: bool = True,
) -> float | None:
    """FDB v1.5 backchannel resume rate (higher is better).

    Returns ``None`` when there are no backchannels.  The FDB v1.5 paper calls
    the corresponding behaviour class ``Resume``; it is the fraction of
    backchannel events after which the model continues its pre-overlap turn.
    """
    flags = backchannel_resume_flags(
        backchannels,
        agent_speech,
        resume_gap_s=resume_gap_s,
        require_active=require_active,
    )
    if not flags:
        return None
    return sum(flags) / len(flags)


def self_interruption_events(
    agent_speech: Iterable[Sequence[float]],
    *,
    min_overlap_s: float = 0.0,
) -> list[tuple[int, float]]:
    """Agent segments whose onset overlaps a still-playing earlier segment.

    Returns ``(segment_index, overlap_seconds)``.  Input is sorted by onset and
    is *not* merged: a self-interruption is exactly the case where two agent
    play-outs coexist on the timeline.  ``min_overlap_s`` filters out touch or
    sub-frame jitter.
    """
    segments = normalize_intervals(agent_speech, merge=False)
    events: list[tuple[int, float]] = []
    previous_end = -math.inf
    for index, (start, end) in enumerate(segments):
        if start < previous_end - _EPS:
            overlap = previous_end - start
            if overlap > min_overlap_s + _EPS:
                events.append((index, overlap))
        previous_end = max(previous_end, end)
    return events


def self_interruption_rate(
    agent_speech: Iterable[Sequence[float]],
    *,
    min_overlap_s: float = 0.0,
) -> float:
    """Fraction of agent segments that begin while agent audio is playing.

    Blueprint Gate 7 asks for *zero* self-interruptions in a live session.
    The denominator is the number of agent segments; an empty timeline scores
    ``0.0`` (nothing was played, so nothing interrupted itself).
    """
    segments = normalize_intervals(agent_speech, merge=False)
    if not segments:
        return 0.0
    return len(self_interruption_events(segments, min_overlap_s=min_overlap_s)) / len(
        segments
    )


def self_interruptions_per_minute(
    agent_speech: Iterable[Sequence[float]],
    duration_s: float,
    *,
    min_overlap_s: float = 0.0,
) -> float:
    """Self-interruption frequency per minute of session audio."""
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    events = self_interruption_events(agent_speech, min_overlap_s=min_overlap_s)
    return len(events) * 60.0 / float(duration_s)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_rates(flags: Sequence[bool], *, name: str = "rate") -> dict[str, Any]:
    """Count and rate for a sequence of binary decisions."""
    n = len(flags)
    successes = sum(1 for flag in flags if flag)
    return {
        "name": name,
        "rate": (successes / n) if n else None,
        "successes": successes,
        "n": n,
    }


def aggregate_latencies(values: Sequence[float | None]) -> dict[str, Any]:
    """Mean/median/p90/min/max over defined values, with a missing count."""
    present = [float(value) for value in values if value is not None]
    missing = len(values) - len(present)
    if not present:
        return {
            "mean_s": None,
            "median_s": None,
            "p90_s": None,
            "min_s": None,
            "max_s": None,
            "n": 0,
            "missing": missing,
        }
    ordered = sorted(present)
    # Nearest-rank p90, which is what a small benchmark subset needs.
    index = min(len(ordered) - 1, max(0, math.ceil(0.9 * len(ordered)) - 1))
    return {
        "mean_s": statistics.fmean(ordered),
        "median_s": statistics.median(ordered),
        "p90_s": ordered[index],
        "min_s": ordered[0],
        "max_s": ordered[-1],
        "n": len(ordered),
        "missing": missing,
    }


def macro_average(values: Mapping[str, float | None]) -> float | None:
    """Unweighted mean over categories that produced a number."""
    present = [float(value) for value in values.values() if value is not None]
    if not present:
        return None
    return statistics.fmean(present)


@dataclass(frozen=True)
class GateCheck:
    """One blueprint gate threshold and its measured value."""

    metric: str
    value: float | None
    threshold: float
    higher_is_better: bool

    @property
    def passed(self) -> bool | None:
        if self.value is None:
            return None
        if self.higher_is_better:
            return self.value >= self.threshold
        return self.value <= self.threshold

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "value": self.value,
            "threshold": self.threshold,
            "higher_is_better": self.higher_is_better,
            "passed": self.passed,
        }


def gate5_checks(
    *,
    pause_tor: float | None,
    interruption_success: float | None,
    ttfa_s: float | None,
    backchannel_resume: float | None,
) -> list[GateCheck]:
    """Blueprint Gate 5 checks (FDB v1 numbers, *not* FDB-defined thresholds).

    ``pause TOR < 60%``, ``interruption success > 90%``, ``TTFA < 400 ms`` and
    ``backchannel resume > 70%`` come from ``BLUEPRINT.md``; the FDB papers are
    descriptive and do not set pass marks.
    """
    return [
        GateCheck("pause_tor", pause_tor, GATE5_PAUSE_TOR_MAX, higher_is_better=False),
        GateCheck(
            "interruption_success",
            interruption_success,
            GATE5_INTERRUPTION_SUCCESS_MIN,
            higher_is_better=True,
        ),
        GateCheck("ttfa_s", ttfa_s, GATE5_TTFA_MAX_S, higher_is_better=False),
        GateCheck(
            "backchannel_resume",
            backchannel_resume,
            GATE5_BACKCHANNEL_RESUME_MIN,
            higher_is_better=True,
        ),
    ]


def summarize_events(
    *,
    pause_flags: Sequence[bool] = (),
    interruption_flags: Sequence[bool] = (),
    ttfa_values: Sequence[float | None] = (),
    backchannel_flags: Sequence[bool] = (),
    agent_speech: Iterable[Sequence[float]] = (),
) -> dict[str, Any]:
    """One JSON-ready summary for a category or for the whole benchmark."""
    pause = aggregate_rates(pause_flags, name="pause_tor")
    interruption = aggregate_rates(interruption_flags, name="interruption_success")
    ttfa = aggregate_latencies(ttfa_values)
    backchannel = aggregate_rates(backchannel_flags, name="backchannel_resume")
    return {
        "pause_tor": pause,
        "interruption_success": interruption,
        "ttfa": ttfa,
        "backchannel_resume": backchannel,
        "self_interruption": {
            "rate": self_interruption_rate(agent_speech),
            "events": len(self_interruption_events(agent_speech)),
        },
    }


__all__ = [
    "GATE5_BACKCHANNEL_RESUME_MIN",
    "GATE5_INTERRUPTION_SUCCESS_MIN",
    "GATE5_PAUSE_TOR_MAX",
    "GATE5_TTFA_MAX_S",
    "GATE7_SELF_INTERRUPTION_MAX",
    "GateCheck",
    "aggregate_latencies",
    "aggregate_rates",
    "as_interval",
    "backchannel_resume",
    "backchannel_resume_flags",
    "gate5_checks",
    "interruption_success",
    "interruption_success_flags",
    "is_backchannel",
    "macro_average",
    "normalize_intervals",
    "overlap_s",
    "overlaps",
    "pause_takeovers",
    "pause_tor",
    "self_interruption_events",
    "self_interruption_rate",
    "self_interruptions_per_minute",
    "speech_span",
    "stop_latencies",
    "stop_latency",
    "summarize_events",
    "time_to_first_audio",
    "total_duration",
    "ttfa_stats",
    "turn_taking_success",
    "turn_taking_success_flags",
]
