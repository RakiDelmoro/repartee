"""Unit tests for the pure FDB interval metrics and adjacent parsers.

No audio is decoded, no model is loaded and no network is touched.  The
synthetic timelines below are the specification for the formulas documented in
``eval.metrics``.
"""

from __future__ import annotations

import json
import math

import pytest

from eval import metrics
from eval.metrics import (
    aggregate_latencies,
    aggregate_rates,
    as_interval,
    backchannel_resume,
    backchannel_resume_flags,
    gate5_checks,
    interruption_success,
    interruption_success_flags,
    is_backchannel,
    macro_average,
    normalize_intervals,
    overlap_s,
    pause_takeovers,
    pause_tor,
    self_interruption_events,
    self_interruption_rate,
    self_interruptions_per_minute,
    speech_span,
    stop_latencies,
    stop_latency,
    time_to_first_audio,
    turn_taking_success,
    turn_taking_success_flags,
)


# ---------------------------------------------------------------------------
# Interval primitives
# ---------------------------------------------------------------------------


def test_as_interval_and_overlap() -> None:
    assert as_interval([1, 2]) == (1.0, 2.0)
    assert overlap_s((0.0, 2.0), (1.5, 3.0)) == 0.5
    assert overlap_s((0.0, 1.0), (1.0, 2.0)) == 0.0
    with pytest.raises(ValueError):
        as_interval([2, 1])
    with pytest.raises(ValueError):
        as_interval([1])
    with pytest.raises(ValueError):
        as_interval([float("nan"), 1.0])


def test_normalize_merges_and_can_preserve_onsets() -> None:
    raw = [(2.0, 3.0), (0.0, 1.0), (0.9, 1.5), (3.0, 4.0)]
    assert normalize_intervals(raw) == [(0.0, 1.5), (2.0, 4.0)]
    assert normalize_intervals(raw, merge=False) == [
        (0.0, 1.0),
        (0.9, 1.5),
        (2.0, 3.0),
        (3.0, 4.0),
    ]
    assert normalize_intervals([(0.0, 0.0)]) == []
    assert speech_span([(4.0, 5.0), (1.0, 2.0)]) == 4.0


def test_is_backchannel_thresholds() -> None:
    assert is_backchannel((0.0, 0.5), n_words=2)
    assert is_backchannel((0.0, 0.5))
    # One second is not "shorter than one second" in the released code.
    assert not is_backchannel((0.0, 1.0), n_words=2)
    # A sub-second but wordy segment is a takeover, not a backchannel.
    assert not is_backchannel((0.0, 0.5), n_words=4)
    # Without word timings, only duration is observable.
    assert is_backchannel((0.0, 0.5), n_words=None)


# ---------------------------------------------------------------------------
# Pause handling
# ---------------------------------------------------------------------------


def test_pause_tor_silence_and_backchannel() -> None:
    pauses = [(1.0, 1.8)]
    assert pause_tor(pauses, []) == 0.0
    assert pause_tor(pauses, [(1.0, 2.5)]) == 1.0
    assert pause_tor(pauses, [(1.2, 1.7)], word_counts=[2]) == 0.0
    # A short but wordy interjection is a takeover under the released rule.
    assert pause_tor(pauses, [(1.2, 1.7)], word_counts=[5]) == 1.0


def test_pause_tor_only_counts_speech_overlapping_the_pause() -> None:
    # Speech entirely before or after the pause is not a pause takeover.
    assert pause_tor([(1.0, 2.0)], [(0.0, 0.5), (2.5, 4.0)]) == 0.0
    # Speech that starts before and continues across the pause does count.
    assert pause_tor([(1.0, 2.0)], [(0.5, 3.0)]) == 1.0


def test_pause_tor_multiple_pauses_and_missing_events() -> None:
    pauses = [(1.0, 2.0), (5.0, 6.0)]
    agent = [(1.2, 2.8), (5.2, 5.5)]
    assert pause_takeovers(pauses, agent) == [True, False]
    assert pause_tor(pauses, agent) == 0.5
    assert pause_tor([], agent) is None


def test_pause_tor_rejects_misaligned_word_counts() -> None:
    with pytest.raises(ValueError):
        pause_tor([(1.0, 2.0)], [(1.2, 1.4)], word_counts=[1, 2])


# ---------------------------------------------------------------------------
# Interruption management
# ---------------------------------------------------------------------------


def test_interruption_success_takes_turn_after_user_end() -> None:
    interruptions = [(2.0, 3.0)]
    flags = interruption_success_flags(interruptions, [(3.2, 5.0)])
    assert flags == [True]
    assert interruption_success(interruptions, [(3.2, 5.0)]) == 1.0
    assert time_to_first_audio(3.0, [(3.2, 5.0)]) == pytest.approx(0.2)


def test_interruption_only_a_backchannel_is_failure() -> None:
    assert interruption_success([(2.0, 3.0)], [(3.2, 3.6)], word_counts=[2]) == 0.0


def test_interruption_talking_over_is_not_a_new_response() -> None:
    # The model was already speaking and never stopped; that is not a turn
    # taken after the interruption.
    assert interruption_success([(2.0, 3.0)], [(1.0, 4.5)]) == 0.0


def test_interruption_response_window_and_empty() -> None:
    assert interruption_success([(2.0, 3.0)], [(20.0, 22.0)], response_window_s=10.0) == 0.0
    assert interruption_success([], [(3.0, 4.0)]) is None


def test_turn_taking_success_uses_turn_boundaries() -> None:
    assert turn_taking_success_flags([2.0], [(2.4, 4.0)]) == [True]
    assert turn_taking_success_flags([2.0], [(0.0, 1.5)]) == [False]
    assert turn_taking_success([2.0, 5.0], [(2.4, 4.0), (5.2, 6.5)]) == 1.0
    assert turn_taking_success([], [(2.4, 4.0)]) is None


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def test_time_to_first_audio_cases() -> None:
    assert time_to_first_audio(1.0, [(1.2, 2.0)]) == pytest.approx(0.2)
    assert time_to_first_audio(1.0, [(0.5, 1.5)]) == 0.0
    assert time_to_first_audio(1.0, []) is None
    assert time_to_first_audio(1.0, [(2.0, 3.0)], window_s=0.5) is None
    assert time_to_first_audio(1.0, [(2.0, 3.0)], window_s=1.0) == pytest.approx(1.0)


def test_stop_latency_measures_until_agent_stops() -> None:
    assert stop_latency(1.5, [(0.0, 2.0)]) == pytest.approx(0.5)
    # Silent at the onset: already stopped, not an error.
    assert stop_latency(1.0, [(2.0, 3.0)]) == 0.0
    # Only user intervals that actually overlap agent speech contribute.
    user = [(1.5, 1.7), (3.0, 3.5)]
    assert stop_latencies(user, [(0.0, 2.0)]) == [pytest.approx(0.5)]
    assert stop_latencies([(3.0, 3.5)], [(0.0, 2.0)]) == []


# ---------------------------------------------------------------------------
# Backchannels and self-interruption
# ---------------------------------------------------------------------------


def test_backchannel_resume_continuation_and_dip() -> None:
    assert backchannel_resume([(2.0, 2.5)], [(0.0, 5.0)]) == 1.0
    # 100 ms dip inside the cue, next onset still inside it.
    assert backchannel_resume([(2.0, 2.5)], [(0.0, 2.1), (2.4, 5.0)]) == 1.0
    # Long stop is a new turn, not a resume.
    assert backchannel_resume([(2.0, 2.5)], [(0.0, 2.1), (4.0, 5.0)]) == 0.0
    # A cue while the model is silent cannot be resumed.
    assert backchannel_resume([(1.0, 1.5)], [(3.0, 4.0)]) == 0.0
    assert backchannel_resume([], [(0.0, 1.0)]) is None


def test_backchannel_resume_multiple_cues() -> None:
    cues = [(2.0, 2.5), (7.0, 7.4)]
    agent = [(0.0, 5.0), (8.0, 9.0)]
    assert backchannel_resume_flags(cues, agent) == [True, False]
    assert backchannel_resume(cues, agent) == 0.5


def test_self_interruption_detects_overlapping_agent_playout() -> None:
    assert self_interruption_events([(0.0, 2.0), (1.0, 3.0)]) == [(1, 1.0)]
    assert self_interruption_rate([(0.0, 2.0), (1.0, 3.0)]) == 0.5
    assert self_interruption_rate([(0.0, 1.0), (1.2, 2.0)]) == 0.0
    assert self_interruption_rate([]) == 0.0
    assert self_interruptions_per_minute([(0.0, 2.0), (1.0, 3.0)], 60.0) == 1.0


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_aggregate_rates_and_latencies() -> None:
    summary = aggregate_rates([True, False, True], name="x")
    assert summary == {"name": "x", "rate": pytest.approx(2 / 3), "successes": 2, "n": 3}
    assert aggregate_rates([])["rate"] is None

    latencies = aggregate_latencies([0.2, None, 0.4])
    assert latencies["n"] == 2 and latencies["missing"] == 1
    assert latencies["mean_s"] == pytest.approx(0.3)
    assert latencies["median_s"] == pytest.approx(0.3)
    assert latencies["p90_s"] == pytest.approx(0.4)
    assert aggregate_latencies([])["mean_s"] is None


def test_macro_average_skips_missing_categories() -> None:
    assert macro_average({"a": 0.2, "b": None, "c": 0.4}) == pytest.approx(0.3)
    assert macro_average({"a": None}) is None


def test_gate5_checks_equality_and_missing_values() -> None:
    checks = gate5_checks(
        pause_tor=0.60,
        interruption_success=0.90,
        ttfa_s=0.40,
        backchannel_resume=0.70,
    )
    assert all(check.passed for check in checks)
    missing = gate5_checks(
        pause_tor=None,
        interruption_success=None,
        ttfa_s=None,
        backchannel_resume=None,
    )
    assert all(check.passed is None for check in missing)


def test_summarize_events_is_json_ready() -> None:
    summary = metrics.summarize_events(
        pause_flags=[False, True],
        interruption_flags=[True, True],
        ttfa_values=[0.2, None],
        backchannel_flags=[True, False],
        agent_speech=[(0.0, 2.0), (1.0, 3.0)],
    )
    assert summary["pause_tor"]["rate"] == 0.5
    assert summary["interruption_success"]["rate"] == 1.0
    assert summary["ttfa"]["missing"] == 1
    assert summary["self_interruption"]["events"] == 1
    json.dumps(summary)


# ---------------------------------------------------------------------------
# A small end-to-end synthetic timeline
# ---------------------------------------------------------------------------


def test_synthetic_timeline_metrics() -> None:
    user_pause = [(2.0, 2.4)]
    agent = [(0.0, 1.2), (2.2, 3.4), (5.0, 7.0)]
    # Pause takeover: the 2.2..3.4 segment overlaps the pause and is >1 s.
    assert pause_tor(user_pause, agent) == 1.0
    # Interruption at 4..5 s: response starts at 5.0 s.
    assert interruption_success_flags([(4.0, 5.0)], agent) == [True]
    assert time_to_first_audio(5.0, agent) == 0.0
    # Backchannel 4..5 s is during the 3.4..5.0 s silence -> not a resume.
    assert backchannel_resume([(4.0, 5.0)], agent) == 0.0
    # Turn end at 1.2 s: the 2.2 s onset is a valid response.
    assert turn_taking_success_flags([1.2], agent) == [True]
    assert self_interruption_rate(agent) == 0.0
    assert math.isclose(speech_span(agent), 7.0)


# ---------------------------------------------------------------------------
# Adjacent pure parsers (still interval-only, no downloads)
# ---------------------------------------------------------------------------


def test_fdb_jsonl_session_parser(tmp_path) -> None:
    from eval.fdb import load_jsonl_sessions

    path = tmp_path / "sessions.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "a",
                        "category": "pause_handling",
                        "pause": [[1.0, 1.5]],
                        "agent_speech": [[1.1, 2.3]],
                        "agent_word_counts": [4],
                    }
                ),
                json.dumps(
                    {
                        "id": "b",
                        "version": "v1.5",
                        "category": "user_backchannel",
                        "backchannels": [[2.0, 2.4]],
                        "agent_speech": [[0.0, 3.0]],
                        "user_wav": "b.wav",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    sessions = load_jsonl_sessions(path)
    assert [session.session_id for session in sessions] == ["a", "b"]
    assert sessions[0].version == "v1"
    assert sessions[0].pauses == [(1.0, 1.5)]
    assert sessions[0].agent_word_counts == [4]
    assert sessions[1].version == "v1.5"
    assert sessions[1].backchannels == [(2.0, 2.4)]
    assert str(sessions[1].user_wav).endswith("b.wav")


def test_fdb_missing_data_error_names_url(tmp_path) -> None:
    from eval.fdb import FDB_DATA_URL, FdbError, discover_sessions

    with pytest.raises(FdbError) as error:
        discover_sessions(tmp_path / "missing")
    assert FDB_DATA_URL in str(error.value)


def test_fdb_backchannel_rule() -> None:
    from eval.fdb import _backchannel_sample

    takeover, frequency, jsd, reason = _backchannel_sample(
        [(0.0, 0.5)], [2], duration_s=10.0, ground_truth=None
    )
    assert takeover is False
    assert frequency == pytest.approx(0.1)
    assert jsd is None and reason is not None
    takeover, _, jsd, _ = _backchannel_sample(
        [(0.0, 2.0)], [4], duration_s=10.0, ground_truth=None
    )
    assert takeover is True


def test_text_regression_pure_helpers() -> None:
    from eval.text_regression import (
        extract_gsm8k_answer,
        gsm8k_exact_match,
        mmlu_prompt,
        normalize_numeric,
        within_tolerance,
    )

    assert extract_gsm8k_answer("work\n#### 1,234") == "1234"
    assert extract_gsm8k_answer("the answer is 7.") == "7"
    assert extract_gsm8k_answer("no digits here") is None
    assert normalize_numeric("$1,234.0") == "1234"
    assert gsm8k_exact_match("1234", "1234") is True
    assert gsm8k_exact_match("1234", "1235") is False

    prompt = mmlu_prompt("Q?", ["alpha", "beta", "gamma", "delta"])
    assert "A. alpha" in prompt[1]["content"]
    assert "D. delta" in prompt[1]["content"]

    passing = within_tolerance(0.705, 0.72)
    assert passing["delta_points"] == pytest.approx(-1.5)
    assert passing["passes"] is True
    assert passing["no_regression"] is True
    failing = within_tolerance(0.69, 0.72)
    assert failing["delta_points"] == pytest.approx(-3.0)
    assert failing["passes"] is False
    assert within_tolerance(None, 0.7)["passes"] is None

    from eval.text_regression import gate4_artifact

    gate = gate4_artifact(
        {
            "status": "ok",
            "suite": "mmlu",
            "model": "candidate",
            "examples": 200,
            "offset": 0,
            "limit": 200,
            "seed": 0,
            "accuracy": 0.705,
            "comparison": {
                "status": "ok",
                "delta_points": -1.5,
                "passes": True,
                "no_regression": True,
                "tolerance_points": 2.0,
                "baseline_accuracy": 0.72,
                "candidate_accuracy": 0.705,
            },
        }
    )
    assert gate["gate"] == "Gate 4" and gate["result"] == "PASS" and gate["passed"] is True
    failed = gate4_artifact(
        {"status": "ok", "suite": "mmlu", "comparison": {"status": "ok", "passes": False, "delta_points": -3.0}}
    )
    assert failed["result"] == "FAIL" and failed["passed"] is False
    unknown = gate4_artifact({"status": "ok", "suite": "mmlu"})
    assert unknown["result"] == "UNVERIFIED" and unknown["passed"] is False


def test_pilot_scenarios_and_keyword_scoring() -> None:
    from eval.pilot import (
        SCENARIOS,
        keyword_score,
        load_scenarios,
        scenarios_json,
        validate_scenarios,
    )

    assert len(SCENARIOS) == 20
    counts: dict[str, int] = {}
    for scenario in SCENARIOS:
        counts[scenario.category] = counts.get(scenario.category, 0) + 1
    assert counts == {
        "corrections": 4,
        "interruptions": 4,
        "noise": 3,
        "names": 3,
        "refusals": 3,
        "backchannels": 3,
    }
    validate_scenarios(SCENARIOS)
    assert len({scenario.scenario_id for scenario in SCENARIOS}) == 20

    score, matched = keyword_score("Sure, I can't help with that.", ["can't"], mode="any")
    assert score == 1.0 and matched == ["can't"]
    score, matched = keyword_score("It is Ko-ro.", ["Ko-ro", "Koro"], mode="any")
    assert score == 1.0 and matched == ["Ko-ro"]
    score, _ = keyword_score("Lena will email Leo.", ["Lena", "Leo"], mode="all")
    assert score == 1.0
    assert keyword_score("", []) == (None, [])


def test_pilot_scenario_json_roundtrip(tmp_path) -> None:
    from eval.pilot import (
        SCENARIOS,
        load_scenarios,
        scenarios_json,
        validate_scenarios,
    )

    path = tmp_path / "scenarios.json"
    path.write_text(json.dumps(scenarios_json(SCENARIOS)), encoding="utf-8")
    loaded = load_scenarios(path)
    assert [scenario.scenario_id for scenario in loaded] == [
        scenario.scenario_id for scenario in SCENARIOS
    ]
    assert loaded[0].turns[0].text == SCENARIOS[0].turns[0].text

    with pytest.raises(Exception):
        validate_scenarios(SCENARIOS[:-1])
