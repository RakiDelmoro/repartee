"""Pure-function tests for :mod:`src.build_synth` (no TTS downloads)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from src import build_synth as bs


# ---------------------------------------------------------------------------
# Dialogue mix and writers
# ---------------------------------------------------------------------------


def test_dialogue_mix_exact_quotas_and_deterministic() -> None:
    assert bs.dialogue_mix(10, seed=3).count("assistant") == 7
    assert bs.dialogue_mix(10, seed=3).count("casual") == 2
    assert bs.dialogue_mix(10, seed=3).count("backchannel") == 1
    assert bs.dialogue_mix(10, seed=3) == bs.dialogue_mix(10, seed=3)
    assert bs.dialogue_mix(100, seed=1).count("assistant") == 70
    assert bs.dialogue_mix(100, seed=1).count("casual") == 20
    assert bs.dialogue_mix(100, seed=1).count("backchannel") == 10
    assert bs.dialogue_mix(0) == []
    with pytest.raises(ValueError):
        bs.dialogue_mix(-1)


def test_category_for_index_covers_a_block() -> None:
    block = [bs.category_for_index(index, seed=5) for index in range(10)]
    assert block.count("assistant") == 7
    assert block.count("casual") == 2
    assert block.count("backchannel") == 1
    assert block == [bs.category_for_index(index, seed=5) for index in range(10)]
    with pytest.raises(ValueError):
        bs.category_for_index(-1)


def test_scripted_writer_is_deterministic_and_valid() -> None:
    writer = bs.ScriptedDialogueWriter(seed=0)
    first = writer.write("synth-0-0000000", category="casual", seed=11)
    second = writer.write("synth-0-0000000", category="casual", seed=11)
    assert first == second
    assert 2 <= len(first.utterances) <= 8
    assert first.agent_voice == ""  # filled from the backend at render time
    assert all(u.speaker in ("user", "agent") for u in first.utterances)
    assert "mm-hmm" in " ".join(
        u.text for u in writer.write("x", category="backchannel", seed=1).utterances
    )


def test_dialogue_validation() -> None:
    with pytest.raises(ValueError):
        bs.Dialogue("id", "casual", (bs.Utterance("user", "hi"),), "p", 0)
    with pytest.raises(ValueError):
        bs.Dialogue("id", "unknown", (bs.Utterance("user", "hi"), bs.Utterance("agent", "yo")), "p", 0)
    with pytest.raises(ValueError):
        bs.Utterance("robot", "hi")
    with pytest.raises(ValueError):
        bs.Utterance("user", "  ")


def test_parse_dialogue_text_valid_and_invalid() -> None:
    utterances = bs.parse_dialogue_text(
        '```json\n{"turns": [{"speaker": "user", "text": "Hi"}, '
        '{"speaker": "agent", "text": "Hello"}]}\n```',
        dialogue_id="d1",
    )
    assert [u.speaker for u in utterances] == ["user", "agent"]
    with pytest.raises(bs.DialogueError):
        bs.parse_dialogue_text("not json", dialogue_id="d1")
    with pytest.raises(bs.DialogueError):
        bs.parse_dialogue_text('{"turns": [{"speaker": "robot", "text": "hi"}]}', dialogue_id="d1")
    with pytest.raises(bs.DialogueError):
        bs.parse_dialogue_text('{"turns": []}', dialogue_id="d1")


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------


def test_compose_timeline_gap_rule() -> None:
    turns = [bs.TimingTurn("user" if i % 2 == 0 else "agent", 1.0) for i in range(6)]
    placements = bs.compose_timeline(
        turns,
        seed=1,
        gap_probability=1.0,
        barge_in_probability=0.0,
        overlap_probability=0.0,
        backchannel_probability=0.0,
    )
    starts = [p.start for p in placements]
    for previous, current in zip(placements, placements[1:]):
        assert math.isclose(current.start - previous.end, 0.64, abs_tol=1e-4)
    assert starts[0] == 0.0


def test_compose_timeline_never_overlaps_same_speaker() -> None:
    turns = [
        bs.TimingTurn("user" if i % 3 else "agent", 0.4 + (i % 4) * 0.3)
        for i in range(40)
    ]
    placements = bs.compose_timeline(turns, seed=7, backchannel_probability=0.5)
    for speaker in ("user", "agent"):
        spans = sorted((p.start, p.end) for p in placements if p.speaker == speaker)
        for (_, previous_end), (next_start, _) in zip(spans, spans[1:]):
            assert next_start >= previous_end - 1e-6
    assert placements == sorted(placements, key=lambda p: (p.start, p.speaker, p.end))


def test_compose_timeline_barge_in_cuts_agent_and_pads() -> None:
    placements = bs.compose_timeline(
        [bs.TimingTurn("agent", 2.0), bs.TimingTurn("user", 1.0)],
        seed=0,
        barge_in_probability=1.0,
        overlap_probability=0.0,
        gap_probability=1.0,
        backchannel_probability=0.0,
    )
    agent, user = placements[0], placements[1]
    assert agent.speaker == "agent"
    assert agent.end < agent.nominal_end
    assert agent.pad_after > 0
    assert math.isclose(user.start, agent.end, abs_tol=1e-6)
    # The user starts before the agent would have finished.
    assert user.start < agent.nominal_end


def test_compose_timeline_overlap_rule() -> None:
    placements = bs.compose_timeline(
        [bs.TimingTurn("agent", 2.0), bs.TimingTurn("user", 1.0)],
        seed=0,
        barge_in_probability=0.0,
        overlap_probability=1.0,
        gap_probability=1.0,
        backchannel_probability=0.0,
    )
    agent, user = placements[0], placements[1]
    overlap = agent.end - user.start
    assert 0.10 - 1e-6 <= overlap <= 0.40 + 1e-6


def test_compose_timeline_backchannel_during_user_speech() -> None:
    placements = bs.compose_timeline(
        [bs.TimingTurn("user", 3.0)],
        seed=2,
        backchannel_probability=1.0,
        backchannel_options=(("mm-hmm", 0.4),),
    )
    assert len(placements) == 2
    user, backchannel = placements
    assert user.speaker == "user" and backchannel.speaker == "agent"
    assert backchannel.kind == "backchannel"
    assert backchannel.start >= user.start and backchannel.end <= user.end
    assert backchannel.text == "mm-hmm"


def test_timeline_end_includes_pad() -> None:
    placements = bs.compose_timeline(
        [bs.TimingTurn("agent", 2.0), bs.TimingTurn("user", 1.0)],
        seed=0,
        barge_in_probability=1.0,
        overlap_probability=0.0,
        backchannel_probability=0.0,
    )
    assert bs.timeline_end(placements) == max(p.nominal_end for p in placements)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------


def test_apply_augmentation_gain_only() -> None:
    audio = np.full(1000, 0.1, dtype=np.float32)
    out = bs.apply_augmentation(audio, 16000, bs.AugmentationSpec(gain_db=6.0206), seed=0)
    assert np.allclose(out, 0.2, atol=1e-4)
    assert np.allclose(audio, 0.1)  # input untouched


def test_apply_augmentation_noise_snr() -> None:
    audio = np.full(8000, 0.1, dtype=np.float32)
    out = bs.apply_augmentation(audio, 16000, bs.AugmentationSpec(snr_db=0.0), seed=3)
    noise = out - audio
    signal_rms = float(np.sqrt(np.mean(audio**2)))
    noise_rms = float(np.sqrt(np.mean(noise**2)))
    assert math.isclose(noise_rms, signal_rms, rel_tol=0.15)


def test_apply_augmentation_echo_delay() -> None:
    user = np.zeros(1600, dtype=np.float32)
    agent = np.zeros(1600, dtype=np.float32)
    agent[0] = 1.0
    spec = bs.AugmentationSpec(echo_gain=0.5, echo_delay_s=0.01)
    out = bs.apply_augmentation(user, 16000, spec, agent=agent, seed=0)
    assert out[159] == 0.0
    assert math.isclose(float(out[160]), 0.5, abs_tol=1e-6)


def test_apply_augmentation_echo_requires_agent() -> None:
    with pytest.raises(ValueError):
        bs.apply_augmentation(
            np.zeros(100, dtype=np.float32),
            16000,
            bs.AugmentationSpec(echo_gain=0.1, echo_delay_s=0.1),
        )


def test_apply_augmentation_reverb_changes_signal() -> None:
    audio = np.zeros(4000, dtype=np.float32)
    audio[0] = 1.0
    out = bs.apply_augmentation(
        audio, 16000, bs.AugmentationSpec(reverb=True), seed=4, peak_limit=False
    )
    assert out[0] != 1.0
    assert np.count_nonzero(out) > 1


def test_sample_augmentation_ranges_and_determinism() -> None:
    spec = bs.sample_augmentation(123)
    assert spec == bs.sample_augmentation(123)
    seen = [bs.sample_augmentation(seed) for seed in range(200)]
    gains = [spec.gain_db for spec in seen if spec.gain_db]
    assert gains and min(gains) >= -24.0 and max(gains) <= 15.0
    assert any(spec.snr_db is not None for spec in seen)
    assert any(spec.reverb for spec in seen)
    assert any(spec.echo for spec in seen)
    for spec in seen:
        if spec.echo:
            assert 0.0 < spec.echo_gain <= 0.2
            assert 0.1 <= spec.echo_delay_s <= 0.5


def test_cosine_similarity() -> None:
    assert bs.cosine_similarity(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == pytest.approx(1.0)
    assert bs.cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)
    assert bs.cosine_similarity(np.zeros(2), np.zeros(2)) == 0.0
    with pytest.raises(ValueError):
        bs.cosine_similarity(np.array([1.0]), np.array([1.0, 2.0]))


# ---------------------------------------------------------------------------
# Rendering and dry-run
# ---------------------------------------------------------------------------


def test_render_channels_places_audio_on_frame_grid() -> None:
    # User-only placement at the backend rate avoids any resampling (torch).
    placement = bs.Placement(
        source_index=0,
        speaker="user",
        text="hello",
        kind="normal",
        start=0.0,
        end=0.5,
        nominal_end=0.5,
        pad_after=0.0,
    )
    audio = np.full(8000, 0.1, dtype=np.float32)  # 0.5 s at 16 kHz
    user, agent, duration = bs.render_channels(
        [placement],
        utterance_audio=[audio],
        backchannel_audio={},
        backend_rate=16000,
    )
    assert duration == pytest.approx(0.64)  # ceil((0.5 + 0.1) * 12.5) = 8 frames
    assert user.size == int(duration * 16000)
    assert agent.size == int(duration * 24000)
    assert np.allclose(user[:8000], 0.1)
    assert np.allclose(agent, 0.0)


def test_dry_run_plan_is_labelled_and_has_no_manifest(tmp_path) -> None:
    plan = bs.dry_run_plan(hours=0.1, seed=8, dialogues=2)
    assert plan["dry_run"] is True
    assert plan["audio"] is False
    assert plan["manifest"] is False
    assert len(plan["examples"]) == 2
    assert plan["planned_dialogues_estimate"] >= 1
    for example in plan["examples"]:
        assert 2 <= len(example["turns"]) <= 8
        assert example["timeline"]
        assert example["estimated_duration_s"] > 0
        assert example["turns"][0]["speaker"] == "user"
    assert not list(tmp_path.iterdir())


def test_build_requires_real_writer() -> None:
    with pytest.raises(bs.SynthError):
        bs.build(
            out_dir=Path("unused"),
            raw_root=Path("unused"),
            hours=0.01,
            backend_name="kokoro",
            writer=None,
        )
    with pytest.raises(bs.SynthError):
        bs.build(
            out_dir=Path("unused"),
            raw_root=Path("unused"),
            hours=0.01,
            backend_name="kokoro",
            writer=bs.ScriptedDialogueWriter(),
        )


def test_render_dialogue_with_stub_backend(monkeypatch) -> None:
    import threading

    class StubBackend:
        name = "stub"
        sample_rate = 16000
        agent_voice = "agent"
        user_voices = ("user",)

        def synthesize(self, text: str, *, voice: str, seed: int = 0):
            # Deterministic 0.2 s of low-level audio; not training data, just
            # a test double so the pure pipeline can be exercised offline.
            return np.full(3200, 0.05 if voice == "agent" else 0.04, dtype=np.float32)

    def fake_resample(audio, source_rate, target_rate):
        if source_rate == target_rate:
            return np.asarray(audio, dtype=np.float32)
        target = int(round(len(audio) * target_rate / source_rate))
        x_old = np.linspace(0.0, 1.0, len(audio), endpoint=False)
        x_new = np.linspace(0.0, 1.0, target, endpoint=False)
        return np.interp(x_new, x_old, audio).astype(np.float32)

    monkeypatch.setattr(bs, "create_backend", lambda name, options=None: StubBackend())
    monkeypatch.setattr(bs, "_THREAD_CONTEXT", threading.local())
    monkeypatch.setattr(bs, "_resample", fake_resample)
    dialogue = bs.Dialogue(
        id="synth-0-0000000",
        category="casual",
        utterances=(
            bs.Utterance("user", "hello there"),
            bs.Utterance("agent", "hi, how are you"),
        ),
        persona="test",
        seed=42,
    )
    user, agent, turns, duration, spec, meta = bs._render_dialogue(
        dialogue,
        backend_name="stub",
        backend_options={},
        voice_check=False,
        device="cpu",
        seed=0,
    )
    assert user.dtype == np.float32 and agent.dtype == np.float32
    assert user.size == int(round(duration * 16000))
    assert agent.size == int(round(duration * 24000))
    assert len(turns) >= 2
    assert all(turn.end <= duration for turn in turns)
    assert meta["tts_backend"] == "stub"
    assert meta["augmentation"]["gain_db"] == pytest.approx(spec.gain_db)
    from src import schema

    record = schema.Conversation(
        id=dialogue.id,
        source=bs.SYNTH_SOURCE,
        license=bs.SYNTH_LICENSE,
        duration_s=duration,
        channels={"user": "user.flac", "agent": "agent.flac"},
        turns=turns,
        meta=meta,
    )
    record.validate()
