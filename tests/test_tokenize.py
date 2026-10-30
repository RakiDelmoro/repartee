"""Tests for src.tokenize: RPT1 round-trip, resume, CLI and verify helpers."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from src.audio import write_audio
from src import tokenize
from src.schema import (
    CODEBOOK_SIZE,
    CODEC_SAMPLE_RATE,
    CONTROL_IDS,
    FRAME_RATE,
    NUM_CODEBOOKS,
    Conversation,
    Turn,
    write_manifest,
)
from src.tokenize import (
    HEADER_SIZE,
    check_streaming_decode,
    conversation_arrays,
    decode_record,
    encode_record,
    load_index,
    read_record,
    record_bytes,
    word_error_rate,
)

SAMPLE_RATE = CODEC_SAMPLE_RATE
FRAME_SAMPLES = 1920


class FakeCodec:
    """Deterministic, per-frame-independent stand-in for LiquidCodec."""

    n_codebooks = NUM_CODEBOOKS
    sample_rate = SAMPLE_RATE
    frame_samples = FRAME_SAMPLES

    def encode(self, audio: np.ndarray) -> np.ndarray:
        waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
        if waveform.size % FRAME_SAMPLES:
            raise AssertionError("test codec expects whole 80 ms frames")
        frames = waveform.reshape(-1, FRAME_SAMPLES)
        energy = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
        base = np.floor(energy * 10000.0).astype(np.int64)
        offsets = np.arange(self.n_codebooks, dtype=np.int64) * 13
        codes = (base[:, None] + offsets[None, :]) % CODEBOOK_SIZE
        return codes.astype(np.uint16)

    def decode(self, codes: np.ndarray, streaming_cache=None, finalize: bool = True) -> np.ndarray:
        array = np.asarray(codes)
        if array.size == 0:
            return np.zeros(0, dtype=np.float32)
        values = (array[:, 0].astype(np.float32) - 1000.0) / 1000.0
        return np.repeat(values, FRAME_SAMPLES).astype(np.float32)

    def new_streaming_cache(self):
        return {}


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [1 + (ord(character) % 5000) for character in text]


class FakeASR:
    def transcribe(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
        return "hello world"


def _tone(frames: int, frequency: float, amplitude: float) -> np.ndarray:
    samples = np.arange(frames * FRAME_SAMPLES, dtype=np.float32) / SAMPLE_RATE
    return (amplitude * np.sin(2.0 * np.pi * frequency * samples)).astype(np.float32)


def _conversation(tmp_path: Path, conversation_id: str, frames: int, split: str) -> Conversation:
    raw = tmp_path / "raw" / conversation_id
    raw.mkdir(parents=True)
    user_path = raw / "user.wav"
    agent_path = raw / "agent.wav"
    write_audio(user_path, _tone(frames, 220.0, 0.20), SAMPLE_RATE)
    write_audio(agent_path, _tone(frames, 330.0, 0.12), SAMPLE_RATE)
    duration = frames / FRAME_RATE
    user_end = min(0.32, duration * 0.30)
    agent_start = duration * 0.40
    agent_end = duration * 0.80
    turns = [
        Turn("user", 0.0, user_end, "user says"),
        Turn("agent", agent_start, agent_end, "hello world"),
    ]
    return Conversation(
        id=conversation_id,
        source="toy",
        license="cc0",
        duration_s=duration,
        channels={"user": str(user_path), "agent": str(agent_path)},
        turns=turns,
        split=split,
    )


@pytest.fixture()
def built(tmp_path: Path):
    codec = FakeCodec()
    tokenizer = FakeTokenizer()
    conversations = [
        _conversation(tmp_path, "conv_a", frames=16, split="train"),
        _conversation(tmp_path, "conv_b", frames=24, split="valid"),
    ]
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, conversations)
    out = tmp_path / "tokens"
    summary = tokenize.build_index(
        [manifest],
        out,
        codec,
        text_tokenizer=tokenizer,
        progress=lambda message: None,
    )
    return {
        "tmp_path": tmp_path,
        "out": out,
        "manifest": manifest,
        "conversations": conversations,
        "summary": summary,
        "codec": codec,
        "tokenizer": tokenizer,
    }


def test_round_trip_arrays_and_offsets(built) -> None:
    index = load_index(built["out"])
    assert len(index["conversations"]) == 2
    assert index["n_codebooks"] == NUM_CODEBOOKS
    assert index["frame_rate"] == FRAME_RATE
    assert index["silence_frame"] is not None
    assert len(index["silence_frame"]) == NUM_CODEBOOKS

    expected_offset = 0
    for entry, conversation in zip(index["conversations"], built["conversations"]):
        assert entry["offset"] == expected_offset
        expected_offset += entry["bytes"]
        assert entry["bytes"] == record_bytes(entry["T"], NUM_CODEBOOKS)
        assert entry["id"] == conversation.id
        assert entry["split"] == conversation.split
        raw = (built["out"] / entry["shard"]).read_bytes()
        header = struct.unpack("<4sHHI", raw[entry["offset"] : entry["offset"] + HEADER_SIZE])
        assert header == (b"RPT1", 1, NUM_CODEBOOKS, entry["T"])

        record = read_record(
            built["out"] / entry["shard"],
            entry["offset"],
            n_codebooks=NUM_CODEBOOKS,
            frames=entry["T"],
        )
        expected = conversation_arrays(conversation, built["codec"], built["tokenizer"])
        assert np.array_equal(record["user_codes"], expected["user_codes"])
        assert np.array_equal(record["agent_codes"], expected["agent_codes"])
        assert np.array_equal(record["text"], expected["text"])
        assert np.array_equal(record["control"], expected["control"])
        assert record["user_codes"].dtype == np.uint16
        assert record["text"].dtype == np.int32
        assert record["control"].dtype == np.uint8
    assert expected_offset == index["shards"][0]["bytes"]


def test_summary_reports_bytes_per_hour(built) -> None:
    summary = built["summary"]
    assert summary["conversations"] == 2
    assert summary["new_conversations"] == 2
    assert summary["skipped"] == 0
    assert summary["shards"] == 1
    assert summary["hours"] > 0
    assert summary["record_bytes"] == summary["shard_bytes"]
    assert summary["bytes_per_hour"] > 0
    assert summary["bytes_per_hour"] == pytest.approx(summary["record_bytes"] / summary["hours"])


def test_resume_skips_existing(built) -> None:
    summary = tokenize.build_index(
        [built["manifest"]],
        built["out"],
        built["codec"],
        text_tokenizer=built["tokenizer"],
        progress=lambda message: None,
    )
    assert summary["new_conversations"] == 0
    assert summary["skipped"] == 2
    assert summary["conversations"] == 2
    index = load_index(built["out"])
    assert len(index["conversations"]) == 2


def test_shard_rotation_by_hours(built) -> None:
    out = built["tmp_path"] / "sharded"
    summary = tokenize.build_index(
        [built["manifest"]],
        out,
        built["codec"],
        text_tokenizer=built["tokenizer"],
        shard_hours=0.0001,
        progress=lambda message: None,
    )
    index = load_index(out)
    assert summary["shards"] == 2
    assert len(index["shards"]) == 2
    assert {entry["shard"] for entry in index["conversations"]} == {
        shard["file"] for shard in index["shards"]
    }
    for entry in index["conversations"]:
        assert entry["offset"] == 0
        assert entry["bytes"] == record_bytes(entry["T"], NUM_CODEBOOKS)


def test_limit_and_overwrite(built) -> None:
    out = built["tmp_path"] / "limited"
    summary = tokenize.build_index(
        [built["manifest"]],
        out,
        built["codec"],
        text_tokenizer=built["tokenizer"],
        limit=1,
        progress=lambda message: None,
    )
    assert summary["new_conversations"] == 1
    assert summary["limit_reached"] is True
    assert len(load_index(out)["conversations"]) == 1

    summary = tokenize.build_index(
        [built["manifest"]],
        out,
        built["codec"],
        text_tokenizer=built["tokenizer"],
        overwrite=True,
        progress=lambda message: None,
    )
    assert summary["new_conversations"] == 2
    assert len(load_index(out)["conversations"]) == 2


def test_record_codec_rejects_bad_input() -> None:
    user = np.zeros((2, NUM_CODEBOOKS), dtype=np.uint16)
    agent = np.zeros((2, NUM_CODEBOOKS), dtype=np.uint16)
    text = np.zeros(2, dtype=np.int32)
    control = np.full(2, CONTROL_IDS["SPEAK"], dtype=np.uint8)
    data = encode_record(user, agent, text, control)
    assert len(data) == record_bytes(2, NUM_CODEBOOKS)
    parsed = decode_record(data)
    assert np.array_equal(parsed["control"], control)

    with pytest.raises(ValueError, match="magic|version|truncated"):
        decode_record(b"XXXX" + data[4:])
    with pytest.raises(ValueError, match="truncated"):
        decode_record(data[:-1])
    with pytest.raises(ValueError, match="frames"):
        decode_record(data, frames=3)
    with pytest.raises(ValueError, match="out of range"):
        encode_record(
            np.full((1, NUM_CODEBOOKS), CODEBOOK_SIZE, dtype=np.uint16),
            agent[:1],
            text[:1],
            control[:1],
        )
    with pytest.raises(ValueError, match="control"):
        encode_record(user, agent, text, np.full(2, 99, dtype=np.uint8))


def test_load_index_rejects_unknown_version(built) -> None:
    index_path = built["out"] / "index.json"
    data = json.loads(index_path.read_text())
    data["version"] = 99
    index_path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="version"):
        load_index(built["out"])


def test_word_error_rate() -> None:
    assert word_error_rate("hello world", "hello world") == 0.0
    assert word_error_rate("hello world", "Hello, world!") == 0.0
    assert word_error_rate("a b c", "a c") == pytest.approx(1 / 3)
    assert word_error_rate("a b", "") == 1.0
    assert word_error_rate("", "extra") == 1.0
    assert word_error_rate("", "") == 0.0


def test_streaming_decode_equivalence_and_chunk_limit() -> None:
    codec = FakeCodec()
    audio = _tone(4, 440.0, 0.1)
    codes = codec.encode(audio)
    result = check_streaming_decode(codec, codes, chunk_frames=2)
    assert result["status"] == "PASS"
    assert result["chunks"] == 2
    assert result["max_chunk_seconds"] <= 0.160 + 1e-12
    assert result["max_abs_diff"] == 0.0

    with pytest.raises(ValueError, match="above"):
        check_streaming_decode(codec, codes, chunk_frames=3)


def test_verify_index_with_fake_backend(built) -> None:
    result = tokenize.verify_index(
        built["out"],
        samples=2,
        seed=0,
        backend=built["codec"],
        manifests=[built["manifest"]],
        asr=FakeASR(),
        run_asr=True,
    )
    assert result["core"]["status"] == "PASS"
    assert result["streaming"]["status"] == "PASS"
    assert result["asr"]["status"] == "PASS"
    assert result["asr"]["wer"] == 0.0
    assert result["utmos"]["status"] == "SKIPPED"
    assert result["status"] == "PASS"
    assert result["samples_verified"] == 2


def test_verify_reports_unverified_without_reference(built) -> None:
    result = tokenize.verify_index(
        built["out"],
        samples=1,
        backend=built["codec"],
        run_asr=True,
    )
    assert result["asr"]["status"] == "UNVERIFIED"
    assert "WER" in result["asr"]["reason"]
    assert result["status"] == "UNVERIFIED"


def test_cli_build_uses_injected_backend(built, monkeypatch, capsys) -> None:
    out = built["tmp_path"] / "cli_tokens"
    monkeypatch.setattr(tokenize, "LiquidCodecBackend", lambda *args, **kwargs: built["codec"])
    monkeypatch.setattr(tokenize, "HFTokenizer", lambda *args, **kwargs: built["tokenizer"])
    code = tokenize.main(
        ["--manifests", str(built["manifest"]), "--out", str(out), "--limit", "1"]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["new_conversations"] == 1
    assert (out / "index.json").is_file()
