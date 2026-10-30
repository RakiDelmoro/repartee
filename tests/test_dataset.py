"""Tests for src.dataset: index validation, memmap windows, padding, collate."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from src.audio import write_audio
from src import tokenize
from src.dataset import (
    DuplexShardDataset,
    ShardIndex,
    collate,
    control_id_for_name,
    measure_random_batch_load,
)
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
from src.tokenize import conversation_arrays

SAMPLE_RATE = CODEC_SAMPLE_RATE
FRAME_SAMPLES = 1920


class FakeCodec:
    n_codebooks = NUM_CODEBOOKS
    sample_rate = SAMPLE_RATE
    frame_samples = FRAME_SAMPLES

    def encode(self, audio: np.ndarray) -> np.ndarray:
        waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
        frames = waveform.reshape(-1, FRAME_SAMPLES)
        energy = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
        base = np.floor(energy * 10000.0).astype(np.int64)
        offsets = np.arange(self.n_codebooks, dtype=np.int64) * 13
        return ((base[:, None] + offsets[None, :]) % CODEBOOK_SIZE).astype(np.uint16)

    def decode(self, codes: np.ndarray, streaming_cache=None, finalize: bool = True) -> np.ndarray:
        array = np.asarray(codes)
        if array.size == 0:
            return np.zeros(0, dtype=np.float32)
        return np.repeat((array[:, 0].astype(np.float32) - 1000.0) / 1000.0, FRAME_SAMPLES)

    def new_streaming_cache(self):
        return {}


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [1 + (ord(character) % 5000) for character in text]


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
    turns = [
        Turn("user", 0.0, min(0.32, duration * 0.30), "user says"),
        Turn("agent", duration * 0.40, duration * 0.80, "hello world"),
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
    tokenize.build_index(
        [manifest],
        out,
        codec,
        text_tokenizer=tokenizer,
        progress=lambda message: None,
    )
    index = ShardIndex.load(out)
    expected = {
        conversation.id: conversation_arrays(conversation, codec, tokenizer)
        for conversation in conversations
    }
    return {"out": out, "index": index, "expected": expected, "codec": codec}


def test_shard_index_load_and_split(built) -> None:
    index = built["index"]
    assert len(index) == 2
    assert index.n_codebooks == NUM_CODEBOOKS
    assert index.frame_samples == FRAME_SAMPLES
    assert index.frame_rate == FRAME_RATE
    assert [record.id for record in index.by_split("train")] == ["conv_a"]
    assert [record.id for record in index.by_split("valid")] == ["conv_b"]
    assert len(index.by_split("all")) == 2
    assert index.hours == pytest.approx((16 + 24) / FRAME_RATE / 3600.0)
    assert index.split_hours("train") == pytest.approx(16 / FRAME_RATE / 3600.0)
    assert index.conversation("conv_a").split == "train"
    assert index.bytes > 0
    assert index.silence_frame is not None
    silence = built["codec"].encode(np.zeros(FRAME_SAMPLES, dtype=np.float32))[0]
    assert np.array_equal(np.asarray(index.silence_frame), silence)


def test_random_windows_match_source_arrays(built) -> None:
    dataset = DuplexShardDataset(
        built["index"], seq_frames=8, split="all", samples_per_epoch=120, seed=1
    )
    starts: dict[str, set[int]] = {"conv_a": set(), "conv_b": set()}
    for item_index in range(40):
        item = dataset[item_index]
        record = built["index"].conversation(item["id"])
        expected = built["expected"][item["id"]]
        start = item["start_frame"]
        assert 0 <= start <= record.T - 8
        assert item["user_codes"].shape == (8, NUM_CODEBOOKS)
        assert item["user_codes"].dtype == np.uint16
        assert item["agent_codes"].dtype == np.uint16
        assert item["text"].dtype == np.int32
        assert item["control"].dtype == np.uint8
        assert item["valid"].all()
        assert np.array_equal(item["user_codes"], expected["user_codes"][start : start + 8])
        assert np.array_equal(item["agent_codes"], expected["agent_codes"][start : start + 8])
        assert np.array_equal(item["text"], expected["text"][start : start + 8])
        assert np.array_equal(item["control"], expected["control"][start : start + 8])
        starts[item["id"]].add(start)
    assert len(starts["conv_a"]) > 1


def test_short_conversation_pads_from_silence(built) -> None:
    dataset = DuplexShardDataset(
        built["index"], seq_frames=32, split="all", samples_per_epoch=40, seed=5
    )
    silence = np.asarray(built["index"].silence_frame, dtype=np.uint16)
    for item_index in range(20):
        item = dataset[item_index]
        record = built["index"].conversation(item["id"])
        expected = built["expected"][item["id"]]
        assert item["start_frame"] == 0
        assert int(item["valid"].sum()) == record.T
        assert item["valid"][: record.T].all()
        assert not item["valid"][record.T :].any()
        assert np.array_equal(item["user_codes"][: record.T], expected["user_codes"])
        assert np.array_equal(item["agent_codes"][: record.T], expected["agent_codes"])
        assert np.array_equal(item["user_codes"][record.T :], np.tile(silence, (32 - record.T, 1)))
        assert np.array_equal(item["agent_codes"][record.T :], np.tile(silence, (32 - record.T, 1)))
        assert np.all(item["text"][record.T :] == 0)
        assert np.all(item["control"][record.T :] == CONTROL_IDS["PAD"])


def test_deterministic_items(built) -> None:
    dataset = DuplexShardDataset(
        built["index"], seq_frames=8, split="all", samples_per_epoch=50, seed=11
    )
    first = dataset[7]
    second = dataset[7]
    for key in ("user_codes", "agent_codes", "text", "control", "valid"):
        assert np.array_equal(first[key], second[key])
    assert first["id"] == second["id"]
    assert first["start_frame"] == second["start_frame"]


def test_collate_and_frame_batches(built) -> None:
    dataset = DuplexShardDataset(
        built["index"], seq_frames=8, split="all", samples_per_epoch=6, seed=3
    )
    batch = next(dataset.frame_batches(batch_size=2, shuffle=False))
    assert batch["user_codes"].shape == (2, 8, NUM_CODEBOOKS)
    assert batch["agent_codes"].shape == (2, 8, NUM_CODEBOOKS)
    assert batch["text"].shape == (2, 8)
    assert batch["control"].shape == (2, 8)
    assert batch["valid"].shape == (2, 8)
    assert batch["user_codes"].dtype == torch.uint16
    assert batch["agent_codes"].dtype == torch.uint16
    assert batch["text"].dtype == torch.int32
    assert batch["control"].dtype == torch.uint8
    assert batch["valid"].dtype == torch.bool
    assert torch_equal(batch["lengths"], batch["valid"].sum(dim=1))
    assert len(batch["ids"]) == 2
    assert batch["start_frame"].shape == (2,)
    assert batch["duration_s"].shape == (2,)

    manual = collate([dataset[0], dataset[1]])
    assert manual["user_codes"].shape == batch["user_codes"].shape


def torch_equal(left, right) -> bool:
    import torch

    return bool(torch.equal(left, right))


def test_measure_random_batch_load(built, capsys) -> None:
    stats = measure_random_batch_load(
        built["index"],
        n=20,
        batch_size=4,
        seq_frames=8,
        split="all",
        samples_per_epoch=50,
        seed=0,
    )
    print("batch load stats (seq_frames=8):", stats)  # visible with pytest -s
    assert stats["batches"] == 20
    assert stats["mean_ms"] >= 0.0
    assert stats["median_ms"] <= stats["p95_ms"] <= stats["max_ms"]
    assert stats["gate_ms"] == 5.0
    assert isinstance(stats["passed"], bool)

    gate_stats = measure_random_batch_load(
        built["index"],
        n=100,
        batch_size=8,
        seq_frames=4096,
        split="all",
        samples_per_epoch=100,
        seed=0,
    )
    print("batch load stats (seq_frames=4096):", gate_stats)
    assert gate_stats["batches"] == 100
    assert gate_stats["median_ms"] <= gate_stats["p95_ms"] <= gate_stats["max_ms"]


def test_index_rejects_offset_gap(built, tmp_path) -> None:
    bad = tmp_path / "bad_offsets"
    shutil.copytree(built["out"], bad)
    path = bad / "index.json"
    data = json.loads(path.read_text())
    assert len(data["conversations"]) >= 2
    data["conversations"][1]["offset"] += 1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="offset|follow"):
        ShardIndex.load(bad)


def test_index_rejects_truncated_shard(built, tmp_path) -> None:
    bad = tmp_path / "bad_truncated"
    shutil.copytree(built["out"], bad)
    data = json.loads((bad / "index.json").read_text())
    shard = bad / data["shards"][0]["file"]
    shard.write_bytes(shard.read_bytes()[:-1])
    with pytest.raises(ValueError, match="bytes|truncated|missing"):
        ShardIndex.load(bad)


def test_index_rejects_version(built, tmp_path) -> None:
    bad = tmp_path / "bad_version"
    shutil.copytree(built["out"], bad)
    path = bad / "index.json"
    data = json.loads(path.read_text())
    data["version"] = 2
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="version"):
        ShardIndex.load(bad)


def test_empty_split_raises(built) -> None:
    with pytest.raises(ValueError, match="no conversations"):
        DuplexShardDataset(built["index"], split="test")


def test_control_id_for_name() -> None:
    assert control_id_for_name("SPEAK") == CONTROL_IDS["SPEAK"]
    assert control_id_for_name("speak") == CONTROL_IDS["SPEAK"]
    assert control_id_for_name("<|stop|>") == CONTROL_IDS["STOP"]
    assert control_id_for_name("PAD") == CONTROL_IDS["PAD"]
    with pytest.raises(KeyError):
        control_id_for_name("shout")
