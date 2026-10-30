"""Contract tests for Repartee model surgery and training (no checkpoint download).

Covers: token layout, mean initialisation, freeze sets and the stage-1 2 %
budget, loss weighting/composition, the flattened placeholder/audio-label
contract, valid-mask handling, cosine schedule, Gate-4 refusal, DPO maths and
adapter-disabled references, and actionable failures for missing artifacts.
All models here are toys; nothing is downloaded and nothing is trained beyond
tiny CPU steps.
"""
from __future__ import annotations

import contextlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

import src.model as model_lib
import src.train_common as common
import src.train_dpo as dpo
import src.train_stage1 as stage1
import src.train_stage2 as stage2
from src.schema import CONTROL_IDS, NUM_CODEBOOKS


# ---------------------------------------------------------------------------
# Toy tokenizer / tied model
# ---------------------------------------------------------------------------


class FakeTokenizer:
    """Grows its vocabulary like a real tokenizer without downloading one."""

    def __init__(self, base_vocab: int = 8) -> None:
        self.pad_token_id = 0
        self.unk_token_id = 1
        self._next = int(base_vocab)
        self.mapping: dict[str, int] = {}

    def add_special_tokens(self, value, replace_additional_special_tokens: bool = False) -> int:
        added = 0
        for token in value.get("additional_special_tokens", []):
            if token not in self.mapping:
                self.mapping[token] = self._next
                self._next += 1
                added += 1
        return added

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.mapping.get(token, self.unk_token_id)

    def __len__(self) -> int:
        return self._next


class TinyLM(nn.Module):
    def __init__(self, vocab: int = 8, hidden: int = 4, tied: bool = False) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.config = SimpleNamespace(tie_word_embeddings=tied)
        if tied:
            self.lm_head.weight = self.embed.weight

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def resize_token_embeddings(self, new_size: int) -> nn.Embedding:
        old = self.embed
        new = nn.Embedding(new_size, old.weight.shape[1])
        with torch.no_grad():
            new.weight[: old.weight.shape[0]] = old.weight
        self.embed = new
        if self.config.tie_word_embeddings:
            self.lm_head.weight = new.weight
        return new


def test_control_token_ids_map_pad_and_new_tokens() -> None:
    tokenizer = FakeTokenizer(base_vocab=8)
    control = model_lib.add_control_tokens(tokenizer, None)
    assert control.pad == 0
    assert control.listen == 8
    assert control.speak == 9
    assert control.backchannel == 10
    assert control.stop == 11
    assert control.by_label() == {"PAD": 0, "LISTEN": 8, "SPEAK": 9, "BACKCHANNEL": 10, "STOP": 11}
    # idempotent
    again = model_lib.add_control_tokens(tokenizer, None)
    assert again == control
    assert len(tokenizer) == 12


def test_mean_initialises_new_rows_and_leaves_old_rows() -> None:
    embedding = nn.Embedding(10, 4)
    with torch.no_grad():
        embedding.weight.copy_(torch.arange(40, dtype=torch.float32).reshape(10, 4))
    original = embedding.weight[:8].clone()
    mean = model_lib.mean_initialize_rows(embedding, [8, 9], reference_rows=8)
    assert torch.allclose(embedding.weight[:8], original)
    expected = original.float().mean(dim=0)
    assert torch.allclose(mean, expected)
    assert torch.allclose(embedding.weight[8], expected)
    assert torch.allclose(embedding.weight[9], expected)


def test_add_tokens_resizes_and_mean_inits_through_model() -> None:
    tokenizer = FakeTokenizer(base_vocab=8)
    model = TinyLM(vocab=8, hidden=4)
    with torch.no_grad():
        model.embed.weight.copy_(torch.arange(32, dtype=torch.float32).reshape(8, 4))
    old = model.embed.weight.clone()
    control = model_lib.add_control_tokens(tokenizer, model)
    assert model.embed.weight.shape[0] == 12
    expected = old.float().mean(dim=0)
    for token_id in (control.listen, control.speak, control.backchannel, control.stop):
        assert torch.allclose(model.embed.weight[token_id], expected)


class ReusingSlotTokenizer(FakeTokenizer):
    """Real LFM3 behaviour: added specials take unused slots inside the vocab."""

    def __init__(self, base_vocab: int = 8, first_slot: int = 4) -> None:
        super().__init__(base_vocab=base_vocab)
        self._next = int(first_slot)

    def __len__(self) -> int:
        return max(int(self._next), 8)


def test_add_tokens_mean_inits_existing_slots() -> None:
    tokenizer = ReusingSlotTokenizer(base_vocab=8, first_slot=4)
    model = TinyLM(vocab=8, hidden=4)
    with torch.no_grad():
        model.embed.weight.copy_(torch.arange(32, dtype=torch.float32).reshape(8, 4))
    original = model.embed.weight.clone()
    control = model_lib.add_control_tokens(tokenizer, model)
    assert model.embed.weight.shape[0] == 8  # no resize needed
    expected = original.float().mean(dim=0)
    for token_id in (control.listen, control.speak, control.backchannel, control.stop):
        assert token_id < 8
        assert torch.allclose(model.embed.weight[token_id], expected)
    # Rows other than the four control slots stay untouched.
    untouched = [row for row in range(8) if row not in control.by_label().values()]
    for row in untouched:
        assert torch.allclose(model.embed.weight[row], original[row])


def test_tied_embeddings_stay_tied_after_resize() -> None:
    tokenizer = FakeTokenizer(base_vocab=8)
    model = TinyLM(vocab=8, hidden=4, tied=True)
    model_lib.add_control_tokens(tokenizer, model)
    assert model.lm_head.weight.data_ptr() == model.embed.weight.data_ptr()
    assert model_lib.assert_tied_embeddings(model) is True


# ---------------------------------------------------------------------------
# Freeze sets
# ---------------------------------------------------------------------------


class ToyRMSNorm(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value


class ToyLanguageModel(nn.Module):
    def __init__(self, vocab: int = 1008, hidden: int = 64) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.fc = nn.Linear(hidden, hidden)
        self.norm = ToyRMSNorm(hidden)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.fc(value)


class ToyRepartee(nn.Module):
    """Structurally the subset of LFM3-Omni used by the freeze plan."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=False, use_cache=False)
        inner = nn.Module()
        inner.language_model = ToyLanguageModel()
        inner.audio_encoder = nn.Sequential(nn.Linear(16, 16), ToyRMSNorm(16))
        inner.audio_projector = nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 16))
        inner.depth_decoder = nn.Linear(16, 2017)
        inner.vision_tower = nn.Linear(4, 4)
        inner.multi_modal_projector = nn.Linear(4, 4)
        inner.codebook_embedding = nn.Embedding(64, 16)
        self.model = inner
        self.lm_head = nn.Linear(64, 1008, bias=False)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.language_model.embed

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head


def _control() -> model_lib.ControlTokenIds:
    return model_lib.ControlTokenIds(pad=0, listen=1000, speak=1001, backchannel=1002, stop=1003)


def test_stage1_freeze_set_and_two_percent_budget() -> None:
    model = ToyRepartee()
    control = _control()
    plan = model_lib.configure_trainability(model, 1, control)
    assert plan.trainable_modules == ("audio_projector", "control_token_embeddings")
    assert plan.embedding_rows == (1000, 1001, 1002, 1003)
    assert all(p.requires_grad for p in model.model.audio_projector.parameters())
    # everything else frozen, including encoder/depth/vision and all norms
    for name, parameter in model.named_parameters():
        if "audio_projector" in name:
            continue
        if parameter is model.model.language_model.embed.weight:
            # the parameter is on, but only four rows receive gradient
            continue
        assert not parameter.requires_grad, name
    summary = model_lib.assert_stage1_budget(model)
    assert summary["ratio"] < 0.02
    assert summary["trainable"] == sum(p.numel() for p in model.model.audio_projector.parameters()) + 4 * 64
    # the row mask zeroes gradients outside the control rows
    masked = model_lib.embedding_row_mask(model.model.language_model.embed)
    assert masked is not None and int(masked.sum()) == 4


def test_stage1_gradient_row_mask_is_applied() -> None:
    model = ToyRepartee()
    control = _control()
    model_lib.configure_trainability(model, 1, control)
    embedding = model.model.language_model.embed
    embedding.weight.grad = None
    loss = embedding(torch.tensor([0, 1000])).sum()
    loss.backward()
    grad = embedding.weight.grad
    assert grad is not None
    nonzero = (grad.abs().sum(dim=1) > 0).nonzero().flatten().tolist()
    assert nonzero == [1000]


def test_stage2_freeze_set_unfreezes_llm_but_not_norms_or_encoder() -> None:
    model = ToyRepartee()
    control = _control()
    plan = model_lib.configure_trainability(model, 2, control)
    assert "language_model" in plan.trainable_modules
    assert "audio_projector" in plan.trainable_modules
    assert model.model.language_model.fc.weight.requires_grad
    assert model.model.language_model.embed.weight.requires_grad
    assert not model.model.language_model.norm.weight.requires_grad
    assert not any(p.requires_grad for p in model.model.audio_encoder.parameters())
    assert not any(p.requires_grad for p in model.model.depth_decoder.parameters())
    assert not any(p.requires_grad for p in model.model.codebook_embedding.parameters())
    assert model_lib.embedding_row_mask(model.model.language_model.embed) is None


def test_dpo_stage_freezes_everything_for_lora() -> None:
    model = ToyRepartee()
    model_lib.configure_trainability(model, "dpo", _control())
    assert not any(p.requires_grad for p in model.parameters())


# ---------------------------------------------------------------------------
# Token layout / batch contract
# ---------------------------------------------------------------------------


def test_frame_sequence_layout_and_labels() -> None:
    ids, labels = common.build_frame_sequence(
        np.array([CONTROL_IDS["LISTEN"], CONTROL_IDS["SPEAK"]]),
        np.array([0, 42]),
        control_ids=common.TOY_CONTROL_IDS,
        audio_token_id=common.TOY_AUDIO_TOKEN_ID,
        audio_out_token_id=common.TOY_AUDIO_OUT_TOKEN_ID,
    )
    assert ids == [1, 4, 2, 1, 5, 42, 2]
    assert labels == [-100, 4, -100, -100, 5, 42, -100]
    # only the control and text targets are supervised
    supervised = [(token, label) for token, label in zip(ids, labels) if label != -100]
    assert supervised == [(4, 4), (5, 5), (42, 42)]


def test_batch_builder_honours_valid_mask_and_flattened_counts() -> None:
    components = common.make_smoke_components(frames=12, batch_size=2, seed=3)
    builder = components["builder"]
    items = common.make_synthetic_items(count=2, frames=12, seed=5, text_frames=2)
    # mark the last 4 frames as padding for one item
    items[0] = dict(items[0])
    items[0]["valid"] = np.array([True] * 8 + [False] * 4)
    items[0]["valid_frames"] = 8
    prepared = [builder.prepare(item) for item in items]
    assert prepared[0].frames == 8
    assert prepared[1].frames == 12
    batch = builder.build(prepared, [item["control"] for item in items])
    audio_placeholders = int((batch.input_ids == common.TOY_AUDIO_OUT_TOKEN_ID).sum())
    user_placeholders = int((batch.input_ids == common.TOY_AUDIO_TOKEN_ID).sum())
    assert user_placeholders == 20
    assert audio_placeholders == 20
    assert batch.audio_out_codes.shape == (20, NUM_CODEBOOKS)
    assert batch.audio_labels.shape == (20, NUM_CODEBOOKS)
    counts = common.validate_duplex_batch(
        batch.input_ids,
        audio_feature_tokens=20,
        audio_token_id=common.TOY_AUDIO_TOKEN_ID,
        audio_out_token_id=common.TOY_AUDIO_OUT_TOKEN_ID,
        audio_out_codes=batch.audio_out_codes,
        audio_labels=batch.audio_labels,
    )
    assert counts == {"user_tokens": 20, "audio_frames": 20}


def test_validate_duplex_batch_rejects_count_mismatches() -> None:
    input_ids = torch.tensor([[common.TOY_AUDIO_TOKEN_ID, 4, common.TOY_AUDIO_OUT_TOKEN_ID]])
    codes = torch.zeros(1, NUM_CODEBOOKS, dtype=torch.long)
    with pytest.raises(ValueError, match="encoder tokens"):
        common.validate_duplex_batch(
            input_ids,
            audio_feature_tokens=2,
            audio_token_id=common.TOY_AUDIO_TOKEN_ID,
            audio_out_token_id=common.TOY_AUDIO_OUT_TOKEN_ID,
            audio_out_codes=codes,
        )
    with pytest.raises(ValueError, match="code rows"):
        common.validate_duplex_batch(
            input_ids,
            audio_feature_tokens=1,
            audio_token_id=common.TOY_AUDIO_TOKEN_ID,
            audio_out_token_id=common.TOY_AUDIO_OUT_TOKEN_ID,
            audio_out_codes=torch.zeros(2, NUM_CODEBOOKS, dtype=torch.long),
        )
    with pytest.raises(ValueError, match=r"\[0, 2017\)"):
        common.validate_duplex_batch(
            input_ids,
            audio_feature_tokens=1,
            audio_token_id=common.TOY_AUDIO_TOKEN_ID,
            audio_out_token_id=common.TOY_AUDIO_OUT_TOKEN_ID,
            audio_out_codes=torch.full((1, NUM_CODEBOOKS), 5000),
        )


def test_toy_model_accepts_the_built_batch() -> None:
    components = common.make_smoke_components(frames=8, batch_size=2, seed=1)
    builder = components["builder"]
    model = components["model"]
    batch = builder.build_items(common.make_synthetic_items(count=2, frames=8, seed=2, text_frames=2))
    outputs = model(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        input_features=batch.input_features,
        feature_attention_mask=batch.feature_attention_mask,
        audio_out_codes=batch.audio_out_codes,
        audio_labels=batch.audio_labels,
    )
    assert outputs.logits.shape[:2] == batch.input_ids.shape
    assert outputs.audio_loss is not None


def test_window_item_slices_valid_mask() -> None:
    item = common.make_synthetic_items(count=1, frames=10, seed=0)[0]
    item["valid"] = np.array([True] * 7 + [False] * 3)
    item["valid_frames"] = 7
    window = common.window_item(item, 2, 4)
    assert window["user_codes"].shape[0] == 4
    assert window["valid"].tolist() == [True, True, True, True]
    assert window["valid_frames"] == 4
    assert window["start_frame"] == 2


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def _zero_logits(vocab: int = 8) -> torch.Tensor:
    return torch.zeros(1, 2, vocab)


def test_control_loss_uses_blueprint_class_weights() -> None:
    vocab = 8
    logits = _zero_logits(vocab)
    ce = math.log(vocab)
    speak_loss, speak_count = common.control_loss(
        logits,
        torch.tensor([[-100, common.TOY_CONTROL_IDS["SPEAK"]]]),
        common.TOY_CONTROL_IDS,
    )
    listen_loss, _ = common.control_loss(
        logits,
        torch.tensor([[-100, common.TOY_CONTROL_IDS["LISTEN"]]]),
        common.TOY_CONTROL_IDS,
    )
    pad_loss, _ = common.control_loss(
        logits,
        torch.tensor([[-100, common.TOY_CONTROL_IDS["PAD"]]]),
        common.TOY_CONTROL_IDS,
    )
    assert speak_count == 1
    assert speak_loss.item() == pytest.approx(ce)
    assert listen_loss.item() == pytest.approx(ce)
    # a single weighted token is normalised by its own weight, so compare the
    # ratio through the unnormalised numerator instead
    assert speak_loss.item() == pytest.approx(listen_loss.item())
    assert pad_loss.item() == pytest.approx(ce)
    # two positions with different CEs: normalise by the applied weight sum
    mixed_logits = torch.randn(1, 3, vocab)
    mixed_labels = torch.tensor(
        [[-100, common.TOY_CONTROL_IDS["SPEAK"], common.TOY_CONTROL_IDS["PAD"]]]
    )
    weighted, count = common.control_loss(mixed_logits, mixed_labels, common.TOY_CONTROL_IDS)
    per_token = torch.nn.functional.cross_entropy(
        mixed_logits[:, :-1].reshape(-1, vocab),
        mixed_labels[:, 1:].reshape(-1),
        reduction="none",
    )
    expected = (8.0 * per_token[0] + 0.3 * per_token[1]) / 8.3
    assert count == 2
    assert weighted.item() == pytest.approx(float(expected), rel=1e-5)


def test_duplex_loss_composition() -> None:
    logits = torch.zeros(1, 2, 10)
    # control-only position: 2*CE
    loss, parts = common.duplex_loss(
        logits,
        torch.tensor([[-100, common.TOY_CONTROL_IDS["SPEAK"]]]),
        common.TOY_CONTROL_IDS,
        audio_loss=torch.tensor(5.0),
    )
    ce = math.log(10)
    assert parts["text_tokens"] == 0
    assert loss.item() == pytest.approx(2.0 * ce + 5.0)
    # text-only position: 3*CE
    loss, parts = common.duplex_loss(
        logits,
        torch.tensor([[-100, 9]]),
        common.TOY_CONTROL_IDS,
        audio_loss=None,
    )
    assert parts["control_tokens"] == 0
    assert parts["audio_loss"] is None
    assert loss.item() == pytest.approx(3.0 * ce)


def test_text_loss_ignores_control_ids() -> None:
    logits = torch.zeros(1, 3, 10)
    labels = torch.tensor([[-100, 9, common.TOY_CONTROL_IDS["SPEAK"]]])
    loss, count = common.text_loss(logits, labels, common.TOY_CONTROL_IDS)
    assert count == 1
    assert loss.item() == pytest.approx(math.log(10))


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------


def test_cosine_schedule_warmup_and_decay() -> None:
    assert common.cosine_lr_at(0, 100, 10, 1.0) == pytest.approx(0.1)
    assert common.cosine_lr_at(9, 100, 10, 1.0) == pytest.approx(1.0)
    assert common.cosine_lr_at(100, 100, 10, 1.0) == pytest.approx(0.0, abs=1e-8)
    assert common.cosine_lr_at(55, 100, 10, 1.0) == pytest.approx(0.5, abs=1e-6)
    values = [common.cosine_lr_at(step, 100, 10, 1.0) for step in range(0, 101)]
    assert values[0] < values[9]
    assert all(values[i] >= values[i + 1] - 1e-9 for i in range(9, 100))


# ---------------------------------------------------------------------------
# DPO
# ---------------------------------------------------------------------------


def test_dpo_loss_zero_reference_is_log_two() -> None:
    zero = torch.zeros(1)
    loss, metrics = dpo.dpo_loss(zero, zero, zero, zero, beta=0.1)
    assert loss.item() == pytest.approx(math.log(2), rel=1e-6)
    assert metrics["reward_accuracy"] == 0.0


def test_dpo_loss_prefers_higher_policy_margin() -> None:
    better = torch.tensor([1.0])
    worse = torch.tensor([-1.0])
    zero = torch.zeros(1)
    good, _ = dpo.dpo_loss(better, zero, zero, zero, beta=0.1)
    bad, _ = dpo.dpo_loss(worse, zero, zero, zero, beta=0.1)
    assert good.item() < math.log(2) < bad.item()


def test_sequence_log_prob_uses_causal_shift() -> None:
    logits = torch.zeros(1, 2, 4)
    labels = torch.tensor([[-100, 2]])
    value = dpo.sequence_log_prob(logits, labels)
    expected = torch.log_softmax(logits[0, 0], dim=-1)[2]
    assert value.item() == pytest.approx(float(expected))


def test_adapter_disabled_reference_context() -> None:
    class Stub:
        def __init__(self) -> None:
            self.enabled = True

        @contextlib.contextmanager
        def disable_adapter(self):
            self.enabled = False
            try:
                yield self
            finally:
                self.enabled = True

    stub = Stub()
    with dpo.adapter_disabled(stub):
        assert stub.enabled is False
    assert stub.enabled is True


def test_timing_variants_differ_only_in_control() -> None:
    item = common.make_synthetic_items(count=1, frames=20, seed=0)[0]
    item["control"] = np.zeros(20, dtype=np.uint8)
    early, late = dpo.timing_variants(item, onset_frame=2, early_delay_frames=3, late_delay_frames=15)
    assert np.array_equal(early["user_codes"], late["user_codes"])
    assert np.array_equal(early["agent_codes"], late["agent_codes"])
    assert np.array_equal(early["text"], late["text"])
    changed = np.nonzero(early["control"] != late["control"])[0].tolist()
    assert changed == [5, 17]
    assert early["control"][5] == CONTROL_IDS["SPEAK"]
    assert late["control"][17] == CONTROL_IDS["SPEAK"]


def test_iter_dpo_pairs_validates_context(tmp_path: Path) -> None:
    item = common.make_synthetic_items(count=1, frames=10, seed=0)[0]
    item["control"] = np.zeros(10, dtype=np.uint8)
    early, late = dpo.timing_variants(item, onset_frame=1, early_delay_frames=2, late_delay_frames=5)
    path = tmp_path / "pairs.jsonl"
    good = {"id": "p0", "chosen": _jsonable(early), "rejected": _jsonable(late)}
    path.write_text(json.dumps(good) + "\n", encoding="utf-8")
    pairs = list(dpo.iter_dpo_pairs(path))
    assert len(pairs) == 1
    assert pairs[0].id == "p0"
    assert not np.array_equal(pairs[0].chosen["control"], pairs[0].rejected["control"])

    bad_item = dict(early)
    bad_item["text"] = np.asarray(bad_item["text"]) + 1
    path.write_text(json.dumps({"id": "bad", "chosen": _jsonable(early), "rejected": _jsonable(bad_item)}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="text differs"):
        list(dpo.iter_dpo_pairs(path))


def _jsonable(item: dict) -> dict:
    result = {}
    for key, value in item.items():
        result[key] = value.tolist() if isinstance(value, np.ndarray) else value
    return result


def test_dpo_missing_pairs_is_actionable(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="DPO pairs not found"):
        dpo.main(["--pairs", str(tmp_path / "missing.jsonl"), "--model", str(tmp_path / "model")])


def test_dpo_smoke_runs_two_tensor_steps() -> None:
    assert dpo.main(["--smoke", "--steps", "2"]) == 0


# ---------------------------------------------------------------------------
# Gate 4 and actionable preflight failures
# ---------------------------------------------------------------------------


def test_gate4_refusal(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="Gate-4 artifact not found"):
        stage2.check_gate4(tmp_path / "missing.json")
    failing = tmp_path / "fail.json"
    failing.write_text(json.dumps({"gate": "Gate 4", "result": "FAIL", "measured": 7.1}), encoding="utf-8")
    with pytest.raises(stage2.GateArtifactError, match="did not pass"):
        stage2.check_gate4(failing)
    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps({"gate": 5, "result": "PASS"}), encoding="utf-8")
    with pytest.raises(stage2.GateArtifactError, match="not Gate 4"):
        stage2.check_gate4(wrong)
    passing = tmp_path / "pass.json"
    passing.write_text(json.dumps({"gate": "Gate 4", "result": "PASS", "measured": 1.0}), encoding="utf-8")
    assert stage2.check_gate4(passing)["result"] == "PASS"


def test_stage2_cli_refuses_without_gate4() -> None:
    with pytest.raises(SystemExit, match="requires --gate4"):
        stage2.main([])


def test_stage1_missing_tokens_is_actionable(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="Token shard directory"):
        stage1.main(
            [
                "--tokens",
                str(tmp_path / "missing-tokens"),
                "--model",
                "not-a-checkpoint",
                "--out",
                str(tmp_path / "out"),
            ]
        )


def test_stage1_smoke_runs_two_cpu_steps(tmp_path: Path) -> None:
    assert stage1.main(["--smoke", "--out", str(tmp_path / "smoke")]) == 0
    records = (tmp_path / "smoke" / "smoke" / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(records) == 2
    assert json.loads(records[-1])["step"] == 2


def test_resolve_resume_auto(tmp_path: Path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    assert common.resolve_resume("auto", state) is None
    (state / "latest.pt").write_bytes(b"checkpoint")
    assert common.resolve_resume("auto", state) == state / "latest.pt"
    assert common.resolve_resume(None, state) is None


def test_scheduler_resume_uses_new_horizon() -> None:
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=1e-3)
    old = common.WarmupCosineScheduler(optimizer, total_steps=10, warmup_steps=2)
    for _ in range(10):
        old.step()
    state = old.state_dict()
    fresh = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    new = common.WarmupCosineScheduler(fresh, total_steps=300, warmup_steps=20)
    new.load_state_dict(state)
    assert new.total_steps == 300
    assert new.optimizer.param_groups[0]["lr"] > 0
