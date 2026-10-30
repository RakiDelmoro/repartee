"""Pure-helper tests for scripts/gate0.py (no checkpoint, no torch)."""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.gate0 import (  # noqa: E402
    CAUSALITY_MAX_ABS_DIFF,
    check_layout_config,
    classify_license,
    evaluate_causality,
    evaluate_codec_stream,
    evaluate_interrupt,
    group_parameter_counts,
    parse_license_text,
    parse_safetensors_header_bytes,
    read_safetensors_header,
    token_rate_metrics,
)


LICENSE_SAMPLE = """LFM Open License v1.0

"Threshold" shall mean annual revenue of 10 million United States dollars ($10,000,000) or more.

4. Redistribution. You may reproduce and distribute copies of the Work or Derivative Works thereof in any
medium, with or without modifications, and in Source or Object form, provided that You meet the following
conditions:

(a) You must give any other recipients of the Work or Derivative Works a copy of this License; and
(b) You must cause any modified files to carry prominent notices stating that You changed the files; and
(c) You must retain, in the Source form of any Derivative Works that You distribute, all copyright, patent,
trademark, and attribution notices from the Source form of the Work; and
(d) If the Work includes a "NOTICE" text file as part of its distribution, then any Derivative Works that You
distribute must include a readable copy of the attribution notices contained within such NOTICE file.

5. Commercial Use Limitation.
(a) The rights granted under this License for Commercial Use are conditioned upon You or Your Legal Entity
not exceeding the Threshold.
(b) Any Commercial Use of the Work or a Derivative Work by a Legal Entity that exceeds the Threshold is not
licensed under this Agreement.
(c) The Threshold shall not apply to a Qualified Non-Profit Organization's use of the Work or a Derivative
Work for Non-Commercial or Research Purposes.
"""


def test_parse_license_threshold_and_conditions():
    parsed = parse_license_text(LICENSE_SAMPLE)
    assert parsed["version"] == "1.0"
    assert parsed["license_name"] == "LFM Open License v1.0"
    assert parsed["annual_revenue_threshold_usd"] == 10_000_000
    assert parsed["commercial_use_limited"] is True
    assert parsed["nonprofit_exception"] is True
    assert parsed["requires_attribution"] is True
    assert [clause[:3] for clause in parsed["redistribution_conditions"]] == ["(a)", "(b)", "(c)", "(d)"]
    assert parsed["characters"] == len(LICENSE_SAMPLE)
    assert len(parsed["sha256"]) == 64


def test_parse_license_uses_dollar_fallback():
    parsed = parse_license_text('LFM Open License v1.0\n"Threshold" means $10,000,000.')
    assert parsed["annual_revenue_threshold_usd"] == 10_000_000


def test_classify_license_unknown_is_conditional():
    result = classify_license(None)
    assert result["result"] == "CONDITIONAL"
    assert "unknown" in result["reason"].lower()


def test_classify_license_threshold_is_inclusive_fail():
    assert classify_license(9_999_999)["result"] == "PASS"
    assert classify_license(10_000_000)["result"] == "FAIL"
    assert classify_license(50_000_000)["result"] == "FAIL"


def test_classify_license_nonprofit_noncommercial():
    assert classify_license(None, nonprofit=True, noncommercial=True)["result"] == "PASS"
    # A non-profit using the work commercially is still subject to the revenue threshold.
    assert classify_license(None, nonprofit=True, noncommercial=False)["result"] == "CONDITIONAL"
    assert classify_license(20_000_000, nonprofit=True, noncommercial=False)["result"] == "FAIL"


def test_token_rate_exactly_one_per_frame():
    exact = token_rate_metrics(375, 30.0)
    assert exact["ok"] is True
    assert exact["tokens_per_frame"] == pytest.approx(1.0)
    assert exact["token_rate_hz"] == pytest.approx(12.5)

    short = token_rate_metrics(374, 30.0)
    assert short["ok"] is False
    assert short["tokens_per_frame"] == pytest.approx(374 / 375)


def test_evaluate_causality_fails_closed():
    assert evaluate_causality(True, 0.0)["result"] == "PASS"
    assert evaluate_causality(True, CAUSALITY_MAX_ABS_DIFF)["result"] == "PASS"
    assert evaluate_causality(True, CAUSALITY_MAX_ABS_DIFF * 2)["result"] == "FAIL"
    assert evaluate_causality(False, 0.0)["result"] == "FAIL"
    assert evaluate_causality(True, None)["result"] == "FAIL"


def test_codec_stream_helpers():
    passing = evaluate_codec_stream([1920] * 375, all_finite=True, rms_diff_db=0.2)
    assert passing["result"] == "PASS"
    assert passing["max_call_ms"] == pytest.approx(80.0)
    assert passing["calls"] == 375

    at_limit = evaluate_codec_stream([3840], all_finite=True, rms_diff_db=0.0)
    assert at_limit["result"] == "PASS"

    too_long = evaluate_codec_stream([4000], all_finite=True, rms_diff_db=0.0)
    assert too_long["result"] == "FAIL"

    non_finite = evaluate_codec_stream([1920], all_finite=False, rms_diff_db=0.0)
    assert non_finite["result"] == "FAIL"

    unstable = evaluate_codec_stream([1920], all_finite=True, rms_diff_db=2.0)
    assert unstable["result"] == "FAIL"


def test_evaluate_interrupt_threshold():
    assert evaluate_interrupt(0)["result"] == "PASS"
    assert evaluate_interrupt(2)["result"] == "PASS"
    assert evaluate_interrupt(3)["result"] == "FAIL"
    assert evaluate_interrupt(None)["result"] == "FAIL"


def _header_bytes(header: dict) -> bytes:
    payload = json.dumps(header).encode("utf-8")
    return struct.pack("<Q", len(payload)) + payload


def test_parse_safetensors_header_bytes_and_file(tmp_path):
    header = {
        "__metadata__": {"format": "pt"},
        "model.audio_projector.linear_1.weight": {"dtype": "BF16", "shape": [2048, 768], "data_offsets": [0, 8]},
    }
    blob = _header_bytes(header)
    assert parse_safetensors_header_bytes(blob)["__metadata__"]["format"] == "pt"

    path = tmp_path / "model.safetensors"
    path.write_bytes(blob + b"\0" * 8)
    assert read_safetensors_header(path)["model.audio_projector.linear_1.weight"]["shape"] == [2048, 768]

    with pytest.raises(ValueError):
        parse_safetensors_header_bytes(b"short")
    with pytest.raises(ValueError):
        parse_safetensors_header_bytes(struct.pack("<Q", 100) + b"{}")


def test_group_parameter_counts():
    header = {
        "__metadata__": {},
        "lm_head.weight": {"shape": [128000, 2048], "dtype": "BF16"},
        "model.audio_projector.linear_1.weight": {"shape": [2048, 768], "dtype": "BF16"},
        "model.audio_projector.linear_2.weight": {"shape": [2048, 2048], "dtype": "BF16"},
        "model.audio_encoder.conv.weight": {"shape": [4, 5], "dtype": "BF16"},
        "something.else": {"shape": [3], "dtype": "F32"},
    }
    groups = {"lm_head": "lm_head.", "audio_projector": "model.audio_projector.", "audio_encoder": "model.audio_encoder."}
    counts = group_parameter_counts(header, groups)
    assert counts["groups"]["audio_projector"]["params"] == 2048 * 768 + 2048 * 2048
    assert counts["groups"]["audio_projector"]["tensors"] == 2
    assert counts["groups"]["audio_encoder"]["params"] == 20
    assert counts["groups"]["other"]["params"] == 3
    assert counts["total_params"] == 128000 * 2048 + 2048 * 768 + 2048 * 2048 + 20 + 3


def test_check_layout_config_accepts_contract_values():
    config = {
        "num_codebooks": 12,
        "codebook_size": 2017,
        "audio_eos_code": 2016,
        "audio_token_id": 125031,
        "audio_start_token_id": 125012,
        "audio_out_token_id": 125032,
        "text_end_token_id": 125014,
        "text_config": {"hidden_size": 2048},
        "audio_encoder_config": {"hidden_size": 768, "num_mel_bins": 128},
    }
    codec = {"n_q": 12, "frame_samples": 1920, "sample_rate": 24000}
    processor = {
        "feature_extractor": {"hop_length": 160, "encoder_stride": 8, "sampling_rate": 16000}
    }
    checks = check_layout_config(config, codec, processor)
    assert all(item["pass"] for item in checks)
    assert next(item for item in checks if item["name"] == "encoder_frame_ms")["actual"] == 80.0


def test_check_layout_config_detects_drift():
    config = {
        "num_codebooks": 8,
        "codebook_size": 2017,
        "audio_eos_code": 2016,
        "audio_token_id": 1,
        "audio_start_token_id": 2,
        "audio_out_token_id": 3,
        "text_end_token_id": 4,
        "text_config": {"hidden_size": 2048},
        "audio_encoder_config": {"hidden_size": 768, "num_mel_bins": 128},
    }
    codec = {"n_q": 12, "frame_samples": 1920, "sample_rate": 24000}
    processor = {"feature_extractor": {"hop_length": 160, "encoder_stride": 8, "sampling_rate": 16000}}
    checks = {item["name"]: item["pass"] for item in check_layout_config(config, codec, processor)}
    assert checks["num_codebooks"] is False
    assert checks["audio_token_id"] is False
    assert checks["codec_frame_samples"] is True


def test_synthetic_speech_is_deterministic_and_bounded():
    np = pytest.importorskip("numpy")
    from scripts.gate0 import synthetic_speech_like

    first = synthetic_speech_like(0.5, 16000, seed=7)
    second = synthetic_speech_like(0.5, 16000, seed=7)
    assert first.shape == (8000,)
    assert first.dtype == np.float32
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()
    assert float(np.max(np.abs(first))) <= 0.7 + 1e-6
