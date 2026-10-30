"""Tests for :mod:`src.license_audit` using temporary JSONL manifests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src import license_audit as la


def _record(
    *,
    record_id: str,
    source: str,
    license_name: str,
    hours: float = 1.0,
    meta: dict | None = None,
) -> dict:
    return {
        "id": record_id,
        "source": source,
        "license": license_name,
        "duration_s": hours * 3600.0,
        "channels": {"user": "u.flac", "agent": "a.flac"},
        "turns": [],
        "sample_rates": {"user": 16000, "agent": 24000},
        "split": "train",
        "meta": meta or {},
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )


def test_registry_lookup_and_forbidden_tts() -> None:
    assert la.find_entry("open_yap_1k") is la.find_entry("open_yap")
    assert la.find_entry("theagenticdatacompany/open-yap-1k").name == "open_yap_1k"
    assert la.find_entry("xtts-v2").forbidden is True
    assert la.find_entry("f5-tts").forbidden is True
    assert la.find_entry("fish").forbidden is True
    assert la.find_entry("kokoro").commercial_use is True
    assert la.find_entry("not-a-dataset") is None
    lfm = la.find_entry("LiquidAI/LFM3-Omni-2B")
    assert lfm is not None and lfm.revenue_threshold_usd == 10_000_000.0


def test_clean_commercial_manifest_passes(tmp_path: Path) -> None:
    manifest = tmp_path / "open.jsonl"
    _write_jsonl(
        manifest,
        [
            _record(record_id="c1", source="open_yap_1k", license_name="CC-BY-4.0", hours=2.0),
            _record(record_id="c2", source="otospeech_280h", license_name="CC-BY-4.0", hours=3.0),
        ],
    )
    result = la.audit([manifest])
    assert result.ok, result.violations
    assert result.conversations == 2
    assert result.hours == pytest.approx(5.0)
    assert result.violations == []
    # LFM threshold is unverified without an explicit revenue figure.
    codes = [finding.code for finding in result.warnings]
    assert "revenue_threshold_unverified" in codes


def test_non_commercial_source_is_a_single_aggregated_violation(tmp_path: Path) -> None:
    manifest = tmp_path / "turn.jsonl"
    _write_jsonl(
        manifest,
        [
            _record(
                record_id=f"t{i}",
                source="otospeech_turn104h",
                license_name="oto-speech-license-v1.0",
                hours=1.5,
            )
            for i in range(5)
        ],
    )
    result = la.audit([manifest])
    assert not result.ok
    violations = [finding for finding in result.violations if finding.code == "non_commercial_source"]
    assert len(violations) == 1
    assert "7.50 h" in violations[0].message
    assert la.find_entry("otospeech_turn104h").commercial_use is False


def test_missing_and_unknown_licence_are_violations(tmp_path: Path) -> None:
    manifest = tmp_path / "bad.jsonl"
    _write_jsonl(
        manifest,
        [
            _record(record_id="x1", source="mystery_set", license_name=""),
            _record(record_id="x2", source="open_yap_1k", license_name="proprietary-weird"),
        ],
    )
    result = la.audit([manifest])
    codes = {finding.code for finding in result.violations}
    assert "unknown_source" in codes
    assert "missing_license" in codes
    assert "license_mismatch" in codes


def test_manifest_license_alias_is_accepted(tmp_path: Path) -> None:
    manifest = tmp_path / "alias.jsonl"
    _write_jsonl(
        manifest,
        [_record(record_id="c1", source="open_yap", license_name="commercial")],
    )
    result = la.audit([manifest])
    assert result.ok, result.violations


def test_forbidden_tts_in_meta_and_selected(tmp_path: Path) -> None:
    manifest = tmp_path / "synth.jsonl"
    _write_jsonl(
        manifest,
        [
            _record(
                record_id="s1",
                source="synthetic",
                license_name="synthetic-commercial",
                meta={"tts_backend": "xtts-v2"},
            )
        ],
    )
    result = la.audit([manifest], selected_tts=["f5-tts"])
    codes = {finding.code for finding in result.violations}
    assert "forbidden_tts" in codes
    assert result.tts_usage["xtts_v2"]["commercial_use"] is False
    assert result.tts_usage["f5_tts"]["commercial_use"] is False


def test_unknown_tts_is_a_violation(tmp_path: Path) -> None:
    manifest = tmp_path / "synth.jsonl"
    _write_jsonl(
        manifest,
        [
            _record(
                record_id="s1",
                source="synthetic",
                license_name="synthetic-commercial",
                meta={"tts_backend": "mystery-tts"},
            )
        ],
    )
    result = la.audit([manifest])
    assert "unknown_tts" in {finding.code for finding in result.violations}


def test_lfm_revenue_threshold() -> None:
    below = la.audit([], annual_revenue_usd=5_000_000)
    assert below.ok
    exceeded = la.audit([], annual_revenue_usd=10_000_000)
    assert not exceeded.ok
    assert {finding.code for finding in exceeded.violations} == {"revenue_threshold_exceeded"}
    checks = {item["entry"]: item["status"] for item in below.to_dict()["threshold_checks"]}
    assert checks["lfm3_omni_2b"] == "within_threshold"
    unverified = la.audit([])
    assert unverified.ok
    assert "revenue_threshold_unverified" in {finding.code for finding in unverified.warnings}


def test_duplicate_ids_across_manifests_are_counted_once(tmp_path: Path) -> None:
    record = _record(record_id="same", source="open_yap_1k", license_name="CC-BY-4.0", hours=1.0)
    first = tmp_path / "a.jsonl"
    second = tmp_path / "repartee-mix.jsonl"
    _write_jsonl(first, [record])
    _write_jsonl(second, [record])
    result = la.audit([first, second])
    assert result.conversations == 1
    assert result.hours == pytest.approx(1.0)
    assert result.ok

    conflicting = dict(record)
    conflicting["license"] = "proprietary"
    _write_jsonl(second, [conflicting])
    conflict = la.audit([first, second])
    assert "duplicate_conflict" in {finding.code for finding in conflict.violations}


def test_malformed_json_is_reported_with_location(tmp_path: Path) -> None:
    manifest = tmp_path / "broken.jsonl"
    manifest.write_text('{"id": "ok"}\nnot-json\n', encoding="utf-8")
    result = la.audit([manifest])
    codes = {finding.code for finding in result.violations}
    assert "malformed_manifest" in codes
    assert any(finding.record_id == "line:2" for finding in result.violations)


def test_markdown_and_json_reports(tmp_path: Path) -> None:
    manifest = tmp_path / "mix.jsonl"
    _write_jsonl(
        manifest,
        [
            _record(record_id="c1", source="open_yap_1k", license_name="CC-BY-4.0", hours=2.0),
            _record(
                record_id="t1",
                source="otospeech_turn104h",
                license_name="oto-speech-license-v1.0",
                hours=1.0,
            ),
            _record(
                record_id="s1",
                source="synthetic",
                license_name="synthetic-commercial",
                hours=4.0,
                meta={"tts_backend": "kokoro"},
            ),
        ],
    )
    result = la.audit([manifest], annual_revenue_usd=2_000_000)
    markdown = la.render_markdown(result)
    assert "# Repartee licence audit" in markdown
    assert "FAIL" in markdown
    assert "otospeech_turn104h" in markdown
    assert "xtts_v2" in markdown and "forbidden" in markdown.lower()
    assert "$10,000,000" in markdown
    assert "non_commercial_source" in markdown
    json_path, md_path = la.write_reports(result, tmp_path / "audit")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["manifests"]["conversations"] == 3
    assert md_path.read_text(encoding="utf-8") == markdown


def test_cli_strict_exit_code(tmp_path: Path) -> None:
    manifest = tmp_path / "turn.jsonl"
    _write_jsonl(
        manifest,
        [
            _record(
                record_id="t1",
                source="otospeech_turn104h",
                license_name="oto-speech-license-v1.0",
            )
        ],
    )
    out = tmp_path / "audit"
    assert la.main(["--manifests", str(manifest), "--out", str(out), "--strict"]) == 1
    assert (out / "license_audit.json").exists()
    clean = tmp_path / "clean.jsonl"
    _write_jsonl(
        clean,
        [_record(record_id="c1", source="open_yap_1k", license_name="CC-BY-4.0")],
    )
    assert la.main(["--manifests", str(clean), "--out", str(out), "--strict"]) == 0


def test_missing_manifest_path_is_an_error(tmp_path: Path) -> None:
    assert la.main(["--manifests", str(tmp_path / "nope.jsonl")]) == 2
