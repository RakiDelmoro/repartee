"""Tests for the training dashboard's metrics parsing and discovery."""
from __future__ import annotations

import json
from pathlib import Path

from tools.dashboard import discover_runs, load_run, summarize_run


def _write_run(root: Path, name: str, records: list[dict]) -> Path:
    run = root / name
    run.mkdir(parents=True)
    (run / "metrics.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    return run


def test_summarize_and_load_run(tmp_path: Path) -> None:
    run = _write_run(
        tmp_path,
        "stage1",
        [
            {"event": "start", "device": "cuda", "dtype": "fp16", "steps": 100},
            {"step": 1, "loss": 5.0, "text_loss": 3.0, "control_loss": 1.0, "audio_loss": 1.0, "lr": 0.001, "seconds_per_step": 10.0},
            {"step": 2, "loss": 4.0, "text_loss": 2.0, "control_loss": 1.0, "audio_loss": 1.0, "lr": 0.0009, "seconds_per_step": 12.0},
        ],
    )
    summary = summarize_run(run)
    assert summary["step"] == 2
    assert summary["total_steps"] == 100
    assert summary["eta_s"] == 98 * 11.0
    assert summary["checkpoint"]["exists"] is False
    loaded = load_run(run)
    assert loaded["series"]["loss"] == [5.0, 4.0]
    assert loaded["start"]["device"] == "cuda"


def test_discover_runs_newest_first(tmp_path: Path) -> None:
    _write_run(tmp_path, "a", [{"step": 1}])
    import time

    time.sleep(0.01)
    _write_run(tmp_path, "b", [{"step": 1}])
    names = [path.name for path in discover_runs(tmp_path)]
    assert names[0] == "b"
    assert set(names) == {"a", "b"}


def test_load_run_handles_missing_metrics(tmp_path: Path) -> None:
    run = tmp_path / "empty"
    run.mkdir()
    loaded = load_run(run)
    assert loaded["step"] == 0
    assert loaded["series"]["loss"] == []
