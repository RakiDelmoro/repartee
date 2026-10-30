"""Tests for the build-system gate prerequisite check."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.require_gate import gate_passed, main


def test_gate_shapes() -> None:
    assert gate_passed({"result": "PASS"})
    assert gate_passed({"passed": True})
    assert gate_passed({"passes": True})
    assert gate_passed({"overall": {"result": "PASS"}})
    assert gate_passed({"result": "CONDITIONAL"})
    assert not gate_passed({"result": "FAIL"})
    assert not gate_passed({"overall": {"result": "FAIL"}})
    assert not gate_passed({})


def test_main_refuses_failed_artifact(tmp_path: Path) -> None:
    path = tmp_path / "gate.json"
    path.write_text(json.dumps({"result": "FAIL"}))
    assert main(["--artifact", str(path)]) == 1


def test_main_accepts_passed_artifact(tmp_path: Path) -> None:
    path = tmp_path / "gate.json"
    path.write_text(json.dumps({"overall": {"result": "PASS"}}))
    assert main(["--artifact", str(path)]) == 0
