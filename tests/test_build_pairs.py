"""Pure tests for the FDB-labelled DPO pair builder helpers."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.build_dpo_pairs import find_onsets, load_preferences
from src.schema import CONTROL_IDS


def test_find_onsets_marks_each_speak_run() -> None:
    speak = CONTROL_IDS["SPEAK"]
    listen = CONTROL_IDS["LISTEN"]
    control = np.array([listen, speak, speak, listen, speak, CONTROL_IDS["BACKCHANNEL"], speak])
    assert find_onsets(control) == [1, 4, 6]


def test_load_preferences_accepts_mapping(tmp_path: Path) -> None:
    path = tmp_path / "fdb.json"
    path.write_text(json.dumps({"preferences": {"conv:3": "early", "conv:9": "late"}}))
    assert load_preferences(path) == {"conv:3": "early", "conv:9": "late"}


def test_load_preferences_accepts_records(tmp_path: Path) -> None:
    path = tmp_path / "fdb.json"
    path.write_text(
        json.dumps(
            [
                {"item": "conv", "onset_frame": 3, "chosen": "early"},
                {"id": "conv:9", "chosen": "late"},
            ]
        )
    )
    assert load_preferences(path) == {"conv:3": "early", "conv:9": "late"}


def test_load_preferences_rejects_unlabelled(tmp_path: Path) -> None:
    path = tmp_path / "fdb.json"
    path.write_text(json.dumps({"preferences": {"conv:3": "maybe"}}))
    with pytest.raises(ValueError, match="expected 'early' or 'late'"):
        load_preferences(path)
