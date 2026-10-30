#!/usr/bin/env python3
"""Build FDB-labelled DPO timing pairs from token shards.

The blueprint's stage 3 needs 5,000 pairs that share one context but differ in
response timing (200 ms vs 1,200 ms onset, or barge-in vs talk-over), with the
winning side chosen by Full-Duplex-Bench metrics.  This script creates the
candidate timings; it does **not** invent the preference labels.

Input
-----
``--fdb-results`` is a JSON produced by the FDB evaluation for the two timing
variants.  Accepted shapes::

    {"preferences": {"<item_id>:<onset_frame>": "early" | "late"}}
    {"preferences": [{"item": "...", "onset_frame": 12, "chosen": "early"}]}
    [{"id": "...", "onset_frame": 12, "chosen": "early"}]

Output (JSONL, consumable by ``src.train_dpo --pairs``)::

    {"id": "...", "dataset_index": 17,
     "chosen_control": [...], "rejected_control": [...],
     "label_source": "fdb", "onset_frame": 12}

Pairs without an FDB label are skipped, and the command exits non-zero if no
labelled pair can be written.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import DuplexShardDataset, ShardIndex  # noqa: E402
from src.schema import CONTROL_IDS  # noqa: E402
from src.train_dpo import timing_variants  # noqa: E402


def _preference_key(item_id: str, onset: int) -> str:
    return f"{item_id}:{int(onset)}"


def load_preferences(path: Path) -> dict[str, str]:
    """Return ``{"<item_id>:<onset_frame>": "early"|"late"}``."""
    data = json.loads(path.read_text(encoding="utf-8"))
    raw: Iterable[Any]
    if isinstance(data, dict):
        if "preferences" not in data:
            raise ValueError(f"{path}: expected a 'preferences' object")
        value = data["preferences"]
        if isinstance(value, dict):
            raw = [{"id": key, "chosen": choice} for key, choice in value.items()]
        elif isinstance(value, list):
            raw = value
        else:
            raise ValueError(f"{path}: preferences must be an object or list")
    elif isinstance(data, list):
        raw = data
    else:
        raise ValueError(f"{path}: expected a JSON object or list")

    preferences: dict[str, str] = {}
    for index, record in enumerate(raw):
        if not isinstance(record, dict):
            raise ValueError(f"{path}: preference {index} is not an object")
        if "id" in record and "onset_frame" not in record and ":" in str(record["id"]):
            key = str(record["id"])
            choice = record.get("chosen")
        else:
            key = _preference_key(str(record.get("item", record.get("id", ""))), int(record.get("onset_frame", -1)))
            choice = record.get("chosen", record.get("preference"))
        if choice not in ("early", "late"):
            raise ValueError(f"{path}: preference {index} has chosen={choice!r}; expected 'early' or 'late'")
        if key.endswith(":-1"):
            raise ValueError(f"{path}: preference {index} needs id + onset_frame")
        preferences[key] = str(choice)
    if not preferences:
        raise ValueError(f"{path}: no preferences")
    return preferences


def find_onsets(control: np.ndarray, *, limit: int | None = None) -> list[int]:
    """Agent speech onsets, i.e. SPEAK frames after a non-speech frame."""
    labels = np.asarray(control).reshape(-1)
    speak = CONTROL_IDS["SPEAK"]
    onsets = [index for index in range(len(labels)) if labels[index] == speak and (index == 0 or labels[index - 1] != speak)]
    return onsets[:limit] if limit is not None else onsets


def build(
    *,
    tokens: Path,
    fdb_results: Path,
    out: Path,
    count: int = 5000,
    max_items: int = 20000,
    seq_frames: int = 4096,
    seed: int = 0,
    early_delay_frames: int = 3,
    late_delay_frames: int = 15,
) -> dict[str, Any]:
    preferences = load_preferences(fdb_results)
    index = ShardIndex.load(tokens)
    dataset = DuplexShardDataset(
        index,
        seq_frames=seq_frames,
        split="train",
        samples_per_epoch=max_items,
        seed=seed,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    scanned = 0
    with out.open("w", encoding="utf-8") as handle:
        for dataset_index in range(min(len(dataset), max_items)):
            item = dataset[dataset_index]
            scanned += 1
            onsets = find_onsets(item["control"])
            if not onsets:
                skipped += 1
                continue
            for onset in onsets:
                choice = preferences.get(_preference_key(item["id"], onset))
                if choice is None:
                    continue
                try:
                    early, late = timing_variants(
                        item,
                        onset_frame=onset,
                        early_delay_frames=early_delay_frames,
                        late_delay_frames=late_delay_frames,
                        response_label=CONTROL_IDS["SPEAK"],
                    )
                except ValueError:
                    skipped += 1
                    continue
                chosen, rejected = (early, late) if choice == "early" else (late, early)
                record = {
                    "id": f"{item['id']}:{onset}",
                    "dataset_index": dataset_index,
                    "chosen_control": np.asarray(chosen["control"]).astype(int).tolist(),
                    "rejected_control": np.asarray(rejected["control"]).astype(int).tolist(),
                    "label_source": "fdb",
                    "onset_frame": int(onset),
                    "preference": choice,
                }
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                written += 1
                if written >= count:
                    break
            if written >= count:
                break
    if written == 0:
        raise RuntimeError(
            "no FDB-labelled pairs written; the timing pairs and the FDB preference file did not overlap. "
            "Run eval/fdb.py for both timing variants first."
        )
    return {
        "written": written,
        "scanned_items": scanned,
        "skipped_items": skipped,
        "target": count,
        "early_delay_frames": early_delay_frames,
        "late_delay_frames": late_delay_frames,
        "out": str(out),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tokens", type=Path, default=Path("data/tokens"))
    parser.add_argument("--fdb-results", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/pairs/dpo_timing.jsonl"))
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--max-items", type=int, default=20000)
    parser.add_argument("--seq-frames", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--early-delay-frames", type=int, default=3, help="200 ms")
    parser.add_argument("--late-delay-frames", type=int, default=15, help="1,200 ms")
    args = parser.parse_args(argv)
    if args.count <= 0:
        parser.error("--count must be positive")
    try:
        summary = build(
            tokens=args.tokens,
            fdb_results=args.fdb_results,
            out=args.out,
            count=args.count,
            max_items=args.max_items,
            seq_frames=args.seq_frames,
            seed=args.seed,
            early_delay_frames=args.early_delay_frames,
            late_delay_frames=args.late_delay_frames,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"build_dpo_pairs: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
