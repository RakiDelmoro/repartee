#!/usr/bin/env python3
"""Gate 4: text-only MMLU and GSM8K within 2 points of the base model.

The blueprint requires *both* suites.  ``eval.text_regression`` evaluates one
suite at a time, so this script runs the base model (baseline) and the candidate
(stage-1 checkpoint) for each suite and writes one combined artifact that
``src.train_stage2 --gate4`` accepts.

Artifact shape::

    {"gate": "Gate 4", "result": "PASS|FAIL|UNVERIFIED", "passed": bool,
     "threshold_points": 2.0,
     "suites": {"mmlu": {...}, "gsm8k": {...}}}

A suite that cannot run (missing dataset/network/model) is UNVERIFIED and makes
the whole gate UNVERIFIED; it is never silently treated as a pass.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval import text_regression as text_eval  # noqa: E402

DEFAULT_SUITES = ("mmlu", "gsm8k")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="LiquidAI/LFM3-Omni-2B", help="base model for the baseline")
    parser.add_argument("--candidate", required=True, help="candidate checkpoint (e.g. checkpoints/stage1)")
    parser.add_argument("--suites", default=",".join(DEFAULT_SUITES))
    parser.add_argument("--limit", type=int, default=200, help="examples per suite")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="auto", choices=("auto", "float32", "bfloat16", "float16"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/gates/gate4.json"))
    parser.add_argument("--work-dir", type=Path, default=Path("artifacts/gates/text"))
    parser.add_argument("--baseline-dir", type=Path, default=None, help="reuse baseline JSONs from here")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args(argv)


def _run_suite(
    *,
    model: str,
    suite: str,
    out: Path,
    args: argparse.Namespace,
    candidate: Path | None,
    gate_artifact: Path | None = None,
) -> dict[str, Any]:
    return text_eval.run(
        model_id=model,
        suite=suite,
        out=out,
        limit=args.limit,
        offset=args.offset,
        candidate=candidate,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        gate_artifact=gate_artifact,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    suites = [suite.strip() for suite in args.suites.split(",") if suite.strip()]
    unknown = sorted(set(suites) - set(DEFAULT_SUITES))
    if unknown:
        raise SystemExit(f"unknown suite(s) {unknown}; choose from {DEFAULT_SUITES}")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir = args.baseline_dir or args.work_dir

    report: dict[str, Any] = {
        "gate": "Gate 4",
        "threshold_points": text_eval.GATE4_TOLERANCE_POINTS,
        "model": args.model,
        "candidate": args.candidate,
        "limit": args.limit,
        "offset": args.offset,
        "suites": {},
    }
    all_pass = True
    any_unverified = False

    for suite in suites:
        baseline_path = baseline_dir / f"{suite}_baseline.json"
        candidate_path = args.work_dir / f"{suite}_candidate.json"
        suite_report: dict[str, Any] = {
            "baseline": str(baseline_path),
            "candidate": str(candidate_path),
        }
        try:
            if not baseline_path.exists():
                _run_suite(model=args.model, suite=suite, out=baseline_path, args=args, candidate=None)
            _run_suite(
                model=args.candidate,
                suite=suite,
                out=candidate_path,
                args=args,
                candidate=baseline_path,
            )
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
            comparison = candidate.get("comparison", {})
            suite_report.update(
                {
                    "baseline_accuracy": baseline.get("accuracy"),
                    "candidate_accuracy": candidate.get("accuracy"),
                    "delta_points": comparison.get("delta_points"),
                    "passes": comparison.get("passes"),
                    "status": comparison.get("status"),
                    "reason": comparison.get("reason"),
                }
            )
            if baseline.get("status") != "ok" or candidate.get("status") != "ok":
                any_unverified = True
                suite_report["status"] = "UNVERIFIED"
            elif comparison.get("passes") is not True:
                all_pass = False
        except Exception as error:  # noqa: BLE001 - record the reason, never guess
            any_unverified = True
            suite_report.update(
                {
                    "status": "UNVERIFIED",
                    "reason": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(limit=8),
                }
            )
        report["suites"][suite] = suite_report

    if any_unverified:
        report["result"] = "UNVERIFIED"
        report["passed"] = False
    elif all_pass:
        report["result"] = "PASS"
        report["passed"] = True
    else:
        report["result"] = "FAIL"
        report["passed"] = False

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else (2 if any_unverified else 1)


if __name__ == "__main__":
    sys.exit(main())
