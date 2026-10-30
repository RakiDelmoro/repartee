"""Full-Duplex-Bench (FDB) v1.0 / v1.5 runner.

What this module implements
---------------------------
* Discovery and annotation parsing for the public FDB v1.0 and v1.5 data
  layouts (``candor_pause_handling``, ``candor_turn_taking``,
  ``icc_backchannel``, ``synthetic_pause_handling``,
  ``synthetic_user_interruption`` and the four v1.5 overlap scenarios).
* Annotation-driven scoring with the pure interval metrics in
  :mod:`eval.metrics`.  The model's speech intervals come either from an
  existing ``output.wav`` (``--engine wav``) or from the real Repartee duplex
  loop (``--engine repartee``), passed through a deterministic file source.
* A JSON report with per-category and overall numbers, explicit ``n`` and
  ``missing`` counts, and the blueprint Gate 5 checks.

What it does **not** do
-----------------------
* It never invents data.  If the public FDB download is absent, the CLI raises
  an actionable error naming the public folder URL.  ``--download`` will try
  ``gdown`` for that folder and verify the resulting layout; it is never
  silently skipped.
* It refuses to treat the repository's ``example_data`` demo clips as
  evidence.  ``--allow-demo`` runs them but writes ``"evidence": false``.
* It does not reproduce FDB's GPT-4o judged interruption score or the ICC
  Jensen-Shannon number unless the downloaded ICC ground-truth JSON is present.
  Those fields are reported as ``null`` with a reason instead of guessed.

Data URL (v1.0 and v1.5): the public Google Drive folder linked by the
benchmark, ``https://drive.google.com/drive/folders/1DtoxMVO9_Y_nDs2peZtx3pw-U2qYgpd3``.
The upstream repository is ``https://github.com/DanielLin94144/Full-Duplex-Bench``.

Usage
-----
::

    python -m eval.fdb --data /data/full-duplex-bench --dry-run
    python -m eval.fdb --data /data/full-duplex-bench --engine wav --out results.json
    python -m eval.fdb --data /data/full-duplex-bench --engine repartee \
        --model checkpoints/stage2 --out results.json

``--data`` may also be a JSONL file with one session per line::

    {"id": "s1", "version": "v1.5", "category": "user_interruption",
     "user_speech": [[0.0, 2.0], [7.0, 8.0]],
     "agent_speech": [[9.0, 11.0]],
     "interruptions": [[7.0, 8.0]], "agent_word_counts": [4, 8],
     "agent_wav": "out.wav"}

Only the assigned files are touched.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eval.metrics import (
    GATE7_SELF_INTERRUPTION_MAX,
    aggregate_latencies,
    aggregate_rates,
    backchannel_resume_flags,
    gate5_checks,
    interruption_success_flags,
    normalize_intervals,
    pause_takeovers,
    self_interruption_events,
    self_interruption_rate,
    speech_span,
    stop_latencies,
    time_to_first_audio,
    turn_taking_success_flags,
)

FDB_DATA_URL = (
    "https://drive.google.com/drive/folders/1DtoxMVO9_Y_nDs2peZtx3pw-U2qYgpd3"
)
FDB_REPO_URL = "https://github.com/DanielLin94144/Full-Duplex-Bench"

# Directory names used by the public release.  Keys are normalised to lower
# case with dashes/spaces replaced by underscores.
V1_CATEGORY_DIRS: dict[str, str] = {
    "candor_pause_handling": "pause_handling",
    "synthetic_pause_handling": "pause_handling",
    "pause_handling": "pause_handling",
    "candor_turn_taking": "smooth_turn_taking",
    "smooth_turn_taking": "smooth_turn_taking",
    "turn_taking": "smooth_turn_taking",
    "icc_backchannel": "backchannel",
    "backchannel": "backchannel",
    "synthetic_user_interruption": "user_interruption",
    "user_interruption": "user_interruption",
}
V15_CATEGORY_DIRS: dict[str, str] = {
    "user_interruption": "user_interruption",
    "user_backchannel": "user_backchannel",
    "user_backchanneling": "user_backchannel",
    "talking_to_other": "talking_to_other",
    "talking_other": "talking_to_other",
    "background_speech": "background_speech",
    "bkg_speech": "background_speech",
}
V1_VERSIONS = {"v1", "v1.0", "v1_0", "version1", "1.0"}
V15_VERSIONS = {"v1.5", "v1_5", "version1.5", "1.5"}

# Published sample counts (paper Table II).  Downloads may be partial; these
# are used for warnings, never to fabricate missing sessions.
PUBLISHED_COUNTS = {
    ("v1", "pause_handling"): 353,  # 216 Candor + 137 synthetic
    ("v1", "smooth_turn_taking"): 119,
    ("v1", "backchannel"): 55,
    ("v1", "user_interruption"): 200,
    ("v1.5", "user_interruption"): 200,
    ("v1.5", "user_backchannel"): 99,
    ("v1.5", "talking_to_other"): 100,
    ("v1.5", "background_speech"): 100,
}

_FRAME_RATE = 12.5


class FdbError(RuntimeError):
    """Actionable runner failure (missing data, missing real engine, ...)."""


# ---------------------------------------------------------------------------
# Small IO helpers
# ---------------------------------------------------------------------------


def _canonical_dir_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as error:
        raise FdbError(f"required annotation is missing: {path}") from error
    except json.JSONDecodeError as error:
        raise FdbError(f"{path}: invalid JSON: {error}") from error


def _parse_intervals(value: Any, *, path: str = "annotations") -> list[tuple[float, float]]:
    """Parse ``[[s, e], ...]`` or ``[{"timestamp": [s, e]}, ...]``."""
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise FdbError(f"{path}: expected a list of intervals, got {type(value).__name__}")
    if (
        len(value) == 2
        and all(isinstance(item, (int, float)) for item in value)
    ):
        value = [value]
    intervals: list[tuple[float, float]] = []
    for item in value:
        if isinstance(item, Mapping):
            item = item.get("timestamp", item.get("timestamps"))
            if item is None:
                continue
        try:
            start, end = float(item[0]), float(item[1])
        except (TypeError, ValueError, IndexError) as error:
            raise FdbError(f"{path}: bad interval {item!r}") from error
        if start > end:
            start, end = end, start
        if end > start:
            intervals.append((start, end))
    return intervals


def _parse_boundaries(value: Any, *, path: str = "annotations") -> list[float]:
    """Parse FDB turn-end annotations.

    The released v1.0 code uses ``turn_taking.json[0]["timestamp"][0]`` as the
    user turn end (``input_end_time``).  We accept that shape and an explicit
    list of numbers.
    """
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    result: list[float] = []
    if not isinstance(value, (list, tuple)):
        raise FdbError(f"{path}: expected a list of timestamps")
    for item in value:
        if isinstance(item, Mapping):
            item = item.get("timestamp", item.get("timestamps"))
        if item is None:
            continue
        if isinstance(item, (int, float)):
            result.append(float(item))
        else:
            try:
                result.append(float(item[0]))
            except (TypeError, ValueError, IndexError) as error:
                raise FdbError(f"{path}: bad turn-end {item!r}") from error
    return result


def _load_word_chunks(path: Path | None) -> list[dict[str, Any]] | None:
    """Read the FDB ASR format (``{"chunks": [{"text", "timestamp"}, ...]}``)."""
    if path is None or not path.exists():
        return None
    payload = _read_json(path)
    if isinstance(payload, Mapping):
        chunks = payload.get("chunks", [])
    elif isinstance(payload, list):
        chunks = payload
    else:
        raise FdbError(f"{path}: expected an object with a 'chunks' list")
    cleaned: list[dict[str, Any]] = []
    for chunk in chunks:
        if not isinstance(chunk, Mapping):
            continue
        timestamp = chunk.get("timestamp", chunk.get("timestamps"))
        if not timestamp or len(timestamp) < 2:
            continue
        start, end = timestamp[0], timestamp[1]
        if start is None:
            continue
        try:
            start = float(start)
            end = float(end) if end is not None else start
        except (TypeError, ValueError):
            continue
        cleaned.append(
            {"text": str(chunk.get("text", "")).strip(), "start": start, "end": max(start, end)}
        )
    return cleaned


def word_counts_for_segments(
    segments: Sequence[tuple[float, float]],
    chunks: Sequence[Mapping[str, Any]] | None,
) -> list[int] | None:
    """Count ASR chunks whose midpoint falls inside each speech segment."""
    if chunks is None:
        return None
    counts = [0] * len(segments)
    for chunk in chunks:
        midpoint = 0.5 * (float(chunk["start"]) + float(chunk["end"]))
        for index, (start, end) in enumerate(segments):
            if start - 1e-9 <= midpoint <= end + 1e-9:
                counts[index] += 1
                break
    return counts


# ---------------------------------------------------------------------------
# Session description
# ---------------------------------------------------------------------------


@dataclass
class FdbSession:
    """One scored FDB sample, independent of how its output audio is made."""

    session_id: str
    version: str  # "v1" or "v1.5"
    category: str
    sample_dir: Path | None = None
    user_wav: Path | None = None
    output_wav: Path | None = None
    output_json: Path | None = None
    user_speech: list[tuple[float, float]] = field(default_factory=list)
    agent_speech: list[tuple[float, float]] = field(default_factory=list)
    pauses: list[tuple[float, float]] = field(default_factory=list)
    interruptions: list[tuple[float, float]] = field(default_factory=list)
    backchannels: list[tuple[float, float]] = field(default_factory=list)
    turn_ends: list[float] = field(default_factory=list)
    agent_word_counts: list[int] | None = None
    ground_truth_distribution: list[float] | None = None
    duration_s: float | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    def to_plan(self) -> dict[str, Any]:
        return {
            "id": self.session_id,
            "version": self.version,
            "category": self.category,
            "sample_dir": str(self.sample_dir) if self.sample_dir else None,
            "user_wav": str(self.user_wav) if self.user_wav and self.user_wav.exists() else None,
            "output_wav": str(self.output_wav) if self.output_wav and self.output_wav.exists() else None,
            "pause_events": len(self.pauses),
            "interruption_events": len(self.interruptions),
            "backchannel_events": len(self.backchannels),
            "turn_ends": len(self.turn_ends),
        }


# ---------------------------------------------------------------------------
# Data discovery
# ---------------------------------------------------------------------------


def _iter_category_dirs(root: Path) -> Iterator[tuple[str, str, Path]]:
    """Yield ``(version, category, directory)`` for recognised FDB folders.

    Handles the release layout (``v1_0/<category>/<id>/`` and
    ``v1_5/<category>/<id>/``), a dataset root passed one level lower, and a
    single category directory passed directly.
    """
    seen: set[Path] = set()
    for current, dirs, _ in os.walk(root):
        dirs.sort()
        depth = len(Path(current).relative_to(root).parts)
        if depth >= 3:
            dirs[:] = []
            continue
        normalized = _canonical_dir_name(Path(current).name)
        is_version_dir = normalized in V1_VERSIONS or normalized in V15_VERSIONS
        if is_version_dir:
            dirs[:] = [d for d in dirs if d.lower() not in {"__macosx", ".git"}]
            key = "v1" if normalized in V1_VERSIONS else "v1.5"
            for child in sorted(dirs):
                child_dir = Path(current) / child
                canon = _canonical_dir_name(child)
                if key == "v1" and canon in V1_CATEGORY_DIRS:
                    yield key, V1_CATEGORY_DIRS[canon], child_dir
                elif key == "v1.5" and canon in V15_CATEGORY_DIRS:
                    yield key, V15_CATEGORY_DIRS[canon], child_dir
            dirs[:] = []
            continue
        if normalized in V1_CATEGORY_DIRS or normalized in V15_CATEGORY_DIRS:
            path = Path(current)
            if path in seen:
                continue
            seen.add(path)
            version = "v1" if normalized in V1_CATEGORY_DIRS else "v1.5"
            mapping = V1_CATEGORY_DIRS if version == "v1" else V15_CATEGORY_DIRS
            yield version, mapping[normalized], path
            dirs[:] = []
            continue
        if depth == 0:
            dirs[:] = [d for d in dirs if d.lower() not in {"__macosx", ".git", "node_modules"}]


def _sample_dirs(category_dir: Path) -> Iterator[Path]:
    """Sample folders below a category directory, sorted by name.

    A category directory that itself contains ``input.wav`` / ``output.wav``
    is treated as one sample (some mirrors flatten the ID level).
    """
    if any((category_dir / name).exists() for name in ("input.wav", "output.wav", "metadata.json")):
        yield category_dir
        return
    for child in sorted(category_dir.iterdir(), key=lambda p: p.name):
        if not child.is_dir() or child.name.startswith("."):
            continue
        if any(
            (child / name).exists()
            for name in ("input.wav", "output.wav", "clean_input.wav", "metadata.json")
        ):
            yield child


def _load_v1_annotations(session: FdbSession) -> None:
    directory = session.sample_dir
    assert directory is not None
    if session.category == "pause_handling":
        pause_path = directory / "pause.json"
        session.pauses = _parse_intervals(_read_json(pause_path), path=str(pause_path))
    elif session.category == "smooth_turn_taking":
        turn_path = directory / "turn_taking.json"
        session.turn_ends = _parse_boundaries(
            _read_json(turn_path), path=str(turn_path)
        )
    elif session.category == "user_interruption":
        interrupt_path = directory / "interrupt.json"
        session.interruptions = _parse_intervals(
            _read_json(interrupt_path), path=str(interrupt_path)
        )
    elif session.category == "backchannel":
        # ICC samples ship a stimulus transcript but no per-utterance
        # backchannel intervals; scoring uses the model-side TOR/frequency and
        # the ICC ground-truth distribution when present.
        transcription = directory / "transcription.json"
        if transcription.exists():
            session.notes["stimulus_transcription"] = str(transcription)


def _load_v15_annotations(session: FdbSession) -> None:
    directory = session.sample_dir
    assert directory is not None
    metadata_path = directory / "metadata.json"
    metadata = _read_json(metadata_path)
    if isinstance(metadata, list):
        if not metadata:
            raise FdbError(f"{metadata_path}: empty metadata list")
        metadata = metadata[0]
    if not isinstance(metadata, Mapping):
        raise FdbError(f"{metadata_path}: expected an object")
    timestamps = metadata.get("timestamps")
    if timestamps is None:
        raise FdbError(f"{metadata_path}: missing 'timestamps'")
    overlap = _parse_intervals([timestamps], path=str(metadata_path))
    session.notes["context_text"] = metadata.get("context_text", "")
    session.notes["current_turn_text"] = metadata.get("current_turn_text", "")
    if session.category == "user_interruption":
        session.interruptions = overlap
    elif session.category == "user_backchannel":
        session.backchannels = overlap
    else:
        # talking_to_other / background_speech: the model should keep its turn
        # (Resume).  The same interval machinery applies.
        session.backchannels = overlap
    session.user_speech = list(overlap)


def _load_session(sample_dir: Path, version: str, category: str) -> FdbSession:
    session_id = f"{category}:{sample_dir.name}" if sample_dir.name else category
    session = FdbSession(
        session_id=session_id,
        version=version,
        category=category,
        sample_dir=sample_dir,
        user_wav=sample_dir / "input.wav" if (sample_dir / "input.wav").exists() else None,
        output_wav=sample_dir / "output.wav" if (sample_dir / "output.wav").exists() else None,
        output_json=sample_dir / "output.json" if (sample_dir / "output.json").exists() else None,
    )
    if version == "v1":
        _load_v1_annotations(session)
    else:
        _load_v15_annotations(session)
    return session


def load_jsonl_sessions(path: Path) -> list[FdbSession]:
    """Load an explicit, annotation-driven session file.

    Required per record: ``id``, ``category``.  Optional: ``version``
    (defaults to the category's benchmark), ``agent_speech``/``agent_wav``,
    ``pause``/``pauses``, ``interruptions``/``interrupt``, ``backchannels``,
    ``turn_ends``, ``agent_word_counts``, ``user_speech``, ``user_wav``,
    ``output_json``, ``ground_truth_distribution``, ``duration_s``.
    """
    sessions: list[FdbSession] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise FdbError(f"{path}:{line_number}: invalid JSON: {error}") from error
        if not isinstance(record, Mapping):
            raise FdbError(f"{path}:{line_number}: expected an object")
        category = str(record.get("category", "")).strip()
        if not category:
            raise FdbError(f"{path}:{line_number}: missing 'category'")
        version = str(record.get("version") or ("v1" if category in V1_CATEGORY_DIRS.values() else "v1.5"))
        if version.lower() in V1_VERSIONS:
            version = "v1"
        elif version.lower() in V15_VERSIONS:
            version = "v1.5"
        else:
            raise FdbError(f"{path}:{line_number}: unknown version {version!r}")
        base = path.parent
        session = FdbSession(
            session_id=str(record.get("id") or f"line{line_number}"),
            version=version,
            category=category,
            sample_dir=None,
            user_wav=_resolve_optional_path(record.get("user_wav"), base),
            output_wav=_resolve_optional_path(
                record.get("agent_wav", record.get("output_wav", record.get("model_wav"))),
                base,
            ),
            output_json=_resolve_optional_path(record.get("output_json"), base),
            user_speech=_parse_intervals(
                record.get("user_speech", record.get("user_intervals")),
                path=f"{path}:{line_number}",
            ),
            agent_speech=_parse_intervals(
                record.get(
                    "agent_speech",
                    record.get("model_speech", record.get("agent_intervals", record.get("output_intervals"))),
                ),
                path=f"{path}:{line_number}",
            ),
            pauses=_parse_intervals(
                record.get("pauses", record.get("pause", record.get("pause_intervals"))),
                path=f"{path}:{line_number}",
            ),
            interruptions=_parse_intervals(
                record.get(
                    "interruptions",
                    record.get("interrupt", record.get("interruption_intervals")),
                ),
                path=f"{path}:{line_number}",
            ),
            backchannels=_parse_intervals(
                record.get(
                    "backchannels",
                    record.get("backchannel", record.get("backchannel_intervals")),
                ),
                path=f"{path}:{line_number}",
            ),
            turn_ends=_parse_boundaries(record.get("turn_ends"), path=f"{path}:{line_number}"),
            agent_word_counts=(
                [
                    int(value)
                    for value in record.get(
                        "agent_word_counts", record.get("word_counts")
                    )
                ]
                if record.get("agent_word_counts", record.get("word_counts")) is not None
                else None
            ),
            ground_truth_distribution=(
                [float(value) for value in record["ground_truth_distribution"]]
                if record.get("ground_truth_distribution") is not None
                else None
            ),
            duration_s=(
                float(record.get("duration_s", record.get("duration")))
                if record.get("duration_s", record.get("duration")) is not None
                else None
            ),
        )
        sessions.append(session)
    return sessions


def _resolve_optional_path(value: Any, base: Path) -> Path | None:
    if value is None:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else (base / path)


def load_icc_gt(path: Path) -> dict[str, list[float]]:
    """Load the ICC ground-truth backchannel distributions.

    The upstream file is a JSON object mapping ICC sample ids to a list of
    200 ms window weights.  Missing/invalid values are skipped, but a file
    that contains no usable distributions is an error rather than an empty
    success.
    """
    payload = _read_json(path)
    if not isinstance(payload, Mapping):
        raise FdbError(f"{path}: expected a JSON object of sample-id -> distribution")
    distributions: dict[str, list[float]] = {}
    for key, value in payload.items():
        if not isinstance(value, (list, tuple)) or not value:
            continue
        try:
            distribution = [float(item) for item in value]
        except (TypeError, ValueError):
            continue
        if any(item < 0 for item in distribution) or sum(distribution) <= 0:
            continue
        distributions[str(key)] = distribution
    if not distributions:
        raise FdbError(f"{path}: no valid ICC distributions found")
    return distributions


def attach_icc_gt(sessions: Iterable[FdbSession], distributions: Mapping[str, Sequence[float]]) -> int:
    """Attach ICC ground truth to v1 backchannel sessions by sample id."""
    attached = 0
    for session in sessions:
        if session.category != "backchannel":
            continue
        candidates: list[str] = []
        if session.sample_dir is not None:
            candidates.extend([session.sample_dir.name, str(session.sample_dir)])
        candidates.extend([session.session_id, session.session_id.split(":")[-1]])
        for key in candidates:
            if key in distributions:
                session.ground_truth_distribution = [float(v) for v in distributions[key]]
                session.notes["icc_gt_key"] = key
                attached += 1
                break
    return attached


def _find_icc_gt(data: Path) -> Path | None:
    candidates = [
        data / "icc_gt_distribution.json",
        data / "evaluation" / "icc_gt_distribution.json",
        data.parent / "icc_gt_distribution.json",
        data.parent / "evaluation" / "icc_gt_distribution.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def discover_sessions(data: Path) -> list[FdbSession]:
    """Discover FDB sessions from a data root or a JSONL annotation file."""
    if not data.exists():
        raise FdbError(
            f"FDB data not found at {data}. Download FDB v1.0/v1.5 from the public "
            f"folder {FDB_DATA_URL} (repository: {FDB_REPO_URL}) and pass the "
            "downloaded directory with --data, or pass --download to try gdown."
        )
    if data.is_file():
        if data.suffix.lower() not in {".jsonl", ".ndjson"}:
            raise FdbError(
                f"{data}: expected a directory or a .jsonl annotation file; "
                f"download FDB from {FDB_DATA_URL}"
            )
        sessions = load_jsonl_sessions(data)
        if not sessions:
            raise FdbError(f"{data}: no sessions found")
        return sessions

    sessions: list[FdbSession] = []
    found_category_dirs = False
    for version, category, category_dir in _iter_category_dirs(data):
        found_category_dirs = True
        for sample_dir in _sample_dirs(category_dir):
            sessions.append(_load_session(sample_dir, version, category))
    if not found_category_dirs:
        raise FdbError(
            f"{data}: no FDB category directories found. Expected v1_0/ and v1_5/ "
            f"folders from {FDB_DATA_URL} (repository: {FDB_REPO_URL})."
        )
    if not sessions:
        raise FdbError(f"{data}: FDB categories exist but contain no sample folders")
    return sessions


# ---------------------------------------------------------------------------
# Download / verification
# ---------------------------------------------------------------------------


def download_fdb(destination: Path) -> Path:
    """Download the public FDB folder with ``gdown`` and verify the layout.

    This is deliberately opt-in.  The Google Drive folder is the only public
    distribution linked by the benchmark; there is no Hugging Face mirror
    guaranteed to exist.
    """
    try:
        import gdown
    except ImportError as error:
        raise FdbError(
            "Cannot download FDB because gdown is not installed. Install it with "
            "`pip install gdown`, or download the public folder manually:\n"
            f"  {FDB_DATA_URL}\n"
            f"Repository: {FDB_REPO_URL}"
        ) from error
    destination.mkdir(parents=True, exist_ok=True)
    try:
        result = gdown.download_folder(
            url=FDB_DATA_URL,
            output=str(destination),
            quiet=False,
            use_cookies=False,
        )
    except Exception as error:  # gdown raises a variety of network/parse errors
        raise FdbError(
            f"gdown could not download {FDB_DATA_URL}: {error}. Download it "
            "manually and pass the extracted directory with --data."
        ) from error
    if result is None or not any(destination.rglob("*.wav")):
        raise FdbError(
            f"Download from {FDB_DATA_URL} produced no WAV files under {destination}. "
            "The folder is public but may require browser confirmation; download it "
            f"manually or use another mirror linked from {FDB_REPO_URL}."
        )
    # Fail early if the layout is unrecognisable.
    discover_sessions(destination)
    return destination


def verify_layout(sessions: Sequence[FdbSession]) -> dict[str, Any]:
    """Return counts and published-count warnings; never invents sessions."""
    counts: dict[str, int] = defaultdict(int)
    versions: set[str] = set()
    for session in sessions:
        counts[session.category] += 1
        versions.add(session.version)
    warnings: list[str] = []
    for version in sorted(versions):
        for category, count in sorted(counts.items()):
            expected = PUBLISHED_COUNTS.get((version, category))
            if expected is not None and count < expected:
                warnings.append(
                    f"{version} {category}: found {count} sessions; the paper reports "
                    f"{expected}. Partial downloads are scored as-is."
                )
    return {
        "sessions": len(sessions),
        "versions": sorted(versions),
        "category_counts": dict(sorted(counts.items())),
        "warnings": warnings,
    }


def _is_demo_path(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    return "example_data" in parts or "demo" in parts


# ---------------------------------------------------------------------------
# Audio / VAD
# ---------------------------------------------------------------------------


def audio_duration(path: Path) -> float:
    """Duration in seconds using soundfile's header (no full decode)."""
    try:
        import soundfile as sf
    except ImportError as error:  # pragma: no cover - dependency is installed by pyproject
        raise FdbError("soundfile is required to read evaluation audio") from error
    try:
        info = sf.info(str(path))
    except Exception as error:
        raise FdbError(f"cannot read audio header {path}: {error}") from error
    if info.samplerate <= 0:
        raise FdbError(f"{path}: invalid sample rate {info.samplerate}")
    return float(info.frames) / float(info.samplerate)


def resolve_vad(vad: str) -> tuple[str, str | None]:
    """Resolve ``auto`` to a usable backend and report any fallback.

    ``silero`` remains a hard requirement when explicitly requested; ``auto``
    falls back to the labelled energy approximation.
    """
    if vad == "energy":
        return "energy", None
    try:
        import silero_vad  # noqa: F401, PLC0415 - availability probe

        return "silero", None
    except Exception as error:
        if vad == "silero":
            raise FdbError(
                "Silero VAD is unavailable. Install it with `pip install silero-vad` "
                "(and torch), or use --vad energy for a labelled approximation. "
                f"Original error: {error}"
            ) from error
        return "energy", f"Silero VAD unavailable ({error}); used the energy approximation"


def read_speech_intervals(
    path: Path,
    *,
    vad: str = "auto",
    min_speech_s: float = 0.05,
    min_silence_s: float = 0.05,
) -> list[tuple[float, float]]:
    """Detect speech intervals in a mono WAV.

    ``vad="silero"`` uses the same family of VAD as the FDB scripts and is the
    reproducible choice.  ``vad="energy"`` is a deterministic, dependency-light
    approximation and is always labelled as such in the report.  ``auto``
    prefers Silero and falls back to energy with a recorded warning.
    """
    if vad not in {"auto", "silero", "energy"}:
        raise FdbError(f"unknown VAD {vad!r}; choose auto, silero or energy")
    if vad in {"auto", "silero"}:
        try:
            return _silero_intervals(path, min_speech_s=min_speech_s, min_silence_s=min_silence_s)
        except Exception as error:
            if vad == "silero":
                raise FdbError(
                    "Silero VAD is unavailable. Install it with `pip install silero-vad` "
                    "(and torch), or use --vad energy for a labelled approximation. "
                    f"Original error: {error}"
                ) from error
    return _energy_intervals(path, min_speech_s=min_speech_s, min_silence_s=min_silence_s)


def _load_mono_16k(path: Path):
    import numpy as np
    import soundfile as sf

    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    audio = data.mean(axis=1)
    if sample_rate != 16000:
        # Dependency-light linear resampling for VAD only.  FDB inputs are
        # already 16 kHz; generated 24 kHz output is brought down here.  This
        # is not used for training or for audio features.
        target = max(1, int(round(audio.size * 16000 / sample_rate)))
        source_positions = np.linspace(0.0, 1.0, audio.size, endpoint=False)
        target_positions = np.linspace(0.0, 1.0, target, endpoint=False)
        audio = np.interp(target_positions, source_positions, audio).astype(np.float32)
        sample_rate = 16000
    return np.ascontiguousarray(audio, dtype=np.float32), int(sample_rate)


def _silero_intervals(
    path: Path,
    *,
    min_speech_s: float,
    min_silence_s: float,
) -> list[tuple[float, float]]:
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    audio, sample_rate = _load_mono_16k(path)
    model = load_silero_vad()
    timestamps = get_speech_timestamps(
        torch.from_numpy(audio),
        model,
        sampling_rate=sample_rate,
        min_speech_duration_ms=int(max(0.0, min_speech_s) * 1000),
        min_silence_duration_ms=int(max(0.0, min_silence_s) * 1000),
    )
    return [
        (float(item["start"]) / sample_rate, float(item["end"]) / sample_rate)
        for item in timestamps
    ]


def _energy_intervals(
    path: Path,
    *,
    min_speech_s: float,
    min_silence_s: float,
    frame_s: float = 0.02,
    threshold_db: float = -35.0,
    relative_db: float = 10.0,
) -> list[tuple[float, float]]:
    """Frame-energy VAD with an adaptive floor.

    This is *not* Silero.  It is deterministic and useful for smoke tests, but
    results obtained with it must not be presented as FDB-reproducing.
    """
    import numpy as np

    audio, sample_rate = _load_mono_16k(path)
    frame = max(1, int(round(sample_rate * frame_s)))
    if audio.size < frame:
        return []
    count = audio.size // frame
    trimmed = audio[: count * frame].reshape(count, frame)
    rms = np.sqrt(np.mean(np.square(trimmed, dtype=np.float64), axis=1) + 1e-12)
    db = 20.0 * np.log10(np.maximum(rms, 1e-12))
    # The 10th percentile approximates the noise floor; require both an
    # absolute and a relative margin so a noisy file does not become all
    # speech.  Cap the relative threshold by the median minus a margin so a
    # continuously voiced file is not classified as silence.
    floor = float(np.percentile(db, 10))
    median = float(np.percentile(db, 50))
    relative_threshold = min(floor + relative_db, median - 3.0)
    active = db >= max(threshold_db, relative_threshold)

    intervals: list[tuple[float, float]] = []
    start_index: int | None = None
    for index, is_active in enumerate(active.tolist()):
        if is_active and start_index is None:
            start_index = index
        elif not is_active and start_index is not None:
            intervals.append((start_index * frame_s, index * frame_s))
            start_index = None
    if start_index is not None:
        intervals.append((start_index * frame_s, count * frame_s))

    merged: list[tuple[float, float]] = []
    for start, end in intervals:
        if merged and start - merged[-1][1] <= min_silence_s:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return [
        (start, end)
        for start, end in merged
        if end - start >= min_speech_s
    ]


# ---------------------------------------------------------------------------
# Real engine adapter
# ---------------------------------------------------------------------------


class ReparteeEngineRunner:
    """Lazily wrap ``serve.loop.Lfm3DuplexEngine`` behind a file-to-file call.

    The heavy import (torch/transformers) happens on first construction, never
    at module import.  If serving is absent, the error tells the user to run
    ``--engine wav`` or implement the documented surface.
    """

    def __init__(self, model_id: str, *, int4: bool = False, device: str | None = None) -> None:
        try:
            import serve.loop as loop  # noqa: PLC0415 - intentional lazy import
        except Exception as error:
            raise FdbError(
                "The Repartee serving loop is unavailable, so --engine repartee cannot run. "
                "Run with --engine wav against precomputed output.wav files, or make "
                "`serve.loop` importable. Original error: " + repr(error)
            ) from error
        self._loop = loop
        if not hasattr(loop, "run_session") or not hasattr(loop, "Lfm3DuplexEngine"):
            raise FdbError(
                "serve.loop does not expose the expected file-source surface "
                "(`run_session`, `Lfm3DuplexEngine`). Cannot run --engine repartee."
            )
        ctor_kwargs: dict[str, Any] = {}
        try:
            import inspect

            parameters = inspect.signature(loop.Lfm3DuplexEngine).parameters
        except (TypeError, ValueError):
            parameters = {}
        if "int4" in parameters or not parameters:
            ctor_kwargs["int4"] = int4
        if device is not None and "device" in parameters:
            ctor_kwargs["device"] = device
        self.engine = loop.Lfm3DuplexEngine(model_id, **ctor_kwargs)
        self.model_id = model_id

    def run(self, user_wav: Path, output_wav: Path) -> Path:
        loop = self._loop
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        duration = audio_duration(user_wav)
        ticks = max(1, int(math.ceil(duration * _FRAME_RATE)))
        if hasattr(self.engine, "reset"):
            try:
                self.engine.reset()
            except Exception:
                # A stateless engine may not implement a meaningful reset; a
                # fresh session input is still valid.
                pass
        source = loop.WavFileSource(str(user_wav))
        sink = loop.WavFileSink(str(output_wav))
        if hasattr(source, "__enter__"):
            with source as entered_source:  # type: ignore[union-attr]
                with sink as entered_sink:  # type: ignore[union-attr]
                    loop.run_session(
                        entered_source or source,
                        entered_sink or sink,
                        self.engine,
                        ticks=ticks,
                        metrics_path=None,
                        realtime=False,
                    )
        else:  # pragma: no cover - defensive for older serving builds
            loop.run_session(
                source,
                sink,
                self.engine,
                ticks=ticks,
                metrics_path=None,
                realtime=False,
            )
        if not output_wav.exists():
            raise FdbError(
                f"serve.loop completed but wrote no output audio to {output_wav}"
            )
        return output_wav


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@dataclass
class CategoryAccumulator:
    session_ids: list[str] = field(default_factory=list)
    pause_flags: list[bool] = field(default_factory=list)
    interruption_flags: list[bool] = field(default_factory=list)
    turn_flags: list[bool] = field(default_factory=list)
    ttfa_values: list[float | None] = field(default_factory=list)
    backchannel_flags: list[bool] = field(default_factory=list)
    ignore_flags: list[bool] = field(default_factory=list)
    stop_values: list[float] = field(default_factory=list)
    backchannel_tor_flags: list[bool] = field(default_factory=list)
    backchannel_frequencies: list[float] = field(default_factory=list)
    jsd_values: list[float | None] = field(default_factory=list)
    agent_timelines: list[list[tuple[float, float]]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _backchannel_sample(
    agent_speech: Sequence[tuple[float, float]],
    word_counts: Sequence[int] | None,
    *,
    duration_s: float,
    ground_truth: Sequence[float] | None,
    time_threshold_s: float = 3.0,
) -> tuple[bool, float, float | None, str | None]:
    """FDB v1 ICC backchannel sample metrics.

    Returns ``(takeover, frequency, jsd, jsd_reason)``.  This mirrors the
    released ``eval_backchannel.py``: any speech segment longer than three
    seconds, or wordier than three ASR chunks, is a takeover; otherwise short
    segments are candidate backchannels.  JSD is the square-root
    Jensen-Shannon distance (SciPy's ``jensenshannon``) against the ICC
    distribution, or ``1.0`` when no backchannel was produced, matching the
    released code.  Without the ICC ground truth the value is ``None``.
    """
    takeover = False
    backchannels: list[tuple[float, float]] = []
    for index, (start, end) in enumerate(normalize_intervals(agent_speech)):
        segment_duration = end - start
        n_words = None if word_counts is None else int(word_counts[index])
        if segment_duration > time_threshold_s + 1e-9:
            takeover = True
            continue
        if n_words is not None and n_words > 3:
            takeover = True
            continue
        if segment_duration < 1.0 - 1e-9 and (n_words is None or n_words <= 2):
            backchannels.append((start, end))
        else:
            takeover = True

    frequency = (len(backchannels) / duration_s) if duration_s > 0 else 0.0
    if not backchannels:
        return takeover, frequency, 1.0, None
    if ground_truth is None:
        return takeover, frequency, None, "icc_gt_distribution.json not provided"

    bins = max(1, int(duration_s / 0.2) + 1)
    histogram = [0.0] * bins
    for start, end in backchannels:
        first = min(bins - 1, int(start / 0.2))
        last = min(bins - 1, int(end / 0.2))
        for index in range(first, last + 1):
            histogram[index] += 1.0
    total = sum(histogram)
    prediction = [value / total for value in histogram] if total else [1.0 / bins] * bins

    # Linear interpolation of the ground truth onto the prediction grid,
    # matching the released scipy.interpolate.interp1d call.
    gt = [float(value) for value in ground_truth]
    if not gt:
        return takeover, frequency, None, "empty ICC ground-truth distribution"
    resized: list[float] = []
    for index in range(bins):
        position = index / max(1, bins - 1) * (len(gt) - 1)
        low = int(math.floor(position))
        high = min(len(gt) - 1, low + 1)
        weight = position - low
        resized.append(gt[low] * (1.0 - weight) + gt[high] * weight)
    gt_total = sum(resized)
    if gt_total <= 0:
        return takeover, frequency, None, "non-positive ICC ground-truth mass"
    ground = [value / gt_total for value in resized]
    jsd = 0.0
    for p, q in zip(prediction, ground):
        m = 0.5 * (p + q)
        if p > 0:
            jsd += 0.5 * p * math.log(p / m)
        if q > 0:
            jsd += 0.5 * q * math.log(q / m)
    return takeover, frequency, math.sqrt(max(0.0, jsd)), None


def _aggregate_counts_for_merged(
    raw: Sequence[tuple[float, float]],
    merged: Sequence[tuple[float, float]],
    counts: Sequence[int],
) -> list[int]:
    """Sum per-raw-segment word counts into the merged activity segments."""
    aggregated = [0] * len(merged)
    for (start, _), count in zip(raw, counts):
        for index, (m_start, m_end) in enumerate(merged):
            if m_start - 1e-9 <= start <= m_end + 1e-9:
                aggregated[index] += int(count)
                break
    return aggregated


def _score_session(
    session: FdbSession,
    accumulator: CategoryAccumulator,
    *,
    vad: str,
) -> None:
    # Raw segments preserve self-overlap (the Gate 7 signal); the merged view
    # is what FDB's activity-based metrics use.
    raw_agent = normalize_intervals(session.agent_speech, merge=False)
    if not raw_agent and session.output_wav is not None:
        raw_agent = normalize_intervals(
            read_speech_intervals(session.output_wav, vad=vad), merge=False
        )
    agent = normalize_intervals(raw_agent)
    word_counts = session.agent_word_counts
    if word_counts is not None and len(word_counts) != len(agent):
        if len(word_counts) == len(raw_agent):
            word_counts = _aggregate_counts_for_merged(raw_agent, agent, word_counts)
        else:
            raise FdbError(
                f"{session.session_id}: {len(word_counts)} word counts for "
                f"{len(raw_agent)} raw / {len(agent)} merged agent intervals"
            )
    accumulator.session_ids.append(session.session_id)
    accumulator.agent_timelines.append(raw_agent)

    if session.category == "pause_handling":
        accumulator.pause_flags.extend(
            pause_takeovers(session.pauses, agent, word_counts=word_counts)
        )
    elif session.category == "smooth_turn_taking":
        turn_flags = turn_taking_success_flags(
            session.turn_ends, agent, word_counts=word_counts
        )
        accumulator.turn_flags.extend(turn_flags)
        for turn_end, took_turn in zip(session.turn_ends, turn_flags):
            # FDB averages response latency only over TO=1 samples; failures
            # are counted as missing, not as zero-latency successes.
            accumulator.ttfa_values.append(
                time_to_first_audio(turn_end, agent, window_s=10.0)
                if took_turn
                else None
            )
    elif session.category == "user_interruption":
        interrupt_flags = interruption_success_flags(
            session.interruptions, agent, word_counts=word_counts
        )
        accumulator.interruption_flags.extend(interrupt_flags)
        for interruption, took_turn in zip(session.interruptions, interrupt_flags):
            accumulator.ttfa_values.append(
                time_to_first_audio(interruption[1], agent, window_s=10.0)
                if took_turn
                else None
            )
        for interruption in session.interruptions:
            accumulator.stop_values.extend(stop_latencies([interruption], agent))
    elif session.category == "user_backchannel":
        accumulator.backchannel_flags.extend(
            backchannel_resume_flags(session.backchannels, agent)
        )
        for cue in session.backchannels:
            accumulator.stop_values.extend(stop_latencies([cue], agent))
    elif session.category in {"talking_to_other", "background_speech"}:
        accumulator.ignore_flags.extend(
            backchannel_resume_flags(session.backchannels, agent)
        )
        for overlap in session.backchannels:
            accumulator.stop_values.extend(stop_latencies([overlap], agent))
    elif session.category == "backchannel":
        if session.duration_s is not None:
            duration = float(session.duration_s)
        elif session.output_wav is not None:
            duration = audio_duration(session.output_wav)
        else:
            duration = speech_span(agent)
        takeover, frequency, jsd, reason = _backchannel_sample(
            agent,
            word_counts,
            duration_s=duration,
            ground_truth=session.ground_truth_distribution,
        )
        accumulator.backchannel_tor_flags.append(takeover)
        accumulator.backchannel_frequencies.append(frequency)
        accumulator.jsd_values.append(jsd)
        if reason:
            session.notes["jsd_reason"] = reason
    else:
        raise FdbError(f"{session.session_id}: unknown category {session.category!r}")


def _category_report(accumulator: CategoryAccumulator) -> dict[str, Any]:
    return {
        "sessions": len(accumulator.session_ids),
        "session_ids": list(accumulator.session_ids),
        "pause_tor": aggregate_rates(accumulator.pause_flags, name="pause_tor"),
        "interruption_success": aggregate_rates(
            accumulator.interruption_flags, name="interruption_success"
        ),
        "turn_taking_success": aggregate_rates(
            accumulator.turn_flags, name="turn_taking_success"
        ),
        "ttfa": aggregate_latencies(accumulator.ttfa_values),
        "backchannel_resume": aggregate_rates(
            accumulator.backchannel_flags, name="backchannel_resume"
        ),
        "ignore_resume": aggregate_rates(
            accumulator.ignore_flags, name="ignore_resume"
        ),
        "stop_latency": aggregate_latencies(accumulator.stop_values),
        "backchannel_tor": aggregate_rates(
            accumulator.backchannel_tor_flags, name="backchannel_tor"
        ),
        "backchannel_frequency": (
            sum(accumulator.backchannel_frequencies) / len(accumulator.backchannel_frequencies)
            if accumulator.backchannel_frequencies
            else None
        ),
        "jsd": aggregate_latencies(accumulator.jsd_values),
        "self_interruption": {
            "rate": (
                sum(self_interruption_rate(timeline) for timeline in accumulator.agent_timelines)
                / len(accumulator.agent_timelines)
                if accumulator.agent_timelines
                else None
            ),
            "events": sum(
                len(self_interruption_events(timeline))
                for timeline in accumulator.agent_timelines
            ),
        },
        "errors": list(accumulator.errors),
    }


def _pooled_report(accumulators: Iterable[CategoryAccumulator]) -> dict[str, Any]:
    pause_flags: list[bool] = []
    interruption_flags: list[bool] = []
    turn_flags: list[bool] = []
    ttfa_values: list[float | None] = []
    backchannel_flags: list[bool] = []
    ignore_flags: list[bool] = []
    stop_values: list[float] = []
    agent_timelines: list[list[tuple[float, float]]] = []
    for accumulator in accumulators:
        pause_flags.extend(accumulator.pause_flags)
        interruption_flags.extend(accumulator.interruption_flags)
        turn_flags.extend(accumulator.turn_flags)
        ttfa_values.extend(accumulator.ttfa_values)
        backchannel_flags.extend(accumulator.backchannel_flags)
        ignore_flags.extend(accumulator.ignore_flags)
        stop_values.extend(accumulator.stop_values)
        agent_timelines.extend(accumulator.agent_timelines)
    self_events = sum(len(self_interruption_events(t)) for t in agent_timelines)
    self_rate = (
        sum(self_interruption_rate(t) for t in agent_timelines) / len(agent_timelines)
        if agent_timelines
        else None
    )
    return {
        "pause_tor": (
            sum(pause_flags) / len(pause_flags) if pause_flags else None
        ),
        "interruption_success": (
            sum(interruption_flags) / len(interruption_flags)
            if interruption_flags
            else None
        ),
        "turn_taking_success": (
            sum(turn_flags) / len(turn_flags) if turn_flags else None
        ),
        "ttfa": aggregate_latencies(ttfa_values),
        "backchannel_resume": (
            sum(backchannel_flags) / len(backchannel_flags)
            if backchannel_flags
            else None
        ),
        "ignore_resume": (
            sum(ignore_flags) / len(ignore_flags) if ignore_flags else None
        ),
        "stop_latency": aggregate_latencies(stop_values),
        "self_interruption": {
            "rate": self_rate,
            "events": self_events,
            "gate": GATE7_SELF_INTERRUPTION_MAX,
        },
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run(
    *,
    data: Path,
    engine: str,
    out: Path,
    model: str = "LiquidAI/LFM3-Omni-2B",
    int4: bool = False,
    device: str | None = None,
    vad: str = "auto",
    limit: int | None = None,
    offset: int = 0,
    version: str = "auto",
    category: str | None = None,
    download: bool = False,
    allow_demo: bool = False,
    keep_going: bool = False,
    output_audio_dir: Path | None = None,
    baseline_json: Path | None = None,
    icc_gt: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Discover, optionally execute, and score FDB sessions."""
    if baseline_json is not None and not baseline_json.exists():
        raise FdbError(f"baseline result not found: {baseline_json}")
    if download:
        data = download_fdb(data)
    all_sessions = discover_sessions(data)
    layout = verify_layout(all_sessions)

    if data.is_dir():
        resolved_icc = icc_gt or _find_icc_gt(data)
    else:
        resolved_icc = icc_gt
    icc_attached = 0
    if resolved_icc is not None:
        icc_attached = attach_icc_gt(all_sessions, load_icc_gt(resolved_icc))

    demo = _is_demo_path(data)
    if demo and not allow_demo:
        raise FdbError(
            f"{data} looks like the benchmark's example/demo data, not an evaluation "
            "set. Demo clips are never evidence; pass --allow-demo only for a smoke "
            f"test, or download the real data from {FDB_DATA_URL}."
        )
    sessions = list(all_sessions)
    if version != "auto":
        wanted = "v1" if version.lower() in V1_VERSIONS else "v1.5" if version.lower() in V15_VERSIONS else None
        if wanted is None:
            raise FdbError(f"unknown --version {version!r}")
        sessions = [session for session in sessions if session.version == wanted]
    if category:
        sessions = [session for session in sessions if session.category == category]
    sessions = sessions[offset:]
    if limit is not None:
        if limit < 0:
            raise FdbError("--limit must be non-negative")
        sessions = sessions[:limit]
    if not sessions:
        raise FdbError("no FDB sessions matched the requested version/category/offset/limit")

    if dry_run:
        return {
            "dry_run": True,
            "data": str(data),
            "engine": engine,
            "vad": vad,
            "vad_fallback": None,
            "demo": demo,
            "evidence": False,
            "layout": layout,
            "planned_sessions": [session.to_plan() for session in sessions],
            "message": "No numbers were computed (--dry-run).",
        }

    active_vad, vad_fallback = resolve_vad(vad)
    accumulators: dict[str, CategoryAccumulator] = defaultdict(CategoryAccumulator)
    runner: ReparteeEngineRunner | None = None
    if engine == "repartee":
        runner = ReparteeEngineRunner(model, int4=int4, device=device)
    elif engine != "wav":
        raise FdbError(f"unknown engine {engine!r}; choose wav or repartee")

    audio_root = output_audio_dir
    if audio_root is None:
        audio_root = out.parent / "fdb_audio"
    errors: list[str] = []

    for session in sessions:
        accumulator = accumulators[session.category]
        try:
            if engine == "repartee":
                if session.user_wav is None or not session.user_wav.exists():
                    raise FdbError(f"{session.session_id}: no input.wav for --engine repartee")
                if runner is None:  # pragma: no cover - constructed above
                    raise FdbError("repartee runner was not constructed")
                safe_id = "".join(
                    character if character.isalnum() or character in "-._" else "_"
                    for character in session.session_id
                )
                generated = audio_root / safe_id / "output.wav"
                session.output_wav = runner.run(session.user_wav, generated)
                session.notes["derived_output_wav"] = str(session.output_wav)
                session.output_json = None
            else:
                if not session.agent_speech and (
                    session.output_wav is None or not session.output_wav.exists()
                ):
                    raise FdbError(
                        f"{session.session_id}: no output.wav for --engine wav and no "
                        "agent_speech intervals were supplied. Generate model output "
                        "first, or run --engine repartee."
                    )
            if session.agent_word_counts is None and session.output_json is not None:
                chunks = _load_word_chunks(session.output_json)
                if chunks is not None and not session.agent_speech:
                    session.agent_speech = normalize_intervals(
                        read_speech_intervals(session.output_wav, vad=active_vad)  # type: ignore[arg-type]
                    )
                if chunks is not None and session.agent_speech:
                    session.agent_word_counts = word_counts_for_segments(
                        normalize_intervals(session.agent_speech), chunks
                    )
            _score_session(session, accumulator, vad=active_vad)
        except Exception as error:
            message = str(error)
            if not message.startswith(f"{session.session_id}:"):
                message = f"{session.session_id}: {message}"
            accumulator.errors.append(message)
            errors.append(message)
            if not keep_going:
                raise FdbError(message) from error

    categories = {
        name: _category_report(accumulator)
        for name, accumulator in sorted(accumulators.items())
    }
    overall = _pooled_report(accumulators.values())
    gate5 = [
        check.to_dict()
        for check in gate5_checks(
            pause_tor=overall["pause_tor"],
            interruption_success=overall["interruption_success"],
            ttfa_s=overall["ttfa"]["mean_s"],
            backchannel_resume=overall["backchannel_resume"],
        )
    ]
    result: dict[str, Any] = {
        "status": "ok" if not errors else "partial",
        "evidence": bool(not demo),
        "demo": demo,
        "benchmark": "Full-Duplex-Bench v1.0/v1.5",
        "data": str(data),
        "engine": engine,
        "model": model if engine == "repartee" else None,
        "vad": active_vad,
        "vad_fallback": vad_fallback,
        "layout": layout,
        "icc_gt": str(resolved_icc) if resolved_icc is not None else None,
        "icc_sessions_matched": icc_attached,
        "sessions": len(sessions),
        "categories": categories,
        "overall": overall,
        "gate5": gate5,
        "errors": errors,
        "limitations": [
            "Turn-taking metrics are the interval implementations in eval.metrics, "
            "following the FDB papers (arXiv:2503.04721, arXiv:2507.23159) and the "
            "released v1/v1.5 scripts; this is not the upstream harness.",
            "FDB's GPT-4o interruption-relevance score and its ICC JSD are not "
            "reproduced without the corresponding external annotations/API.",
            "VAD is Silero when available; energy-VAD numbers are approximations "
            "and are labelled by the 'vad' field.",
            "Without an output.json word-level transcript, takeover/backchannel "
            "classification falls back to segment duration and can miss wordy "
            "sub-second speech.",
        ],
    }
    if demo:
        result["limitations"].insert(
            0, "This run used demo/example clips; it is not benchmark evidence."
        )
    if baseline_json is not None:
        result["baseline_comparison"] = compare_baseline(result, baseline_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def compare_baseline(current: Mapping[str, Any], baseline_path: Path) -> dict[str, Any]:
    """Compare overall rates/latencies with a previous result JSON.

    Missing metrics produce ``None`` deltas, never zeros.
    """
    if not baseline_path.exists():
        raise FdbError(f"baseline result not found: {baseline_path}")
    baseline = _read_json(baseline_path)
    if not isinstance(baseline, Mapping) or "overall" not in baseline:
        raise FdbError(f"{baseline_path}: not an eval.fdb result (missing 'overall')")
    current_overall = current.get("overall", {})
    base_overall = baseline.get("overall", {})
    comparison: dict[str, Any] = {"baseline": str(baseline_path), "deltas": {}}
    for key in ("pause_tor", "interruption_success", "backchannel_resume", "turn_taking_success"):
        now = current_overall.get(key)
        before = base_overall.get(key)
        comparison["deltas"][key] = (
            (float(now) - float(before)) if now is not None and before is not None else None
        )
    for key in ("ttfa", "stop_latency"):
        now = (current_overall.get(key) or {}).get("mean_s")
        before = (base_overall.get(key) or {}).get("mean_s")
        comparison["deltas"][key + "_mean_s"] = (
            (float(now) - float(before)) if now is not None and before is not None else None
        )
    return comparison


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m eval.fdb",
        description=(
            "Score Full-Duplex-Bench v1.0/v1.5 pause handling, interruption "
            "management, backchannel and turn-taking metrics."
        ),
        epilog=(
            f"Data: {FDB_DATA_URL}\n"
            f"Upstream: {FDB_REPO_URL}\n"
            "Demo clips are refused as evidence unless --allow-demo is given."
        ),
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="FDB dataset root (v1_0/ + v1_5/) or a .jsonl session file.",
    )
    parser.add_argument(
        "--engine",
        choices=("wav", "repartee"),
        default="wav",
        help="wav: score existing output.wav; repartee: run the real duplex loop.",
    )
    parser.add_argument("--out", type=Path, default=Path("artifacts/fdb_results.json"))
    parser.add_argument("--model", default="LiquidAI/LFM3-Omni-2B")
    parser.add_argument("--int4", action="store_true", help="Load the repartee engine in int4.")
    parser.add_argument("--device", default=None, help="Torch device for --engine repartee.")
    parser.add_argument(
        "--vad",
        choices=("auto", "silero", "energy"),
        default="auto",
        help="Speech detection for output audio (energy is a labelled approximation).",
    )
    parser.add_argument("--version", default="auto", help="auto, v1 or v1.5")
    parser.add_argument("--category", default=None, help="Restrict to one category.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Try to download the public FDB folder with gdown before scoring.",
    )
    parser.add_argument(
        "--allow-demo",
        action="store_true",
        help="Allow the repository example_data clips (marked evidence=false).",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Record per-session errors and continue instead of failing fast.",
    )
    parser.add_argument(
        "--audio-out",
        type=Path,
        default=None,
        help="Directory for generated output.wav files (default: alongside --out).",
    )
    parser.add_argument(
        "--baseline-json",
        type=Path,
        default=None,
        help="Previous results.json to compute overall deltas against.",
    )
    parser.add_argument(
        "--icc-gt",
        type=Path,
        default=None,
        help="ICC ground-truth JSON for the v1 backchannel JSD (auto-detected if present).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned sessions and inputs without computing numbers.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run(
            data=args.data,
            engine=args.engine,
            out=args.out,
            model=args.model,
            int4=args.int4,
            device=args.device,
            vad=args.vad,
            limit=args.limit,
            offset=args.offset,
            version=args.version,
            category=args.category,
            download=args.download,
            allow_demo=args.allow_demo,
            keep_going=args.keep_going,
            output_audio_dir=args.audio_out,
            baseline_json=args.baseline_json,
            icc_gt=args.icc_gt,
            dry_run=args.dry_run,
        )
    except FdbError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if result.get("dry_run"):
        print(json.dumps(result, indent=2))
    else:
        print(f"wrote {args.out} ({result['status']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
