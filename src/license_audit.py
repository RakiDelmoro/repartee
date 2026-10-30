"""Commercial-use licence registry and manifest audit for Repartee.

The registry below lists every dataset, TTS backend and base model used (or
explicitly rejected) by the project, with the licence, canonical URL, a
``commercial_use`` flag and notes.  Verified 2026-10-29 from the dataset/model
cards and licence files; see the module tests for the acceptance behaviour.

Audit rules
-----------
For every record in the supplied JSONL manifests:

* unknown ``source`` -> violation,
* missing/empty ``license`` -> violation,
* a licence string not accepted for that source -> violation,
* a source whose registry entry is non-commercial -> violation,
* a TTS backend recorded in ``meta["tts_backend"]`` (or selected with
  ``--tts``) that is forbidden/non-commercial/unknown -> violation.

The LFM3-Omni-2B / LiquidCodec licence grants commercial use only below
**$10,000,000** annual revenue.  Pass ``--annual-revenue-usd`` to turn that
into a hard check; without it the report records the threshold as an
unverified warning.  ``--strict`` exits non-zero when there is any violation
(not merely a warning).

XTTS-v2, F5-TTS and Fish weights are marked forbidden: their weights are not
commercially licensed (Coqui Public Model License, CC-BY-NC-4.0 and
CC-BY-NC-SA-4.0 respectively), and the project must never use them.

CLI
---
``python -m src.license_audit --manifests data/manifests --out data/audit
[--strict] [--annual-revenue-usd N] [--tts kokoro]``

Writes ``license_audit.json`` and ``license_audit.md``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LicenseEntry:
    name: str
    kind: str  # "dataset" | "tts" | "model"
    license: str
    url: str
    commercial_use: bool
    notes: str
    aliases: tuple[str, ...] = ()
    manifest_licenses: tuple[str, ...] = ()
    forbidden: bool = False
    revenue_threshold_usd: float | None = None


def _normalise_key(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[\s/]+", "_", value)
    value = re.sub(r"[^a-z0-9_.+-]", "", value)
    return value


def _normalise_license(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value


REGISTRY: tuple[LicenseEntry, ...] = (
    # ---------------------------- datasets --------------------------------
    LicenseEntry(
        name="open_yap_1k",
        kind="dataset",
        license="CC-BY-4.0 (public sample) / Open Yap 1K Data Use Agreement (full)",
        url="https://theagenticdatacompany.com/open-yap-1k",
        commercial_use=True,
        notes=(
            "Public 8.92 h sample is CC-BY-4.0. The 1,000 h corpus is free for "
            "commercial use but request-only under a Data Use Agreement: no "
            "redistribution/sublicensing, no speaker re-identification, no "
            "voice clones identifiable as a corpus speaker."
        ),
        aliases=(
            "open_yap",
            "open_yap_1k",
            "open-yap-1k",
            "open_yap_1k_sample",
            "theagenticdatacompany/open-yap-1k",
        ),
        manifest_licenses=("cc-by-4.0", "commercial", "open yap 1k data use agreement"),
    ),
    LicenseEntry(
        name="otospeech_280h",
        kind="dataset",
        license="CC-BY-4.0",
        url="https://huggingface.co/datasets/otoearth/otoSpeech-full-duplex-280h",
        commercial_use=True,
        notes=(
            "Gated (auto-approval) Hugging Face dataset. Attribution required. "
            "No transcript is published; the build uses energy-VAD turns."
        ),
        aliases=(
            "otospeech",
            "otospeech_280h",
            "otospeech-full-duplex-280h",
            "otoearth/otospeech-full-duplex-280h",
        ),
        manifest_licenses=("cc-by-4.0",),
    ),
    LicenseEntry(
        name="otospeech_turn104h",
        kind="dataset",
        license="oto-speech-license-v1.0",
        url="https://huggingface.co/datasets/otoearth/otoSpeech-full-duplex-turn-104h",
        commercial_use=False,
        notes=(
            "The Hub card tags the repository 'other'; the actual licence is the "
            "oto Speech License v1.0, a non-commercial research licence. "
            "Production/commercial use needs a separate written licence; no "
            "voice-identity uses. The project records it but must not ship a "
            "commercial model trained on it without that licence."
        ),
        aliases=(
            "otospeech_turn104h",
            "otospeech_turn_104h",
            "otospeech-turn-104h",
            "turn104h",
            "turn_104h",
            "otoearth/otospeech-full-duplex-turn-104h",
        ),
        manifest_licenses=("oto-speech-license-v1.0", "other"),
    ),
    LicenseEntry(
        name="candor",
        kind="dataset",
        license="CANDOR research licence",
        url="https://github.com/Eric-Wallace/CANDOR",
        commercial_use=False,
        notes="Research-only conversational corpus; never use commercially without a separate licence.",
        aliases=("candor",),
        manifest_licenses=("candor research licence",),
        forbidden=True,
    ),
    LicenseEntry(
        name="fisher",
        kind="dataset",
        license="LDC User Agreement (paid)",
        url="https://catalog.ldc.upenn.edu/LDC2004S13",
        commercial_use=False,
        notes="Telephone corpus under an LDC agreement; commercial use needs an LDC commercial licence.",
        aliases=("fisher", "fisher_english"),
        manifest_licenses=("ldc user agreement",),
        forbidden=True,
    ),
    # ------------------------------- TTS ----------------------------------
    LicenseEntry(
        name="kokoro",
        kind="tts",
        license="Apache-2.0",
        url="https://huggingface.co/hexgrad/Kokoro-82M",
        commercial_use=True,
        notes="Kokoro-82M weights and code are Apache-2.0.",
        aliases=("kokoro", "kokoro-82m", "hexgrad/kokoro-82m"),
        manifest_licenses=("apache-2.0",),
    ),
    LicenseEntry(
        name="dia",
        kind="tts",
        license="Apache-2.0",
        url="https://huggingface.co/nari-labs/Dia-1.6B-0626",
        commercial_use=True,
        notes="Dia-1.6B code and weights are Apache-2.0.",
        aliases=("dia", "dia-1.6b", "dia_1_6b", "nari-labs/dia-1.6b-0626"),
        manifest_licenses=("apache-2.0",),
    ),
    LicenseEntry(
        name="chatterbox",
        kind="tts",
        license="MIT",
        url="https://huggingface.co/ResembleAI/chatterbox",
        commercial_use=True,
        notes="Chatterbox code and weights are MIT; voice-prompt wavs need their own rights.",
        aliases=("chatterbox", "resembleai/chatterbox"),
        manifest_licenses=("mit",),
    ),
    LicenseEntry(
        name="cosyvoice2",
        kind="tts",
        license="Apache-2.0",
        url="https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B",
        commercial_use=True,
        notes="CosyVoice2 code and weights are Apache-2.0.",
        aliases=("cosyvoice2", "cosyvoice", "cosyvoice2-0.5b", "funaudiollm/cosyvoice2-0.5b"),
        manifest_licenses=("apache-2.0",),
    ),
    LicenseEntry(
        name="xtts_v2",
        kind="tts",
        license="Coqui Public Model License 1.0",
        url="https://huggingface.co/coqui/XTTS-v2",
        commercial_use=False,
        notes="Coqui's model licence is non-commercial; never use these weights in Repartee.",
        aliases=("xtts", "xtts-v2", "xtts_v2", "coqui/xtts-v2"),
        manifest_licenses=("coqui public model license 1.0",),
        forbidden=True,
    ),
    LicenseEntry(
        name="f5_tts",
        kind="tts",
        license="CC-BY-NC-4.0 (model weights)",
        url="https://huggingface.co/SWivid/F5-TTS",
        commercial_use=False,
        notes="Code is MIT but the released weights are CC-BY-NC-4.0; forbidden.",
        aliases=("f5", "f5-tts", "f5_tts", "swivid/f5-tts"),
        manifest_licenses=("cc-by-nc-4.0",),
        forbidden=True,
    ),
    LicenseEntry(
        name="fish_speech",
        kind="tts",
        license="CC-BY-NC-SA-4.0 (model weights)",
        url="https://huggingface.co/fishaudio/fish-speech-1.5",
        commercial_use=False,
        notes="Fish/OpenAudio weights are non-commercial share-alike; forbidden.",
        aliases=("fish", "fish-speech", "fish_speech", "fishaudio/fish-speech-1.5"),
        manifest_licenses=("cc-by-nc-sa-4.0",),
        forbidden=True,
    ),
    LicenseEntry(
        name="synthetic",
        kind="dataset",
        license="synthetic-commercial",
        url="https://github.com/",
        commercial_use=True,
        notes=(
            "Generated text+TTS output. Commercial status depends on the selected "
            "writer model and TTS backend; this entry is valid only when all "
            "components are commercially licensed."
        ),
        aliases=("synthetic", "synth", "generated"),
        manifest_licenses=("synthetic-commercial",),
    ),
    # ------------------------------ models --------------------------------
    LicenseEntry(
        name="lfm3_omni_2b",
        kind="model",
        license="LFM Open License v1.0",
        url="https://huggingface.co/LiquidAI/LFM3-Omni-2B",
        commercial_use=True,
        notes=(
            "Commercial use is permitted only while the licensee's legal entity "
            "stays below $10,000,000 annual revenue (Section 5(a)); at or above "
            "the Threshold the licence does not grant commercial rights."
        ),
        aliases=(
            "lfm3",
            "lfm3_omni",
            "lfm3_omni_2b",
            "lfm3-omni-2b",
            "liquidai/lfm3-omni-2b",
        ),
        manifest_licenses=("lfm open license v1.0",),
        revenue_threshold_usd=10_000_000.0,
    ),
    LicenseEntry(
        name="liquid_codec",
        kind="model",
        license="LFM Open License v1.0",
        url="https://huggingface.co/LiquidAI/LFM3-Omni-2B",
        commercial_use=True,
        notes="LiquidCodec ships inside LFM3-Omni-2B and is covered by the same $10M revenue threshold.",
        aliases=("liquid_codec", "liquidcodec"),
        manifest_licenses=("lfm open license v1.0",),
        revenue_threshold_usd=10_000_000.0,
    ),
)

_LOOKUP: dict[str, LicenseEntry] = {}
for _entry in REGISTRY:
    for _alias in (_entry.name, *_entry.aliases):
        _LOOKUP[_normalise_key(_alias)] = _entry


def find_entry(name: str | None) -> LicenseEntry | None:
    """Look up a registry entry by canonical name or alias."""
    if not name:
        return None
    return _LOOKUP.get(_normalise_key(str(name)))


def registry_as_dicts() -> list[dict[str, Any]]:
    return [
        {
            "name": entry.name,
            "kind": entry.kind,
            "license": entry.license,
            "url": entry.url,
            "commercial_use": entry.commercial_use,
            "notes": entry.notes,
            "forbidden": entry.forbidden,
            "revenue_threshold_usd": entry.revenue_threshold_usd,
            "manifest_licenses": list(entry.manifest_licenses),
        }
        for entry in REGISTRY
    ]


# ---------------------------------------------------------------------------
# Manifest loading and audit
# ---------------------------------------------------------------------------


@dataclass
class Usage:
    source: str
    license: str
    conversations: int = 0
    hours: float = 0.0


@dataclass
class Finding:
    code: str
    message: str
    severity: str = "violation"
    source: str = ""
    record_id: str = ""
    path: str = ""


@dataclass
class AuditResult:
    files: list[str] = field(default_factory=list)
    conversations: int = 0
    hours: float = 0.0
    usage: dict[str, Usage] = field(default_factory=dict)
    tts_usage: dict[str, dict[str, Any]] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    selected_tts: list[str] = field(default_factory=list)
    annual_revenue_usd: float | None = None

    @property
    def violations(self) -> list[Finding]:
        return [finding for finding in self.findings if finding.severity == "violation"]

    @property
    def warnings(self) -> list[Finding]:
        return [finding for finding in self.findings if finding.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.violations

    def add(
        self,
        code: str,
        message: str,
        *,
        severity: str = "violation",
        source: str = "",
        record_id: str = "",
        path: str = "",
    ) -> None:
        self.findings.append(
            Finding(
                code=code,
                message=message,
                severity=severity,
                source=source,
                record_id=record_id,
                path=path,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        threshold_checks = []
        for entry in REGISTRY:
            if entry.revenue_threshold_usd is None:
                continue
            threshold_checks.append(
                {
                    "entry": entry.name,
                    "threshold_usd": entry.revenue_threshold_usd,
                    "annual_revenue_usd": self.annual_revenue_usd,
                    "status": _threshold_status(entry, self.annual_revenue_usd),
                }
            )
        return {
            "ok": self.ok,
            "strict": not self.ok,
            "manifests": {
                "files": self.files,
                "conversations": self.conversations,
                "hours": round(self.hours, 6),
            },
            "usage": {
                name: {
                    "conversations": usage.conversations,
                    "hours": round(usage.hours, 6),
                    "license": usage.license,
                    "registry_entry": find_entry(name).name if find_entry(name) else None,
                    "commercial_use": find_entry(name).commercial_use if find_entry(name) else None,
                }
                for name, usage in sorted(self.usage.items())
            },
            "tts_usage": {
                name: {**info, "hours": round(float(info.get("hours", 0.0)), 6)}
                for name, info in sorted(self.tts_usage.items())
            },
            "selected_tts": sorted(self.selected_tts),
            "annual_revenue_usd": self.annual_revenue_usd,
            "threshold_checks": threshold_checks,
            "violations": [finding.__dict__ for finding in self.violations],
            "warnings": [finding.__dict__ for finding in self.warnings],
            "registry": registry_as_dicts(),
        }


def _threshold_status(entry: LicenseEntry, revenue: float | None) -> str:
    if entry.revenue_threshold_usd is None:
        return "not_applicable"
    if revenue is None:
        return "unverified"
    if revenue >= entry.revenue_threshold_usd:
        return "exceeded"
    return "within_threshold"


def _manifest_paths(paths: Iterable[str | Path]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.glob("*.jsonl")))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"manifest path does not exist: {path}")
    return sorted({file.resolve() for file in files})


def _duration(record: dict[str, Any]) -> float:
    value = record.get("duration_s", 0.0)
    if not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def audit(
    manifests: Iterable[str | Path],
    *,
    annual_revenue_usd: float | None = None,
    selected_tts: Sequence[str] = (),
    strict: bool = False,
) -> AuditResult:
    """Audit every JSONL manifest under ``manifests``.

    Duplicate record ids across per-source and combined manifests are counted
    once (conflicting duplicates are a violation).  Malformed JSON is a
    violation and is reported with the file and line, never silently skipped.
    """
    result = AuditResult(annual_revenue_usd=annual_revenue_usd)
    manifest_files = _manifest_paths(manifests)
    result.files = [str(path) for path in manifest_files]
    seen: dict[str, dict[str, Any]] = {}

    for path in manifest_files:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    result.add(
                        "malformed_manifest",
                        f"invalid JSON: {exc}",
                        path=str(path),
                        record_id=f"line:{line_number}",
                    )
                    continue
                if not isinstance(record, dict):
                    result.add(
                        "malformed_manifest",
                        "record is not a JSON object",
                        path=str(path),
                        record_id=f"line:{line_number}",
                    )
                    continue
                record_id = str(record.get("id", "")) or f"line:{line_number}"
                source = str(record.get("source", "") or "")
                license_name = str(record.get("license", "") or "")
                if record_id in seen:
                    previous = seen[record_id]
                    if (
                        str(previous.get("source", "")) != source
                        or str(previous.get("license", "")) != license_name
                    ):
                        result.add(
                            "duplicate_conflict",
                            "same id appears with a different source/licence",
                            source=source,
                            record_id=record_id,
                            path=str(path),
                        )
                    continue
                seen[record_id] = record
                _audit_record(result, record, source, license_name, str(path), record_id)

    # TTS selection is checked even before any synthetic manifest exists.
    for name in selected_tts:
        result.selected_tts.append(name)
        _audit_tts(result, name, hours=0.0, record_id="", path="<selected>", required=True)

    # Model revenue threshold.
    for entry in REGISTRY:
        if entry.revenue_threshold_usd is None:
            continue
        status = _threshold_status(entry, annual_revenue_usd)
        if status == "exceeded":
            result.add(
                "revenue_threshold_exceeded",
                f"{entry.name}: annual revenue ${annual_revenue_usd:,.0f} is at or above the "
                f"${entry.revenue_threshold_usd:,.0f} licence threshold; commercial rights are not granted",
                source=entry.name,
            )
        elif status == "unverified":
            result.add(
                "revenue_threshold_unverified",
                f"{entry.name}: commercial use is conditional on annual revenue below "
                f"${entry.revenue_threshold_usd:,.0f}; pass --annual-revenue-usd to verify",
                severity="warning",
                source=entry.name,
            )

    # A non-commercial dataset in the mix makes a commercially deployed model unsafe.
    for name, usage in result.usage.items():
        entry = find_entry(name)
        if entry is not None and not entry.commercial_use:
            result.add(
                "non_commercial_source",
                f"{entry.name}: {usage.hours:.2f} h in the manifests is non-commercial "
                f"({entry.license}); do not train a commercial model on it without a separate licence",
                source=name,
            )
    return result


def _audit_record(
    result: AuditResult,
    record: dict[str, Any],
    source: str,
    license_name: str,
    path: str,
    record_id: str,
) -> None:
    result.conversations += 1
    hours = _duration(record) / 3600.0
    result.hours += hours

    if not source:
        result.add("missing_source", "record has no source", record_id=record_id, path=path)
    entry = find_entry(source)
    if entry is None:
        result.add(
            "unknown_source",
            f"source {source!r} is not in the licence registry",
            source=source,
            record_id=record_id,
            path=path,
        )
    else:
        usage = result.usage.setdefault(entry.name, Usage(source=entry.name, license=license_name))
        usage.conversations += 1
        usage.hours += hours
        if entry.forbidden:
            result.add(
                "forbidden_source",
                f"source {entry.name!r} is explicitly forbidden for this project",
                source=source,
                record_id=record_id,
                path=path,
            )
        if not entry.license:
            result.add(
                "missing_license_registry",
                f"registry entry {entry.name!r} has no licence",
                source=source,
                record_id=record_id,
                path=path,
            )
        # Non-commercial usage is reported once per source after aggregation,
        # so a 1,000-record source does not emit 1,000 identical findings.

    if not license_name:
        result.add(
            "missing_license",
            "record has no licence string",
            source=source,
            record_id=record_id,
            path=path,
        )
    elif entry is not None:
        accepted = {_normalise_license(item) for item in (entry.license, *entry.manifest_licenses)}
        if _normalise_license(license_name) not in accepted:
            result.add(
                "license_mismatch",
                f"manifest licence {license_name!r} is not accepted for {entry.name!r} "
                f"({entry.license})",
                source=source,
                record_id=record_id,
                path=path,
            )

    meta = record.get("meta")
    if isinstance(meta, dict) and meta.get("tts_backend"):
        _audit_tts(
            result,
            str(meta["tts_backend"]),
            hours=hours,
            record_id=record_id,
            path=path,
            required=False,
        )


def _audit_tts(
    result: AuditResult,
    name: str,
    *,
    hours: float,
    record_id: str,
    path: str,
    required: bool,
) -> None:
    entry = find_entry(name)
    key = entry.name if entry is not None else name
    bucket = result.tts_usage.setdefault(
        key,
        {
            "hours": 0.0,
            "commercial_use": entry.commercial_use if entry else None,
            "license": entry.license if entry else None,
        },
    )
    bucket["hours"] = float(bucket["hours"]) + hours
    if entry is None:
        result.add(
            "unknown_tts",
            f"TTS backend {name!r} is not in the licence registry",
            source=name,
            record_id=record_id,
            path=path,
        )
    elif entry.forbidden:
        result.add(
            "forbidden_tts",
            f"TTS backend {entry.name!r} is forbidden ({entry.license})",
            source=name,
            record_id=record_id,
            path=path,
        )
    elif not entry.commercial_use:
        result.add(
            "non_commercial_tts",
            f"TTS backend {entry.name!r} is non-commercial ({entry.license})",
            source=name,
            record_id=record_id,
            path=path,
        )
    if required and entry is not None and entry.commercial_use and not entry.forbidden:
        result.tts_usage[key]["selected"] = True


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def render_markdown(result: AuditResult) -> str:
    status = "PASS" if result.ok else "FAIL"
    lines = [
        "# Repartee licence audit",
        "",
        f"**Status: {status}**  ",
        f"Manifests: {len(result.files)} file(s), {result.conversations} conversation(s), "
        f"{result.hours:.2f} h.  ",
        f"Violations: {len(result.violations)}; warnings: {len(result.warnings)}.",
        "",
        "## Manifest usage",
        "",
        "| source | conversations | hours | licence | commercial |",
        "|---|---:|---:|---|---|",
    ]
    for name, usage in sorted(result.usage.items()):
        entry = find_entry(name)
        commercial = "yes" if entry and entry.commercial_use else "**no**"
        lines.append(
            f"| `{name}` | {usage.conversations} | {usage.hours:.2f} | "
            f"{usage.license or '—'} | {commercial} |"
        )
    if not result.usage:
        lines.append("| — | 0 | 0.00 | — | — |")

    lines += ["", "## TTS usage", "", "| backend | hours | licence | commercial |", "|---|---:|---|---|"]
    for name, info in sorted(result.tts_usage.items()):
        commercial = "yes" if info.get("commercial_use") else "**no**"
        lines.append(
            f"| `{name}` | {float(info.get('hours', 0.0)):.2f} | "
            f"{info.get('license') or '—'} | {commercial} |"
        )
    if not result.tts_usage:
        lines.append("| — | 0.00 | — | — |")

    lines += ["", "## Model licence thresholds", ""]
    for entry in REGISTRY:
        if entry.revenue_threshold_usd is None:
            continue
        status_value = _threshold_status(entry, result.annual_revenue_usd)
        threshold = f"${entry.revenue_threshold_usd:,.0f}"
        lines.append(
            f"- `{entry.name}`: commercial use only below **{threshold}** annual revenue "
            f"(status: {status_value})."
        )

    lines += ["", "## Violations", ""]
    if result.violations:
        for finding in result.violations:
            location = f" ({finding.source}" + (f"/{finding.record_id}" if finding.record_id else "") + ")"
            lines.append(f"- **{finding.code}**{location}: {finding.message}")
    else:
        lines.append("None.")

    lines += ["", "## Warnings", ""]
    if result.warnings:
        for finding in result.warnings:
            lines.append(f"- **{finding.code}**: {finding.message}")
    else:
        lines.append("None.")

    lines += ["", "## Forbidden / non-commercial components", ""]
    flagged = [
        entry
        for entry in REGISTRY
        if entry.forbidden or not entry.commercial_use
    ]
    for entry in flagged:
        lines.append(
            f"- `{entry.name}` ({entry.kind}): {entry.license} — {entry.notes} "
            f"<{entry.url}>"
        )

    lines += ["", "## Registry", ""]
    for kind, title in (("dataset", "Datasets"), ("tts", "TTS backends"), ("model", "Base models")):
        lines += [f"### {title}", "", "| name | licence | commercial | url |", "|---|---|---|---|"]
        for entry in REGISTRY:
            if entry.kind != kind:
                continue
            commercial = "yes" if entry.commercial_use else "**no**"
            if entry.forbidden:
                commercial = "**forbidden**"
            lines.append(f"| `{entry.name}` | {entry.license} | {commercial} | <{entry.url}> |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def write_reports(result: AuditResult, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "license_audit.json"
    markdown_path = out_dir / "license_audit.md"
    json_path.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(render_markdown(result), encoding="utf-8")
    return json_path, markdown_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--manifests",
        nargs="+",
        default=["data/manifests"],
        help="manifest JSONL file(s) or directory(ies)",
    )
    parser.add_argument("--out", type=Path, default=Path("data/audit"))
    parser.add_argument("--strict", action="store_true", help="exit non-zero on violations")
    parser.add_argument("--annual-revenue-usd", type=float, default=None)
    parser.add_argument("--tts", action="append", default=[], help="TTS backend to check")
    args = parser.parse_args(argv)
    if args.annual_revenue_usd is not None and args.annual_revenue_usd < 0:
        parser.error("--annual-revenue-usd must be >= 0")
    try:
        result = audit(
            args.manifests,
            annual_revenue_usd=args.annual_revenue_usd,
            selected_tts=args.tts,
            strict=args.strict,
        )
        json_path, markdown_path = write_reports(result, args.out)
    except FileNotFoundError as exc:
        print(f"audit failed: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"audit failed (I/O): {exc}", file=sys.stderr)
        return 2
    print(
        f"license audit: {'PASS' if result.ok else 'FAIL'} "
        f"({result.conversations} conversations, {result.hours:.2f} h, "
        f"{len(result.violations)} violation(s), {len(result.warnings)} warning(s))"
    )
    print(f"  {json_path}")
    print(f"  {markdown_path}")
    for finding in result.violations:
        print(f"  VIOLATION {finding.code}: {finding.message}", file=sys.stderr)
    if args.strict and not result.ok:
        return 1
    return 0


__all__ = [
    "LicenseEntry",
    "REGISTRY",
    "find_entry",
    "registry_as_dicts",
    "Usage",
    "Finding",
    "AuditResult",
    "audit",
    "render_markdown",
    "write_reports",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
