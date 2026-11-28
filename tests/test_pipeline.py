"""Pipeline: done markers, idempotence, summary fallback, and path parsing."""

from __future__ import annotations

from pathlib import Path

from app import knowledge_base, pipeline, stt


class FakeTranscriber:
    def __init__(self, text="We discussed the visa paperwork.", summary="Visa paperwork."):
        self.text = text
        self.summary = summary
        self.calls = 0
        self.summary_calls = 0

    def transcribe(self, path: Path) -> stt.Transcript:
        self.calls += 1
        return stt.Transcript(
            text=self.text,
            language="English",
            duration_s=12.5,
            chunks=(stt.Chunk(0.0, 12.5),),
            model="fake",
            elapsed_s=0.1,
        )

    def summarize(self, text: str) -> str:
        self.summary_calls += 1
        return self.summary


class FailingSummary(FakeTranscriber):
    def summarize(self, text: str) -> str:
        self.summary_calls += 1
        raise RuntimeError("summary model unavailable")


def _kb(tmp_path: Path) -> Path:
    return tmp_path / "knowledge-base"


def _recording(root: Path, name: str = "20261128-120000-rep01.wav") -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not real pcm; the fake transcriber never decodes it")
    return path


def test_parse_receiver_filename():
    device, started = pipeline.parse_filename("20261128-120000-rep01.wav")

    assert device == "rep01"
    assert started == "2026-11-28T12:00:00"


def test_parse_foreign_filename_falls_back():
    device, started = pipeline.parse_filename("conv_d4005da6db98-60s.wav")

    assert device == "conv_d4005da6db98-60s"
    assert started is None


def test_process_writes_one_markdown_entry(tmp_path):
    root = _kb(tmp_path)
    recordings = tmp_path / "recordings"
    path = _recording(recordings)
    pipeline.done_path(path).write_text("done\n")

    processed = pipeline.process_ready(root, FakeTranscriber(), recordings)

    assert processed == 1
    stored = knowledge_base.find(root, path.stem)
    assert stored == root / "2026-11-28" / f"{path.stem}.md"
    entry = knowledge_base.get(root, path.stem)
    assert entry["device"] == "rep01"
    assert entry["started_at"] == "2026-11-28T12:00:00"
    assert entry["transcript"] == "We discussed the visa paperwork."
    assert entry["summary"] == "Visa paperwork."
    assert entry["status"] == "summarized"
    assert entry["audio"] == str(path.resolve())


def test_worker_requires_done_marker(tmp_path):
    root = _kb(tmp_path)
    recordings = tmp_path / "recordings"
    _recording(recordings)
    fake = FakeTranscriber()

    assert pipeline.process_ready(root, fake, recordings) == 0
    assert fake.calls == 0


def test_worker_skips_error_marker(tmp_path):
    root = _kb(tmp_path)
    recordings = tmp_path / "recordings"
    path = _recording(recordings)
    pipeline.done_path(path).write_text("done\n")
    pipeline.error_path(path).write_text("boom\n")
    fake = FakeTranscriber()

    assert pipeline.process_ready(root, fake, recordings) == 0
    assert fake.calls == 0


def test_worker_is_idempotent(tmp_path):
    root = _kb(tmp_path)
    recordings = tmp_path / "recordings"
    path = _recording(recordings)
    pipeline.done_path(path).write_text("done\n")
    fake = FakeTranscriber()

    assert pipeline.process_ready(root, fake, recordings) == 1
    assert pipeline.process_ready(root, fake, recordings) == 0
    assert fake.calls == 1


def test_force_rewrites_and_clears_error(tmp_path):
    root = _kb(tmp_path)
    recordings = tmp_path / "recordings"
    path = _recording(recordings)
    pipeline.error_path(path).write_text("old failure\n")

    entry = pipeline.process(path, FakeTranscriber(), root, force=True)

    assert entry["summary"] == "Visa paperwork."
    assert not pipeline.error_path(path).exists()


def test_summary_failure_keeps_transcript(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline.config, "SUMMARY_ENABLED", True)
    root = _kb(tmp_path)
    recordings = tmp_path / "recordings"
    path = _recording(recordings)
    pipeline.done_path(path).write_text("done\n")

    processed = pipeline.process_ready(root, FailingSummary(), recordings)

    assert processed == 1
    entry = knowledge_base.get(root, path.stem)
    assert entry["transcript"] == "We discussed the visa paperwork."
    assert entry["summary"] == ""
    assert entry["status"] == "transcribed"


def test_summary_can_be_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline.config, "SUMMARY_ENABLED", False)
    root = _kb(tmp_path)
    recordings = tmp_path / "recordings"
    path = _recording(recordings)
    pipeline.done_path(path).write_text("done\n")

    pipeline.process_ready(root, FakeTranscriber(), recordings)

    entry = knowledge_base.get(root, path.stem)
    assert entry["summary"] == ""
    assert entry["status"] == "transcribed"


def test_short_transcript_is_flagged_without_summary(tmp_path):
    root = _kb(tmp_path)
    recordings = tmp_path / "recordings"
    path = _recording(recordings)
    pipeline.done_path(path).write_text("done\n")
    fake = FakeTranscriber(text="Thank you.")

    pipeline.process_ready(root, fake, recordings)

    entry = knowledge_base.get(root, path.stem)
    assert entry["status"] == "short"
    assert entry["summary"] == ""
    assert fake.summary_calls == 0


def test_empty_transcript_is_quarantined(tmp_path):
    root = _kb(tmp_path)
    recordings = tmp_path / "recordings"
    path = _recording(recordings)
    pipeline.done_path(path).write_text("done\n")

    processed = pipeline.process_ready(root, FakeTranscriber(text="   "), recordings)

    assert processed == 0
    assert pipeline.error_path(path).exists()
    assert knowledge_base.find(root, path.stem) is None
