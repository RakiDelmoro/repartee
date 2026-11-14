"""Device side: low-power listen, record the utterance, spool it for the bigger device.

The VAD is pure energy on 30 ms frames — no model, no audio device library. Playback
and capture go through any command that writes 16 kHz mono signed-16-bit PCM to
stdout (arecord, parecord, ffmpeg), so the same code runs on the device, in a
container, or over a pipe. Finished recordings land atomically in the spool with the
device filename the ingest side expects: ``YYYYMMDD-HHMMSS-device.wav``.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import subprocess
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import soundfile as sf

from . import config

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000
FRAME_S = FRAME_MS / 1000.0
DEVICE_RE = re.compile(r"[^A-Za-z0-9]")


@dataclass(frozen=True)
class VadConfig:
    threshold: float = 0.012       # RMS floor that counts as speech
    min_speech_ms: int = 90        # speech this long starts a recording
    stop_silence_s: float = 60.0   # silence this long closes it
    preroll_s: float = 1.0         # audio kept before the first speech frame
    max_record_s: float = 1800.0   # hard cap, so a stuck stream cannot fill the disk
    min_record_s: float = 1.0      # shorter segments are discarded as noise


class Segmenter:
    """Frame-by-frame VAD state machine. Feed 30 ms frames, get finished PCM or None."""

    def __init__(self, cfg: VadConfig | None = None) -> None:
        self.cfg = cfg or VadConfig()
        self.floor = 0.0
        self.preroll: deque[np.ndarray] = deque(
            maxlen=max(1, int(round(self.cfg.preroll_s / FRAME_S)))
        )
        self.buf: list[np.ndarray] = []
        self.recording = False
        self.loud = 0
        self.silent = 0

    def _threshold(self) -> float:
        return max(self.cfg.threshold, self.floor * 3.0)

    def feed(self, frame: np.ndarray) -> np.ndarray | None:
        frame = np.asarray(frame, dtype=np.float32)
        rms = float(np.sqrt(np.mean(frame ** 2))) if frame.size else 0.0
        thr = self._threshold()

        if not self.recording:
            # Track the quiet floor upward slowly and downward fast.
            self.floor = min(rms, self.floor * 1.02 + 1e-6)
            self.preroll.append(frame)
            if rms > thr:
                self.loud += 1
                if self.loud * FRAME_MS >= self.cfg.min_speech_ms:
                    self.recording = True
                    self.buf = list(self.preroll)
                    self.preroll.clear()
                    self.silent = 0
                    self.loud = 0
            else:
                self.loud = 0
            return None

        self.buf.append(frame)
        self.silent = 0 if rms > thr else self.silent + 1
        too_long = len(self.buf) * FRAME_S >= self.cfg.max_record_s
        if self.silent * FRAME_S >= self.cfg.stop_silence_s or too_long:
            return self.finish()
        return None

    def finish(self) -> np.ndarray | None:
        """Close the open recording, or return None when it was too short to keep."""
        if not self.recording:
            return None
        audio = np.concatenate(self.buf) if self.buf else np.zeros(0, dtype=np.float32)
        self.reset()
        if len(audio) / SAMPLE_RATE < self.cfg.min_record_s:
            return None
        return audio

    def reset(self) -> None:
        self.recording = False
        self.buf = []
        self.loud = 0
        self.silent = 0
        self.preroll.clear()


_last_stamp: datetime | None = None


def spool_path(spool: Path, device: str) -> Path:
    """One-second stamps, forced monotonic so two recordings can never collide.

    The ingest side parses exactly six time digits, so a sub-second suffix is not
    an option; bumping the second is the only collision-free choice.
    """
    global _last_stamp
    now = datetime.now().replace(microsecond=0)
    if _last_stamp is not None and now <= _last_stamp:
        now = _last_stamp + timedelta(seconds=1)
    _last_stamp = now
    name = f"{now.strftime('%Y%m%d-%H%M%S')}-{DEVICE_RE.sub('', device) or 'device'}.wav"
    return spool / name


def write_recording(spool: Path, device: str, audio: np.ndarray) -> Path:
    """Write atomically: the worker only ever sees complete ``.wav`` files."""
    spool.mkdir(parents=True, exist_ok=True)
    dest = spool_path(spool, device)
    tmp = dest.with_suffix(".part")
    sf.write(tmp, np.clip(audio, -1.0, 1.0), SAMPLE_RATE, format="WAV",
             subtype="PCM_16")
    os.replace(tmp, dest)
    return dest


def frames_from_bytes(stream, frame_bytes: int):
    while True:
        chunk = stream.read(frame_bytes)
        if not chunk or len(chunk) < frame_bytes:
            return
        yield np.frombuffer(chunk, dtype="<i2").astype(np.float32) / 32768.0


def capture(cmd: str, spool: Path, device: str, cfg: VadConfig,
            realtime: bool = False) -> int:
    """Record from a command that emits 16 kHz mono s16le on stdout."""
    seg = Segmenter(cfg)
    written = 0
    proc = subprocess.Popen(shlex_split(cmd), stdout=subprocess.PIPE)
    try:
        for frame in frames_from_bytes(proc.stdout, FRAME_SAMPLES * 2):
            if realtime:
                time.sleep(FRAME_S)
            audio = seg.feed(frame)
            if audio is not None:
                path = write_recording(spool, device, audio)
                written += 1
                log.info("spooled %s (%.1fs)", path.name, len(audio) / SAMPLE_RATE)
        tail = seg.finish()
        if tail is not None:
            path = write_recording(spool, device, tail)
            written += 1
            log.info("spooled %s (%.1fs, end of stream)", path.name,
                     len(tail) / SAMPLE_RATE)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    return written


def capture_file(path: Path, spool: Path, device: str, cfg: VadConfig,
                 realtime: bool = False) -> int:
    """Replay a WAV as if it came from the microphone."""
    wav, sr = sf.read(str(path), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)
    if sr != SAMPLE_RATE:
        import librosa

        wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)
    seg = Segmenter(cfg)
    written = 0
    for i in range(0, len(wav), FRAME_SAMPLES):
        frame = wav[i:i + FRAME_SAMPLES]
        if len(frame) < FRAME_SAMPLES:
            break
        if realtime:
            time.sleep(FRAME_S)
        audio = seg.feed(frame)
        if audio is not None:
            p = write_recording(spool, device, audio)
            written += 1
            log.info("spooled %s (%.1fs)", p.name, len(audio) / SAMPLE_RATE)
    tail = seg.finish()
    if tail is not None:
        p = write_recording(spool, device, tail)
        written += 1
        log.info("spooled %s (%.1fs, end of file)", p.name, len(tail) / SAMPLE_RATE)
    return written


def shlex_split(cmd: str) -> list[str]:
    import shlex

    return shlex.split(cmd)


def default_capture_cmd() -> str:
    """First capture backend that exists on this machine."""
    if os.environ.get("REPARTEE_LISTENER_CMD"):
        return os.environ["REPARTEE_LISTENER_CMD"]
    if shutil.which("arecord"):
        return "arecord -q -f S16_LE -r 16000 -c 1 -t raw"
    if shutil.which("parecord"):
        return "parecord --format=s16le --rate=16000 --channels=1 --raw"
    if shutil.which("ffmpeg"):
        return ("ffmpeg -hide_banner -loglevel error -f alsa -i default "
                "-f s16le -acodec pcm_s16le -ac 1 -ar 16000 -")
    raise SystemExit("no capture backend: set REPARTEE_LISTENER_CMD")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="repartee listen", description=__doc__)
    ap.add_argument("--device", default=os.environ.get("HOSTNAME", "device"))
    ap.add_argument("--spool", type=Path, default=config.SPOOL_DIR)
    ap.add_argument("--input", type=Path, help="replay a WAV instead of capturing")
    ap.add_argument("--cmd", default=None, help="capture command (16 kHz mono s16le)")
    ap.add_argument("--stop-silence", type=float, default=60.0)
    ap.add_argument("--threshold", type=float, default=0.012)
    ap.add_argument("--min-speech-ms", type=int, default=90)
    ap.add_argument("--preroll", type=float, default=1.0)
    ap.add_argument("--realtime", action="store_true",
                    help="pace file replay at real time")
    args = ap.parse_args(argv)

    cfg = VadConfig(threshold=args.threshold, min_speech_ms=args.min_speech_ms,
                    stop_silence_s=args.stop_silence, preroll_s=args.preroll)
    if args.input:
        n = capture_file(args.input, args.spool, args.device, cfg, args.realtime)
    else:
        n = capture(args.cmd or default_capture_cmd(), args.spool, args.device, cfg,
                    args.realtime)
    log.info("listener stopped; %d recording(s) in %s", n, args.spool)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
