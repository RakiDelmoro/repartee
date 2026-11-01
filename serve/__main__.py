"""``python -m serve``: run the fixed 80 ms duplex loop.

Examples
--------
File to file with the real checkpoint::

    python -m serve --model LiquidAI/LFM3-Omni-2B \
        --input user.wav --output agent.wav --metrics metrics.json

CI smoke test with the deterministic fake engine::

    python -m serve --fake --input user.wav --output agent.wav

Live audio uses the optional ``sounddevice`` package when ``--input`` or
``--output`` is omitted.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def _dtype_from_name(name: str):
    if name == "auto":
        return None
    import torch

    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m serve",
        description="Fixed 80 ms full-duplex serving loop (Repartee).",
    )
    parser.add_argument("--model", default="LiquidAI/LFM3-Omni-2B", help="checkpoint id or local path")
    parser.add_argument("--input", type=Path, default=None, help="16 kHz mono user WAV; omit for live mic")
    parser.add_argument("--output", type=Path, default=None, help="24 kHz mono agent WAV; omit for live speaker")
    parser.add_argument("--metrics", type=Path, default=None, help="write session metrics JSON here")
    parser.add_argument("--ticks", type=int, default=None, help="number of 80 ms ticks (default: input length or 7500)")
    parser.add_argument("--fake", action="store_true", help="use the deterministic scripted engine (CI only)")
    parser.add_argument("--int4", action="store_true", help="load the real checkpoint in 4-bit (bitsandbytes)")
    parser.add_argument("--device", default=None, help="torch device for the real checkpoint, e.g. cuda:0")
    parser.add_argument(
        "--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"], help="model dtype"
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="audio sampling temperature (0 = greedy)")
    parser.add_argument("--top-k", type=int, default=None, help="audio sampling top-k (default: greedy)")
    parser.add_argument(
        "--realtime",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="sleep on absolute 80 ms deadlines (default: live audio yes, files no)",
    )
    parser.add_argument("--aec", action=argparse.BooleanOptionalAction, default=None, help="enable AEC (default: live only)")
    parser.add_argument(
        "--aec-backend", choices=["auto", "webrtc", "nlms"], default="auto", help="AEC backend (default: auto)"
    )
    parser.add_argument("--user-speech-dbfs", type=float, default=-40.0, help="RMS threshold for user activity")
    parser.add_argument("--jitter-target-ms", type=float, default=30.0, help="jitter reserve target (20-40 ms)")
    parser.add_argument("--jitter-max-ms", type=float, default=1000.0, help="hard jitter-buffer bound")
    parser.add_argument("--input-device", default=None, help="sounddevice input device")
    parser.add_argument("--output-device", default=None, help="sounddevice output device")
    parser.add_argument("--seed", type=int, default=None, help="seed numpy/torch sampling")
    return parser


def _input_ticks(path: Path) -> int:
    import soundfile as sf

    from src.schema import FRAME_SECONDS

    info = sf.info(str(path))
    return max(1, int(math.ceil(info.duration / FRAME_SECONDS - 1e-9)))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.seed is not None:
        import numpy as np

        np.random.seed(args.seed)
        try:
            import torch

            torch.manual_seed(args.seed)
        except ImportError:
            pass

    if args.fake:
        from .loop import FakeDuplexEngine

        engine = FakeDuplexEngine()
    else:
        from .loop import Lfm3DuplexEngine

        engine = Lfm3DuplexEngine(
            args.model,
            int4=args.int4,
            device=args.device,
            dtype=_dtype_from_name(args.dtype),
            temperature=args.temperature,
            top_k=args.top_k,
            user_speech_dbfs=min(args.user_speech_dbfs, -50.0),
        )

    live = args.input is None or args.output is None
    use_aec = live if args.aec is None else bool(args.aec)
    aec = None
    if use_aec:
        from .aec import create_echo_canceller

        aec = create_echo_canceller(backend=args.aec_backend)
    realtime = live if args.realtime is None else args.realtime
    ticks = args.ticks
    if ticks is None:
        ticks = _input_ticks(args.input) if args.input is not None else 7500

    from .loop import SoundDeviceSink, SoundDeviceSource, WavFileSink, WavFileSource, run_session

    source = (
        WavFileSource(args.input) if args.input is not None else SoundDeviceSource(device=args.input_device)
    )
    sink = (
        WavFileSink(args.output) if args.output is not None else SoundDeviceSink(device=args.output_device)
    )
    try:
        metrics = run_session(
            source,
            sink,
            engine,
            ticks,
            metrics_path=args.metrics,
            aec=aec,
            realtime=realtime,
            user_speech_dbfs=args.user_speech_dbfs,
            jitter_target_ms=args.jitter_target_ms,
            jitter_max_ms=args.jitter_max_ms,
        )
    finally:
        source.close()
        sink.close()

    summary = {
        key: metrics[key]
        for key in (
            "ticks",
            "audio_seconds",
            "rtf_engine",
            "rtf_wall",
            "jitter_ms",
            "ttfa_ms",
            "controls",
            "jitter_buffer",
            "self_interruptions",
            "barge_in",
            "rss",
            "gate7_claimed",
        )
    }
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
