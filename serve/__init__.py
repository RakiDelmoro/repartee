"""Repartee serving runtime: AEC, fixed 80 ms loop, LFM3 duplex engine."""

from __future__ import annotations

from .aec import AcousticEchoCanceller, NLMSAEC, WebRtcAec3, create_echo_canceller
from .loop import (
    DuplexEngine,
    EngineOutput,
    FakeDuplexEngine,
    JitterBuffer,
    Lfm3DuplexEngine,
    SoundDeviceSink,
    SoundDeviceSource,
    TickClock,
    WavFileSink,
    WavFileSource,
    run_session,
)

__version__ = "0.1.0"

__all__ = [
    "AcousticEchoCanceller",
    "DuplexEngine",
    "EngineOutput",
    "FakeDuplexEngine",
    "JitterBuffer",
    "Lfm3DuplexEngine",
    "NLMSAEC",
    "SoundDeviceSink",
    "SoundDeviceSource",
    "TickClock",
    "WavFileSink",
    "WavFileSource",
    "WebRtcAec3",
    "create_echo_canceller",
    "run_session",
]
