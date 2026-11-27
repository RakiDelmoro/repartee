"""USB CDC receiver for the conversation-recorder firmware.

Reads START/AUDIO/END records from the XIAO's USB serial console, writes one
16-bit PCM WAV per session into data/recordings, and acknowledges received sample
counts so the device can resume after a link loss. The ESP-IDF console shares
the port, so the framer resyncs past log lines by scanning for the record magic.

Recordings land in ``data/recordings`` as ``YYYYMMDD-HHMMSS-<device>.wav``.
The file appears at START and is playable while it grows; its RIFF sizes are
refreshed every few seconds and fixed at END.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import select
import struct
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import BinaryIO

from . import config

DEVICE_RE = re.compile(r"[^A-Za-z0-9]")
_last_stamp: datetime | None = None

HDR = struct.Struct("<IBBHI")  # magic, version, type, flags, length
START = struct.Struct("<IIIHHQQ16s")
END = struct.Struct("<IIB3xI")
ACK = struct.Struct("<II")

MAGIC = 0x43455243
MAGIC_BYTES = struct.pack("<I", MAGIC)
VERSION = 1

TYPE_START = 1
TYPE_AUDIO = 2
TYPE_END = 3
TYPE_ACK = 0x81

MAX_AUDIO_BYTES = 6400
RETRY_OPEN_SECONDS = 1.0

FINAL_ACK_FLAG = 0x0001
FINALIZE_IDLE_SECONDS = 300.0
CLEANUP_INTERVAL_SECONDS = 60.0

log = logging.getLogger(__name__)


def wav_header(rate: int, channels: int, bits: int, data_bytes: int) -> bytes:
    block_align = channels * bits // 8
    byte_rate = rate * block_align
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_bytes,
        b"WAVE",
        b"fmt ",
        16,
        1,
        channels,
        rate,
        byte_rate,
        block_align,
        bits,
        b"data",
        data_bytes,
    )


class Framer:
    """Splits a byte stream into protocol records.

    With ``resync`` the stream may also carry device console text (USB shares
    the port with the ESP-IDF log), so sync is recovered by scanning for a
    plausible record header.
    """

    def __init__(self, reader: asyncio.StreamReader, resync: bool = False):
        self.reader = reader
        self.resync = resync
        self.buf = bytearray()

    async def _fill(self, size: int) -> None:
        while len(self.buf) < size:
            chunk = await self.reader.read(65536)
            if not chunk:
                raise asyncio.IncompleteReadError(bytes(self.buf), size)
            self.buf.extend(chunk)

    @staticmethod
    def _plausible(typ: int, length: int) -> bool:
        if typ == TYPE_START:
            return length == START.size
        if typ == TYPE_END:
            return length == END.size
        if typ == TYPE_ACK:
            return length == ACK.size
        if typ == TYPE_AUDIO:
            return 0 < length <= MAX_AUDIO_BYTES and length % 2 == 0
        return False

    async def _resync(self) -> None:
        while True:
            await self._fill(HDR.size)
            idx = self.buf.find(MAGIC_BYTES)
            if idx < 0:
                del self.buf[: max(0, len(self.buf) - (len(MAGIC_BYTES) - 1))]
                continue
            if idx:
                log.debug("skipping %d non-protocol bytes", idx)
                del self.buf[:idx]
            await self._fill(HDR.size)
            _magic, version, typ, _flags, length = HDR.unpack_from(self.buf)
            if version == VERSION and self._plausible(typ, length):
                return
            del self.buf[0]

    async def next_record(self) -> tuple[int, int, bytes]:
        if self.resync:
            await self._resync()
        else:
            await self._fill(HDR.size)
            magic, version, typ, _flags, length = HDR.unpack_from(self.buf)
            if (
                magic != MAGIC
                or version != VERSION
                or not self._plausible(typ, length)
            ):
                raise ValueError(
                    f"bad header magic={magic:08x} version={version} "
                    f"type={typ:#04x} length={length}"
                )
        _magic, _version, typ, flags, length = HDR.unpack_from(self.buf)
        await self._fill(HDR.size + length)
        payload = bytes(self.buf[HDR.size : HDR.size + length])
        del self.buf[: HDR.size + length]
        return typ, flags, payload


@dataclass
class Session:
    key: tuple[str, int]
    path: Path
    file: BinaryIO
    rate: int
    channels: int
    bits: int
    received: int = 0
    last_activity: float = field(default_factory=time.monotonic)
    last_report: float = 0.0
    active: bool = True
    finalized: bool = False

    @property
    def sample_bytes(self) -> int:
        return self.channels * self.bits // 8


def recording_path(out_dir: Path, device: str) -> Path:
    """``YYYYMMDD-HHMMSS-<device>.wav``: one-second stamps, forced monotonic.

    The name carries the start time, and a non-alphanumeric device id is dropped
    so the filename stays parseable and path-safe. An existing file is never
    opened: the stamp moves forward instead.
    """
    global _last_stamp
    while True:
        now = datetime.now().replace(microsecond=0)
        if _last_stamp is not None and now <= _last_stamp:
            now = _last_stamp + timedelta(seconds=1)
        _last_stamp = now
        slug = DEVICE_RE.sub("", device) or "device"
        path = out_dir / f"{now.strftime('%Y%m%d-%H%M%S')}-{slug}.wav"
        if not path.exists():
            return path


class IngestServer:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.sessions: dict[tuple[str, int], Session] = {}
        self._cleanup_task: asyncio.Task | None = None

    def start_cleanup(self) -> None:
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self) -> None:
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
        for session in list(self.sessions.values()):
            if not session.finalized:
                self._finalize(session)

    async def _cleanup_loop(self) -> None:
        while True:
            await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
            now = time.monotonic()
            for session in list(self.sessions.values()):
                if (
                    not session.finalized
                    and not session.active
                    and now - session.last_activity > FINALIZE_IDLE_SECONDS
                ):
                    session.last_activity = now
                    self._finalize(session)

    def _write_header(self, session: Session) -> None:
        data_bytes = session.received * session.sample_bytes
        pos = session.file.tell()
        session.file.seek(0)
        session.file.write(
            wav_header(session.rate, session.channels, session.bits, data_bytes)
        )
        session.file.flush()
        session.file.seek(pos)

    def _finalize(self, session: Session) -> None:
        if session.finalized:
            return
        data_bytes = session.received * session.sample_bytes
        try:
            session.file.seek(0)
            session.file.write(
                wav_header(session.rate, session.channels, session.bits, data_bytes)
            )
            session.file.flush()
            os.fsync(session.file.fileno())
            session.file.close()
        except OSError as exc:
            log.error("finalize %s failed: %s", session.path, exc)
        session.active = False
        session.finalized = True
        duration = session.received / session.rate if session.rate else 0.0
        log.info(
            "finalized %s (%.2f s, %d samples)",
            session.path,
            duration,
            session.received,
        )
        print(f"{session.path} ({duration:.2f} s)")

    async def _send_ack(
        self,
        writer,
        session_id: int,
        acked_samples: int,
        final: bool = False,
    ) -> None:
        flags = FINAL_ACK_FLAG if final else 0
        writer.write(
            HDR.pack(MAGIC, VERSION, TYPE_ACK, flags, ACK.size)
            + ACK.pack(session_id, acked_samples)
        )
        await writer.drain()

    async def handle(
        self,
        reader: asyncio.StreamReader,
        writer,
        resync: bool = False,
    ) -> None:
        peer = writer.get_extra_info("peername")
        log.info("connection from %s", peer)
        current: Session | None = None
        framer = Framer(reader, resync=resync)
        try:
            while True:
                try:
                    typ, flags, payload = await framer.next_record()
                except ValueError as exc:
                    log.error("bad record from %s: %s", peer, exc)
                    break

                if typ == TYPE_START:
                    if len(payload) != START.size:
                        log.error("bad START length %d", len(payload))
                        break
                    (
                        session_id,
                        first_sample,
                        rate,
                        channels,
                        bits,
                        start_unix_ms,
                        device_uptime_ms,
                        device_id_raw,
                    ) = START.unpack(payload)
                    device = device_id_raw.split(b"\0", 1)[0].decode(
                        "ascii", "replace"
                    )
                    key = (device, session_id)
                    log.info(
                        "START %s session=%08x first=%d rate=%d ch=%d bits=%d uptime=%d",
                        device,
                        session_id,
                        first_sample,
                        rate,
                        channels,
                        bits,
                        device_uptime_ms,
                    )

                    if (
                        current is not None
                        and current.key != key
                        and current.active
                        and not current.finalized
                    ):
                        self._finalize(current)
                        current = None

                    existing = self.sessions.get(key)
                    if existing is not None and existing.finalized:
                        log.info("START for finalized session %s", key)
                        await self._send_ack(
                            writer, session_id, existing.received, final=True
                        )
                        current = None
                        continue
                    if existing is not None and existing.active:
                        if first_sample > existing.received:
                            log.error(
                                "reject duplicate START gap for %s: first=%d received=%d",
                                key,
                                first_sample,
                                existing.received,
                            )
                            break
                        if first_sample < existing.received:
                            offset = 44 + first_sample * existing.sample_bytes
                            existing.file.seek(offset)
                            existing.file.truncate(offset)
                            existing.received = first_sample
                        existing.active = True
                        existing.last_activity = time.monotonic()
                        current = existing
                        log.info(
                            "duplicate START for active %s; resuming at sample %d",
                            key,
                            existing.received,
                        )
                        await self._send_ack(writer, session_id, existing.received)
                        continue

                    if existing is None:
                        if first_sample != 0:
                            log.error(
                                "reject START gap for new session %s: first=%d",
                                key,
                                first_sample,
                            )
                            break
                        path = recording_path(self.out_dir, device)
                        file = open(path, "wb+")
                        file.write(b"\0" * 44)
                        file.flush()
                        session = Session(
                            key=key,
                            path=path,
                            file=file,
                            rate=rate,
                            channels=channels,
                            bits=bits,
                            received=0,
                            active=True,
                        )
                        self.sessions[key] = session
                        log.info("new recording %s", path)
                    else:
                        if first_sample > existing.received:
                            log.error(
                                "reject START resume gap for %s: first=%d received=%d",
                                key,
                                first_sample,
                                existing.received,
                            )
                            break
                        offset = 44 + first_sample * existing.sample_bytes
                        existing.file.seek(offset)
                        existing.file.truncate(offset)
                        existing.received = first_sample
                        existing.active = True
                        existing.last_activity = time.monotonic()
                        session = existing
                        log.info("resume %s at sample %d", session.path, first_sample)

                    current = session
                    await self._send_ack(writer, session_id, session.received)

                elif typ == TYPE_AUDIO:
                    if current is None or current.finalized:
                        log.warning(
                            "AUDIO without active session from %s; waiting for START",
                            peer,
                        )
                        continue
                    if not current.active:
                        log.error("AUDIO for inactive session %s", current.key)
                        break
                    if len(payload) % current.sample_bytes != 0:
                        log.warning(
                            "AUDIO payload not sample-aligned: %d bytes", len(payload)
                        )
                    current.file.write(payload)
                    current.file.flush()
                    current.received += len(payload) // current.sample_bytes
                    current.last_activity = time.monotonic()
                    if current.last_activity - current.last_report >= 5.0:
                        current.last_report = current.last_activity
                        try:
                            self._write_header(current)
                        except OSError as exc:
                            log.error("header update %s failed: %s", current.path, exc)
                        duration = (
                            current.received / current.rate if current.rate else 0.0
                        )
                        log.info(
                            "recording %s: %.1f s, %d samples",
                            current.path,
                            duration,
                            current.received,
                        )
                    await self._send_ack(writer, current.key[1], current.received)

                elif typ == TYPE_END:
                    if len(payload) != END.size:
                        log.error("bad END length %d", len(payload))
                        break
                    session_id, total_samples, reason, _crc = END.unpack(payload)
                    log.info(
                        "END session=%08x total=%d reason=%d",
                        session_id,
                        total_samples,
                        reason,
                    )
                    session = None
                    if current is not None and current.key[1] == session_id:
                        session = current
                    if session is None:
                        for candidate in self.sessions.values():
                            if candidate.key[1] == session_id:
                                session = candidate
                                break
                    if session is None:
                        log.error("END for unknown session %08x", session_id)
                        continue
                    if session.finalized:
                        await self._send_ack(
                            writer, session_id, session.received, final=True
                        )
                        continue
                    if session.received != total_samples:
                        log.warning(
                            "END total %d disagrees with received %d",
                            total_samples,
                            session.received,
                        )
                    self._finalize(session)
                    await self._send_ack(
                        writer, session_id, session.received, final=True
                    )
                    current = session

                else:
                    log.error("unsupported record type 0x%02x", typ)
                    break
        except asyncio.IncompleteReadError:
            log.info("connection %s closed", peer)
        except (ConnectionError, OSError) as exc:
            log.info("connection %s error: %s", peer, exc)
        finally:
            if current is not None and current.active:
                current.active = False
                current.last_activity = time.monotonic()
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass


def open_serial(path: str):
    """Open a USB CDC device on POSIX or Windows.

    POSIX uses a raw tty. Windows uses pyserial, because termios is unavailable.
    """
    if os.name == "nt":
        try:
            import serial
        except ImportError as exc:
            raise OSError(
                "pyserial is required on Windows: python -m pip install pyserial"
            ) from exc
        return serial.Serial(
            path,
            baudrate=115200,
            timeout=0.2,
            write_timeout=1.0,
            rtscts=False,
            dsrdtr=False,
            xonxoff=False,
        )

    import termios
    import tty

    fd = os.open(path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    try:
        tty.setraw(fd)
        attrs = termios.tcgetattr(fd)
        attrs[2] |= termios.CLOCAL | termios.CREAD
        termios.tcsetattr(fd, termios.TCSANOW, attrs)
    except termios.error:
        pass
    return fd


def _pump(fd, loop, reader: asyncio.StreamReader, stop: threading.Event) -> None:
    """Move bytes from the serial device into the asyncio reader."""
    try:
        if os.name == "nt":
            while not stop.is_set():
                try:
                    data = fd.read(8192)
                except Exception:
                    break
                if data:
                    try:
                        loop.call_soon_threadsafe(reader.feed_data, data)
                    except RuntimeError:
                        return
        else:
            while not stop.is_set():
                ready, _, _ = select.select([fd], [], [], 0.2)
                if not ready:
                    continue
                data = os.read(fd, 8192)
                if not data:
                    break
                try:
                    loop.call_soon_threadsafe(reader.feed_data, data)
                except RuntimeError:
                    return
    except OSError:
        pass
    finally:
        try:
            loop.call_soon_threadsafe(reader.feed_eof)
        except RuntimeError:
            pass


class SerialBytes:
    """StreamWriter-compatible sink for a raw tty or pyserial port."""

    def __init__(self, fd, label: str):
        self.fd = fd
        self.label = label

    def write(self, data: bytes) -> None:
        if os.name == "nt":
            self.fd.write(data)
            self.fd.flush()
            return
        view = memoryview(data)
        while view:
            try:
                n = os.write(self.fd, view)
            except BlockingIOError:
                select.select([], [self.fd], [], 0.1)
                continue
            view = view[n:]

    async def drain(self) -> None:
        return None

    def get_extra_info(self, name, default=None):
        return self.label if name == "peername" else default

    def close(self) -> None:
        return None

    async def wait_closed(self) -> None:
        return None


async def run_serial(serial_path: str, out_dir: Path) -> None:
    """Record every session the device sends, reconnecting until cancelled."""
    server = IngestServer(out_dir)
    server.start_cleanup()
    loop = asyncio.get_running_loop()
    log.info("serial: reading %s, writing to %s", serial_path, out_dir)
    try:
        while True:
            try:
                fd = open_serial(serial_path)
            except OSError as exc:
                log.error("serial: open %s failed: %s", serial_path, exc)
                await asyncio.sleep(RETRY_OPEN_SECONDS)
                continue
            log.info("serial: %s open", serial_path)
            reader = asyncio.StreamReader()
            stop = threading.Event()
            thread = threading.Thread(
                target=_pump, args=(fd, loop, reader, stop), daemon=True
            )
            thread.start()
            try:
                await server.handle(reader, SerialBytes(fd, serial_path), resync=True)
            finally:
                stop.set()
                thread.join(timeout=1.0)
                if os.name == "nt":
                    try:
                        fd.close()
                    except Exception:
                        pass
                else:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
            log.warning("serial: %s closed; reopening", serial_path)
            await asyncio.sleep(RETRY_OPEN_SECONDS)
    finally:
        await server.stop()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="repartee receiver", description=__doc__)
    parser.add_argument(
        "--serial",
        default=config.SERIAL_DEVICE,
        metavar="DEVICE",
        help="USB serial device, e.g. /dev/ttyACM0 or COM3 (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=config.RECORDINGS_DIR,
        help="recordings directory (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config.ensure_dirs()
    try:
        asyncio.run(run_serial(args.serial, args.out))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
