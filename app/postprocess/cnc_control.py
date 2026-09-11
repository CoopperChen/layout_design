"""Mach4 load/run control over localhost UDP (JSON).

Companion to ``scripts/mach4_work_pose_publisher.lua`` ``PollCncCommandUdp()``.
Does not jog or MDI; Cycle Start is never invoked from ``run``.
"""

from __future__ import annotations

import json
import socket
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app import paths

DEFAULT_CNC_HOST = "127.0.0.1"
DEFAULT_CNC_PORT = 62110
ALLOWED_CMDS = frozenset({"status", "load", "start", "hold", "stop"})
_STOP_HOLD_TRIES = 2

_TIMEOUT_SEC = {
    "status": 1.0,
    "load": 60.0,
    "start": 2.0,
    "hold": 2.0,
    "stop": 2.0,
}


class CncControlError(RuntimeError):
    """Mach4 command failed or timed out."""


@dataclass(frozen=True)
class CncAck:
    ok: bool
    request_id: str
    cmd: str
    error: str
    state: str
    enabled: bool
    file: str
    x: float
    y: float
    z: float
    b_deg: float
    c_deg: float

    def format_line(self) -> str:
        bits = [
            f"ok={self.ok}",
            f"cmd={self.cmd or '-'}",
            f"state={self.state or '-'}",
            f"enabled={self.enabled}",
        ]
        if self.file:
            bits.append(f"file={self.file}")
        if self.error:
            bits.append(f"error={self.error}")
        bits.append(
            f"X={self.x:.3f} Y={self.y:.3f} Z={self.z:.3f} "
            f"B={self.b_deg:.2f} C={self.c_deg:.2f}"
        )
        return "  ".join(bits)


def build_command(
    cmd: str,
    *,
    request_id: str | None = None,
    path: str | None = None,
) -> dict[str, str]:
    if cmd not in ALLOWED_CMDS:
        raise ValueError(f"unsupported cnc cmd {cmd!r}")
    record: dict[str, str] = {
        "id": request_id or uuid.uuid4().hex,
        "cmd": cmd,
    }
    if cmd == "load":
        if not path:
            raise ValueError("load requires path")
        record["path"] = path
    return record


def parse_ack(data: bytes | str) -> CncAck:
    if isinstance(data, bytes):
        text = data.decode("utf-8").strip()
    else:
        text = data.strip()
    if not text:
        raise ValueError("empty cnc ack")
    record = json.loads(text)
    if not isinstance(record, dict):
        raise ValueError("cnc ack must be a JSON object")
    return CncAck(
        ok=bool(record.get("ok")),
        request_id=str(record.get("id", "")),
        cmd=str(record.get("cmd", "")),
        error=str(record.get("error", "")),
        state=str(record.get("state", "")),
        enabled=bool(record.get("enabled")),
        file=str(record.get("file", "")),
        x=float(record.get("x", 0.0)),
        y=float(record.get("y", 0.0)),
        z=float(record.get("z", 0.0)),
        b_deg=float(record.get("b", record.get("b_deg", 0.0))),
        c_deg=float(record.get("c", record.get("c_deg", 0.0))),
    )


def resolve_gcode_path(gcode: str | Path) -> Path:
    path = Path(gcode)
    if not path.is_absolute():
        path = paths.REPO_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"G-code file not found: {path}")
    return path


def encode_command(record: dict[str, str]) -> bytes:
    return json.dumps(record, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


class CncControlClient:
    """Send one JSON command per datagram to Mach4; wait for one ACK."""

    def __init__(
        self,
        *,
        host: str = DEFAULT_CNC_HOST,
        port: int = DEFAULT_CNC_PORT,
    ) -> None:
        if port < 0 or port > 65535:
            raise ValueError("cnc command UDP port must be 0..65535")
        self.host = host
        self.port = int(port)

    def request(
        self,
        cmd: str,
        *,
        path: str | None = None,
        timeout_sec: float | None = None,
        request_id: str | None = None,
    ) -> CncAck:
        record = build_command(cmd, request_id=request_id, path=path)
        timeout = (
            float(timeout_sec)
            if timeout_sec is not None
            else _TIMEOUT_SEC.get(cmd, 2.0)
        )
        ack = self._exchange(record, timeout)
        if ack.request_id and ack.request_id != record["id"]:
            raise CncControlError(
                f"ack id mismatch: sent {record['id']}, got {ack.request_id}"
            )
        return ack

    def status(self, **kwargs: Any) -> CncAck:
        return self.request("status", **kwargs)

    def load(self, gcode: str | Path, **kwargs: Any) -> CncAck:
        resolved = resolve_gcode_path(gcode)
        return self.request("load", path=str(resolved), **kwargs)

    def start(self, **kwargs: Any) -> CncAck:
        return self.request("start", **kwargs)

    def hold(self, **kwargs: Any) -> CncAck:
        return self._repeat("hold", **kwargs)

    def stop(self, **kwargs: Any) -> CncAck:
        return self._repeat("stop", **kwargs)

    def _repeat(self, cmd: str, **kwargs: Any) -> CncAck:
        last: CncAck | None = None
        first_error: BaseException | None = None
        for _ in range(_STOP_HOLD_TRIES):
            try:
                last = self.request(cmd, **kwargs)
                first_error = None
            except CncControlError as exc:
                if last is None:
                    first_error = exc
                time.sleep(0.05)
                continue
            time.sleep(0.05)
        if last is None:
            assert first_error is not None
            raise first_error
        return last

    def _exchange(self, record: dict[str, str], timeout_sec: float) -> CncAck:
        payload = encode_command(record)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.bind((self.host if self.host == "127.0.0.1" else "0.0.0.0", 0))
            sock.settimeout(max(0.05, timeout_sec))
            sock.sendto(payload, (self.host, self.port))
            try:
                data, _addr = sock.recvfrom(4096)
            except (ConnectionResetError, ConnectionRefusedError) as exc:
                raise CncControlError(
                    f"Mach4 is not listening on {self.host}:{self.port} "
                    f"(cmd={record['cmd']}). PLC must call PollCncCommandUdp() "
                    "every cycle; check Mach4 for a bind-failed message."
                ) from exc
            except TimeoutError as exc:
                raise CncControlError(
                    f"timed out waiting for Mach4 ACK at {self.host}:{self.port} "
                    f"(cmd={record['cmd']}). If this is load, the controller "
                    "may still be opening the file — try: python -m app cnc status"
                ) from exc
        finally:
            sock.close()
        try:
            return parse_ack(data)
        except (ValueError, json.JSONDecodeError, UnicodeDecodeError, KeyError) as exc:
            raise CncControlError(f"invalid ack from Mach4: {exc}") from exc
