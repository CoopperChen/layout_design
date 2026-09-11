"""Mach4 CNC command protocol and path resolution (no live controller)."""

from __future__ import annotations

import json
import socket
import threading
from pathlib import Path

import pytest

from app.cli import build_parser
from app.postprocess.cnc_control import (
    CncControlClient,
    CncControlError,
    build_command,
    encode_command,
    parse_ack,
    resolve_gcode_path,
)


def test_build_command_status_has_id():
    rec = build_command("status", request_id="abc")
    assert rec == {"id": "abc", "cmd": "status"}


def test_build_command_load_requires_path():
    with pytest.raises(ValueError, match="path"):
        build_command("load")
    rec = build_command("load", path=r"D:\gcode\allinterconnects.txt")
    assert rec["cmd"] == "load"
    assert rec["path"].endswith("allinterconnects.txt")


def test_build_command_rejects_jog():
    with pytest.raises(ValueError, match="unsupported"):
        build_command("jog")


def test_parse_ack_lua_style():
    ack = parse_ack(
        '{"ok":true,"id":"abc","cmd":"status","error":"",'
        '"state":"idle","enabled":true,"file":"C:\\\\gcode\\\\a.txt",'
        '"x":1.25,"y":2.5,"z":-3,"b":90,"c":0.5}'
    )
    assert ack.ok is True
    assert ack.request_id == "abc"
    assert ack.cmd == "status"
    assert ack.state == "idle"
    assert ack.enabled is True
    assert ack.file == r"C:\gcode\a.txt"
    assert ack.x == 1.25
    assert ack.b_deg == 90.0
    assert ack.c_deg == 0.5
    line = ack.format_line()
    assert "ok=True" in line
    assert "state=idle" in line


def test_parse_ack_false_error():
    ack = parse_ack('{"ok":false,"id":"1","cmd":"start","error":"not idle (state=running)"}')
    assert ack.ok is False
    assert "not idle" in ack.error


def test_encode_command_compact_json():
    raw = encode_command({"id": "a", "cmd": "stop"})
    assert raw == b'{"id":"a","cmd":"stop"}'


def test_resolve_gcode_path_absolute(tmp_path: Path):
    gcode = tmp_path / "allinterconnects.txt"
    gcode.write_text("G94 G1 X0 Y0 Z0 B0 C0 F100\n", encoding="utf-8")
    assert resolve_gcode_path(gcode) == gcode.resolve()


def test_resolve_gcode_path_relative_to_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from app.postprocess import cnc_control as mod

    gcode = tmp_path / "wire.txt"
    gcode.write_text("G1 X1\n", encoding="utf-8")
    monkeypatch.setattr(mod.paths, "REPO_ROOT", tmp_path)
    assert resolve_gcode_path("wire.txt") == gcode.resolve()


def test_resolve_gcode_path_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        resolve_gcode_path(tmp_path / "missing.txt")


def test_start_requires_confirm():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["cnc", "start"])
    args = parser.parse_args(["cnc", "start", "--confirm"])
    assert args.cnc_cmd == "start"
    assert args.confirm is True


def test_load_cli_keeps_gcode_path():
    parser = build_parser()
    args = parser.parse_args(["cnc", "load", "--gcode", "data/output/gcode/s.txt"])
    assert args.cnc_cmd == "load"
    assert Path(args.gcode) == Path("data/output/gcode/s.txt")


def test_client_roundtrip_status():
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.bind(("127.0.0.1", 0))
    port = server.getsockname()[1]
    received: list[bytes] = []

    def serve() -> None:
        data, addr = server.recvfrom(4096)
        received.append(data)
        rec = json.loads(data.decode())
        ack = {
            "ok": True,
            "id": rec["id"],
            "cmd": rec["cmd"],
            "error": "",
            "state": "idle",
            "enabled": True,
            "file": "",
            "x": 0,
            "y": 0,
            "z": 0,
            "b": 0,
            "c": 0,
        }
        server.sendto(json.dumps(ack).encode(), addr)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        client = CncControlClient(host="127.0.0.1", port=port)
        ack = client.status(timeout_sec=2.0)
        assert ack.ok is True
        assert ack.state == "idle"
        assert json.loads(received[0].decode())["cmd"] == "status"
    finally:
        thread.join(timeout=2.0)
        server.close()


def test_client_invalid_ack_includes_raw():
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.bind(("127.0.0.1", 0))
    port = server.getsockname()[1]

    def serve() -> None:
        _data, addr = server.recvfrom(4096)
        server.sendto(b'{"ok":true,"id":"","cmd":"load","error":"[string "PLC"]"}', addr)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        client = CncControlClient(host="127.0.0.1", port=port)
        with pytest.raises(CncControlError, match=r"raw=.*string") as caught:
            client.status(timeout_sec=2.0)
        assert "Expecting" in str(caught.value) or "delimiter" in str(caught.value)
    finally:
        thread.join(timeout=2.0)
        server.close()


def test_client_timeout():
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    client = CncControlClient(host="127.0.0.1", port=port)
    with pytest.raises(CncControlError, match="not listening|timed out"):
        client.status(timeout_sec=0.1)
