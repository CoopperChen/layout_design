"""CNC work-pose UDP parse and pm relative landmark math."""

from __future__ import annotations

import json
import socket
import time
from pathlib import Path

import numpy as np

from app.postprocess.cnc_work_pose import (
    WorkPoseUdpClient,
    parse_work_pose_payload,
)
from app.postprocess.print_config import load_physical_landmarks, save_physical_landmarks
from app.postprocess.record_pm import pm_from_raw_touches


def test_parse_work_pose_payload_mm():
    sample = parse_work_pose_payload(
        json.dumps(
            {
                "coord": "work",
                "units": "mm",
                "x": 10.0,
                "y": 20.0,
                "z": -5.0,
                "b": 90.0,
                "c": 1.5,
            }
        )
    )
    assert sample.pose.xyz == (10.0, 20.0, -5.0)
    assert sample.pose.b_deg == 90.0
    assert sample.pose.c_deg == 1.5


def test_parse_work_pose_payload_inches():
    sample = parse_work_pose_payload(
        '{"coord":"work","units":"in","x":1,"y":0,"z":0,"b":0,"c":0}'
    )
    assert abs(sample.pose.x - 25.4) < 1e-9


def test_pm_from_raw_touches_centers_origin():
    raw = np.array(
        [
            [100.0, 50.0, -10.0],
            [110.0, 50.0, -10.0],
            [100.0, 60.0, -10.0],
        ]
    )
    pm = pm_from_raw_touches(raw)
    np.testing.assert_allclose(pm[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(pm[1], [10.0, 0.0, 0.0])
    np.testing.assert_allclose(pm[2], [0.0, 10.0, 0.0])


def test_save_and_load_physical_landmarks(tmp_path: Path):
    pm = np.array([[0.0, 0.0, 0.0], [12.0, -3.0, 1.0], [0.5, 8.0, 0.0]])
    path = tmp_path / "subject_9.yaml"
    save_physical_landmarks(
        path,
        pm,
        subject_id=9,
        capture_meta={
            "raw_work_xyz_mm": [[1, 2, 3], [13, -1, 4], [1.5, 10, 3]],
            "work_bc_deg": [{"b": 90, "c": 0}, {"b": 90, "c": 1}, {"b": 88, "c": 0}],
            "udp_port": 62100,
        },
    )
    loaded = load_physical_landmarks(path)
    np.testing.assert_allclose(loaded, pm)
    text = path.read_text(encoding="utf-8")
    assert "raw_work_xyz_mm" in text
    assert "udp_port: 62100" in text


def test_udp_client_receives_packet():
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()

    with WorkPoseUdpClient(bind_ip="127.0.0.1", port=port, stale_sec=1.0) as client:
        payload = json.dumps(
            {"coord": "work", "units": "mm", "x": 1, "y": 2, "z": 3, "b": 4, "c": 5}
        ).encode()
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.sendto(payload, ("127.0.0.1", port))
            deadline = time.monotonic() + 2.0
            pose = None
            while time.monotonic() < deadline:
                pose = client.latest()
                if pose is not None:
                    break
                time.sleep(0.02)
        finally:
            sock.close()
        assert pose is not None
        assert pose.xyz == (1.0, 2.0, 3.0)
        assert client.packets >= 1
