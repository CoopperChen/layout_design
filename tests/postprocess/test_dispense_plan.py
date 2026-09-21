"""Constant tip-speed dispensing and jet-off orientation slews."""

from __future__ import annotations

import numpy as np

from app.postprocess.gcode.kinematics.dispense_plan import plan_constant_tip_dispense
from app.postprocess.gcode.kinematics.feed_rate import (
    dispense_pivot_feed,
    tip_positions_from_poses,
)
from app.postprocess.gcode.models import MachineConfig
from app.postprocess.gcode.pipeline.merge_traces import _reverse_trace_keeping_jet_cycles
from app.postprocess.gcode.pipeline.process_traces import process_trace


def _machine(*, max_speed_mm_min: float = 1500.0) -> MachineConfig:
    return MachineConfig(
        a_mm=180.7,
        d_mm=57.59,
        gap_size_mm=15.0,
        speed_mm_min=750.0,
        max_speed_mm_min=max_speed_mm_min,
        transition_speed_mm_min=1500.0,
        retract_speed_mm_min=750.0,
    )


def _jet_on_tip_speeds(rows: np.ndarray, machine: MachineConfig) -> list[float]:
    """Tip speed on moves that run after merge has turned the jet on."""
    jet = True
    speeds: list[float] = []
    for i in range(1, len(rows)):
        disp_pivot = float(np.linalg.norm(rows[i, :3] - rows[i - 1, :3]))
        if jet and disp_pivot > 1e-3:
            tips = tip_positions_from_poses(
                rows[i - 1 : i + 1, :3],
                rows[i - 1 : i + 1, 3],
                rows[i - 1 : i + 1, 4],
                machine,
            )
            disp_tip = float(np.linalg.norm(tips[1] - tips[0]))
            speeds.append(disp_tip * float(rows[i, 5]) / disp_pivot)
        marker = int(rows[i, 6])
        if marker == 11:
            jet = False
        elif marker == 10:
            jet = True
    return speeds


def test_dispense_feed_rejects_over_cap():
    machine = _machine(max_speed_mm_min=750.0)
    assert dispense_pivot_feed(10.0, 10.0, machine) == 750.0
    assert dispense_pivot_feed(20.0, 10.0, machine) is None


def test_straight_move_prints_at_tip_speed():
    machine = _machine()
    scalp = np.array([[0.0, 0.0, 90.0], [10.0, 0.0, 90.0], [22.0, 0.0, 90.0]])
    normal = np.array([0.0, 0.2, 1.0])
    normal = normal / np.linalg.norm(normal)
    normals = np.tile(normal, (3, 1))
    rows = plan_constant_tip_dispense(
        scalp, normals, np.zeros(3), np.full(3, 30.0), machine
    )
    assert 11 not in rows[:, 6]
    speeds = _jet_on_tip_speeds(rows, machine)
    assert speeds
    assert max(abs(s - machine.speed_mm_min) for s in speeds) < 2.0


def test_large_c_step_prints_then_slews_jet_off():
    machine = _machine()
    scalp = np.array([[0.0, 0.0, 95.0], [3.0, 0.0, 95.0], [6.0, 0.0, 95.0]])
    normal = np.array([0.15, 0.0, 1.0])
    normal = normal / np.linalg.norm(normal)
    normals = np.tile(normal, (3, 1))
    c = np.array([0.0, 12.0, 12.0])
    rows = plan_constant_tip_dispense(scalp, normals, np.zeros(3), c, machine)
    assert 11 in rows[:, 6]
    assert 10 in rows[:, 6]
    speeds = _jet_on_tip_speeds(rows, machine)
    assert speeds
    assert min(speeds) > machine.speed_mm_min * 0.95
    # The 12° catch-up itself is not a dispensing move.
    jet = True
    for i in range(1, len(rows)):
        dc = abs((rows[i, 4] - rows[i - 1, 4] + 180.0) % 360.0 - 180.0)
        moved = float(np.linalg.norm(rows[i, :3] - rows[i - 1, :3])) > 1e-3
        if jet and moved:
            assert dc < 2.0
        marker = int(rows[i, 6])
        if marker == 11:
            jet = False
        elif marker == 10:
            jet = True


def test_tip_setpoint_above_pivot_cap_stays_jet_off():
    """When even a straight move needs more pivot feed than the cap, do not dispense."""
    machine = _machine(max_speed_mm_min=400.0)
    scalp = np.array([[0.0, 0.0, 90.0], [8.0, 0.0, 90.0]])
    normal = np.array([0.0, 0.3, 1.0])
    normal = normal / np.linalg.norm(normal)
    normals = np.tile(normal, (2, 1))
    rows = plan_constant_tip_dispense(
        scalp, normals, np.zeros(2), np.full(2, 20.0), machine
    )
    assert 11 in rows[:, 6]
    assert _jet_on_tip_speeds(rows, machine) == []


def test_reverse_swaps_jet_markers():
    rows = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 750.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 1500.0, 11.0],
            [1.0, 0.0, 0.0, 0.0, 20.0, 1500.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 20.0, 750.0, 10.0],
        ]
    )
    rev = _reverse_trace_keeping_jet_cycles(rows)
    assert rev[:, 6].tolist() == [11.0, 0.0, 10.0, 0.0]


def test_process_trace_crown_keeps_jet_on_tip_speed():
    machine = _machine()
    radius = 95.0
    alpha = np.deg2rad(np.linspace(30.0, 2.0, 12))
    phi = np.zeros_like(alpha)
    alpha2 = np.full(8, np.deg2rad(2.0))
    phi2 = np.deg2rad(np.linspace(0.0, 70.0, 8))
    alpha3 = np.deg2rad(np.linspace(2.0, 35.0, 12))
    phi3 = np.full_like(alpha3, np.deg2rad(70.0))
    alpha_all = np.concatenate([alpha, alpha2, alpha3])
    phi_all = np.concatenate([phi, phi2, phi3])
    pos = np.stack(
        [
            radius * np.sin(alpha_all) * np.cos(phi_all),
            radius * np.sin(alpha_all) * np.sin(phi_all),
            radius * np.cos(alpha_all),
        ],
        axis=1,
    )
    normals = pos / radius
    data = np.hstack([pos, normals])
    rows = process_trace(data, machine)
    speeds = _jet_on_tip_speeds(rows, machine)
    assert speeds
    assert min(speeds) > machine.speed_mm_min * 0.95
    assert max(speeds) < machine.speed_mm_min * 1.05
