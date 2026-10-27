"""Engage clearance offset for upward-nozzle trace starts."""

from __future__ import annotations

import numpy as np
import pytest

from app import paths
from app.gcode.config_loader import load_machine_config
from app.gcode.kinematics.engage_clearance import (
    EngageClearanceConfig,
    build_disengage_offset_rows,
    build_engage_offset_rows,
    compute_disengage_xy_offset,
    compute_engage_xy_offset,
)


class _TipDistanceChecker:
    """Return a fixed signed distance for every tip query."""

    def __init__(self, min_sd: float) -> None:
        self._min_sd = float(min_sd)

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=float)
        if pts.ndim == 1:
            pts = pts.reshape(1, 3)
        return np.full(pts.shape[0], self._min_sd, dtype=float)


def test_compute_engage_xy_offset_zero_when_b_below_threshold():
    machine = load_machine_config(paths.machine_config())
    config = EngageClearanceConfig()
    row = np.array([10.0, -20.0, -180.0, 45.0, 10.0, 500.0])
    off = compute_engage_xy_offset(
        row, 50.0, np.zeros(2), None, machine, config
    )
    assert np.allclose(off, 0.0)


def test_compute_engage_xy_offset_zero_when_z_descent_already_clear():
    machine = load_machine_config(paths.machine_config())
    config = EngageClearanceConfig(descent_min_clearance_mm=20.0)
    row = np.array([10.0, -20.0, -180.0, 105.0, 8.0, 500.0])
    checker = _TipDistanceChecker(22.0)
    off = compute_engage_xy_offset(
        row, 50.0, np.zeros(2), checker, machine, config
    )
    assert np.allclose(off, 0.0)


def test_compute_engage_xy_offset_uses_default_when_needed_and_no_mesh():
    machine = load_machine_config(paths.machine_config())
    config = EngageClearanceConfig(default_offset_mm=20.0)
    row = np.array([10.0, -20.0, -180.0, 105.0, 8.0, 500.0])
    head_center = np.array([0.0, 0.0])
    off = compute_engage_xy_offset(
        row, 50.0, head_center, None, machine, config
    )
    assert np.linalg.norm(off) == pytest.approx(20.0)
    outward = row[:2] - head_center
    outward /= np.linalg.norm(outward)
    assert np.allclose(off / np.linalg.norm(off), outward)


def test_compute_engage_xy_offset_finds_minimum_for_z_descent_clearance(
    monkeypatch: pytest.MonkeyPatch,
):
    machine = load_machine_config(paths.machine_config())
    config = EngageClearanceConfig(
        default_offset_mm=20.0,
        step_mm=5.0,
        descent_min_clearance_mm=20.0,
    )
    row = np.array([10.0, 20.0, -180.0, 105.0, 8.0, 500.0])
    head_center = np.array([0.0, 0.0])

    def _fake_descent(
        _target,
        _b,
        _c,
        _zsafe,
        offset_xy,
        _checker,
        _machine,
        *,
        samples_per_segment=20,
    ) -> float:
        del samples_per_segment
        dist = float(np.linalg.norm(offset_xy))
        if dist < 1e-9:
            return 12.0
        if dist < 10.0:
            return 18.0
        return 22.0

    monkeypatch.setattr(
        "app.gcode.kinematics.engage_clearance.min_tip_clearance_during_z_descent",
        _fake_descent,
    )
    monkeypatch.setattr(
        "app.gcode.kinematics.engage_clearance.min_tip_clearance_during_xy_slide",
        lambda *a, **k: float("inf"),
    )
    off = compute_engage_xy_offset(
        row,
        50.0,
        head_center,
        object(),
        machine,
        config,
    )
    assert np.linalg.norm(off) == pytest.approx(10.0)


def test_build_engage_offset_rows_empty_when_no_offset():
    machine = load_machine_config(paths.machine_config())
    row = np.array([1.0, 2.0, -3.0, 100.0, 5.0, 500.0])
    out = build_engage_offset_rows(row, 40.0, np.zeros(2), machine)
    assert out.shape == (0, 7)


def test_build_engage_offset_rows_three_leg_path():
    machine = load_machine_config(paths.machine_config())
    row = np.array([10.0, -20.0, -180.0, 105.0, 8.0, 500.0])
    out = build_engage_offset_rows(row, 50.0, np.array([0.0, 20.0]), machine)
    assert out.shape == (3, 7)
    assert out[0, 2] == pytest.approx(50.0)
    assert out[1, 0] == pytest.approx(10.0)
    assert out[1, 1] == pytest.approx(0.0)
    assert out[1, 2] == pytest.approx(-180.0)
    assert out[2, 0] == pytest.approx(10.0)
    assert out[2, 1] == pytest.approx(-20.0)
    assert out[0, 5] == pytest.approx(machine.transition_speed_mm_min)
    assert out[1, 5] == pytest.approx(machine.retract_speed_mm_min)
    assert out[2, 5] == pytest.approx(machine.transition_speed_mm_min)


def test_build_disengage_offset_rows_empty_when_no_offset():
    machine = load_machine_config(paths.machine_config())
    row = np.array([1.0, 2.0, -3.0, 100.0, 5.0, 500.0])
    out = build_disengage_offset_rows(row, 40.0, np.zeros(2), machine)
    assert out.shape == (0, 7)


def test_build_disengage_offset_rows_two_leg_path():
    machine = load_machine_config(paths.machine_config())
    row = np.array([10.0, -20.0, -180.0, 105.0, 8.0, 500.0])
    out = build_disengage_offset_rows(row, 50.0, np.array([0.0, 20.0]), machine)
    assert out.shape == (2, 7)
    assert out[0, 0] == pytest.approx(10.0)
    assert out[0, 1] == pytest.approx(0.0)
    assert out[0, 2] == pytest.approx(-180.0)
    assert out[1, 0] == pytest.approx(10.0)
    assert out[1, 1] == pytest.approx(0.0)
    assert out[1, 2] == pytest.approx(50.0)
    assert out[0, 5] == pytest.approx(machine.transition_speed_mm_min)
    assert out[1, 5] == pytest.approx(machine.retract_speed_mm_min)


def test_compute_disengage_xy_offset_zero_when_z_ascent_already_clear():
    machine = load_machine_config(paths.machine_config())
    config = EngageClearanceConfig(descent_min_clearance_mm=20.0)
    row = np.array([10.0, -20.0, -180.0, 105.0, 8.0, 500.0])
    checker = _TipDistanceChecker(22.0)
    off = compute_disengage_xy_offset(
        row, 50.0, np.zeros(2), checker, machine, config
    )
    assert np.allclose(off, 0.0)


def test_compute_disengage_xy_offset_finds_minimum_for_z_ascent_clearance(
    monkeypatch: pytest.MonkeyPatch,
):
    machine = load_machine_config(paths.machine_config())
    config = EngageClearanceConfig(
        default_offset_mm=20.0,
        step_mm=5.0,
        descent_min_clearance_mm=20.0,
    )
    row = np.array([10.0, 20.0, -180.0, 105.0, 8.0, 500.0])
    head_center = np.array([0.0, 0.0])

    def _fake_ascent(
        _target,
        _b,
        _c,
        _zsafe,
        offset_xy,
        _checker,
        _machine,
        *,
        samples_per_segment=20,
    ) -> float:
        del samples_per_segment
        dist = float(np.linalg.norm(offset_xy))
        if dist < 1e-9:
            return 12.0
        if dist < 10.0:
            return 18.0
        return 22.0

    monkeypatch.setattr(
        "app.gcode.kinematics.engage_clearance.min_tip_clearance_during_z_ascent",
        _fake_ascent,
    )
    monkeypatch.setattr(
        "app.gcode.kinematics.engage_clearance.min_tip_clearance_during_xy_slide",
        lambda *a, **k: float("inf"),
    )
    off = compute_disengage_xy_offset(
        row,
        50.0,
        head_center,
        object(),
        machine,
        config,
    )
    assert np.linalg.norm(off) == pytest.approx(10.0)


