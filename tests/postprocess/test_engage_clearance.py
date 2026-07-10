"""Engage clearance offset for upward-nozzle trace starts."""

from __future__ import annotations

import numpy as np
import pytest

from app import paths
from app.postprocess.bundle.load import load_bundle
from app.postprocess.gcode.config_loader import load_machine_config
from app.postprocess.gcode.kinematics.arm_clearance import HeadMeshInsideChecker
from app.postprocess.gcode.kinematics.engage_clearance import (
    EngageClearanceConfig,
    build_disengage_offset_rows,
    build_engage_offset_rows,
    compute_disengage_xy_offset,
    compute_engage_xy_offset,
    head_center_xy_from_mesh,
    min_tip_clearance_during_z_ascent,
    min_tip_clearance_during_z_descent,
)
from app.postprocess.gcode.kinematics.machine_fk import registration_to_machine_frame
from app.postprocess.gcode.models import JobConfig
from app.postprocess.gcode.pipeline.align import align_subject
from app.postprocess.gcode.pipeline.merge_traces import merge_traces
from app.postprocess.print_config import load_physical_landmarks


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
    machine = load_machine_config(paths.postprocessor_machine_config())
    config = EngageClearanceConfig()
    row = np.array([10.0, -20.0, -180.0, 45.0, 10.0, 500.0])
    off = compute_engage_xy_offset(
        row, 50.0, np.zeros(2), None, machine, config
    )
    assert np.allclose(off, 0.0)


def test_compute_engage_xy_offset_zero_when_z_descent_already_clear():
    machine = load_machine_config(paths.postprocessor_machine_config())
    config = EngageClearanceConfig(descent_min_clearance_mm=20.0)
    row = np.array([10.0, -20.0, -180.0, 105.0, 8.0, 500.0])
    checker = _TipDistanceChecker(22.0)
    off = compute_engage_xy_offset(
        row, 50.0, np.zeros(2), checker, machine, config
    )
    assert np.allclose(off, 0.0)


def test_compute_engage_xy_offset_uses_default_when_needed_and_no_mesh():
    machine = load_machine_config(paths.postprocessor_machine_config())
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
    machine = load_machine_config(paths.postprocessor_machine_config())
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
        "app.postprocess.gcode.kinematics.engage_clearance.min_tip_clearance_during_z_descent",
        _fake_descent,
    )
    monkeypatch.setattr(
        "app.postprocess.gcode.kinematics.engage_clearance.min_tip_clearance_during_xy_slide",
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
    machine = load_machine_config(paths.postprocessor_machine_config())
    row = np.array([1.0, 2.0, -3.0, 100.0, 5.0, 500.0])
    out = build_engage_offset_rows(row, 40.0, np.zeros(2), machine)
    assert out.shape == (0, 7)


def test_build_engage_offset_rows_three_leg_path():
    machine = load_machine_config(paths.postprocessor_machine_config())
    row = np.array([10.0, -20.0, -180.0, 105.0, 8.0, 500.0])
    out = build_engage_offset_rows(row, 50.0, np.array([0.0, 20.0]), machine)
    assert out.shape == (3, 7)
    assert out[0, 2] == pytest.approx(50.0)
    assert out[1, 0] == pytest.approx(10.0)
    assert out[1, 1] == pytest.approx(0.0)
    assert out[1, 2] == pytest.approx(-180.0)
    assert out[2, 0] == pytest.approx(10.0)
    assert out[2, 1] == pytest.approx(-20.0)
    assert out[0, 5] == pytest.approx(0.5 * machine.transition_speed_mm_min)
    assert out[1, 5] == pytest.approx(0.5 * machine.transition_speed_mm_min)
    assert out[2, 5] == pytest.approx(0.5 * machine.transition_speed_mm_min)


def test_build_disengage_offset_rows_empty_when_no_offset():
    machine = load_machine_config(paths.postprocessor_machine_config())
    row = np.array([1.0, 2.0, -3.0, 100.0, 5.0, 500.0])
    out = build_disengage_offset_rows(row, 40.0, np.zeros(2), machine)
    assert out.shape == (0, 7)


def test_build_disengage_offset_rows_two_leg_path():
    machine = load_machine_config(paths.postprocessor_machine_config())
    row = np.array([10.0, -20.0, -180.0, 105.0, 8.0, 500.0])
    out = build_disengage_offset_rows(row, 50.0, np.array([0.0, 20.0]), machine)
    assert out.shape == (2, 7)
    assert out[0, 0] == pytest.approx(10.0)
    assert out[0, 1] == pytest.approx(0.0)
    assert out[0, 2] == pytest.approx(-180.0)
    assert out[1, 0] == pytest.approx(10.0)
    assert out[1, 1] == pytest.approx(0.0)
    assert out[1, 2] == pytest.approx(50.0)
    assert out[0, 5] == pytest.approx(0.5 * machine.transition_speed_mm_min)
    assert out[1, 5] == pytest.approx(0.5 * machine.transition_speed_mm_min)


def test_compute_disengage_xy_offset_zero_when_z_ascent_already_clear():
    machine = load_machine_config(paths.postprocessor_machine_config())
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
    machine = load_machine_config(paths.postprocessor_machine_config())
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
        "app.postprocess.gcode.kinematics.engage_clearance.min_tip_clearance_during_z_ascent",
        _fake_ascent,
    )
    monkeypatch.setattr(
        "app.postprocess.gcode.kinematics.engage_clearance.min_tip_clearance_during_xy_slide",
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


def test_merge_traces_adds_disengage_suffix_for_crown_exit():
    bundle_dir = paths.REPO_ROOT / "data/output/bundles/subject_4"
    if not (bundle_dir / "manifest.json").is_file():
        pytest.skip("subject_4 bundle not present")
    pm_path = paths.postprocessor_subject_pm(4)
    if not pm_path.is_file():
        pytest.skip("subject_4 pm config not present")

    bundle = load_bundle(bundle_dir)
    pm = load_physical_landmarks(pm_path)
    machine = load_machine_config(paths.postprocessor_machine_config())
    channels, mesh = align_subject(bundle, JobConfig(physical_landmarks_mm=pm))
    mesh_machine = registration_to_machine_frame(
        mesh,
        a_mm=machine.a_mm,
        d_mm=machine.d_mm,
        calgap_z_mm=machine.calgap_z_mm,
    )
    checker = HeadMeshInsideChecker(mesh_machine, bundle.mesh_faces)
    config = EngageClearanceConfig(descent_min_clearance_mm=20.0)
    head_xy = head_center_xy_from_mesh(mesh_machine)
    zsafe = round(float(np.max(mesh[:, 2])) + machine.zsafe_margin_mm)

    t7 = next(ch for ch in channels if ch.name == "T7")
    from app.postprocess.gcode.pipeline.process_traces import process_trace

    rows = process_trace(
        t7.interconnect,
        machine,
        mesh_points=mesh_machine,
        mesh_faces=bundle.mesh_faces,
    )
    exit_row = rows[-1]
    offset_xy = compute_disengage_xy_offset(
        exit_row, zsafe, head_xy, checker, machine, config
    )
    assert np.linalg.norm(offset_xy) > 1e-9

    assert min_tip_clearance_during_z_ascent(
        exit_row[:3],
        float(exit_row[3]),
        float(exit_row[4]),
        zsafe,
        offset_xy,
        checker,
        machine,
    ) >= 20.0

    merged = merge_traces(
        [rows],
        float(np.max(mesh[:, 2])),
        machine,
        0,
        mesh_points=mesh_machine,
        mesh_faces=bundle.mesh_faces,
    )
    x, y, z = exit_row[0], exit_row[1], exit_row[2]
    dx, dy = offset_xy
    has_slide = any(
        abs(merged[i, 0] - (x + dx)) < 0.2
        and abs(merged[i, 1] - (y + dy)) < 0.2
        and abs(merged[i, 2] - z) < 0.2
        for i in range(len(merged))
    )
    has_ascend = any(
        abs(merged[i, 0] - (x + dx)) < 0.2
        and abs(merged[i, 1] - (y + dy)) < 0.2
        and abs(merged[i, 2] - zsafe) < 0.2
        for i in range(len(merged))
    )
    assert has_slide and has_ascend


def test_merge_traces_adds_offset_prefix_for_crown_engage():
    bundle_dir = paths.REPO_ROOT / "data/output/bundles/subject_4"
    if not (bundle_dir / "manifest.json").is_file():
        pytest.skip("subject_4 bundle not present")
    pm_path = paths.postprocessor_subject_pm(4)
    if not pm_path.is_file():
        pytest.skip("subject_4 pm config not present")

    bundle = load_bundle(bundle_dir)
    pm = load_physical_landmarks(pm_path)
    machine = load_machine_config(paths.postprocessor_machine_config())
    channels, mesh = align_subject(bundle, JobConfig(physical_landmarks_mm=pm))
    mesh_machine = registration_to_machine_frame(
        mesh,
        a_mm=machine.a_mm,
        d_mm=machine.d_mm,
        calgap_z_mm=machine.calgap_z_mm,
    )
    checker = HeadMeshInsideChecker(mesh_machine, bundle.mesh_faces)
    config = EngageClearanceConfig(descent_min_clearance_mm=20.0)
    head_xy = head_center_xy_from_mesh(mesh_machine)
    zsafe = round(float(np.max(mesh[:, 2])) + machine.zsafe_margin_mm)

    t7 = next(ch for ch in channels if ch.name == "T7")
    from app.postprocess.gcode.pipeline.process_traces import process_trace

    rows = process_trace(
        t7.interconnect,
        machine,
        mesh_points=mesh_machine,
        mesh_faces=bundle.mesh_faces,
    )
    engage = rows[0]
    offset_xy = compute_engage_xy_offset(
        engage, zsafe, head_xy, checker, machine, config
    )
    assert np.linalg.norm(offset_xy) > 1e-9

    descent_clear = min_tip_clearance_during_z_descent(
        engage[:3],
        float(engage[3]),
        float(engage[4]),
        zsafe,
        offset_xy,
        checker,
        machine,
    )
    assert descent_clear >= 20.0

    merged = merge_traces(
        [rows],
        float(np.max(mesh[:, 2])),
        machine,
        0,
        mesh_points=mesh_machine,
        mesh_faces=bundle.mesh_faces,
    )
    x, y = engage[0], engage[1]
    dx, dy = offset_xy
    has_prefix = any(
        abs(merged[i, 0] - (x + dx)) < 0.2
        and abs(merged[i, 1] - (y + dy)) < 0.2
        and abs(merged[i, 2] - zsafe) < 0.2
        for i in range(len(merged))
    )
    assert has_prefix


def test_merge_traces_skips_offset_when_z_descent_already_clear():
    bundle_dir = paths.REPO_ROOT / "data/output/bundles/subject_4"
    if not (bundle_dir / "manifest.json").is_file():
        pytest.skip("subject_4 bundle not present")
    pm_path = paths.postprocessor_subject_pm(4)
    if not pm_path.is_file():
        pytest.skip("subject_4 pm config not present")

    bundle = load_bundle(bundle_dir)
    pm = load_physical_landmarks(pm_path)
    machine = load_machine_config(paths.postprocessor_machine_config())
    channels, mesh = align_subject(bundle, JobConfig(physical_landmarks_mm=pm))
    mesh_machine = registration_to_machine_frame(
        mesh,
        a_mm=machine.a_mm,
        d_mm=machine.d_mm,
        calgap_z_mm=machine.calgap_z_mm,
    )
    checker = HeadMeshInsideChecker(mesh_machine, bundle.mesh_faces)
    config = EngageClearanceConfig(descent_min_clearance_mm=15.0)
    head_xy = head_center_xy_from_mesh(mesh_machine)
    zsafe = round(float(np.max(mesh[:, 2])) + machine.zsafe_margin_mm)

    t8 = next(ch for ch in channels if ch.name == "T8")
    from app.postprocess.gcode.pipeline.process_traces import process_trace

    rows = process_trace(
        t8.interconnect,
        machine,
        mesh_points=mesh_machine,
        mesh_faces=bundle.mesh_faces,
    )
    engage = rows[0]
    assert abs(engage[3]) > 90
    offset_xy = compute_engage_xy_offset(
        engage, zsafe, head_xy, checker, machine, config
    )
    assert np.linalg.norm(offset_xy) < 1e-9
    direct_descent = min_tip_clearance_during_z_descent(
        engage[:3],
        float(engage[3]),
        float(engage[4]),
        zsafe,
        np.zeros(2),
        checker,
        machine,
    )
    assert direct_descent >= 15.0


def test_handoff_feeds_and_offset_leg_clearance():
    bundle_dir = paths.REPO_ROOT / "data/output/bundles/subject_4"
    if not (bundle_dir / "manifest.json").is_file():
        pytest.skip("subject_4 bundle not present")
    pm_path = paths.postprocessor_subject_pm(4)
    if not pm_path.is_file():
        pytest.skip("subject_4 pm config not present")

    bundle = load_bundle(bundle_dir)
    pm = load_physical_landmarks(pm_path)
    machine = load_machine_config(paths.postprocessor_machine_config())
    channels, mesh = align_subject(bundle, JobConfig(physical_landmarks_mm=pm))
    mesh_machine = registration_to_machine_frame(
        mesh,
        a_mm=machine.a_mm,
        d_mm=machine.d_mm,
        calgap_z_mm=machine.calgap_z_mm,
    )
    from app.postprocess.gcode.pipeline.process_traces import process_all_traces

    gcode_list = process_all_traces(
        channels, machine, 1, 0, mesh_points=mesh_machine, mesh_faces=bundle.mesh_faces
    )
    merged = merge_traces(
        gcode_list[:5],
        float(np.max(mesh[:, 2])),
        machine,
        0,
        mesh_points=mesh_machine,
        mesh_faces=bundle.mesh_faces,
    )
    trans = machine.transition_speed_mm_min
    half = 0.5 * trans

    for i in range(1, len(merged)):
        prev, cur = merged[i - 1], merged[i]
        if np.linalg.norm(cur[:3] - prev[:3]) < 0.5:
            continue
        if prev[6] == 11 or cur[6] == 10:
            assert cur[5] in (trans, half)

    for i in range(len(merged)):
        if merged[i, 6] != 10:
            continue
        if i + 1 < len(merged) and merged[i + 1, 6] == 0:
            same_pose = np.allclose(merged[i, :3], merged[i + 1, :3], atol=0.2)
            if same_pose:
                assert merged[i, 5] == merged[i + 1, 5]
