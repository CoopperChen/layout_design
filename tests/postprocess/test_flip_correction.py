"""C-axis flip correction during trace processing."""

from __future__ import annotations

import numpy as np
import pytest

from app import paths
from app.postprocess.bundle.load import load_bundle
from app.postprocess.gcode.config_loader import load_machine_config
from app.postprocess.gcode.kinematics.axis_angles import compute_axis_angles
from app.postprocess.gcode.kinematics.flip_correction import (
    correct_flip,
    enforce_axis_continuity,
    limit_c_slew,
    max_c_step_deg,
    retarget_normals_to_c,
    validate_axis_continuity,
    walk_pinned_crown_heading,
)
from app.postprocess.gcode.kinematics.machine_fk import registration_to_machine_frame
from app.postprocess.gcode.models import JobConfig
from app.postprocess.gcode.pipeline.align import align_subject
from app.postprocess.gcode.pipeline.process_traces import process_trace
from app.postprocess.print_config import load_physical_landmarks


def _c_delta_deg(c1: float, c2: float) -> float:
    delta = abs(float(c1) - float(c2)) % 360.0
    return min(delta, 360.0 - delta)


def test_correct_flip_negates_prefix_on_sign_crossing():
    b = np.array([10.0, -11.0, 9.0])
    c = np.array([-71.0, -73.0, 86.0])
    b_out, c_out = correct_flip(b, c)
    assert np.allclose(c_out[:2], -c[:2])
    assert np.allclose(b_out[:2], -b[:2])
    assert c_out[2] == c[2]
    assert _c_delta_deg(c_out[1], c_out[2]) < 20.0


def test_correct_flip_handles_sign_crossing_at_trace_start():
    b = np.array([78.72, -77.61, -77.34])
    c = np.array([-88.37, 89.51, 84.63])
    b_out, c_out = correct_flip(b, c)
    assert np.allclose(c_out[0], -c[0])
    assert np.allclose(b_out[0], -b[0])
    assert _c_delta_deg(c_out[0], c_out[1]) < 5.0


def test_enforce_axis_continuity_picks_equivalent_branch_at_trace_start():
    b = np.array([78.72, -77.61, -77.34])
    c = np.array([-88.37, 89.51, 84.63])
    b_out, c_out = enforce_axis_continuity(b, c)
    assert _c_delta_deg(c_out[0], c_out[1]) < 5.0
    assert max_c_step_deg(c_out) < 5.0


def test_validate_axis_continuity_raises_on_large_jump():
    b = np.array([0.0, 0.0])
    c = np.array([-80.0, 80.0])
    with pytest.raises(ValueError, match="C-axis step"):
        validate_axis_continuity(b, c, max_c_step_deg=45.0)


def test_walk_pinned_crown_heading_takes_small_steps_toward_exit():
    """A 12° crown staircase becomes a 1° walk toward the settled heading."""
    b = np.array([40, 30, 18, 15, 13, 11, 9, 7, 5, 3, 3, 4, 6, 8], dtype=float)
    c = np.array(
        [-15, -12, -9, -3, 9, 21, 33, 45, 57, 69, 57, 49, 43, 42],
        dtype=float,
    )
    _, c_out, catchups = walk_pinned_crown_heading(
        b, c, max_step_deg=12.0, b_upright_deg=20.0, step_deg=1.0
    )
    assert catchups == []
    assert c_out[2] == -9.0
    assert c_out[-1] == pytest.approx(-9.0 + 11.0)
    walked = [_c_delta_deg(c_out[i - 1], c_out[i]) for i in range(3, len(c_out))]
    assert max(walked) <= 1.0 + 1e-9


def test_retarget_normals_to_c_keeps_tilt_and_matches_c():
    normal = np.array([[0.20, 0.10, 0.97]], dtype=float)
    normal /= np.linalg.norm(normal)
    b, c = compute_axis_angles(normal)
    commanded = np.array([float(c[0]) + 8.0])
    out = retarget_normals_to_c(normal, commanded, np.array([True]))
    b2, c2 = compute_axis_angles(out)
    assert b2[0] == pytest.approx(b[0])
    assert _c_delta_deg(c2[0], commanded[0]) < 1e-6


def test_limit_c_slew_caps_steps():
    b = np.zeros(4)
    c = np.array([0.0, 40.0, 80.0, 120.0])
    _, c_out = limit_c_slew(b, c, max_step_deg=12.0)
    for i in range(1, len(c_out)):
        assert _c_delta_deg(c_out[i - 1], c_out[i]) <= 12.0 + 1e-9


def test_process_trace_applies_correct_flip_on_subject_4():
    bundle_dir = paths.REPO_ROOT / "data/output/bundles/subject_4"
    if not (bundle_dir / "manifest.json").is_file():
        import pytest

        pytest.skip("subject_4 bundle not present")

    bundle = load_bundle(bundle_dir)
    pm_path = paths.postprocessor_subject_pm(4)
    if not pm_path.is_file():
        import pytest

        pytest.skip("subject_4 pm config not present")

    pm = load_physical_landmarks(pm_path)
    machine = load_machine_config(paths.postprocessor_machine_config())
    job = JobConfig(physical_landmarks_mm=pm, subject="4", trace_type="interconnect")
    channels, mesh = align_subject(bundle, job)
    mesh_m = registration_to_machine_frame(
        mesh,
        a_mm=machine.a_mm,
        d_mm=machine.d_mm,
        calgap_z_mm=machine.calgap_z_mm,
    )

    fz = next(ch for ch in channels if ch.name == "Fz")
    rows = process_trace(
        fz.interconnect,
        machine,
        mesh_points=mesh_m,
        mesh_faces=bundle.mesh_faces,
    )
    jet = True
    steps: list[float] = []
    for i in range(1, len(rows)):
        if jet:
            steps.append(_c_delta_deg(rows[i - 1, 4], rows[i, 4]))
        marker = int(rows[i, 6])
        if marker == 11:
            jet = False
        elif marker == 10:
            jet = True
    assert steps
    assert max(steps) < 45.0


def test_process_trace_o2_no_start_c_jump_on_subject_5():
    bundle_dir = paths.REPO_ROOT / "data/output/bundles/subject_5"
    if not (bundle_dir / "manifest.json").is_file():
        import pytest

        pytest.skip("subject_5 bundle not present")

    bundle = load_bundle(bundle_dir)
    pm_path = paths.postprocessor_subject_pm(5)
    if not pm_path.is_file():
        import pytest

        pytest.skip("subject_5 pm config not present")

    pm = load_physical_landmarks(pm_path)
    machine = load_machine_config(paths.postprocessor_machine_config())
    job = JobConfig(physical_landmarks_mm=pm, subject="5", trace_type="interconnect")
    channels, mesh = align_subject(bundle, job)
    mesh_m = registration_to_machine_frame(
        mesh,
        a_mm=machine.a_mm,
        d_mm=machine.d_mm,
        calgap_z_mm=machine.calgap_z_mm,
    )

    o2 = next(ch for ch in channels if ch.name == "O2")
    rows = process_trace(
        o2.interconnect,
        machine,
        mesh_points=mesh_m,
        mesh_faces=bundle.mesh_faces,
    )
    assert _c_delta_deg(rows[0, 4], rows[1, 4]) < 5.0


def test_process_trace_slew_limits_c_on_subject_5_fz():
    """Fz crown can demand large C changes; slew limit caps printer steps."""
    bundle_dir = paths.REPO_ROOT / "data/output/bundles/subject_5"
    if not (bundle_dir / "manifest.json").is_file():
        pytest.skip("subject_5 bundle not present")

    bundle = load_bundle(bundle_dir)
    pm_path = paths.postprocessor_subject_pm(5)
    if not pm_path.is_file():
        pytest.skip("subject_5 pm config not present")

    pm = load_physical_landmarks(pm_path)
    machine = load_machine_config(paths.postprocessor_machine_config())
    job = JobConfig(physical_landmarks_mm=pm, subject="5", trace_type="interconnect")
    channels, mesh = align_subject(bundle, job)
    mesh_m = registration_to_machine_frame(
        mesh,
        a_mm=machine.a_mm,
        d_mm=machine.d_mm,
        calgap_z_mm=machine.calgap_z_mm,
    )

    fz = next(ch for ch in channels if ch.name == "Fz")
    rows = process_trace(
        fz.interconnect,
        machine,
        mesh_points=mesh_m,
        mesh_faces=bundle.mesh_faces,
        channel_name="Fz",
    )
    jet = True
    steps: list[float] = []
    for i in range(1, len(rows)):
        if jet:
            steps.append(_c_delta_deg(rows[i - 1, 4], rows[i, 4]))
        marker = int(rows[i, 6])
        if marker == 11:
            jet = False
        elif marker == 10:
            jet = True
    assert steps
    assert max(steps) <= 12.0 + 1e-6
