"""C-axis flip correction during trace processing."""

from __future__ import annotations

import numpy as np
import pytest

from app import paths
from app.postprocess.bundle.load import load_bundle
from app.postprocess.gcode.config_loader import load_machine_config
from app.postprocess.gcode.kinematics.flip_correction import (
    correct_flip,
    enforce_axis_continuity,
    limit_c_slew,
    max_c_step_deg,
    validate_axis_continuity,
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
    max_step = max(_c_delta_deg(rows[i, 4], rows[i - 1, 4]) for i in range(1, len(rows)))
    assert max_step < 45.0


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
    max_step = max(_c_delta_deg(rows[i, 4], rows[i - 1, 4]) for i in range(1, len(rows)))
    assert max_step <= 12.0 + 1e-6
