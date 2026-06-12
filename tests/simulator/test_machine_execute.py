"""Rigid machine kinematics tests."""

from __future__ import annotations

import numpy as np
import pytest

from app import paths
from app.postprocess.gcode.config_loader import load_machine_config
from app.postprocess.gcode.io.write_gcode import format_gcode_lines
from app.postprocess.gcode.kinematics.machine_fk import (
    structural_arm_joints_batch,
    structural_tip_offset_batch,
)
from app.postprocess.gcode.models import JobConfig
from app.postprocess.gcode.pipeline.align import align_subject
from app.postprocess.gcode.pipeline.process_traces import process_trace
from app.postprocess.print_config import load_physical_landmarks
from app.simulator.kinematics.machine_execute import (
    RigidMachineState,
    forward_states_from_gcode,
    rigid_geometry_checks,
)
from app.simulator.kinematics.inverse import nozzle_tip_print_positions
from app.simulator.parser import parse_gcode_text


@pytest.fixture
def subject_4_bundle():
    bundle_dir = paths.REPO_ROOT / "data/output/bundles/subject_4"
    if not (bundle_dir / "manifest.json").is_file():
        pytest.skip("subject_4 bundle not present")
    from app.postprocess.bundle.load import load_bundle

    return load_bundle(bundle_dir)


@pytest.fixture
def machine_config():
    return load_machine_config(paths.postprocessor_machine_config())


def test_rigid_arm_fixed_lengths_and_perpendicular(subject_4_bundle, machine_config):
    pm_path = paths.postprocessor_subject_pm(4)
    if not pm_path.is_file():
        pytest.skip("subject_4 pm config not present")
    pm = load_physical_landmarks(pm_path)

    job = JobConfig(physical_landmarks_mm=pm)
    channels, _mesh = align_subject(subject_4_bundle, job)
    rows = process_trace(channels[0].interconnect, machine_config)
    parsed = parse_gcode_text("\n".join(format_gcode_lines(rows)) + "\n")

    _cnc, b_pivots, tips, states = forward_states_from_gcode(
        parsed,
        a_mm=machine_config.a_mm,
        d_mm=machine_config.d_mm,
    )

    checks = rigid_geometry_checks(
        states,
        a_mm=machine_config.a_mm,
        d_mm=machine_config.d_mm,
        b0_deg=machine_config.b0_deg,
        c0_deg=machine_config.c0_deg,
    )
    assert checks["arm_length_max_err"] < 1e-6
    assert checks["tool_length_max_err"] < 1e-6
    assert checks["perp_dot_max"] < 1e-6

    for s in states:
        arm = s.b_pivot - s.c_center
        tool = s.tip - s.b_pivot
        np.testing.assert_allclose(np.linalg.norm(arm), machine_config.a_mm, atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(tool), machine_config.d_mm, atol=1e-6)
        assert abs(np.dot(arm, tool)) < 1e-6


def test_print_aligned_rigid_tip_matches_inverse(subject_4_bundle, machine_config):
    pm_path = paths.postprocessor_subject_pm(4)
    if not pm_path.is_file():
        pytest.skip("subject_4 pm config not present")
    pm = load_physical_landmarks(pm_path)

    job = JobConfig(physical_landmarks_mm=pm)
    channels, mesh = align_subject(subject_4_bundle, job)
    rows = process_trace(channels[0].interconnect, machine_config)
    parsed = parse_gcode_text("\n".join(format_gcode_lines(rows)) + "\n")
    faces = subject_4_bundle.mesh_faces

    _prog, print_tips, _normals = nozzle_tip_print_positions(
        parsed,
        machine_config,
        mesh_points=mesh,
        mesh_faces=faces,
    )
    a_mm, d_mm = machine_config.a_mm, machine_config.d_mm
    offsets = structural_tip_offset_batch(parsed[:, 3], parsed[:, 4], a_mm, d_mm)
    c_centers = print_tips - offsets
    c_centers, b_pivots, tips = structural_arm_joints_batch(
        c_centers, parsed[:, 3], parsed[:, 4], a_mm, d_mm
    )
    states = [
        RigidMachineState(
            c_centers[i], b_pivots[i], tips[i], float(parsed[i, 3]), float(parsed[i, 4])
        )
        for i in range(parsed.shape[0])
    ]

    np.testing.assert_allclose(tips, print_tips, atol=1e-6)
    checks = rigid_geometry_checks(
        states,
        a_mm=a_mm,
        d_mm=d_mm,
        b0_deg=machine_config.b0_deg,
        c0_deg=machine_config.c0_deg,
    )
    assert checks["arm_length_max_err"] < 1e-6
    assert checks["tool_length_max_err"] < 1e-6

    _cnc_raw, _bp_raw, tips_raw, _ = forward_states_from_gcode(
        parsed,
        a_mm=a_mm,
        d_mm=d_mm,
    )
    assert np.median(np.linalg.norm(tips_raw - print_tips, axis=1)) > 50.0


def test_forward_map_uses_programmed_xyz_as_c_pivot(machine_config):
    """Forward map: C pivot equals G-code X,Y,Z."""
    gcode = np.array([[5.0, -3.0, 12.0, 0.0, 90.0, 50.0, 0.0]])
    c_path, _, _, states = forward_states_from_gcode(
        gcode,
        a_mm=machine_config.a_mm,
        d_mm=machine_config.d_mm,
    )
    np.testing.assert_allclose(c_path[0], gcode[0, :3])
    np.testing.assert_allclose(states[0].c_center, gcode[0, :3])
