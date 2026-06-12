"""Forward → parse → inverse roundtrip against bundle traces."""

from __future__ import annotations

import numpy as np
import pytest

from app import paths
from app.postprocess.bundle.load import load_bundle
from app.postprocess.gcode.config_loader import load_machine_config
from app.postprocess.gcode.io.write_gcode import format_gcode_lines
from app.postprocess.gcode.models import JobConfig
from app.postprocess.gcode.pipeline.align import align_subject
from app.postprocess.gcode.pipeline.process_traces import process_trace
from app.postprocess.print_config import load_physical_landmarks
from app.simulator.kinematics.inverse import gcode_to_poses, nozzle_tip_print_positions
from app.simulator.parser import parse_gcode_text


@pytest.fixture
def subject_4_bundle():
    bundle_dir = paths.REPO_ROOT / "data/output/bundles/subject_4"
    if not (bundle_dir / "manifest.json").is_file():
        pytest.skip("subject_4 bundle not present")
    return load_bundle(bundle_dir)


@pytest.fixture
def machine_config():
    return load_machine_config(paths.postprocessor_machine_config())


def test_forward_parse_inverse_roundtrip(subject_4_bundle, machine_config):
    pm_path = paths.postprocessor_subject_pm(4)
    if not pm_path.is_file():
        pytest.skip("subject_4 pm config not present")
    pm = load_physical_landmarks(pm_path)

    job = JobConfig(physical_landmarks_mm=pm)
    channels, _mesh = align_subject(subject_4_bundle, job)

    # Use first channel interconnect trace
    trace = channels[0].interconnect
    surface_pts = trace[:, :3].copy()

    gcode_rows = process_trace(trace, machine_config)
    text = "\n".join(format_gcode_lines(gcode_rows)) + "\n"
    parsed = parse_gcode_text(text)

    _cnc, nozzle, _normals = gcode_to_poses(
        parsed,
        machine_config,
        mesh_points=_mesh,
        mesh_faces=subject_4_bundle.mesh_faces,
    )

    err = np.linalg.norm(nozzle - surface_pts, axis=1)
    # G-code XYZ/B/C are rounded to 2 decimals in the postprocessor.
    assert float(np.max(err)) < 0.3, f"max roundtrip error {np.max(err):.4f} mm"


def test_nozzle_tip_print_positions_gap_standoff(subject_4_bundle, machine_config):
    pm_path = paths.postprocessor_subject_pm(4)
    if not pm_path.is_file():
        pytest.skip("subject_4 pm config not present")
    pm = load_physical_landmarks(pm_path)

    job = JobConfig(physical_landmarks_mm=pm)
    channels, mesh = align_subject(subject_4_bundle, job)
    trace = channels[0].interconnect

    gcode_rows = process_trace(trace, machine_config)
    text = "\n".join(format_gcode_lines(gcode_rows)) + "\n"
    parsed = parse_gcode_text(text)

    _cnc, surface, normals = gcode_to_poses(
        parsed,
        machine_config,
        mesh_points=mesh,
        mesh_faces=subject_4_bundle.mesh_faces,
    )
    _cnc2, tip, normals2 = nozzle_tip_print_positions(
        parsed,
        machine_config,
        mesh_points=mesh,
        mesh_faces=subject_4_bundle.mesh_faces,
    )

    standoff = np.linalg.norm(tip - surface, axis=1)
    np.testing.assert_allclose(
        standoff,
        machine_config.gap_size_mm,
        atol=1e-6,
    )
    head_center = np.mean(mesh, axis=0)
    radial = surface - head_center
    radial_u = radial / np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-12)
    assert float(np.min(np.sum(normals2 * radial_u, axis=1))) > 0.0


def test_undo_machine_zero_inverse(machine_config):
    from app.postprocess.gcode.kinematics.machine_zero import apply_machine_zero_offset
    from app.simulator.kinematics.inverse import undo_machine_zero_offset

    pts = np.array([[10.0, 20.0, -30.0], [0.0, 5.0, 100.0]])
    shifted = apply_machine_zero_offset(pts, machine_config)
    restored = undo_machine_zero_offset(shifted, machine_config)
    np.testing.assert_allclose(restored, pts, atol=1e-9)
