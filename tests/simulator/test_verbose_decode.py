"""Postprocessor decode uses landmark frame; verbose metrics use machine frame."""

from __future__ import annotations

import numpy as np
import pytest

from app import paths
from app.postprocess.bundle.load import load_bundle
from app.postprocess.gcode.config_loader import load_machine_config
from app.postprocess.gcode.io.write_gcode import format_gcode_lines
from app.postprocess.gcode.kinematics.machine_fk import registration_to_machine_frame
from app.postprocess.gcode.models import JobConfig
from app.postprocess.gcode.pipeline.align import align_subject
from app.postprocess.gcode.pipeline.process_traces import process_trace
from app.postprocess.print_config import load_physical_landmarks
from app.simulator.kinematics.inverse import (
    decode_postprocessor_paths,
    gcode_to_poses,
)
from app.simulator.parser import parse_gcode_text
from app.simulator.registration.mesh import register_mesh_full


@pytest.fixture
def subject_4_bundle():
    bundle_dir = paths.REPO_ROOT / "data/output/bundles/subject_4"
    if not (bundle_dir / "manifest.json").is_file():
        pytest.skip("subject_4 bundle not present")
    return load_bundle(bundle_dir)


@pytest.fixture
def machine_config():
    return load_machine_config(paths.postprocessor_machine_config())


def test_decode_postprocessor_paths_machine_frame(subject_4_bundle, machine_config):
    pm_path = paths.postprocessor_subject_pm(4)
    if not pm_path.is_file():
        pytest.skip("subject_4 pm config not present")
    pm = load_physical_landmarks(pm_path)

    reg = register_mesh_full(
        subject_4_bundle,
        pm,
        a_mm=machine_config.a_mm,
        d_mm=machine_config.d_mm,
        calgap_z_mm=machine_config.calgap_z_mm,
    )
    job = JobConfig(physical_landmarks_mm=pm)
    channels, mesh_lm = align_subject(subject_4_bundle, job)
    rows = process_trace(channels[0].interconnect, machine_config)
    parsed = parse_gcode_text("\n".join(format_gcode_lines(rows)) + "\n")
    faces = subject_4_bundle.mesh_faces

    scalp_m, tips_m, scalp_lm, tips_lm = decode_postprocessor_paths(
        parsed,
        machine_config,
        mesh_points_machine=reg.mesh_points,
        mesh_faces=faces,
    )

    fk_kw = {
        "a_mm": machine_config.a_mm,
        "d_mm": machine_config.d_mm,
        "calgap_z_mm": machine_config.calgap_z_mm,
    }
    np.testing.assert_allclose(
        scalp_m,
        registration_to_machine_frame(scalp_lm, **fk_kw),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        tips_m,
        registration_to_machine_frame(tips_lm, **fk_kw),
        atol=1e-6,
    )

    from scipy.spatial import cKDTree

    mesh_d_lm = cKDTree(mesh_lm).query(scalp_lm)[0]
    assert float(np.median(mesh_d_lm)) < 1.0

    standoff = np.linalg.norm(tips_lm - scalp_lm, axis=1)
    np.testing.assert_allclose(
        standoff, machine_config.gap_size_mm, atol=1e-6
    )

    # Wrong frame: inverse against machine mesh inflates scalp→mesh distance.
    _, scalp_wrong, _ = gcode_to_poses(
        parsed,
        machine_config,
        mesh_points=reg.mesh_points,
        mesh_faces=faces,
    )
    wrong_mesh_d = cKDTree(reg.mesh_points).query(scalp_wrong)[0]
    assert float(np.median(wrong_mesh_d)) > float(np.median(mesh_d_lm)) + 50.0
