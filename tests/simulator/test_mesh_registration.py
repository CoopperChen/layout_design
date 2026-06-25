"""Mesh registration lands landmarks on measured pm in machine frame."""

from __future__ import annotations

import numpy as np
import pytest

from app import paths
from app.postprocess.bundle.load import load_bundle
from app.postprocess.gcode.config_loader import load_machine_config
from app.postprocess.print_config import load_physical_landmarks
from app.simulator.registration.mesh import register_mesh_full


@pytest.fixture
def machine():
    return load_machine_config(paths.postprocessor_machine_config())


@pytest.fixture
def subject_4_bundle():
    bundle_dir = paths.REPO_ROOT / "data/output/bundles/subject_4"
    if not (bundle_dir / "manifest.json").is_file():
        pytest.skip("subject_4 bundle not present")
    return load_bundle(bundle_dir)


def test_registered_landmarks_in_machine_frame(subject_4_bundle, machine):
    pm_path = paths.postprocessor_subject_pm(4)
    if not pm_path.is_file():
        pytest.skip("subject_4 pm config not present")
    pm = load_physical_landmarks(pm_path)

    reg = register_mesh_full(
        subject_4_bundle,
        pm,
        a_mm=machine.a_mm,
        d_mm=machine.d_mm,
        calgap_z_mm=machine.calgap_z_mm,
    )

    expected_central = np.array(
        [0.0, -machine.a_mm, -(machine.d_mm + machine.calgap_z_mm)]
    )
    np.testing.assert_allclose(reg.calibration_registered[0], expected_central, atol=1e-3)
    np.testing.assert_allclose(reg.pm_machine[0], expected_central, atol=1e-6)
    assert reg.landmark_fit_error_mm < 10.0


def test_register_mesh_matches_align_subject_before_machine_shift(subject_4_bundle, machine):
    """scan2phys stage still matches convert-gcode; machine shift is simulator-only."""
    from app.postprocess.gcode.models import JobConfig
    from app.postprocess.gcode.pipeline.align import align_subject

    pm_path = paths.postprocessor_subject_pm(4)
    if not pm_path.is_file():
        pytest.skip("subject_4 pm config not present")
    pm = load_physical_landmarks(pm_path)

    reg = register_mesh_full(
        subject_4_bundle,
        pm,
        a_mm=machine.a_mm,
        d_mm=machine.d_mm,
        calgap_z_mm=machine.calgap_z_mm,
        machine_frame=False,
    )
    _channels, mesh_align = align_subject(
        subject_4_bundle, JobConfig(physical_landmarks_mm=pm)
    )
    np.testing.assert_allclose(reg.mesh_points, mesh_align, atol=1e-6)
    np.testing.assert_allclose(reg.calibration_registered[0], pm[0], atol=1e-6)
