"""Arm-in-mesh normal flip policy."""

from __future__ import annotations

import numpy as np
import pytest

from app import paths
from app.postprocess.bundle.load import load_bundle
from app.postprocess.gcode.config_loader import load_machine_config
from app.postprocess.gcode.kinematics.arm_clearance import resolve_normal_arm_clearance
from app.postprocess.gcode.kinematics.machine_fk import registration_to_machine_frame
from app.postprocess.gcode.models import JobConfig
from app.postprocess.gcode.pipeline.align import align_subject
from app.postprocess.gcode.pipeline.process_traces import process_trace
from app.postprocess.print_config import load_physical_landmarks


class _FakeChecker:
    """First arm pose inside mesh, flipped pose clear."""

    def __init__(self) -> None:
        self._calls = 0

    def arm_is_inside(self, _c_pivot: np.ndarray, _b_pivot: np.ndarray, *, samples: int = 5) -> bool:
        del samples
        self._calls += 1
        return self._calls == 1


def test_resolve_normal_flips_when_arm_inside_mesh():
    machine = load_machine_config(paths.postprocessor_machine_config())
    scalp = np.array([10.0, -20.0, 50.0])
    normal = np.array([0.1, 0.2, 0.97])
    normal /= np.linalg.norm(normal)
    checker = _FakeChecker()

    resolved = resolve_normal_arm_clearance(scalp, normal, machine, checker)
    assert checker._calls == 2
    np.testing.assert_allclose(resolved, -normal, atol=1e-9)


def test_process_trace_with_mesh_runs():
    bundle_dir = paths.REPO_ROOT / "data/output/bundles/subject_4"
    if not (bundle_dir / "manifest.json").is_file():
        pytest.skip("subject_4 bundle not present")
    bundle = load_bundle(bundle_dir)
    pm_path = paths.postprocessor_subject_pm(4)
    if not pm_path.is_file():
        pytest.skip("subject_4 pm config not present")
    pm = load_physical_landmarks(pm_path)
    machine = load_machine_config(paths.postprocessor_machine_config())
    channels, mesh = align_subject(bundle, JobConfig(physical_landmarks_mm=pm))
    mesh_machine = registration_to_machine_frame(
        mesh,
        a_mm=machine.a_mm,
        d_mm=machine.d_mm,
        calgap_z_mm=machine.calgap_z_mm,
    )
    rows = process_trace(
        channels[0].interconnect,
        machine,
        mesh_points=mesh_machine,
        mesh_faces=bundle.mesh_faces,
    )
    assert rows.shape[1] == 7
    assert len(rows) == len(channels[0].interconnect)
