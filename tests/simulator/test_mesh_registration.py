"""Registration places the central landmark on the machine zero of that point."""

from __future__ import annotations

import numpy as np

from app import paths
from app.gcode.config_loader import load_machine_config
from app.gcode.pipeline.align import register_mesh_points
from app.simulator.registration.mesh import register_mesh_full


def test_registered_central_landmark_matches_machine_zero():
    machine = load_machine_config(paths.machine_config())
    landmarks = np.array(
        [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 80.0, 10.0]],
        dtype=float,
    )
    pm = np.array(
        [[0.0, 0.0, 0.0], [90.0, 5.0, 0.0], [4.0, 70.0, 2.0]],
        dtype=float,
    )
    mesh = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 20.0], [0.0, 10.0, 5.0]])
    faces = np.array([[0, 1, 2]])
    reg = register_mesh_full(
        mesh,
        faces,
        landmarks,
        pm,
        a_mm=machine.a_mm,
        d_mm=machine.d_mm,
        calgap_z_mm=machine.calgap_z_mm,
    )
    expected = np.array([0.0, -machine.a_mm, -(machine.d_mm + machine.calgap_z_mm)])
    np.testing.assert_allclose(reg.calibration_registered[0], expected, atol=1e-6)
    scan = register_mesh_points(mesh, landmarks, pm)
    assert scan.shape == mesh.shape
