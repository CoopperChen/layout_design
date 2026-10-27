"""Inverse kinematics helpers that do not need a stored subject bundle."""

from __future__ import annotations

import numpy as np

from app import paths
from app.gcode.config_loader import load_machine_config
from app.gcode.kinematics.machine_zero import apply_machine_zero_offset
from app.simulator.kinematics.inverse import undo_machine_zero_offset


def test_undo_machine_zero_inverse():
    machine = load_machine_config(paths.machine_config())
    pts = np.array([[10.0, 20.0, -30.0], [0.0, 5.0, 100.0]])
    shifted = apply_machine_zero_offset(pts, machine)
    restored = undo_machine_zero_offset(shifted, machine)
    np.testing.assert_allclose(restored, pts, atol=1e-9)
