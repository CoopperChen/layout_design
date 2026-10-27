"""Rigid machine kinematics tests."""

from __future__ import annotations

import numpy as np

from app import paths
from app.gcode.config_loader import load_machine_config
from app.simulator.kinematics.machine_execute import forward_states_from_gcode


def test_forward_map_uses_programmed_xyz_as_c_pivot():
    machine = load_machine_config(paths.machine_config())
    gcode = np.array([[5.0, -3.0, 12.0, 0.0, 90.0, 50.0, 0.0]])
    c_path, _, _, states = forward_states_from_gcode(
        gcode,
        a_mm=machine.a_mm,
        d_mm=machine.d_mm,
    )
    np.testing.assert_allclose(c_path[0], gcode[0, :3])
    np.testing.assert_allclose(states[0].c_center, gcode[0, :3])
