"""Tool offset forward/inverse parity."""

from __future__ import annotations

import numpy as np

from app.postprocess.gcode.kinematics.tool_offset import apply_tool_offset
from app.postprocess.gcode.models import MachineConfig
from app.simulator.kinematics.inverse import undo_tool_offset


def test_tool_offset_roundtrip():
    machine = MachineConfig()
    positions = np.array(
        [
            [10.0, -20.0, -50.0],
            [0.0, 5.0, 100.0],
            [-30.0, 40.0, -80.0],
        ]
    )
    normals = np.array(
        [
            [0.2, 0.3, 0.93],
            [-0.1, 0.5, 0.86],
            [0.6, -0.2, 0.78],
        ]
    )
    for i in range(3):
        normals[i] /= np.linalg.norm(normals[i])
    c_angles = np.array([12.0, -33.5, 88.0])

    shifted = apply_tool_offset(positions, normals, c_angles, machine)
    restored = undo_tool_offset(shifted, normals, c_angles, machine)
    # Forward path rounds to 2 decimals; inverse is exact for that quantized output.
    np.testing.assert_allclose(restored, positions, atol=0.01)
