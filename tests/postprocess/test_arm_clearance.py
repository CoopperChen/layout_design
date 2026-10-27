"""Arm-in-mesh normal flip policy."""

from __future__ import annotations

import numpy as np

from app import paths
from app.gcode.config_loader import load_machine_config
from app.gcode.kinematics.arm_clearance import resolve_normal_arm_clearance


class _FakeChecker:
    """First arm pose inside mesh, flipped pose clear."""

    def __init__(self) -> None:
        self._calls = 0

    def arm_is_inside(self, _c_pivot: np.ndarray, _b_pivot: np.ndarray, *, samples: int = 5) -> bool:
        del samples
        self._calls += 1
        return self._calls == 1


def test_resolve_normal_flips_when_arm_inside_mesh():
    machine = load_machine_config(paths.machine_config())
    scalp = np.array([10.0, -20.0, 50.0])
    normal = np.array([0.1, 0.2, 0.97])
    normal /= np.linalg.norm(normal)
    checker = _FakeChecker()

    resolved = resolve_normal_arm_clearance(scalp, normal, machine, checker)
    assert checker._calls == 2
    np.testing.assert_allclose(resolved, -normal, atol=1e-9)


