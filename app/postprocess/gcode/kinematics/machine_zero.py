"""Machine zero offset based on c0/b0 configuration."""

from __future__ import annotations

import numpy as np

from ..models import MachineConfig


def apply_machine_zero_offset(
    positions: np.ndarray, machine: MachineConfig
) -> np.ndarray:
    """Shift XYZ for machine zero pose (c0, b0)."""
    g = positions.copy()
    c0, b0 = machine.c0_deg, machine.b0_deg
    a, d, calgap_z = machine.a_mm, machine.d_mm, machine.calgap_z_mm

    if c0 == 0 and b0 == 0:
        g[:, 0] -= a
        g[:, 2] += d + calgap_z
    elif c0 == 90 and b0 == 0:
        g[:, 1] -= a
        g[:, 2] -= d + calgap_z
    elif c0 == 0 and b0 == 90:
        g[:, 1] -= a + d + calgap_z
    else:
        raise ValueError(
            "Choose the B,C angles that the tool will be in when setting x,y,z machine zeros"
        )
    return g
