"""Variable feed rate to keep nozzle speed constant."""

from __future__ import annotations

import numpy as np

from ..models import MachineConfig


def compute_feed_rates(
    machine_positions: np.ndarray,
    nozzle_positions: np.ndarray,
    machine: MachineConfig,
) -> np.ndarray:
    npts = machine_positions.shape[0]
    disp_c = np.zeros(npts, dtype=float)
    disp_nozzle = np.zeros(npts, dtype=float)
    speed_c = np.zeros(npts, dtype=float)

    for i in range(1, npts):
        disp_c[i] = np.linalg.norm(machine_positions[i] - machine_positions[i - 1])
        disp_nozzle[i] = np.linalg.norm(nozzle_positions[i] - nozzle_positions[i - 1])
        if disp_nozzle[i] > 0:
            speed_c[i] = machine.speed_mm_min * disp_c[i] / disp_nozzle[i]

    bounded = speed_c.copy()
    for i in range(1, npts):
        if bounded[i] > machine.max_speed_mm_min:
            bounded[i] = machine.max_speed_mm_min

    bounded[0] = machine.transition_speed_mm_min
    return np.round(bounded, 0)
