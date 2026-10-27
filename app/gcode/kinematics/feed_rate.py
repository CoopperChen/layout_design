"""Variable C-pivot feed rate for constant nozzle tip speed."""

from __future__ import annotations

import numpy as np

from ..models import MachineConfig
from .machine_fk import structural_arm_joints_batch


def tip_positions_from_poses(
    c_pivot_xyz: np.ndarray,
    b_deg: np.ndarray,
    c_deg: np.ndarray,
    machine: MachineConfig,
) -> np.ndarray:
    """Nozzle tip XYZ for each programmed C-pivot / B / C pose (rigid FK)."""
    _c, _b, tips = structural_arm_joints_batch(
        np.asarray(c_pivot_xyz, dtype=float),
        np.asarray(b_deg, dtype=float),
        np.asarray(c_deg, dtype=float),
        float(machine.a_mm),
        float(machine.d_mm),
    )
    return tips


def compute_feed_rates(
    machine_positions: np.ndarray,
    nozzle_positions: np.ndarray,
    machine: MachineConfig,
) -> np.ndarray:
    """
    C-pivot feed so average tip speed equals ``machine.speed_mm_min``.

    The controller feedrate commands **C-pivot (XYZ)** speed. For a coordinated
    move from pose ``i-1`` to ``i`` finished in time ``dt``:

        dt = ||Δtip|| / V_tip
        F  = ||ΔC_pivot|| / dt
           = V_tip * ||ΔC_pivot|| / ||Δtip||

    Use **forward-FK tip** positions (not the raw path chord) so B/C swings that
    move the tip are counted in ``||Δtip||``. Otherwise F is too high during
    orientation changes and the tip serpentine runs at max feed.

    Segments with negligible tip motion but nonzero C-pivot travel use
    ``max_speed_mm_min`` (pure reorientation). All feeds are capped at
    ``max_speed_mm_min``; when capped, tip speed is below ``V_tip`` for that hop.
    """
    pivots = np.asarray(machine_positions, dtype=float)
    tips = np.asarray(nozzle_positions, dtype=float)
    if pivots.ndim == 1:
        pivots = pivots.reshape(1, -1)
    if tips.ndim == 1:
        tips = tips.reshape(1, -1)

    npts = pivots.shape[0]
    if tips.shape[0] != npts:
        raise ValueError("machine_positions and nozzle_positions length mismatch")

    v_tip = float(machine.speed_mm_min)
    v_max = float(machine.max_speed_mm_min)
    feeds = np.zeros(npts, dtype=float)
    tip_eps = 1e-6

    for i in range(1, npts):
        disp_c = float(np.linalg.norm(pivots[i] - pivots[i - 1]))
        disp_tip = float(np.linalg.norm(tips[i] - tips[i - 1]))
        if disp_tip > tip_eps:
            feeds[i] = v_tip * disp_c / disp_tip
        elif disp_c > tip_eps:
            # Orientation-dominated hop: finish as fast as allowed.
            feeds[i] = v_max
        else:
            feeds[i] = v_tip

        if feeds[i] > v_max:
            feeds[i] = v_max
        elif feeds[i] < 1.0:
            feeds[i] = 1.0

    # First print row inherits the nominal tip speed (approach feeds set in merge).
    feeds[0] = v_tip
    return np.round(feeds, 0)


def compute_print_feed_rates(
    c_pivot_xyz: np.ndarray,
    b_deg: np.ndarray,
    c_deg: np.ndarray,
    machine: MachineConfig,
) -> np.ndarray:
    """Feed rates from programmed poses using rigid tip FK."""
    tips = tip_positions_from_poses(c_pivot_xyz, b_deg, c_deg, machine)
    return compute_feed_rates(c_pivot_xyz, tips, machine)
