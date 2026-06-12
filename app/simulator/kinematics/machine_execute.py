"""
G-code machine execution for simulate-gcode.

Forward map: programmed X,Y,Z is the C-axis pivot; B,C command the rigid arm/tool chain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.postprocess.gcode.kinematics.machine_fk import structural_arm_joints_batch


@dataclass
class RigidMachineState:
    """Rigid arm + tool at one G-code step."""

    c_center: np.ndarray
    b_pivot: np.ndarray
    tip: np.ndarray
    b_deg: float
    c_deg: float


def _build_states(
    gcode_matrix: np.ndarray,
    c_centers: np.ndarray,
    b_pivots: np.ndarray,
    tips: np.ndarray,
) -> list[RigidMachineState]:
    states: list[RigidMachineState] = []
    for i in range(gcode_matrix.shape[0]):
        states.append(
            RigidMachineState(
                c_center=c_centers[i],
                b_pivot=b_pivots[i],
                tip=tips[i],
                b_deg=float(gcode_matrix[i, 3]),
                c_deg=float(gcode_matrix[i, 4]),
            )
        )
    return states


def forward_states_from_gcode(
    gcode_matrix: np.ndarray,
    *,
    a_mm: float,
    d_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[RigidMachineState]]:
    """
    Forward map from programmed G-code to rigid machine joints.

    Each row ``(X, Y, Z, B, C)``:
      C pivot  = (X, Y, Z)
      B pivot  = C + Rz(−C) · [a, 0, 0]
      tip      = B + d · tool_dir(B, C)

    Returns (c_pivot_path, b_pivot_path, tip_path, per-step states).
    """
    if gcode_matrix.ndim == 1:
        gcode_matrix = gcode_matrix.reshape(1, -1)

    c_pivot_path = gcode_matrix[:, :3].copy()
    c_centers, b_pivots, tips = structural_arm_joints_batch(
        c_pivot_path,
        gcode_matrix[:, 3],
        gcode_matrix[:, 4],
        a_mm,
        d_mm,
    )
    states = _build_states(gcode_matrix, c_centers, b_pivots, tips)
    return c_pivot_path, b_pivots, tips, states


def rigid_geometry_checks(
    states: list[RigidMachineState],
    *,
    a_mm: float,
    d_mm: float,
    b0_deg: float | None = None,
    c0_deg: float | None = None,
    atol: float = 1e-3,
) -> dict[str, float]:
    """Sanity metrics: fixed segment lengths and arm ⊥ tool."""
    from app.postprocess.gcode.kinematics.machine_fk import _at_machine_zero_pose

    arm_lens: list[float] = []
    tool_lens: list[float] = []
    perp_dots: list[float] = []

    for s in states:
        arm = s.b_pivot - s.c_center
        tool = s.tip - s.b_pivot
        arm_lens.append(float(np.linalg.norm(arm)))
        if b0_deg is not None and c0_deg is not None and _at_machine_zero_pose(
            s.b_deg, s.c_deg, b0_deg, c0_deg
        ):
            continue
        tool_lens.append(float(np.linalg.norm(tool)))
        na = np.linalg.norm(arm)
        nt = np.linalg.norm(tool)
        if na > 1e-12 and nt > 1e-12:
            perp_dots.append(abs(float(np.dot(arm / na, tool / nt))))

    return {
        "arm_length_max_err": max(abs(x - a_mm) for x in arm_lens) if arm_lens else 0.0,
        "tool_length_max_err": max(abs(x - d_mm) for x in tool_lens) if tool_lens else 0.0,
        "perp_dot_max": max(perp_dots) if perp_dots else 0.0,
    }
