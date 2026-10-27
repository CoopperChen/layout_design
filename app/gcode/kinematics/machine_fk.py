"""
Machine kinematics matching gcodeConverter_final14.m (updated 20260506).

Reference: Postprocessor/matlab/gcodeConverter_final14_updated20260506.m
  - findCaxisAngle / findBaxisAngle  -> axis_angles.py
  - tool offset sin/cos(C) arm term  -> tool_offset.py (B not used for arm a)
  - machine zero c0/b0               -> machine_zero.py
"""

from __future__ import annotations

import numpy as np


def rot_z(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def arm_offset_xy_matlab(c_deg: float, a_mm: float) -> np.ndarray:
    """
    Horizontal arm term from gcodeConverter_final14 tool offset (lines 402–410).

    Adds to programmed XYZ:
      dX = -|cos(C)| * a
      dY = sin(C) * a

    Positive C rotates the arm contribution from -X toward +Y (RH about +Z).
    """
    c_rad = np.deg2rad(float(c_deg))
    return np.array(
        [-abs(np.cos(c_rad)) * float(a_mm), np.sin(c_rad) * float(a_mm), 0.0]
    )


def _rodrigues(v: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis_u = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis_u)
    if n < 1e-12:
        raise ValueError("rotation axis must be non-zero")
    axis_u = axis_u / n
    v = np.asarray(v, dtype=float)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return v * c + np.cross(axis_u, v) * s + axis_u * np.dot(axis_u, v) * (1.0 - c)


def _tool_direction_from_arm(arm_vec: np.ndarray, b_deg: float) -> np.ndarray:
    """Unit tool direction at B=0 reference (−Z projected ⊥ arm), then Rx(−B) about arm."""
    na = np.linalg.norm(arm_vec)
    if na < 1e-12:
        raise ValueError("arm vector must be non-zero")
    arm_hat = arm_vec / na
    ref = np.array([0.0, 0.0, -1.0])
    tool_ref = ref - np.dot(ref, arm_hat) * arm_hat
    nt = np.linalg.norm(tool_ref)
    if nt < 1e-9:
        tool_ref = np.cross(arm_hat, np.array([1.0, 0.0, 0.0]))
        nt = np.linalg.norm(tool_ref)
    tool_ref = tool_ref / nt
    return _rodrigues(tool_ref, arm_hat, -np.deg2rad(float(b_deg)))


def structural_arm_offset(c_deg: float, a_mm: float) -> np.ndarray:
    """
    C→B arm vector for rigid FK (horizontal, length ``a_mm``).

    C=0 → **+X**; C=90 → **−Y** via ``Rz(−C) · [1, 0, 0]``.
    """
    c_rad = np.deg2rad(float(c_deg))
    arm_dir = rot_z(-c_rad) @ np.array([1.0, 0.0, 0.0])
    return float(a_mm) * arm_dir


def structural_arm_joints(
    center_xyz: np.ndarray,
    b_deg: float,
    c_deg: float,
    a_mm: float,
    d_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (C_center, B_pivot, nozzle_tip) in machine frame.

    Kinematic chain (rigid, arm ⊥ tool):
      1. C axis: rotate about +Z at C pivot.
      2. C–B arm: length ``a_mm`` along ``Rz(−C) · [1, 0, 0]`` (C=0 → **+X**; C=90 → **−Y**).
      3. B axis: rotate tool about the C–B arm (**+B** about arm, −Z at B=0).
      4. B–tip: length ``d_mm``, ⊥ arm.
    """
    center = np.asarray(center_xyz, dtype=float).reshape(3)
    arm_vec = structural_arm_offset(c_deg, a_mm)
    tool_dir = _tool_direction_from_arm(arm_vec, b_deg)
    b_pivot = center + arm_vec
    tip = b_pivot + float(d_mm) * tool_dir
    return center, b_pivot, tip


def machine_zero_head_frame(
    *,
    a_mm: float,
    d_mm: float,
    b0_deg: float,
    c0_deg: float,
    calgap_z_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Machine-frame reference poses at ``(b0, c0)`` when X/Y/Z zeros were set.

    For ``c0=90``, ``b0=0`` (default config) with C pivot at machine X0 Y0 Z0:

      C pivot           = (0, 0, 0)
      nozzle tip        = (0, −a, −d)
      central landmark  = (0, −a, −(d + calgap_z))   — tip is ``calgap_z`` above central in Z
    """
    if not (
        abs(float(b0_deg)) <= 1e-3
        and abs(float(c0_deg) - 90.0) <= 1e-3
    ):
        raise ValueError(
            "machine_zero_head_frame is defined for c0=90, b0=0; "
            f"got b0={b0_deg}, c0={c0_deg}"
        )

    cal = float(calgap_z_mm)
    c_pivot = np.array([0.0, 0.0, 0.0])
    _c, b_pivot, tip = structural_arm_joints(
        c_pivot, b0_deg, c0_deg, a_mm, d_mm
    )
    central = tip + np.array([0.0, 0.0, -cal])
    return central, tip, c_pivot, b_pivot


def central_landmark_machine_coords(
    *,
    a_mm: float,
    d_mm: float,
    calgap_z_mm: float,
) -> np.ndarray:
    """Central landmark in machine frame at ``(b0,c0)=(0,90)`` machine-zero setup."""
    return np.array(
        [0.0, -float(a_mm), -(float(d_mm) + float(calgap_z_mm))]
    )


def registration_to_machine_frame(
    points: np.ndarray,
    *,
    a_mm: float,
    d_mm: float,
    calgap_z_mm: float,
) -> np.ndarray:
    """
    Map landmark / scan2phys frame → machine frame.

    Landmark frame: ``pm[0]`` central at origin.
    Machine frame: C pivot at X0 Y0 Z0; central at ``(0, −a, −(d+calgap))``.
    """
    shift = central_landmark_machine_coords(
        a_mm=a_mm, d_mm=d_mm, calgap_z_mm=calgap_z_mm
    )
    pts = np.asarray(points, dtype=float)
    if pts.ndim == 1:
        return pts + shift
    return pts + shift


def machine_to_registration_frame(
    points: np.ndarray,
    *,
    a_mm: float,
    d_mm: float,
    calgap_z_mm: float,
) -> np.ndarray:
    """Inverse of ``registration_to_machine_frame`` (central landmark at origin)."""
    shift = central_landmark_machine_coords(
        a_mm=a_mm, d_mm=d_mm, calgap_z_mm=calgap_z_mm
    )
    pts = np.asarray(points, dtype=float)
    if pts.ndim == 1:
        return pts - shift
    return pts - shift


def structural_arm_joints_batch(
    centers: np.ndarray,
    b_deg: np.ndarray,
    c_deg: np.ndarray,
    a_mm: float,
    d_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized C/B/tip joints; each row is one G-code step."""
    n = centers.shape[0]
    c_out = np.zeros((n, 3), dtype=float)
    b_out = np.zeros((n, 3), dtype=float)
    tip_out = np.zeros((n, 3), dtype=float)
    for i in range(n):
        c_out[i], b_out[i], tip_out[i] = structural_arm_joints(
            centers[i],
            b_deg[i],
            c_deg[i],
            a_mm,
            d_mm,
        )
    return c_out, b_out, tip_out


def _at_machine_zero_pose(
    b_deg: float,
    c_deg: float,
    b0_deg: float,
    c0_deg: float,
    *,
    tol: float = 1e-3,
) -> bool:
    return abs(float(b_deg) - float(b0_deg)) <= tol and abs(
        float(c_deg) - float(c0_deg)
    ) <= tol


def structural_tip_offset(
    b_deg: float,
    c_deg: float,
    a_mm: float,
    d_mm: float,
) -> np.ndarray:
    """Vector from C pivot to nozzle tip (linear in center; depends on B/C only)."""
    _center, _b_pivot, tip = structural_arm_joints(
        np.zeros(3), b_deg, c_deg, a_mm, d_mm
    )
    return tip


def structural_tip_offset_batch(
    b_deg: np.ndarray,
    c_deg: np.ndarray,
    a_mm: float,
    d_mm: float,
) -> np.ndarray:
    """Per-step C→tip offset for vectorized inverse placement."""
    n = b_deg.shape[0]
    out = np.zeros((n, 3), dtype=float)
    for i in range(n):
        out[i] = structural_tip_offset(b_deg[i], c_deg[i], a_mm, d_mm)
    return out


def c_center_from_tip(
    tip_xyz: np.ndarray,
    b_deg: float,
    c_deg: float,
    a_mm: float,
    d_mm: float,
) -> np.ndarray:
    """C pivot that places the rigid nozzle tip at ``tip_xyz``."""
    tip = np.asarray(tip_xyz, dtype=float).reshape(3)
    return tip - structural_tip_offset(b_deg, c_deg, a_mm, d_mm)
