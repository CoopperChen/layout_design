"""Upward-nozzle engage/disengage clearance via outward XY offset."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.config_loader import load_defaults
from app.postprocess.gcode.kinematics.arm_clearance import HeadMeshInsideChecker
from app.postprocess.gcode.kinematics.machine_fk import structural_arm_joints
from app.postprocess.gcode.models import MachineConfig
from app.postprocess.mesh_normals import head_center_from_points


@dataclass(frozen=True)
class EngageClearanceConfig:
    b_threshold_deg: float = 90.0
    default_offset_mm: float = 20.0
    max_offset_mm: float = 40.0
    step_mm: float = 1.0
    descent_min_clearance_mm: float = 20.0


def load_engage_clearance_config(machine: MachineConfig) -> EngageClearanceConfig:
    del machine
    pp = load_defaults().get("postprocess", {})
    return EngageClearanceConfig(
        b_threshold_deg=float(pp.get("engage_b_threshold_deg", 90.0)),
        default_offset_mm=float(pp.get("engage_xy_offset_default_mm", 20.0)),
        max_offset_mm=float(pp.get("engage_xy_offset_max_mm", 40.0)),
        step_mm=float(pp.get("engage_xy_offset_step_mm", 1.0)),
        descent_min_clearance_mm=float(
            pp.get("engage_descent_min_clearance_mm", 20.0)
        ),
    )


def _unit_outward_xy(target_xy: np.ndarray, head_center_xy: np.ndarray) -> np.ndarray:
    delta = np.asarray(target_xy, dtype=float).reshape(2) - np.asarray(
        head_center_xy, dtype=float
    ).reshape(2)
    length = float(np.linalg.norm(delta))
    if length < 1e-9:
        return np.array([0.0, 1.0])
    return delta / length


def min_tip_clearance_during_z_leg(
    c_pivot_target: np.ndarray,
    b_deg: float,
    c_deg: float,
    zsafe: float,
    offset_xy: np.ndarray,
    checker: HeadMeshInsideChecker | None,
    machine: MachineConfig,
    *,
    ascending: bool,
    samples_per_segment: int = 20,
) -> float:
    """Minimum tip clearance during a vertical Z move at fixed offset XY."""
    if checker is None:
        return float("inf")

    target = np.asarray(c_pivot_target, dtype=float).reshape(3)
    off = np.asarray(offset_xy, dtype=float).reshape(2)
    x, y, z = target
    dx, dy = off
    min_sd = float("inf")
    for t in np.linspace(0.0, 1.0, samples_per_segment):
        if ascending:
            zi = z * (1.0 - t) + float(zsafe) * t
        else:
            zi = float(zsafe) * (1.0 - t) + z * t
        c_xyz = np.array([x + dx, y + dy, zi], dtype=float)
        _c, _b, tip = structural_arm_joints(
            c_xyz, b_deg, c_deg, machine.a_mm, machine.d_mm
        )
        sd = float(checker.signed_distance(tip)[0])
        min_sd = min(min_sd, sd)
    return min_sd


def min_tip_clearance_during_z_descent(
    c_pivot_target: np.ndarray,
    b_deg: float,
    c_deg: float,
    zsafe: float,
    offset_xy: np.ndarray,
    checker: HeadMeshInsideChecker | None,
    machine: MachineConfig,
    *,
    samples_per_segment: int = 20,
) -> float:
    """Minimum tip clearance while descending Z at fixed offset XY (engage leg 2)."""
    return min_tip_clearance_during_z_leg(
        c_pivot_target,
        b_deg,
        c_deg,
        zsafe,
        offset_xy,
        checker,
        machine,
        ascending=False,
        samples_per_segment=samples_per_segment,
    )


def min_tip_clearance_during_z_ascent(
    c_pivot_target: np.ndarray,
    b_deg: float,
    c_deg: float,
    zsafe: float,
    offset_xy: np.ndarray,
    checker: HeadMeshInsideChecker | None,
    machine: MachineConfig,
    *,
    samples_per_segment: int = 20,
) -> float:
    """Minimum tip clearance while ascending Z at fixed offset XY (disengage leg 2)."""
    return min_tip_clearance_during_z_leg(
        c_pivot_target,
        b_deg,
        c_deg,
        zsafe,
        offset_xy,
        checker,
        machine,
        ascending=True,
        samples_per_segment=samples_per_segment,
    )


def min_tip_clearance_during_xy_slide(
    c_pivot_target: np.ndarray,
    b_deg: float,
    c_deg: float,
    offset_xy: np.ndarray,
    checker: HeadMeshInsideChecker | None,
    machine: MachineConfig,
    *,
    outward: bool,
    samples_per_segment: int = 20,
) -> float:
    """Minimum tip clearance during horizontal slide at print Z."""
    if checker is None:
        return float("inf")

    target = np.asarray(c_pivot_target, dtype=float).reshape(3)
    off = np.asarray(offset_xy, dtype=float).reshape(2)
    x, y, z = target
    dx, dy = off
    if float(np.linalg.norm(off)) < 1e-9:
        return float("inf")

    min_sd = float("inf")
    for t in np.linspace(0.0, 1.0, samples_per_segment):
        if outward:
            xi = x * (1.0 - t) + (x + dx) * t
            yi = y * (1.0 - t) + (y + dy) * t
        else:
            xi = (x + dx) * (1.0 - t) + x * t
            yi = (y + dy) * (1.0 - t) + y * t
        c_xyz = np.array([xi, yi, z], dtype=float)
        _c, _b, tip = structural_arm_joints(
            c_xyz, b_deg, c_deg, machine.a_mm, machine.d_mm
        )
        sd = float(checker.signed_distance(tip)[0])
        min_sd = min(min_sd, sd)
    return min_sd


def min_tip_clearance_during_disengage(
    c_pivot_target: np.ndarray,
    b_deg: float,
    c_deg: float,
    zsafe: float,
    offset_xy: np.ndarray,
    checker: HeadMeshInsideChecker | None,
    machine: MachineConfig,
    *,
    samples_per_segment: int = 20,
) -> float:
    """Minimum tip clearance along slide@print Z + Z-up at offset."""
    slide = min_tip_clearance_during_xy_slide(
        c_pivot_target,
        b_deg,
        c_deg,
        offset_xy,
        checker,
        machine,
        outward=True,
        samples_per_segment=samples_per_segment,
    )
    ascent = min_tip_clearance_during_z_ascent(
        c_pivot_target,
        b_deg,
        c_deg,
        zsafe,
        offset_xy,
        checker,
        machine,
        samples_per_segment=samples_per_segment,
    )
    return min(slide, ascent)


def min_tip_clearance_during_engage(
    c_pivot_target: np.ndarray,
    b_deg: float,
    c_deg: float,
    zsafe: float,
    offset_xy: np.ndarray,
    checker: HeadMeshInsideChecker | None,
    machine: MachineConfig,
    *,
    samples_per_segment: int = 20,
) -> float:
    """Minimum tip clearance along Z-down at offset + slide@print Z."""
    descent = min_tip_clearance_during_z_descent(
        c_pivot_target,
        b_deg,
        c_deg,
        zsafe,
        offset_xy,
        checker,
        machine,
        samples_per_segment=samples_per_segment,
    )
    slide = min_tip_clearance_during_xy_slide(
        c_pivot_target,
        b_deg,
        c_deg,
        offset_xy,
        checker,
        machine,
        outward=False,
        samples_per_segment=samples_per_segment,
    )
    return min(descent, slide)


def _offset_path_clearance(
    row: np.ndarray,
    zsafe: float,
    offset_xy: np.ndarray,
    checker: HeadMeshInsideChecker | None,
    machine: MachineConfig,
    *,
    ascending: bool,
) -> float:
    if ascending:
        return min_tip_clearance_during_disengage(
            row[:3],
            float(row[3]),
            float(row[4]),
            zsafe,
            offset_xy,
            checker,
            machine,
        )
    return min_tip_clearance_during_engage(
        row[:3],
        float(row[3]),
        float(row[4]),
        zsafe,
        offset_xy,
        checker,
        machine,
    )


def _compute_nozzle_xy_offset(
    row: np.ndarray,
    zsafe: float,
    head_center_xy: np.ndarray,
    checker: HeadMeshInsideChecker | None,
    machine: MachineConfig,
    config: EngageClearanceConfig,
    *,
    ascending: bool,
) -> np.ndarray:
    """
    Outward XY offset when |B| exceeds threshold.

    Offset is chosen so tip clearance during the Z-up/Z-down leg at offset XY is
    at least ``descent_min_clearance_mm``. The short horizontal slide at print
    Z uses the same offset distance; it is typically ~15 mm on crown traces.
    """
    b_deg = float(row[3])
    if abs(b_deg) <= config.b_threshold_deg:
        return np.zeros(2, dtype=float)

    min_clear = float(config.descent_min_clearance_mm)
    target = np.asarray(row[:3], dtype=float)
    outward = _unit_outward_xy(target[:2], head_center_xy)
    z_leg_clearance = (
        min_tip_clearance_during_z_ascent
        if ascending
        else min_tip_clearance_during_z_descent
    )

    if checker is not None:
        direct_clear = z_leg_clearance(
            target,
            b_deg,
            float(row[4]),
            zsafe,
            np.zeros(2),
            checker,
            machine,
        )
        if direct_clear >= min_clear:
            return np.zeros(2, dtype=float)

        step = max(float(config.step_mm), 1e-6)
        dist = step
        while dist <= config.max_offset_mm + 1e-9:
            off = dist * outward
            if (
                z_leg_clearance(
                    target,
                    b_deg,
                    float(row[4]),
                    zsafe,
                    off,
                    checker,
                    machine,
                )
                >= min_clear
            ):
                return off
            dist += step

    return float(config.default_offset_mm) * outward


def compute_engage_xy_offset(
    engage_row: np.ndarray,
    zsafe: float,
    head_center_xy: np.ndarray,
    checker: HeadMeshInsideChecker | None,
    machine: MachineConfig,
    config: EngageClearanceConfig,
) -> np.ndarray:
    """Outward XY offset before trace engagement (Z-down leg)."""
    return _compute_nozzle_xy_offset(
        engage_row,
        zsafe,
        head_center_xy,
        checker,
        machine,
        config,
        ascending=False,
    )


def compute_disengage_xy_offset(
    exit_row: np.ndarray,
    zsafe: float,
    head_center_xy: np.ndarray,
    checker: HeadMeshInsideChecker | None,
    machine: MachineConfig,
    config: EngageClearanceConfig,
) -> np.ndarray:
    """Outward XY offset after trace end before Z-safe retract (Z-up leg)."""
    return _compute_nozzle_xy_offset(
        exit_row,
        zsafe,
        head_center_xy,
        checker,
        machine,
        config,
        ascending=True,
    )


def build_engage_offset_rows(
    engage_row: np.ndarray,
    zsafe: float,
    offset_xy: np.ndarray,
    machine: MachineConfig,
) -> np.ndarray:
    """
    Three-row engage prefix: offset@Z_safe, Z-down at offset XY, slide to target.

    Returns shape (0, 7) when ``offset_xy`` is zero.
    """
    off = np.asarray(offset_xy, dtype=float).reshape(2)
    if float(np.linalg.norm(off)) < 1e-9:
        return np.zeros((0, 7), dtype=float)

    target = np.asarray(engage_row[:6], dtype=float).reshape(6).copy()
    x, y, z = target[0], target[1], target[2]
    dx, dy = off
    feed = 0.5 * float(machine.transition_speed_mm_min)

    safe = target.copy()
    safe[0], safe[1] = x + dx, y + dy
    safe[2] = float(zsafe)
    safe[5] = feed

    descend = safe.copy()
    descend[2] = z

    slide = target.copy()
    slide[5] = feed

    rows6 = np.vstack([safe, descend, slide])
    markers = np.zeros((rows6.shape[0], 1), dtype=float)
    return np.hstack([rows6, markers])


def build_disengage_offset_rows(
    exit_row: np.ndarray,
    zsafe: float,
    offset_xy: np.ndarray,
    machine: MachineConfig,
) -> np.ndarray:
    """
    Two-row disengage suffix: slide to offset at print Z, then Z-up at offset XY.

    Returns shape (0, 7) when ``offset_xy`` is zero.
    """
    off = np.asarray(offset_xy, dtype=float).reshape(2)
    if float(np.linalg.norm(off)) < 1e-9:
        return np.zeros((0, 7), dtype=float)

    target = np.asarray(exit_row[:6], dtype=float).reshape(6).copy()
    x, y = target[0], target[1]
    dx, dy = off
    feed = 0.5 * float(machine.transition_speed_mm_min)

    slide = target.copy()
    slide[0], slide[1] = x + dx, y + dy
    slide[5] = feed

    ascend = slide.copy()
    ascend[2] = float(zsafe)
    ascend[5] = feed

    rows6 = np.vstack([slide, ascend])
    markers = np.zeros((rows6.shape[0], 1), dtype=float)
    return np.hstack([rows6, markers])


def head_center_xy_from_mesh(mesh_points: np.ndarray | None) -> np.ndarray:
    if mesh_points is None:
        return np.zeros(2, dtype=float)
    center = head_center_from_points(mesh_points)
    return np.asarray(center[:2], dtype=float)
