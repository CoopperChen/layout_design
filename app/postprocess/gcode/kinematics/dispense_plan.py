"""Jet-on blocks at constant tip speed; over-limit motion with the jet off."""

from __future__ import annotations

import numpy as np

from ..models import MachineConfig
from .feed_rate import dispense_pivot_feed, tip_positions_from_poses
from .flip_correction import c_step_deg
from .tool_offset import apply_tool_offset

_JET_ON = 10
_JET_OFF = 11


def _pose_at(
    scalp: np.ndarray,
    normal: np.ndarray,
    b_deg: float,
    c_deg: float,
    machine: MachineConfig,
    gap_mm: float | None,
) -> np.ndarray:
    pivot = apply_tool_offset(
        np.asarray(scalp, dtype=float).reshape(1, 3),
        np.asarray(normal, dtype=float).reshape(1, 3),
        np.array([c_deg], dtype=float),
        machine,
        gap_mm=gap_mm,
    )[0]
    return np.array(
        [pivot[0], pivot[1], pivot[2], float(b_deg), float(c_deg)],
        dtype=float,
    )


def _same_pose(a: np.ndarray, b: np.ndarray) -> bool:
    if float(np.linalg.norm(a[:3] - b[:3])) > 1e-3:
        return False
    if abs(float(a[3]) - float(b[3])) > 1e-3:
        return False
    return c_step_deg(float(a[4]), float(b[4])) <= 1e-3


def _pivot_and_tip(a: np.ndarray, b: np.ndarray, machine: MachineConfig) -> tuple[float, float]:
    pivots = np.vstack([a[:3], b[:3]])
    tips = tip_positions_from_poses(
        pivots,
        np.array([a[3], b[3]], dtype=float),
        np.array([a[4], b[4]], dtype=float),
        machine,
    )
    disp_pivot = float(np.linalg.norm(pivots[1] - pivots[0]))
    disp_tip = float(np.linalg.norm(tips[1] - tips[0]))
    return disp_pivot, disp_tip


def _pack(pose: np.ndarray, feed: float, marker: int) -> np.ndarray:
    row = np.zeros(7, dtype=float)
    row[:5] = pose
    row[5] = float(feed)
    row[6] = float(marker)
    return row


def _jet_off_to(
    rows: list[np.ndarray],
    current: np.ndarray,
    target: np.ndarray,
    *,
    travel: float,
    resume_feed: float,
    jet_on: bool,
) -> bool:
    """Slew to ``target`` with the jet off, then arm the jet at the arrival pose."""
    if jet_on:
        # Zero-length block. Keep the print feed so a reversed trace's swapped
        # M10 still matches the dispense row that shares this pose.
        rows.append(_pack(current, resume_feed, _JET_OFF))
    rows.append(_pack(target, travel, 0))
    rows.append(_pack(target, resume_feed, _JET_ON))
    return True


def plan_constant_tip_dispense(
    scalp_xyz: np.ndarray,
    normals: np.ndarray,
    b_deg: np.ndarray,
    c_deg: np.ndarray,
    machine: MachineConfig,
    *,
    gap_mm: float | None = None,
) -> np.ndarray:
    """
    Build gcode rows whose jet-on motion holds ``speed_mm_min`` at the tip.

    A block is dispensed only when the C-pivot feed required for that tip
    speed is within ``max_speed_mm_min``. Otherwise the tip advances with the
    previous B/C when that translation fits, and the orientation catch-up is
    a parked jet-off slew at travel feed. When even the frozen-orientation
    advance exceeds the cap, the whole hop is jet-off.
    """
    scalp = np.asarray(scalp_xyz, dtype=float)
    en = np.asarray(normals, dtype=float)
    b = np.round(np.asarray(b_deg, dtype=float), 2)
    c = np.round(np.asarray(c_deg, dtype=float), 2)
    npts = scalp.shape[0]
    if npts == 0:
        return np.zeros((0, 7), dtype=float)

    travel = float(machine.transition_speed_mm_min)
    resume = float(machine.speed_mm_min)

    def pose_at(index: int, b_i: float, c_i: float) -> np.ndarray:
        return _pose_at(scalp[index], en[index], b_i, c_i, machine, gap_mm)

    def segment_feed(start: np.ndarray, end: np.ndarray) -> float | None:
        disp_pivot, disp_tip = _pivot_and_tip(start, end, machine)
        return dispense_pivot_feed(disp_pivot, disp_tip, machine)

    current = pose_at(0, float(b[0]), float(c[0]))
    rows: list[np.ndarray] = [_pack(current, resume, 0)]
    jet_on = True

    for i in range(1, npts):
        desired = pose_at(i, float(b[i]), float(c[i]))
        if _same_pose(current, desired):
            continue
        feed = segment_feed(current, desired)
        if feed is not None:
            rows.append(_pack(desired, feed, 0))
            current = desired
            continue

        held = pose_at(i, float(current[3]), float(current[4]))
        if not _same_pose(current, held):
            hold_feed = segment_feed(current, held)
            if hold_feed is not None:
                rows.append(_pack(held, hold_feed, 0))
                current = held

        if _same_pose(current, desired):
            continue
        jet_on = _jet_off_to(
            rows,
            current,
            desired,
            travel=travel,
            resume_feed=resume,
            jet_on=jet_on,
        )
        current = desired

    return np.vstack(rows)
