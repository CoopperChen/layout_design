"""Merge traces with transitions, M-codes, and path reversal."""

from __future__ import annotations

import numpy as np

from ..kinematics.arm_clearance import HeadMeshInsideChecker
from ..kinematics.engage_clearance import (
    EngageClearanceConfig,
    build_disengage_offset_rows,
    build_engage_offset_rows,
    compute_disengage_xy_offset,
    compute_engage_xy_offset,
    head_center_xy_from_mesh,
    load_engage_clearance_config,
)
from ..models import MachineConfig


def _transition_feed(machine: MachineConfig) -> float:
    return float(machine.transition_speed_mm_min)


def _offset_feed(machine: MachineConfig) -> float:
    return 0.5 * float(machine.transition_speed_mm_min)


def _ensure7(trace: np.ndarray) -> np.ndarray:
    if trace.ndim == 1:
        trace = trace.reshape(1, -1)
    if trace.shape[1] >= 7:
        return trace
    pad = np.zeros((trace.shape[0], 7 - trace.shape[1]))
    return np.hstack([trace, pad])


def _append_row(trace: np.ndarray, updates: dict[int, float] | None = None) -> np.ndarray:
    row = np.zeros((1, 7))
    row[0, :6] = trace[-1, :6]
    if updates:
        for col, val in updates.items():
            row[0, col] = val
    return np.vstack([trace, row])


def _engage_offset_xy(
    engage_row: np.ndarray,
    zsafe: float,
    machine: MachineConfig,
    *,
    head_center_xy: np.ndarray,
    checker: HeadMeshInsideChecker | None,
    engage_config: EngageClearanceConfig,
) -> np.ndarray:
    return compute_engage_xy_offset(
        engage_row[:6],
        zsafe,
        head_center_xy,
        checker,
        machine,
        engage_config,
    )


def _disengage_offset_xy(
    exit_row: np.ndarray,
    zsafe: float,
    machine: MachineConfig,
    *,
    head_center_xy: np.ndarray,
    checker: HeadMeshInsideChecker | None,
    engage_config: EngageClearanceConfig,
) -> np.ndarray:
    return compute_disengage_xy_offset(
        exit_row[:6],
        zsafe,
        head_center_xy,
        checker,
        machine,
        engage_config,
    )


def _append_engage_approach(
    trace: np.ndarray,
    engage_row: np.ndarray,
    zsafe: float,
    offset_xy: np.ndarray,
    machine: MachineConfig,
) -> np.ndarray:
    """Append engage rows ending at print pose (no M10 duplicate)."""
    prefix = build_engage_offset_rows(engage_row[:6], zsafe, offset_xy, machine)
    if prefix.shape[0] > 0:
        return np.vstack([trace, prefix])

    x, y, z = engage_row[0], engage_row[1], engage_row[2]
    trace = _append_row(
        trace,
        {0: float(x), 1: float(y), 5: _transition_feed(machine)},
    )
    if abs(float(trace[-1, 2]) - float(z)) > 0.5:
        trace = _append_row(trace, {2: float(z), 5: _offset_feed(machine)})
    return trace


def _append_disengage_offset_retract(
    trace: np.ndarray,
    exit_row: np.ndarray,
    zsafe: float,
    offset_xy: np.ndarray,
    machine: MachineConfig,
) -> np.ndarray:
    suffix = build_disengage_offset_rows(exit_row[:6], zsafe, offset_xy, machine)
    if suffix.shape[0] == 0:
        return _append_row(
            trace, {5: _transition_feed(machine), 2: zsafe}
        )
    return np.vstack([trace, suffix])


def _apply_last_trace_end(
    trace: np.ndarray,
    zsafe: float,
    machine: MachineConfig,
    *,
    head_center_xy: np.ndarray,
    checker: HeadMeshInsideChecker | None,
    engage_config: EngageClearanceConfig,
) -> np.ndarray:
    """M11 on duplicated last row, then optional offset retract to Zsafe."""
    exit_row = trace[-1]
    t = _append_row(trace)
    t[-1, 6] = 11
    offset_xy = _disengage_offset_xy(
        exit_row,
        zsafe,
        machine,
        head_center_xy=head_center_xy,
        checker=checker,
        engage_config=engage_config,
    )
    return _append_disengage_offset_retract(t, exit_row, zsafe, offset_xy, machine)


def _apply_first_trace_approach(
    trace: np.ndarray,
    zsafe: float,
    machine: MachineConfig,
    *,
    head_center_xy: np.ndarray,
    checker: HeadMeshInsideChecker | None,
    engage_config: EngageClearanceConfig,
) -> np.ndarray:
    """Zsafe on row 1, approach rows, M10 on first print pose before jetting."""
    if trace.shape[0] < 1:
        return trace

    engage_row = trace[0]
    offset_xy = _engage_offset_xy(
        engage_row,
        zsafe,
        machine,
        head_center_xy=head_center_xy,
        checker=checker,
        engage_config=engage_config,
    )
    prefix = build_engage_offset_rows(engage_row[:6], zsafe, offset_xy, machine)

    jet_on = np.zeros((1, 7))
    jet_on[0, :6] = engage_row[:6]
    jet_on[0, 5] = float(engage_row[5])
    jet_on[0, 6] = 10

    if prefix.shape[0] > 0:
        return np.vstack([prefix, jet_on, trace[1:]])

    row_zsafe = trace[0:1].copy()
    row_zsafe[0, 2] = zsafe
    row_zsafe[0, 5] = _transition_feed(machine)
    row_print = trace[0:1].copy()
    if abs(float(row_zsafe[0, 2]) - float(row_print[0, 2])) > 0.5:
        row_print[0, 5] = _offset_feed(machine)
    return np.vstack([row_zsafe, row_print, jet_on, trace[1:]])


def merge_traces(
    gcode_list: list[np.ndarray],
    mesh_z_max: float,
    machine: MachineConfig,
    choose_print: int,
    *,
    mesh_points: np.ndarray | None = None,
    mesh_faces: np.ndarray | None = None,
    alternate_flip: bool = True,
    skip_origin_between: list[bool] | None = None,
) -> np.ndarray:
    if not gcode_list:
        raise ValueError("No traces to merge")

    traces = [_ensure7(g.copy()) for g in gcode_list]

    if alternate_flip:
        for i in range(len(traces)):
            if (i + 1) % 2 == 0:
                traces[i] = np.flipud(traces[i])

    zsafe = round(mesh_z_max + machine.zsafe_margin_mm)
    engage_config = load_engage_clearance_config(machine)
    head_center_xy = head_center_xy_from_mesh(mesh_points)
    checker: HeadMeshInsideChecker | None = None
    if mesh_points is not None and mesh_faces is not None:
        checker = HeadMeshInsideChecker(mesh_points, mesh_faces)

    if choose_print != 0 or len(traces) == 1:
        merged = _apply_last_trace_end(
            traces[0],
            zsafe,
            machine,
            head_center_xy=head_center_xy,
            checker=checker,
            engage_config=engage_config,
        )
        merged = _apply_first_trace_approach(
            merged,
            zsafe,
            machine,
            head_center_xy=head_center_xy,
            checker=checker,
            engage_config=engage_config,
        )
        return merged

    for i in range(len(traces) - 1):
        t = traces[i]
        exit_row = t[-1]
        t = _append_row(t)
        t[-1, 6] = 11
        offset_xy = _disengage_offset_xy(
            exit_row,
            zsafe,
            machine,
            head_center_xy=head_center_xy,
            checker=checker,
            engage_config=engage_config,
        )
        t = _append_disengage_offset_retract(
            t, exit_row, zsafe, offset_xy, machine
        )
        skip_origin = (
            skip_origin_between is not None
            and i < len(skip_origin_between)
            and skip_origin_between[i]
        )
        if not skip_origin:
            t = _append_row(
                t,
                {
                    5: _transition_feed(machine),
                    0: 0,
                    1: -machine.a_mm,
                },
            )
        t = _append_row(
            t,
            {
                3: traces[i + 1][0, 3],
                4: traces[i + 1][0, 4],
                5: _transition_feed(machine),
            },
        )

        engage_row = traces[i + 1][0]
        offset_xy = _engage_offset_xy(
            engage_row,
            zsafe,
            machine,
            head_center_xy=head_center_xy,
            checker=checker,
            engage_config=engage_config,
        )
        t = _append_engage_approach(t, engage_row, zsafe, offset_xy, machine)

        jet_on = np.zeros((1, 7))
        jet_on[0, :6] = t[-1, :6]
        jet_on[0, 5] = float(engage_row[5])
        jet_on[0, 6] = 10
        t = np.vstack([t, jet_on])
        traces[i] = t

    traces[-1] = _apply_last_trace_end(
        traces[-1],
        zsafe,
        machine,
        head_center_xy=head_center_xy,
        checker=checker,
        engage_config=engage_config,
    )
    traces[0] = _apply_first_trace_approach(
        traces[0],
        zsafe,
        machine,
        head_center_xy=head_center_xy,
        checker=checker,
        engage_config=engage_config,
    )

    return np.vstack(traces)
