"""Merge traces with transitions, M-codes, and path reversal."""

from __future__ import annotations

import numpy as np

from ..models import MachineConfig


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


def _apply_last_trace_end(trace: np.ndarray, zsafe: float) -> np.ndarray:
    """MATLAB lines 549-555: M11 on duplicated last row, then Zsafe transition."""
    t = _append_row(trace)
    t[-1, 6] = 11
    t = _append_row(t)
    t[-1, 2] = zsafe
    return t


def _apply_first_trace_approach(trace: np.ndarray, zsafe: float) -> np.ndarray:
    """MATLAB lines 557-559: Zsafe on row 1, approach rows, M10 before jetting."""
    if trace.shape[0] < 2:
        trace = trace.copy()
        trace[0, 2] = zsafe
        return trace

    row1 = trace[0:1].copy()
    row2 = trace[1:2].copy()
    row2_m10 = row2.copy()
    row2_m10[0, 6] = 10
    merged = np.vstack([row1, row2, row2_m10, trace[1:]])
    merged[0, 2] = zsafe
    return merged


def merge_traces(
    gcode_list: list[np.ndarray],
    mesh_z_max: float,
    machine: MachineConfig,
    choose_print: int,
) -> np.ndarray:
    if not gcode_list:
        raise ValueError("No traces to merge")

    traces = [_ensure7(g.copy()) for g in gcode_list]

    for i in range(len(traces)):
        if (i + 1) % 2 == 0:
            traces[i] = np.flipud(traces[i])

    zsafe = round(mesh_z_max + machine.zsafe_margin_mm)

    if choose_print != 0 or len(traces) == 1:
        merged = _apply_last_trace_end(traces[0], zsafe)
        merged = _apply_first_trace_approach(merged, zsafe)
        return merged

    for i in range(len(traces) - 1):
        t = traces[i]
        t = _append_row(t)
        t[-1, 6] = 11
        t = _append_row(t, {5: machine.transition_speed_mm_min, 2: zsafe})
        t = _append_row(t, {5: machine.transition_speed_mm_min, 0: 0, 1: -machine.a_mm})
        t = _append_row(t, {3: traces[i + 1][0, 3], 4: traces[i + 1][0, 4]})
        t = _append_row(t, {0: traces[i + 1][0, 0], 1: traces[i + 1][0, 1]})

        approach = np.zeros((1, 7))
        approach[0, :6] = traces[i + 1][0, :6]
        approach[0, 5] = 0.5 * machine.transition_speed_mm_min
        t = np.vstack([t, approach])

        jet_on = np.zeros((1, 7))
        jet_on[0, :6] = traces[i + 1][0, :6]
        jet_on[0, 6] = 10
        t = np.vstack([t, jet_on])
        traces[i] = t

    traces[-1] = _apply_last_trace_end(traces[-1], zsafe)
    traces[0] = _apply_first_trace_approach(traces[0], zsafe)

    return np.vstack(traces)
