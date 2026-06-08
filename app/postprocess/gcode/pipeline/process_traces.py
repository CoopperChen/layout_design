"""Per-trace kinematics processing."""

from __future__ import annotations

import numpy as np

from ..kinematics.axis_angles import compute_axis_angles
from ..kinematics.feed_rate import compute_feed_rates
from ..kinematics.flip_correction import correct_flip
from ..kinematics.machine_zero import apply_machine_zero_offset
from ..kinematics.tool_offset import apply_tool_offset
from ..models import MachineConfig, TraceChannel


def process_trace(
    data: np.ndarray,
    machine: MachineConfig,
) -> np.ndarray:
    """
    Process one Nx6 trace into gcode rows [X, Y, Z, B, C, F, marker].

    marker column is 0 throughout; merge stage adds M10/M11.
    """
    g = data[:, :3].copy()
    en = data[:, 3:6].copy()
    for i in range(en.shape[0]):
        n = np.linalg.norm(en[i])
        if n > 0:
            en[i] /= n

    g = apply_machine_zero_offset(g, machine)
    b_angles, c_angles = compute_axis_angles(en)
    b_angles, c_angles = correct_flip(b_angles, c_angles)
    g = apply_tool_offset(g, en, c_angles, machine)

    b_angles = np.round(b_angles, 2)
    c_angles = np.round(c_angles, 2)

    feed = compute_feed_rates(g, data[:, :3], machine)

    return np.column_stack([g, b_angles, c_angles, feed, np.zeros(len(g))])


def process_all_traces(
    channels: list[TraceChannel],
    machine: MachineConfig,
    choose_trace: int,
    choose_print: int,
) -> list[np.ndarray]:
    """
    Process channels into gcode row arrays.

    choose_trace: 1=interconnect, 2=electrode
    choose_print: 0=all, else 1-based index
    """
    gcode_list: list[np.ndarray] = []
    names = [ch.name for ch in channels]

    for m, ch in enumerate(channels):
        if choose_trace == 1:
            if choose_print == 0:
                data = ch.interconnect
            elif m + 1 == choose_print or names[m] == str(choose_print):
                data = ch.interconnect
                gcode_list.append(process_trace(data, machine))
                break
            else:
                continue
        else:
            if choose_print == 0:
                data = ch.electrode
            elif m + 1 == choose_print or names[m] == str(choose_print):
                data = ch.electrode
                gcode_list.append(process_trace(data, machine))
                break
            else:
                continue

        gcode_list.append(process_trace(data, machine))

    return gcode_list
