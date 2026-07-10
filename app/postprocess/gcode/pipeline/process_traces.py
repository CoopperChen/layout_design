"""Per-trace kinematics processing."""

from __future__ import annotations

import numpy as np

from app.config_loader import load_defaults
from app.postprocess.gcode.kinematics.arm_clearance import (
    HeadMeshInsideChecker,
    resolve_normal_arm_clearance,
)

from ..kinematics.axis_angles import compute_axis_angles
from ..kinematics.feed_rate import compute_feed_rates
from ..kinematics.flip_correction import (
    correct_flip,
    enforce_axis_continuity,
    validate_axis_continuity,
)
from ..kinematics.machine_zero import apply_machine_zero_offset
from ..kinematics.tool_offset import apply_tool_offset
from ..models import MachineConfig, TraceChannel


def _normalize_rows(normals: np.ndarray) -> np.ndarray:
    out = normals.copy()
    for i in range(out.shape[0]):
        length = np.linalg.norm(out[i])
        if length > 0.0:
            out[i] /= length
    return out


def process_trace(
    data: np.ndarray,
    machine: MachineConfig,
    *,
    coords_include_gap: bool = False,
    mesh_points: np.ndarray | None = None,
    mesh_faces: np.ndarray | None = None,
) -> np.ndarray:
    """
    Process one Nx6 trace into gcode rows [X, Y, Z, B, C, F, marker].

    Uses synthesized XYZ and normals as-is. When the rigid C–B arm falls inside
    the registered head mesh, the normal is flipped and B/C recomputed.
    """
    g = data[:, :3].copy()
    en = _normalize_rows(data[:, 3:6].copy())

    checker: HeadMeshInsideChecker | None = None
    if mesh_points is not None and mesh_faces is not None:
        margin = float(
            load_defaults().get("postprocess", {}).get("arm_inside_margin_mm", 0.0)
        )
        checker = HeadMeshInsideChecker(
            mesh_points, mesh_faces, inside_margin_mm=margin
        )

    if checker is not None:
        for i in range(en.shape[0]):
            en[i] = resolve_normal_arm_clearance(
                g[i],
                en[i],
                machine,
                checker,
                coords_include_gap=coords_include_gap,
            )

    g = apply_machine_zero_offset(g, machine)
    b_angles, c_angles = compute_axis_angles(en)
    b_angles, c_angles = correct_flip(b_angles, c_angles)
    b_angles, c_angles = enforce_axis_continuity(b_angles, c_angles)
    max_c_step = float(
        load_defaults().get("postprocess", {}).get("axis_max_c_step_deg", 45.0)
    )
    validate_axis_continuity(b_angles, c_angles, max_c_step_deg=max_c_step)
    offset_gap = 0.0 if coords_include_gap else None
    g = apply_tool_offset(g, en, c_angles, machine, gap_mm=offset_gap)

    b_angles = np.round(b_angles, 2)
    c_angles = np.round(c_angles, 2)

    feed = compute_feed_rates(g, data[:, :3], machine)

    return np.column_stack([g, b_angles, c_angles, feed, np.zeros(len(g))])


def process_all_traces(
    channels: list[TraceChannel],
    machine: MachineConfig,
    choose_trace: int,
    choose_print: int,
    *,
    mesh_points: np.ndarray | None = None,
    mesh_faces: np.ndarray | None = None,
) -> list[np.ndarray]:
    """
    Process channels into gcode row arrays.

    choose_trace: 1=interconnect, 2=electrode
    choose_print: 0=all, else 1-based index
    """
    gcode_list: list[np.ndarray] = []
    names = [ch.name for ch in channels]
    coords_include_gap = choose_trace == 2

    for m, ch in enumerate(channels):
        if choose_trace == 1:
            if choose_print == 0:
                data = ch.interconnect
            elif m + 1 == choose_print or names[m] == str(choose_print):
                data = ch.interconnect
                gcode_list.append(
                    process_trace(
                        data,
                        machine,
                        mesh_points=mesh_points,
                        mesh_faces=mesh_faces,
                    )
                )
                break
            else:
                continue
        else:
            if choose_print == 0:
                data = ch.electrode
            elif m + 1 == choose_print or names[m] == str(choose_print):
                data = ch.electrode
                gcode_list.append(
                    process_trace(
                        data,
                        machine,
                        coords_include_gap=True,
                        mesh_points=mesh_points,
                        mesh_faces=mesh_faces,
                    )
                )
                break
            else:
                continue

        gcode_list.append(
            process_trace(
                data,
                machine,
                coords_include_gap=coords_include_gap,
                mesh_points=mesh_points,
                mesh_faces=mesh_faces,
            )
        )

    return gcode_list
