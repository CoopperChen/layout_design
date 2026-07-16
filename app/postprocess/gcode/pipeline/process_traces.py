"""Per-trace kinematics processing."""

from __future__ import annotations

import numpy as np

from app.config_loader import load_defaults
from app.postprocess.gcode.kinematics.arm_clearance import (
    HeadMeshInsideChecker,
    resolve_normal_arm_clearance,
)
from app.postprocess.mesh_normals import (
    smooth_normals_along_path,
    stabilize_normals_near_pole,
)

from ..kinematics.axis_angles import compute_axis_angles
from ..kinematics.feed_rate import compute_print_feed_rates
from ..kinematics.flip_correction import (
    correct_flip,
    enforce_axis_continuity,
    limit_c_slew,
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
    channel_name: str | None = None,
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

    pp = load_defaults().get("postprocess", {})
    smooth_alpha = float(pp.get("normal_path_smooth_alpha", 0.5))
    smooth_passes = int(pp.get("normal_path_smooth_passes", 1))
    if smooth_passes > 0 and smooth_alpha > 0.0:
        en = smooth_normals_along_path(
            en, alpha=smooth_alpha, passes=smooth_passes
        )
    nxy_min = float(pp.get("c_pole_nxy_min", 0.08))
    if nxy_min > 0.0:
        en = stabilize_normals_near_pole(en, nxy_min=nxy_min)

    g = apply_machine_zero_offset(g, machine)
    b_angles, c_angles = compute_axis_angles(en)
    b_angles, c_angles = correct_flip(b_angles, c_angles)
    b_angles, c_angles = enforce_axis_continuity(b_angles, c_angles)
    c_slew = float(pp.get("c_max_slew_deg", 12.0))
    if c_slew > 0.0:
        b_angles, c_angles = limit_c_slew(
            b_angles, c_angles, max_step_deg=c_slew
        )
    max_c_step = float(pp.get("axis_max_c_step_deg", 90.0))
    try:
        validate_axis_continuity(b_angles, c_angles, max_c_step_deg=max_c_step)
    except ValueError as exc:
        label = channel_name or "trace"
        raise ValueError(f"{label}: {exc}") from exc
    offset_gap = 0.0 if coords_include_gap else None
    g = apply_tool_offset(g, en, c_angles, machine, gap_mm=offset_gap)

    b_angles = np.round(b_angles, 2)
    c_angles = np.round(c_angles, 2)

    feed = compute_print_feed_rates(g, b_angles, c_angles, machine)

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
                        channel_name=ch.name,
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
                        channel_name=ch.name,
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
                channel_name=ch.name,
            )
        )

    return gcode_list
