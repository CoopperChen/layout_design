"""
Preset v4: 4-landmark rigid registration + 3D chord shape replay + mesh snap.

No fiducial UV, no UV→polar bridge for geometry. Target needs only:
  - cleaned STL, 4 anatomical fiducials, standard 10-20 electrodes.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

import PYTHON.tools.new2dAlterations as new2d
import PYTHON.tools.reconstructUsingUVmesh as recon
from PYTHON.tools.layoutPreset import (
    anatomical_fiducials_to_json,
    build_layout_2d,
    extract_anatomical_fiducials,
    extract_terminal_positions,
    load_subject_data,
    map_preset_terminals_to_target,
    resolve_ga_log_path,
    uv_grid_for_context,
    validate_preset_on_subject,
    _pyvista_read_stl,
)

ENTRY_MODE_TARGET_FIDUCIALS = "target_fiducial_terminals"
TERMINAL_2D_FIDUCIAL = "fiducial_native"
TERMINAL_2D_INFLATED = "inflated_legacy"

PRESET_VERSION_V4 = 4
ENTRY_MODE_ARC_OFFSET = "terminal_arc_offset_slots"
ENTRY_MODE_SYNTHESIZE = "straight_synthesize"
ENTRY_MODE_TARGET_SLOTS = "target_native_slots"
TERMINAL_ANGLE_SEARCH_DEG = 36.0
TERMINAL_ANGLE_STEP_DEG = 3.0
SYNTH_PATH_POINTS = 32
SYNTH_ENTRY_SLIDE_MAX_ROUNDS = 120
SYNTH_MIN_SLOT_GAP = 0.45
SYNTH_TAIL_SWAP_SEPARATION = 1.0
SYNTH_TAIL_SWAP_MAX_ROUNDS = 200


def _metrics_dict(analysis: dict) -> dict:
    return {
        "collision_score": analysis.get("collision_score"),
        "crossing_count": analysis.get("crossing_count"),
        "electrode_violations": analysis.get("electrode_violations"),
        "layout_collision_free": analysis.get("layout_collision_free"),
    }


def head_up_vector(anatomical: dict[str, np.ndarray]) -> np.ndarray:
    """Approximate outward normal from nasion toward inion (AP on scalp)."""
    ap = np.asarray(anatomical["nasion"], dtype=float) - np.asarray(
        anatomical["inion"], dtype=float
    )
    n = np.linalg.norm(ap)
    if n < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return ap / n


def chord_basis_3d(
    e3d: np.ndarray,
    end3d: np.ndarray,
    up_ref: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    delta = np.asarray(end3d, dtype=float) - np.asarray(e3d, dtype=float)
    length = float(np.linalg.norm(delta))
    if length < 1e-12:
        u = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        u = delta / length
    up = np.asarray(up_ref, dtype=float)
    n1 = np.cross(u, up)
    n1n = float(np.linalg.norm(n1))
    if n1n < 1e-12:
        n1 = np.array([0.0, 1.0, 0.0], dtype=float)
    else:
        n1 = n1 / n1n
    n2 = np.cross(u, n1)
    n2 /= np.linalg.norm(n2) + 1e-12
    return u, n1, n2, length


def _arc_length_params_3d(path: np.ndarray) -> np.ndarray:
    path = np.asarray(path, dtype=float)
    if len(path) <= 1:
        return np.zeros(len(path), dtype=float)
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if total < 1e-12:
        return np.linspace(0.0, 1.0, len(path))
    t = s / total
    t[0] = 0.0
    t[-1] = 1.0
    return t


def normalize_path_3d(
    path_3d: np.ndarray,
    e3d: np.ndarray,
    end3d: np.ndarray,
    up_ref: np.ndarray,
    scale_invariant: bool = True,
) -> dict:
    path = np.asarray(path_3d, dtype=float).copy()
    e3d = np.asarray(e3d, dtype=float)
    end3d = np.asarray(end3d, dtype=float)
    if len(path) > 0:
        path[0] = e3d
        path[-1] = end3d
    t_values = _arc_length_params_3d(path)
    u, n1, n2, chord_len = chord_basis_3d(e3d, end3d, up_ref)
    delta = end3d - e3d
    offsets = []
    for p, t in zip(path, t_values):
        on_chord = e3d + float(t) * delta
        residual = np.asarray(p, dtype=float) - on_chord
        off = [
            float(np.dot(residual, n1)),
            float(np.dot(residual, n2)),
        ]
        if scale_invariant and chord_len > 1e-12:
            off = [off[0] / chord_len, off[1] / chord_len]
        offsets.append(off)
    return {
        "t_values": t_values.tolist(),
        "offsets": offsets,
        "chord_length": chord_len,
        "scale_invariant": scale_invariant,
    }


def denormalize_path_3d(
    entry: dict,
    e3d: np.ndarray,
    end3d: np.ndarray,
    up_ref: np.ndarray,
    chord_scale: float = 1.0,
) -> np.ndarray:
    e3d = np.asarray(e3d, dtype=float)
    end3d = np.asarray(end3d, dtype=float)
    t_values = np.asarray(entry["t_values"], dtype=float)
    offsets = np.asarray(entry["offsets"], dtype=float)
    u, n1, n2, chord_len = chord_basis_3d(e3d, end3d, up_ref)
    if entry.get("scale_invariant", True) and chord_len > 1e-12:
        offsets = offsets * (chord_len * float(chord_scale))
    delta = end3d - e3d
    pts = []
    for t, off in zip(t_values, offsets):
        on_chord = e3d + float(t) * delta
        pts.append(on_chord + float(off[0]) * n1 + float(off[1]) * n2)
    out = np.asarray(pts, dtype=float)
    if len(out) > 0:
        out[0] = e3d
        out[-1] = end3d
    return out


def snap_path_to_mesh(
    path_3d: np.ndarray,
    mesh,
    *,
    pin_endpoints: bool = True,
) -> np.ndarray:
    """Snap interior samples to STL; keep endpoints if pin_endpoints."""
    path = np.asarray(path_3d, dtype=float).copy()
    n = len(path)
    if n == 0 or mesh is None:
        return path
    start, end = 0, n
    if pin_endpoints and n >= 2:
        start, end = 1, n - 1
    for i in range(start, end):
        path[i] = np.asarray(
            mesh.points[mesh.find_closest_point(path[i])], dtype=float
        )
    return path


def path_3d_to_entry_index(
    path_3d: np.ndarray,
    path_2d: np.ndarray,
    entry_2d: np.ndarray,
) -> int:
    ep = np.asarray(entry_2d, dtype=float)
    return int(np.argmin(np.linalg.norm(path_2d - ep, axis=1)))


def export_layout_preset_v4(
    source_subject_id: int,
    preset_path: str,
    individual_key: str,
    log_dir: str | None = None,
    preset_id: str | None = None,
    uv_resolution: int = 100,
) -> dict:
    ga_path = resolve_ga_log_path(source_subject_id, individual_key, log_dir)
    with open(ga_path, "r") as f:
        ga_data = json.load(f)

    electrodes, fiducials = load_subject_data(source_subject_id)
    anatomical = extract_anatomical_fiducials(fiducials)
    up_ref = head_up_vector(anatomical)
    terminals_3d = extract_terminal_positions(fiducials)
    electrodes_2d, terminals_2d, cz_pos = build_layout_2d(electrodes, fiducials)

    mesh = _pyvista_read_stl(source_subject_id)
    mesh.compute_normals(inplace=True)
    uv_grid = uv_grid_for_context(
        new2d.create_uv_grid(mesh, cz_pos, resolution=uv_resolution)
    )
    uv_ctx = recon.UVReconstructionContext(uv_grid, mesh)

    paths_chord_3d: dict[str, dict] = {}
    terminal_assignments: dict[str, str] = {}
    export_electrodes: list[str] = []
    export_terminals: list[str] = []
    export_entries_2d: dict[str, np.ndarray] = {}
    export_slot_index: dict[str, int] = {}

    for conn in ga_data.get("paths", []):
        electrode = conn.get("electrode")
        terminal = conn.get("terminal")
        mod_2d = conn.get("modified_path_2d")
        if not electrode or not terminal or mod_2d is None:
            continue
        if electrode not in electrodes_2d or terminal not in terminals_2d:
            continue

        e3d = np.asarray(electrodes[electrode], dtype=float)
        t3d = np.asarray(terminals_3d[terminal], dtype=float)
        e2d = electrodes_2d[electrode]
        t2d = terminals_2d[terminal]
        path_2d = new2d.pin_path_endpoints_2d(
            np.asarray(mod_2d, dtype=float), e2d, t2d
        )
        path_3d = uv_ctx.reconstruct(e3d, t3d, path_2d)

        end3d = t3d
        entry_meta: dict[str, Any] = {}
        if conn.get("entry_point_2d") is not None:
            idx = path_3d_to_entry_index(
                path_3d, path_2d, np.asarray(conn["entry_point_2d"], dtype=float)
            )
            end3d = np.asarray(path_3d[idx], dtype=float)
            entry_meta = {
                "entry_position_3d": end3d.tolist(),
                "entry_point_2d": np.asarray(conn["entry_point_2d"], dtype=float).tolist(),
            }
            if conn.get("slot_index") is not None:
                entry_meta["slot_index"] = int(conn["slot_index"])

        norm = normalize_path_3d(path_3d, e3d, end3d, up_ref, scale_invariant=True)
        paths_chord_3d[electrode] = {
            "terminal": terminal,
            **norm,
            **entry_meta,
        }
        terminal_assignments[electrode] = terminal
        export_electrodes.append(electrode)
        export_terminals.append(terminal)
        if conn.get("slot_index") is not None:
            export_slot_index[electrode] = int(conn["slot_index"])
        if conn.get("entry_point_2d") is not None:
            export_entries_2d[electrode] = np.asarray(
                conn["entry_point_2d"], dtype=float
            )

    if export_entries_2d and export_slot_index:
        _, terminal_zones = new2d.create_zones(electrodes_2d, terminals_2d)
        arc_offsets = export_entry_arc_offsets(
            export_electrodes,
            export_terminals,
            export_entries_2d,
            electrodes_2d,
            terminals_2d,
            terminal_zones,
            export_slot_index,
        )
        for electrode, off in arc_offsets.items():
            paths_chord_3d[electrode]["entry_arc_offset_slots"] = off

    if not paths_chord_3d:
        raise ValueError(f"No paths exported from {ga_path}")

    preset = {
        "preset_version": PRESET_VERSION_V4,
        "preset_id": preset_id or f"subject{source_subject_id}_{individual_key}",
        "electrode_layout": "standard_10-20",
        "source_subject_id": source_subject_id,
        "source_individual": individual_key,
        "source_ga_log": ga_path.replace("\\", "/"),
        "source_anatomical_fiducials": anatomical_fiducials_to_json(anatomical),
        "terminal_positions_3d": {k: v.tolist() for k, v in terminals_3d.items()},
        "terminal_inheritance": "rigid_anatomical_landmarks",
        "terminal_entry_mode": ENTRY_MODE_ARC_OFFSET,
        "terminal_assignments": terminal_assignments,
        "paths_chord_3d": paths_chord_3d,
    }

    os.makedirs(os.path.dirname(preset_path) or ".", exist_ok=True)
    with open(preset_path, "w") as f:
        json.dump(preset, f, indent=2)
    print(
        f"Exported v4 preset '{preset['preset_id']}' "
        f"({len(paths_chord_3d)} paths) → {preset_path}"
    )
    return preset


def _path_list_from_preset(
    preset: dict,
    electrodes: dict,
) -> tuple[list[str], list[str]]:
    """Electrode names and terminals from preset (assignments-only or full v4)."""
    assignments = preset.get("terminal_assignments") or {}
    path_electrodes: list[str] = []
    path_terminals: list[str] = []
    for electrode, terminal in assignments.items():
        if electrode not in electrodes or not terminal:
            continue
        path_electrodes.append(electrode)
        path_terminals.append(terminal)
    if path_electrodes:
        return path_electrodes, path_terminals
    for electrode, entry in preset.get("paths_chord_3d", {}).items():
        terminal = entry.get("terminal") or assignments.get(electrode)
        if not terminal or electrode not in electrodes:
            continue
        path_electrodes.append(electrode)
        path_terminals.append(terminal)
    return path_electrodes, path_terminals


def _preset_slot_metadata(
    preset: dict,
) -> tuple[dict[str, int], dict[str, list[str]]]:
    """slot_index per electrode and ordered electrode names per terminal."""
    slot_index: dict[str, int] = {}
    by_terminal: dict[str, list[tuple[int, str]]] = {}
    for electrode, entry in preset.get("paths_chord_3d", {}).items():
        terminal = entry.get("terminal") or preset.get("terminal_assignments", {}).get(
            electrode
        )
        if entry.get("slot_index") is not None:
            slot_index[electrode] = int(entry["slot_index"])
        if terminal and electrode in slot_index:
            by_terminal.setdefault(terminal, []).append(
                (slot_index[electrode], electrode)
            )
    slot_order = {
        term: [name for _, name in sorted(items, key=lambda row: row[0])]
        for term, items in by_terminal.items()
    }
    return slot_index, slot_order


def _kendall_inversions(sequence: list[str], desired_order: list[str]) -> int:
    rank = {name: i for i, name in enumerate(desired_order)}
    filtered = [name for name in sequence if name in rank]
    inv = 0
    for i in range(len(filtered)):
        for j in range(i + 1, len(filtered)):
            if rank[filtered[i]] > rank[filtered[j]]:
                inv += 1
    return inv


def _rotate_terminal_3d_about_cz(
    terminal_3d: np.ndarray,
    cz_pos: np.ndarray,
    delta_rad: float,
) -> np.ndarray:
    """Rotate terminal in the Cz-centered polar plane (scalp tangent plane)."""
    rel = np.asarray(terminal_3d, dtype=float) - np.asarray(cz_pos, dtype=float)
    c = float(np.cos(delta_rad))
    s = float(np.sin(delta_rad))
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return np.asarray(cz_pos, dtype=float) + rot @ rel


def _natural_strip_order_on_terminal(
    terminal: str,
    electrodes_on_terminal: list[str],
    electrodes_2d: dict[str, np.ndarray],
    terminals_2d: dict[str, np.ndarray],
    terminal_zones: dict,
) -> list[str]:
    zone = terminal_zones[terminal]
    terminal_pos = np.asarray(terminals_2d[terminal], dtype=float)
    hits = []
    for name in electrodes_on_terminal:
        start = np.asarray(electrodes_2d[name], dtype=float)
        end = terminal_pos
        chord = np.stack([start, end], axis=0)
        hit = new2d._path_terminal_zone_entry_point(chord, zone)
        if hit is None:
            hit = end
        hits.append(np.asarray(hit, dtype=float))
    strip_sort = new2d._strip_sort_key_for_terminal(terminal_pos, hits, zone)
    rows = sorted(
        zip(electrodes_on_terminal, hits),
        key=lambda row: strip_sort(row[1]),
    )
    return [name for name, _ in rows]


def _terminal_hub_inversion_score(
    terminal: str,
    desired_order: list[str],
    electrodes_2d: dict[str, np.ndarray],
    terminals_2d: dict[str, np.ndarray],
    terminal_zones: dict,
) -> int:
    natural = _natural_strip_order_on_terminal(
        terminal, desired_order, electrodes_2d, terminals_2d, terminal_zones
    )
    return _kendall_inversions(natural, desired_order)


def fit_terminals_to_preset_slot_order(
    preset: dict,
    electrodes: dict[str, np.ndarray],
    terminals_3d: dict[str, np.ndarray],
    layout_fiducials: dict,
    *,
    search_deg: float = TERMINAL_ANGLE_SEARCH_DEG,
    step_deg: float = TERMINAL_ANGLE_STEP_DEG,
) -> tuple[dict[str, np.ndarray], dict, dict[str, np.ndarray], np.ndarray, dict]:
    """
    Slide each terminal hub in polar angle so target chord-hit order matches
    preset slot order (sequence preserved, geometry aligned).
    """
    slot_index, slot_order = _preset_slot_metadata(preset)
    if not slot_order:
        electrodes_2d, terminals_2d, cz_pos = build_layout_2d(electrodes, layout_fiducials)
        return terminals_3d, layout_fiducials, terminals_2d, cz_pos, slot_index

    electrodes_2d, terminals_2d, cz_pos = build_layout_2d(electrodes, layout_fiducials)
    _, terminal_zones = new2d.create_zones(electrodes_2d, terminals_2d)

    terminals_3d = {k: np.asarray(v, dtype=float).copy() for k, v in terminals_3d.items()}
    layout_fiducials = dict(layout_fiducials)

    steps = max(1, int(round(2 * search_deg / step_deg)) + 1)
    angles = np.linspace(-np.deg2rad(search_deg), np.deg2rad(search_deg), steps)

    for terminal, desired in slot_order.items():
        if terminal not in terminals_3d or len(desired) < 2:
            continue
        base_3d = terminals_3d[terminal].copy()
        base_2d = np.asarray(terminals_2d[terminal], dtype=float)
        base_angle = float(np.arctan2(base_2d[1], base_2d[0]))
        radius = float(np.linalg.norm(base_2d))

        best_delta = 0.0
        best_score = _terminal_hub_inversion_score(
            terminal, desired, electrodes_2d, terminals_2d, terminal_zones
        )

        for delta in angles:
            trial_2d = dict(terminals_2d)
            ang = base_angle + float(delta)
            trial_2d[terminal] = radius * np.array(
                [np.cos(ang), np.sin(ang)], dtype=float
            )
            _, trial_zones = new2d.create_zones(electrodes_2d, trial_2d)
            score = _terminal_hub_inversion_score(
                terminal, desired, electrodes_2d, trial_2d, trial_zones
            )
            if score < best_score:
                best_score = score
                best_delta = float(delta)

        if abs(best_delta) > 1e-9:
            terminals_3d[terminal] = _rotate_terminal_3d_about_cz(
                base_3d, cz_pos, best_delta
            )
            layout_fiducials[terminal] = terminals_3d[terminal]
            print(
                f"  Terminal {terminal}: angle {np.rad2deg(best_delta):+.1f}° "
                f"(inversions {best_score})"
            )

    electrodes_2d, terminals_2d, cz_pos = build_layout_2d(electrodes, layout_fiducials)
    return terminals_3d, layout_fiducials, terminals_2d, cz_pos, slot_index


def assign_entry_slots_fixed_order(
    path_electrodes: list[str],
    path_terminals: list[str],
    electrodes_2d: dict[str, np.ndarray],
    terminals_2d: dict[str, np.ndarray],
    terminal_zones: dict,
    slot_index_by_electrode: dict[str, int],
    spacing: float = new2d.TERMINAL_ENTRY_SLOT_SPACING,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Place entry slots on the strip using preset slot_index order (no reorder)."""
    from shapely.geometry import Point

    entry_points: dict[str, np.ndarray] = {}
    slot_index_out: dict[str, int] = dict(slot_index_by_electrode)

    for terminal in sorted(set(path_terminals)):
        if terminal not in terminal_zones:
            continue
        zone = terminal_zones[terminal]
        terminal_pos = np.asarray(terminals_2d[terminal], dtype=float)
        boundary = zone.boundary
        blen = float(boundary.length)

        indices = [i for i, t in enumerate(path_terminals) if t == terminal]
        hit_rows = []
        for idx in indices:
            electrode = path_electrodes[idx]
            start = np.asarray(electrodes_2d[electrode], dtype=float)
            end = terminal_pos
            chord = np.stack([start, end], axis=0)
            hit = new2d._path_terminal_zone_entry_point(chord, zone)
            if hit is None:
                hit = end
            hit = np.asarray(hit, dtype=float)
            si = slot_index_by_electrode.get(electrode, idx)
            hit_rows.append((electrode, si, hit, float(boundary.project(Point(hit)))))

        if not hit_rows:
            continue

        hit_rows.sort(key=lambda row: row[1])
        n = len(hit_rows)
        mid = n // 2
        anchor_hit = hit_rows[mid][2]
        anchor_s = float(boundary.project(Point(anchor_hit)))

        natural_arcs = [row[3] for row in hit_rows]
        unwrapped = new2d._unwrap_arc_distances(natural_arcs, blen)
        arc_sign = 1.0
        if len(unwrapped) > 1 and float(unwrapped[-1] - unwrapped[0]) < 0:
            arc_sign = -1.0

        for slot_idx, (name, _, _, _) in enumerate(hit_rows):
            entry_s = anchor_s + arc_sign * (slot_idx - mid) * spacing
            pt = boundary.interpolate(entry_s % blen)
            entry_points[name] = np.array([pt.x, pt.y], dtype=float)
            slot_index_out[name] = slot_idx

    return entry_points, slot_index_out


def entry_3d_from_chord_fraction(
    e3d: np.ndarray,
    terminal_3d: np.ndarray,
    e2d: np.ndarray,
    entry_2d: np.ndarray,
    cz_pos: np.ndarray,
) -> np.ndarray:
    """Lift 2D entry to 3D by matching arc fraction along the electrode→terminal chord."""
    term_2d = new2d.polar_projection(np.array([terminal_3d]), cz_pos)[0]
    d_total = float(np.linalg.norm(term_2d - e2d))
    d_entry = float(np.linalg.norm(entry_2d - e2d))
    alpha = (d_entry / d_total) if d_total > 1e-12 else 1.0
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return np.asarray(e3d, dtype=float) + alpha * (
        np.asarray(terminal_3d, dtype=float) - np.asarray(e3d, dtype=float)
    )


def entry_3d_from_uv(
    entry_2d: np.ndarray,
    uv_context: recon.UVReconstructionContext,
    mesh,
) -> np.ndarray:
    """Map 2D strip entry to scalp using the same UV grid as path reconstruction."""
    entry_2d = np.asarray(entry_2d, dtype=float)
    _, idx = uv_context.kdtree_2d.query(entry_2d)
    pt = np.asarray(uv_context.grid_3d[int(idx)], dtype=float)
    snapped = snap_path_to_mesh(np.array([pt]), mesh, pin_endpoints=False)
    return np.asarray(snapped[0], dtype=float) if len(snapped) else pt


def entry_3d_for_strip(
    entry_2d: np.ndarray,
    uv_context: recon.UVReconstructionContext | None,
    mesh,
    *,
    e3d: np.ndarray,
    terminal_3d: np.ndarray,
    e2d: np.ndarray,
    cz_pos: np.ndarray,
    terminal_2d_mode: str = TERMINAL_2D_INFLATED,
    terminal_zone_size: float | None = None,
) -> np.ndarray:
    """Surface entry at strip; UV lift when context available, else chord fraction."""
    if uv_context is not None and mesh is not None:
        end3d = entry_3d_from_uv(entry_2d, uv_context, mesh)
    else:
        end3d = entry_3d_from_chord_fraction(e3d, terminal_3d, e2d, entry_2d, cz_pos)

    if (
        terminal_2d_mode == TERMINAL_2D_FIDUCIAL
        and mesh is not None
        and terminal_zone_size is not None
        and terminal_zone_size > 0
    ):
        term_2d = new2d.polar_projection(np.array([terminal_3d]), cz_pos)[0]
        d_hub = float(np.linalg.norm(np.asarray(entry_2d, dtype=float) - term_2d))
        blend = max(0.0, 1.0 - d_hub / (terminal_zone_size * 2.5))
        if blend > 1e-6:
            hub_snap = snap_path_to_mesh(
                np.array([terminal_3d]), mesh, pin_endpoints=False
            )[0]
            end3d = (1.0 - blend) * np.asarray(end3d, dtype=float) + blend * hub_snap
            end3d = snap_path_to_mesh(np.array([end3d]), mesh, pin_endpoints=False)[0]
    return np.asarray(end3d, dtype=float)


def pin_path_endpoints_3d(
    path_3d: np.ndarray,
    e3d: np.ndarray,
    end3d: np.ndarray,
    mesh,
) -> np.ndarray:
    """Pin path ends; snap terminal entry to mesh (avoids off-surface chord endpoint)."""
    path = np.asarray(path_3d, dtype=float).copy()
    if len(path) == 0:
        return path
    path[0] = np.asarray(e3d, dtype=float)
    if len(path) >= 2:
        if mesh is not None:
            path[-1] = snap_path_to_mesh(
                np.array([end3d]), mesh, pin_endpoints=False
            )[0]
        else:
            path[-1] = np.asarray(end3d, dtype=float)
    return path


def _chord_hit_on_terminal_zone(
    electrode: str,
    terminal: str,
    electrodes_2d: dict[str, np.ndarray],
    terminals_2d: dict[str, np.ndarray],
    terminal_zones: dict,
) -> np.ndarray:
    zone = terminal_zones[terminal]
    start = np.asarray(electrodes_2d[electrode], dtype=float)
    end = np.asarray(terminals_2d[terminal], dtype=float)
    chord = np.stack([start, end], axis=0)
    hit = new2d._path_terminal_zone_entry_point(chord, zone)
    if hit is None:
        hit = end
    return np.asarray(hit, dtype=float)


def _terminal_arc_sign(
    terminal: str,
    electrodes_on_terminal: list[str],
    electrodes_2d: dict[str, np.ndarray],
    terminals_2d: dict[str, np.ndarray],
    terminal_zones: dict,
) -> float:
    zone = terminal_zones[terminal]
    boundary = zone.boundary
    blen = float(boundary.length)
    arcs = []
    for name in electrodes_on_terminal:
        hit = _chord_hit_on_terminal_zone(
            name, terminal, electrodes_2d, terminals_2d, terminal_zones
        )
        from shapely.geometry import Point

        arcs.append(float(boundary.project(Point(hit))))
    unwrapped = new2d._unwrap_arc_distances(arcs, blen)
    if len(unwrapped) > 1 and float(unwrapped[-1] - unwrapped[0]) < 0:
        return -1.0
    return 1.0


def _anchor_electrode_for_terminal(
    electrodes_on_terminal: list[tuple[int, str]],
) -> str:
    ordered = sorted(electrodes_on_terminal, key=lambda row: row[0])
    mid = len(ordered) // 2
    return ordered[mid][1]


def export_entry_arc_offsets(
    path_electrodes: list[str],
    path_terminals: list[str],
    entry_points_2d: dict[str, np.ndarray],
    electrodes_2d: dict[str, np.ndarray],
    terminals_2d: dict[str, np.ndarray],
    terminal_zones: dict,
    slot_index_by_electrode: dict[str, int],
    spacing: float = new2d.TERMINAL_ENTRY_SLOT_SPACING,
) -> dict[str, float]:
    """
    Subject-agnostic strip coordinates: signed slot offset from hub anchor on zone boundary.
    """
    from shapely.geometry import Point

    offsets: dict[str, float] = {}
    by_terminal: dict[str, list[tuple[int, str]]] = {}
    for electrode, terminal in zip(path_electrodes, path_terminals):
        si = slot_index_by_electrode.get(electrode)
        if si is None or electrode not in entry_points_2d:
            continue
        by_terminal.setdefault(terminal, []).append((int(si), electrode))

    for terminal, rows in by_terminal.items():
        if terminal not in terminal_zones:
            continue
        zone = terminal_zones[terminal]
        boundary = zone.boundary
        names = [name for _, name in rows]
        arc_sign = _terminal_arc_sign(
            terminal, names, electrodes_2d, terminals_2d, terminal_zones
        )
        anchor_name = _anchor_electrode_for_terminal(rows)
        anchor_hit = _chord_hit_on_terminal_zone(
            anchor_name, terminal, electrodes_2d, terminals_2d, terminal_zones
        )
        anchor_s = float(boundary.project(Point(anchor_hit)))

        for _slot_idx, electrode in rows:
            entry_s = float(boundary.project(Point(entry_points_2d[electrode])))
            delta = entry_s - anchor_s
            if arc_sign < 0:
                delta = -delta
            offsets[electrode] = float(delta / spacing)

    return offsets


def apply_entry_slots_from_arc_offsets(
    preset_paths: dict[str, dict],
    path_electrodes: list[str],
    path_terminals: list[str],
    electrodes_2d: dict[str, np.ndarray],
    terminals_2d: dict[str, np.ndarray],
    terminal_zones: dict,
    slot_index_by_electrode: dict[str, int],
    spacing: float = new2d.TERMINAL_ENTRY_SLOT_SPACING,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Map preset strip offsets onto target terminal zones (sequence + relative position)."""
    from shapely.geometry import Point

    entry_points: dict[str, np.ndarray] = {}
    slot_index_out: dict[str, int] = dict(slot_index_by_electrode)
    by_terminal: dict[str, list[tuple[int, str]]] = {}

    for electrode, terminal in zip(path_electrodes, path_terminals):
        si = slot_index_by_electrode.get(electrode)
        if si is None:
            continue
        by_terminal.setdefault(terminal, []).append((int(si), electrode))

    for terminal, rows in by_terminal.items():
        if terminal not in terminal_zones:
            continue
        zone = terminal_zones[terminal]
        boundary = zone.boundary
        names = [name for _, name in rows]
        arc_sign = _terminal_arc_sign(
            terminal, names, electrodes_2d, terminals_2d, terminal_zones
        )
        anchor_name = _anchor_electrode_for_terminal(rows)
        anchor_hit = _chord_hit_on_terminal_zone(
            anchor_name, terminal, electrodes_2d, terminals_2d, terminal_zones
        )
        anchor_s = float(boundary.project(Point(anchor_hit)))
        blen = float(boundary.length)

        for slot_idx, electrode in rows:
            preset_entry = preset_paths.get(electrode, {})
            if preset_entry.get("entry_arc_offset_slots") is not None:
                off_slots = float(preset_entry["entry_arc_offset_slots"])
            else:
                mid = len(rows) // 2
                off_slots = float(slot_idx - mid)

            entry_s = anchor_s + arc_sign * off_slots * spacing
            pt = boundary.interpolate(entry_s % blen)
            entry_points[electrode] = np.array([pt.x, pt.y], dtype=float)
            slot_index_out[electrode] = int(slot_idx)

    return entry_points, slot_index_out


def _straight_path_2d(start: np.ndarray, end: np.ndarray, n: int = SYNTH_PATH_POINTS) -> np.ndarray:
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    t = np.linspace(0.0, 1.0, max(2, n))
    return start + t[:, None] * (end - start)


def _path_foreign_electrode_hits(
    path: np.ndarray, electrode: str, electrode_zones: dict
) -> int:
    return int(
        new2d.count_single_trace_electrode_violations(
            path, electrode, electrode_zones
        )
    )


def _path_with_electrode_detour(
    start: np.ndarray,
    end: np.ndarray,
    electrode: str,
    electrode_zones: dict,
    *,
    n: int = SYNTH_PATH_POINTS,
) -> np.ndarray:
    """Straight chord, or one perpendicular midpoint bump if a foreign zone blocks."""
    straight = _straight_path_2d(start, end, n=n)
    if _path_foreign_electrode_hits(straight, electrode, electrode_zones) == 0:
        return straight

    from shapely.geometry import LineString, Point

    chord = LineString([start, end])
    best_path = straight
    best_hits = _path_foreign_electrode_hits(straight, electrode, electrode_zones)
    mid = 0.5 * (np.asarray(start, dtype=float) + np.asarray(end, dtype=float))
    direction = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    perp = np.array([-direction[1], direction[0]], dtype=float)
    pn = float(np.linalg.norm(perp))
    if pn < 1e-9:
        return straight
    perp = perp / pn

    for sign in (1.0, -1.0):
        for scale in (8.0, 16.0, 28.0, 44.0):
            waypoint = mid + sign * scale * perp
            leg_a = _straight_path_2d(start, waypoint, n=max(2, n // 2))
            leg_b = _straight_path_2d(waypoint, end, n=max(2, n // 2))
            trial = np.vstack([leg_a[:-1], leg_b])
            hits = _path_foreign_electrode_hits(trial, electrode, electrode_zones)
            if hits < best_hits:
                best_hits = hits
                best_path = trial
            if hits == 0:
                return trial

    for name, zone in electrode_zones["zones"].items():
        if name == electrode:
            continue
        try:
            if not chord.intersects(zone):
                continue
            center = np.asarray(
                electrode_zones["metadata"][name]["center"], dtype=float
            )
            push = mid - center
            norm = float(np.linalg.norm(push))
            if norm < 1e-9:
                continue
            waypoint = mid + (18.0 / norm) * push
            leg_a = _straight_path_2d(start, waypoint, n=max(2, n // 2))
            leg_b = _straight_path_2d(waypoint, end, n=max(2, n // 2))
            trial = np.vstack([leg_a[:-1], leg_b])
            hits = _path_foreign_electrode_hits(trial, electrode, electrode_zones)
            if hits < best_hits:
                best_hits = hits
                best_path = trial
            if hits == 0:
                return trial
        except Exception:
            continue
    return best_path


def _count_pair_crossings(paths: list[np.ndarray]) -> int:
    from shapely.geometry import LineString

    n_cross = 0
    for i in range(len(paths)):
        li = LineString(paths[i])
        for j in range(i + 1, len(paths)):
            inter = li.intersection(LineString(paths[j]))
            if inter.is_empty:
                continue
            if inter.geom_type == "Point":
                n_cross += 1
            elif inter.geom_type == "MultiPoint" and len(inter.geoms) > 0:
                n_cross += 1
    return n_cross


def _layout_violation_score(
    paths: list[np.ndarray],
    path_electrodes: list[str],
    path_terminals: list[str],
    electrode_zones: dict,
    terminal_zones: dict,
) -> tuple[int, int]:
    cross = _count_pair_crossings(paths)
    ev = new2d.count_electrode_violations(
        paths, electrode_zones, path_electrodes
    )
    return int(cross), int(ev)


def _offsets_from_preset(
    preset_paths: dict[str, dict],
    path_electrodes: list[str],
    path_terminals: list[str],
    slot_index_by_electrode: dict[str, int],
) -> dict[str, float]:
    offsets: dict[str, float] = {}
    for electrode, terminal in zip(path_electrodes, path_terminals):
        entry = preset_paths.get(electrode, {})
        if entry.get("entry_arc_offset_slots") is not None:
            offsets[electrode] = float(entry["entry_arc_offset_slots"])
        elif electrode in slot_index_by_electrode:
            offsets[electrode] = float(slot_index_by_electrode[electrode])
    return offsets


def _entries_from_offset_map(
    offset_map: dict[str, float],
    path_electrodes: list[str],
    path_terminals: list[str],
    electrodes_2d: dict[str, np.ndarray],
    terminals_2d: dict[str, np.ndarray],
    terminal_zones: dict,
    slot_index_by_electrode: dict[str, int],
    spacing: float = new2d.TERMINAL_ENTRY_SLOT_SPACING,
) -> dict[str, np.ndarray]:
    from shapely.geometry import Point

    entry_points: dict[str, np.ndarray] = {}
    by_terminal: dict[str, list[tuple[int, str]]] = {}
    for electrode, terminal in zip(path_electrodes, path_terminals):
        si = slot_index_by_electrode.get(electrode)
        if si is None:
            continue
        by_terminal.setdefault(terminal, []).append((int(si), electrode))

    for terminal, rows in by_terminal.items():
        if terminal not in terminal_zones:
            continue
        zone = terminal_zones[terminal]
        boundary = zone.boundary
        blen = float(boundary.length)
        names = [name for _, name in rows]
        arc_sign = _terminal_arc_sign(
            terminal, names, electrodes_2d, terminals_2d, terminal_zones
        )
        anchor_name = _anchor_electrode_for_terminal(rows)
        anchor_hit = _chord_hit_on_terminal_zone(
            anchor_name, terminal, electrodes_2d, terminals_2d, terminal_zones
        )
        anchor_s = float(boundary.project(Point(anchor_hit)))

        for slot_idx, electrode in rows:
            off = float(offset_map.get(electrode, float(slot_idx - len(rows) // 2)))
            entry_s = anchor_s + arc_sign * off * spacing
            pt = boundary.interpolate(entry_s % blen)
            entry_points[electrode] = np.array([pt.x, pt.y], dtype=float)
    return entry_points


def _offsets_order_ok(
    rows: list[tuple[int, str]],
    offset_map: dict[str, float],
    arc_sign: float,
    min_gap: float = SYNTH_MIN_SLOT_GAP,
) -> bool:
    ordered = sorted(rows, key=lambda row: row[0])
    values = [float(offset_map[name]) for _, name in ordered]
    for i in range(len(values) - 1):
        if arc_sign >= 0 and values[i + 1] < values[i] + min_gap:
            return False
        if arc_sign < 0 and values[i + 1] > values[i] - min_gap:
            return False
    return True


def _slide_entry_offsets_for_clearance(
    offset_map: dict[str, float],
    path_electrodes: list[str],
    path_terminals: list[str],
    electrodes_2d: dict[str, np.ndarray],
    terminals_2d: dict[str, np.ndarray],
    terminal_zones: dict,
    electrode_zones: dict,
    slot_index_by_electrode: dict[str, int],
    *,
    max_rounds: int = SYNTH_ENTRY_SLIDE_MAX_ROUNDS,
) -> dict[str, float]:
    from shapely.geometry import LineString

    offset_map = dict(offset_map)
    by_terminal: dict[str, list[tuple[int, str]]] = {}
    for electrode, terminal in zip(path_electrodes, path_terminals):
        si = slot_index_by_electrode.get(electrode)
        if si is None:
            continue
        by_terminal.setdefault(terminal, []).append((int(si), electrode))

    for round_idx in range(max_rounds):
        entries = _entries_from_offset_map(
            offset_map,
            path_electrodes,
            path_terminals,
            electrodes_2d,
            terminals_2d,
            terminal_zones,
            slot_index_by_electrode,
        )
        paths = [
            _straight_path_2d(electrodes_2d[e], entries[e]) for e in path_electrodes
        ]
        cross, _ = _layout_violation_score(
            paths, path_electrodes, path_terminals, electrode_zones, terminal_zones
        )
        if cross == 0:
            break

        lines = [LineString(p) for p in paths]
        improved = False
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                inter = lines[i].intersection(lines[j])
                if inter.is_empty or inter.geom_type not in ("Point", "MultiPoint"):
                    continue
                for idx in (i, j):
                    electrode = path_electrodes[idx]
                    terminal = path_terminals[idx]
                    rows = by_terminal.get(terminal, [])
                    arc_sign = _terminal_arc_sign(
                        terminal,
                        [n for _, n in rows],
                        electrodes_2d,
                        terminals_2d,
                        terminal_zones,
                    )
                    for step in (0.5, -0.5, 1.0, -1.0):
                        trial = dict(offset_map)
                        trial[electrode] = float(trial.get(electrode, 0.0)) + step
                        if not _offsets_order_ok(rows, trial, arc_sign):
                            continue
                        trial_entries = _entries_from_offset_map(
                            trial,
                            path_electrodes,
                            path_terminals,
                            electrodes_2d,
                            terminals_2d,
                            terminal_zones,
                            slot_index_by_electrode,
                        )
                        trial_paths = list(paths)
                        trial_paths[idx] = _straight_path_2d(
                            electrodes_2d[electrode], trial_entries[electrode]
                        )
                        c2, _ = _layout_violation_score(
                            trial_paths,
                            path_electrodes,
                            path_terminals,
                            electrode_zones,
                            terminal_zones,
                        )
                        if c2 < cross:
                            offset_map = trial
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            print(f"  Entry slide stopped at round {round_idx + 1} (crossings={cross})")
            break
    return offset_map


def _bent_path_2d(
    start: np.ndarray,
    end: np.ndarray,
    perp_sign: float,
    scale: float,
    *,
    n: int = SYNTH_PATH_POINTS,
) -> np.ndarray:
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    direction = end - start
    perp = np.array([-direction[1], direction[0]], dtype=float)
    pn = float(np.linalg.norm(perp))
    if pn < 1e-9:
        return _straight_path_2d(start, end, n=n)
    perp = perp_sign * scale * (perp / pn)
    mid = 0.5 * (start + end) + perp
    leg_a = _straight_path_2d(start, mid, n=max(2, n // 2))
    leg_b = _straight_path_2d(mid, end, n=max(2, n // 2))
    return np.vstack([leg_a[:-1], leg_b])


def _first_crossing_point(path_a: np.ndarray, path_b: np.ndarray) -> np.ndarray | None:
    from shapely.geometry import LineString

    inter = LineString(path_a).intersection(LineString(path_b))
    if inter.is_empty:
        return None
    if inter.geom_type == "Point":
        return np.array([inter.x, inter.y], dtype=float)
    if inter.geom_type == "MultiPoint" and len(inter.geoms) > 0:
        g = inter.geoms[0]
        return np.array([g.x, g.y], dtype=float)
    return None


def _split_path_at_point(path: np.ndarray, point: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    path = np.asarray(path, dtype=float)
    point = np.asarray(point, dtype=float)
    idx = int(np.argmin(np.linalg.norm(path - point, axis=1)))
    idx = max(0, min(idx, len(path) - 1))
    return path[: idx + 1].copy(), path[idx:].copy()


def _separate_entry_points(
    entry_a: np.ndarray,
    entry_b: np.ndarray,
    terminal: str,
    terminals_2d: dict[str, np.ndarray],
    terminal_zones: dict,
    separation: float = SYNTH_TAIL_SWAP_SEPARATION,
) -> tuple[np.ndarray, np.ndarray]:
    """Shift two hub entries apart along the strip tangent."""
    if terminal not in terminal_zones:
        return entry_a, entry_b
    zone = terminal_zones[terminal]
    terminal_pos = np.asarray(terminals_2d[terminal], dtype=float)
    tangent = new2d._base_strip_tangent(terminal_pos)
    half = 0.5 * float(separation)
    return (
        np.asarray(entry_a, dtype=float) + half * tangent,
        np.asarray(entry_b, dtype=float) - half * tangent,
    )


def _uncross_by_tail_swap(
    paths: list[np.ndarray],
    path_electrodes: list[str],
    path_terminals: list[str],
    electrodes_2d: dict[str, np.ndarray],
    entry_points: dict[str, np.ndarray],
    terminals_2d: dict[str, np.ndarray],
    terminal_zones: dict,
    electrode_zones: dict,
    *,
    separation: float = SYNTH_TAIL_SWAP_SEPARATION,
    max_rounds: int = SYNTH_TAIL_SWAP_MAX_ROUNDS,
) -> tuple[list[np.ndarray], dict[str, np.ndarray]]:
    """
    At a crossing, swap downstream tails (and hub entry slots) then nudge entries apart.

    Works best when the cross lies on/near the terminal approach; scalp-mid crosses
    may need a body reroute because heads are unchanged.
    """
    paths = [np.asarray(p, dtype=float).copy() for p in paths]
    entries = {k: np.asarray(v, dtype=float).copy() for k, v in entry_points.items()}

    for round_idx in range(max_rounds):
        cross_total = _count_pair_crossings(paths)
        if cross_total == 0:
            break

        improved = False
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                pt = _first_crossing_point(paths[i], paths[j])
                if pt is None:
                    continue

                name_a = path_electrodes[i]
                name_b = path_electrodes[j]
                term = path_terminals[i]
                if path_terminals[j] != term:
                    continue

                head_a, tail_a = _split_path_at_point(paths[i], pt)
                head_b, tail_b = _split_path_at_point(paths[j], pt)

                entries[name_a], entries[name_b] = (
                    entries[name_b].copy(),
                    entries[name_a].copy(),
                )
                entries[name_a], entries[name_b] = _separate_entry_points(
                    entries[name_a],
                    entries[name_b],
                    term,
                    terminals_2d,
                    terminal_zones,
                    separation=separation,
                )

                start_a = electrodes_2d[name_a]
                start_b = electrodes_2d[name_b]
                cross_pt = np.asarray(pt, dtype=float)
                tail_to_a = _straight_path_2d(cross_pt, entries[name_a], n=12)[1:]
                tail_to_b = _straight_path_2d(cross_pt, entries[name_b], n=12)[1:]
                new_a = np.vstack([head_a, tail_to_a])
                new_b = np.vstack([head_b, tail_to_b])
                new_a = new2d.pin_path_endpoints_2d(new_a, start_a, entries[name_a])
                new_b = new2d.pin_path_endpoints_2d(new_b, start_b, entries[name_b])

                if _path_foreign_electrode_hits(new_a, name_a, electrode_zones):
                    continue
                if _path_foreign_electrode_hits(new_b, name_b, electrode_zones):
                    continue

                trial = list(paths)
                trial[i] = new_a
                trial[j] = new_b
                new_cross = _count_pair_crossings(trial)
                if new_cross < cross_total:
                    paths = trial
                    improved = True
                    print(
                        f"  Tail swap {name_a}<->{name_b}: "
                        f"crossings {cross_total}->{new_cross}"
                    )
                    break
            if improved:
                break
        if not improved:
            print(f"  Tail swap stopped at round {round_idx + 1}")
            break

    return paths, entries


def _uncross_paths_with_detours(
    paths: list[np.ndarray],
    path_electrodes: list[str],
    electrodes_2d: dict[str, np.ndarray],
    entry_points: dict[str, np.ndarray],
    electrode_zones: dict,
    *,
    max_rounds: int = 40,
) -> list[np.ndarray]:
    from shapely.geometry import LineString

    paths = [np.asarray(p, dtype=float).copy() for p in paths]
    for _ in range(max_rounds):
        cross = _count_pair_crossings(paths)
        if cross == 0:
            break
        lines = [LineString(p) for p in paths]
        improved = False
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                inter = lines[i].intersection(lines[j])
                if inter.is_empty or inter.geom_type not in ("Point", "MultiPoint"):
                    continue
                for idx in (i, j):
                    name = path_electrodes[idx]
                    start = electrodes_2d[name]
                    end = entry_points[name]
                    for sign in (1.0, -1.0):
                        for scale in (8.0, 16.0, 28.0, 42.0):
                            trial = new2d.pin_path_endpoints_2d(
                                _bent_path_2d(start, end, sign, scale),
                                start,
                                end,
                            )
                            if _path_foreign_electrode_hits(trial, name, electrode_zones):
                                continue
                            trial_list = list(paths)
                            trial_list[idx] = trial
                            if _count_pair_crossings(trial_list) < cross:
                                paths = trial_list
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    return paths


def assign_target_terminal_entries(
    path_electrodes: list[str],
    path_terminals: list[str],
    electrodes_2d: dict[str, np.ndarray],
    terminals_2d: dict[str, np.ndarray],
    terminal_zones: dict,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """GA-style slot order from target geometry (chord hit sort on each hub)."""
    straight_paths = []
    for electrode, terminal in zip(path_electrodes, path_terminals):
        straight_paths.append(
            np.stack(
                [electrodes_2d[electrode], terminals_2d[terminal]],
                axis=0,
            )
        )
    entry_points, slot_index, _ = new2d.assign_terminal_entry_slots(
        path_electrodes,
        path_terminals,
        straight_paths,
        terminal_zones,
        terminals_2d=terminals_2d,
    )
    return entry_points, slot_index


def _resolve_entries_for_synth(
    path_electrodes: list[str],
    path_terminals: list[str],
    electrodes_2d: dict[str, np.ndarray],
    terminals_2d: dict[str, np.ndarray],
    terminal_zones: dict,
    *,
    preserve_entry_order: bool,
    preset_paths: dict[str, dict],
    slot_index_preset: dict[str, int],
    offset_map: dict[str, float],
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    if preserve_entry_order:
        entries = _entries_from_offset_map(
            offset_map,
            path_electrodes,
            path_terminals,
            electrodes_2d,
            terminals_2d,
            terminal_zones,
            slot_index_preset,
        )
        return entries, dict(slot_index_preset)
    return assign_target_terminal_entries(
        path_electrodes,
        path_terminals,
        electrodes_2d,
        terminals_2d,
        terminal_zones,
    )


def optimize_terminals_for_clearance(
    preset: dict,
    electrodes: dict[str, np.ndarray],
    terminals_3d: dict[str, np.ndarray],
    layout_fiducials: dict,
    path_electrodes: list[str],
    path_terminals: list[str],
    offset_map: dict[str, float],
    slot_index_by_electrode: dict[str, int],
    *,
    preserve_entry_order: bool = False,
    preset_paths: dict[str, dict] | None = None,
    search_deg: float = TERMINAL_ANGLE_SEARCH_DEG,
    step_deg: float = TERMINAL_ANGLE_STEP_DEG,
    terminal_2d_mode: str = TERMINAL_2D_INFLATED,
) -> tuple[dict[str, np.ndarray], dict, dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    """Coarse hub-angle search minimizing crossings + electrode violations."""
    _, slot_order = _preset_slot_metadata(preset)
    preset_paths = preset_paths or preset.get("paths_chord_3d", {})
    electrodes_2d, terminals_2d, cz_pos = build_layout_2d(
        electrodes, layout_fiducials, terminal_2d_mode=terminal_2d_mode
    )
    electrode_zones, terminal_zones = new2d.create_zones(electrodes_2d, terminals_2d)

    entries, _ = _resolve_entries_for_synth(
        path_electrodes,
        path_terminals,
        electrodes_2d,
        terminals_2d,
        terminal_zones,
        preserve_entry_order=preserve_entry_order,
        preset_paths=preset_paths,
        slot_index_preset=slot_index_by_electrode,
        offset_map=offset_map,
    )
    paths = [
        _straight_path_2d(electrodes_2d[e], entries[e]) for e in path_electrodes
    ]
    cr0, ev0 = _layout_violation_score(
        paths, path_electrodes, path_terminals, electrode_zones, terminal_zones
    )
    best_score = cr0 * 100 + ev0

    terminals_3d = {k: np.asarray(v, dtype=float).copy() for k, v in terminals_3d.items()}
    layout_fiducials = dict(layout_fiducials)
    steps = max(1, int(round(2 * search_deg / step_deg)) + 1)
    angles = np.linspace(-np.deg2rad(search_deg), np.deg2rad(search_deg), steps)

    for terminal in slot_order:
        if terminal not in terminals_3d:
            continue
        base_3d = terminals_3d[terminal].copy()
        base_2d = np.asarray(terminals_2d[terminal], dtype=float)
        base_angle = float(np.arctan2(base_2d[1], base_2d[0]))
        local_best = best_score
        local_delta = 0.0

        for delta in angles:
            trial_fid = dict(layout_fiducials)
            trial_fid[terminal] = _rotate_terminal_3d_about_cz(base_3d, cz_pos, float(delta))
            e2d, t2d, _ = build_layout_2d(
                electrodes, trial_fid, terminal_2d_mode=terminal_2d_mode
            )
            _, tz = new2d.create_zones(e2d, t2d)
            ent, _ = _resolve_entries_for_synth(
                path_electrodes,
                path_terminals,
                e2d,
                t2d,
                tz,
                preserve_entry_order=preserve_entry_order,
                preset_paths=preset_paths,
                slot_index_preset=slot_index_by_electrode,
                offset_map=offset_map,
            )
            pths = [_straight_path_2d(e2d[e], ent[e]) for e in path_electrodes]
            ez, _ = new2d.create_zones(e2d, t2d)
            cr, ev = _layout_violation_score(
                pths, path_electrodes, path_terminals, ez, tz
            )
            score = cr * 100 + ev
            if score < local_best:
                local_best = score
                local_delta = float(delta)
                best_score = min(best_score, score)

        if abs(local_delta) > 1e-9:
            terminals_3d[terminal] = _rotate_terminal_3d_about_cz(
                base_3d, cz_pos, local_delta
            )
            layout_fiducials[terminal] = terminals_3d[terminal]
            ang = base_angle + local_delta
            print(
                f"  Terminal {terminal}: {np.rad2deg(local_delta):+.1f}° "
                f"(score→{local_best})"
            )

    electrodes_2d, terminals_2d, cz_pos = build_layout_2d(
        electrodes, layout_fiducials, terminal_2d_mode=terminal_2d_mode
    )
    return terminals_3d, layout_fiducials, electrodes_2d, terminals_2d, cz_pos


def apply_layout_preset_v4_synthesize(
    preset_path: str,
    target_subject_id: int,
    output_path: str | None = None,
    *,
    preserve_entry_order: bool = False,
    use_tail_swap: bool = False,
    use_target_terminals: bool = False,
    optimize_terminals: bool = True,
) -> dict[str, Any]:
    """
    Target layout: free hub angle + preset strip offsets + straight/detour 2D paths.
    No GA, no repair, no source chord-shape replay.

    use_target_terminals: TERMINAL_LEFT/RIGHT from target fiducials_{id}.json;
      preset supplies terminal_assignments only (no rigid hub map from reference).
    optimize_terminals: if True with use_target_terminals, ±search hub angle for clearance.
    """
    with open(preset_path, "r") as f:
        preset = json.load(f)

    electrodes, fiducials_file = load_subject_data(target_subject_id)
    if use_target_terminals:
        layout_fiducials = {
            k: np.asarray(v, dtype=float) for k, v in fiducials_file.items()
        }
        terminals_3d = extract_terminal_positions(fiducials_file)
        target_anatomical = extract_anatomical_fiducials(fiducials_file)
    else:
        layout_fiducials, terminals_3d, target_anatomical = map_preset_terminals_to_target(
            preset, fiducials_file
        )
    cz_pos = np.asarray(electrodes["Cz"], dtype=float)

    slot_index_preset, _ = _preset_slot_metadata(preset)
    path_electrodes, path_terminals = _path_list_from_preset(preset, electrodes)
    if not path_electrodes:
        raise ValueError(
            "Preset has no terminal_assignments (or paths_chord_3d) for this electrode set."
        )

    offset_map = _offsets_from_preset(
        preset.get("paths_chord_3d", {}),
        path_electrodes,
        path_terminals,
        slot_index_preset,
    )

    entry_mode = (
        "preset arc offsets"
        if preserve_entry_order
        else "target-native slots (assign_terminal_entry_slots)"
    )
    terminal_mode = (
        ENTRY_MODE_TARGET_FIDUCIALS
        if use_target_terminals
        else "rigid_preset_terminals"
    )
    hub_msg = (
        "target fiducial hubs"
        if use_target_terminals and not optimize_terminals
        else (
            "target fiducial hubs + angle search"
            if use_target_terminals
            else "preset rigid hubs + angle search"
        )
    )
    terminal_2d_mode = (
        TERMINAL_2D_FIDUCIAL if use_target_terminals else TERMINAL_2D_INFLATED
    )
    if terminal_2d_mode == TERMINAL_2D_FIDUCIAL:
        hub_msg = f"{hub_msg} (fiducial strip zones)"
    print(f"v4 synthesize: {entry_mode} → {hub_msg} → straight/detour paths")

    if optimize_terminals:
        terminals_3d, layout_fiducials, electrodes_2d, terminals_2d, cz_pos = (
            optimize_terminals_for_clearance(
                preset,
                electrodes,
                terminals_3d,
                layout_fiducials,
                path_electrodes,
                path_terminals,
                offset_map,
                slot_index_preset,
                preserve_entry_order=preserve_entry_order,
                terminal_2d_mode=terminal_2d_mode,
            )
        )
    else:
        electrodes_2d, terminals_2d, cz_pos = build_layout_2d(
            electrodes, layout_fiducials, terminal_2d_mode=terminal_2d_mode
        )

    electrode_zones, terminal_zones = new2d.create_zones(electrodes_2d, terminals_2d)
    terminal_zone_size = float(
        electrode_zones["metadata"].get("terminal_zone_size", 0.0)
    )
    if preserve_entry_order:
        offset_map = _slide_entry_offsets_for_clearance(
            offset_map,
            path_electrodes,
            path_terminals,
            electrodes_2d,
            terminals_2d,
            terminal_zones,
            electrode_zones,
            slot_index_preset,
        )

    entry_points_2d, slot_index_live = _resolve_entries_for_synth(
        path_electrodes,
        path_terminals,
        electrodes_2d,
        terminals_2d,
        terminal_zones,
        preserve_entry_order=preserve_entry_order,
        preset_paths=preset.get("paths_chord_3d", {}),
        slot_index_preset=slot_index_preset,
        offset_map=offset_map,
    )

    paths_2d: list[np.ndarray] = []
    for electrode in path_electrodes:
        e2d = electrodes_2d[electrode]
        end2d = entry_points_2d[electrode]
        paths_2d.append(
            _path_with_electrode_detour(e2d, end2d, electrode, electrode_zones)
        )
    if use_tail_swap and preserve_entry_order:
        paths_2d, entry_points_2d = _uncross_by_tail_swap(
            paths_2d,
            path_electrodes,
            path_terminals,
            electrodes_2d,
            entry_points_2d,
            terminals_2d,
            terminal_zones,
            electrode_zones,
        )
    paths_2d = _uncross_paths_with_detours(
        paths_2d,
        path_electrodes,
        electrodes_2d,
        entry_points_2d,
        electrode_zones,
    )
    for i, electrode in enumerate(path_electrodes):
        paths_2d[i] = new2d.pin_path_endpoints_2d(
            paths_2d[i],
            electrodes_2d[electrode],
            entry_points_2d[electrode],
        )

    mesh = _pyvista_read_stl(target_subject_id)
    uv_grid_raw = new2d.create_uv_grid(mesh, cz_pos, resolution=100)
    uv_grid_ctx = uv_grid_for_context(uv_grid_raw)
    uv_context = recon.UVReconstructionContext(uv_grid_ctx, mesh)

    output_paths: list[dict] = []
    terminal_assignments = preset.get("terminal_assignments", {})
    preset_paths = preset.get("paths_chord_3d", {})
    for electrode, path_2d in zip(path_electrodes, paths_2d):
        terminal = terminal_assignments.get(electrode) or preset_paths.get(
            electrode, {}
        ).get("terminal")
        e2d = electrodes_2d[electrode]
        e3d = np.asarray(electrodes[electrode], dtype=float)
        end2d = entry_points_2d[electrode]
        t3d = np.asarray(terminals_3d[terminal], dtype=float)
        end3d = entry_3d_for_strip(
            end2d,
            uv_context,
            mesh,
            e3d=e3d,
            terminal_3d=t3d,
            e2d=e2d,
            cz_pos=cz_pos,
            terminal_2d_mode=terminal_2d_mode,
            terminal_zone_size=terminal_zone_size,
        )

        path_3d = uv_context.reconstruct(e3d, end3d, np.asarray(path_2d, dtype=float))
        path_3d = snap_path_to_mesh(path_3d, mesh, pin_endpoints=True)
        path_3d = pin_path_endpoints_3d(path_3d, e3d, end3d, mesh)

        out: dict[str, Any] = {
            "electrode": electrode,
            "terminal": terminal,
            "modified_path_2d": path_2d.tolist(),
            "path_points": path_3d.tolist(),
            "entry_point_2d": end2d.tolist(),
            "entry_position_3d": end3d.tolist(),
        }
        if electrode in slot_index_live:
            out["slot_index"] = int(slot_index_live[electrode])
        output_paths.append(out)

    cross, ev = _layout_violation_score(
        paths_2d, path_electrodes, path_terminals, electrode_zones, terminal_zones
    )
    try:
        analysis = validate_preset_on_subject(
            target_subject_id,
            paths_2d,
            path_electrodes,
            path_terminals,
            electrodes_2d=electrodes_2d,
            terminals_2d=terminals_2d,
        )
    except RecursionError:
        analysis = {
            "collision_score": cross * 10 + ev,
            "crossing_count": cross,
            "electrode_violations": ev,
            "layout_collision_free": cross == 0 and ev == 0,
        }

    if output_path is None:
        output_path = f"data/output/applied_v4_{target_subject_id}_synth.json"

    result = {
        "metadata": {
            "target_subject_id": target_subject_id,
            "preset_version": PRESET_VERSION_V4,
            "preset_id": preset.get("preset_id"),
            "source_subject_id": preset.get("source_subject_id"),
            "path_lift": "uv_surface_synthesize",
            "terminal_mode": terminal_mode,
            "entry_slot_mode": (
                ENTRY_MODE_ARC_OFFSET
                if preserve_entry_order
                else ENTRY_MODE_TARGET_SLOTS
            ),
            "terminal_positions_3d": {
                k: v.tolist() for k, v in terminals_3d.items()
            },
            "terminal_2d_mode": terminal_2d_mode,
            "timestamp": datetime.now().isoformat(),
        },
        "preset_path": preset_path.replace("\\", "/"),
        "collision_metrics": _metrics_dict(analysis),
        "uv_grid": uv_grid_ctx,
        "paths": output_paths,
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    cm = result["collision_metrics"]
    print(
        f"Synthesized → subject {target_subject_id}: "
        f"crossings={cross} (simple) / {cm.get('crossing_count')} (full), "
        f"electrode_violations={ev} / {cm.get('electrode_violations')}"
    )
    print(f"Wrote {output_path}")
    return result


def apply_layout_preset_v4(
    preset_path: str,
    target_subject_id: int,
    output_path: str | None = None,
    metrics_mode: str = "clearance",
    snap_interior: bool = True,
    fit_terminals: bool = False,
) -> dict[str, Any]:
    with open(preset_path, "r") as f:
        preset = json.load(f)
    if preset.get("preset_version") != PRESET_VERSION_V4:
        raise ValueError(f"Expected preset_version {PRESET_VERSION_V4}")

    electrodes, fiducials_file = load_subject_data(target_subject_id)
    layout_fiducials, terminals_3d, target_anatomical = map_preset_terminals_to_target(
        preset, fiducials_file
    )
    up_ref = head_up_vector(target_anatomical)

    mesh = _pyvista_read_stl(target_subject_id)
    mesh.compute_normals(inplace=True)

    electrodes_2d, terminals_2d, cz_pos = build_layout_2d(electrodes, layout_fiducials)

    from PYTHON.tools.layoutPreset import rigid_transform_from_landmarks, transform_point

    src_anat = {
        k: np.asarray(v, dtype=float)
        for k, v in preset["source_anatomical_fiducials"].items()
    }
    r, t = rigid_transform_from_landmarks(src_anat, target_anatomical)

    slot_index_preset, _ = _preset_slot_metadata(preset)
    if fit_terminals and slot_index_preset:
        print("v4 apply: fitting terminal hub angles (legacy) …")
        terminals_3d, layout_fiducials, terminals_2d, cz_pos, slot_index_preset = (
            fit_terminals_to_preset_slot_order(
                preset, electrodes, terminals_3d, layout_fiducials
            )
        )

    path_electrodes: list[str] = []
    path_terminals: list[str] = []
    for electrode, entry in preset.get("paths_chord_3d", {}).items():
        terminal = entry.get("terminal") or preset["terminal_assignments"].get(electrode)
        if not terminal or electrode not in electrodes:
            continue
        path_electrodes.append(electrode)
        path_terminals.append(terminal)

    _, terminal_zones = new2d.create_zones(electrodes_2d, terminals_2d)
    entry_points_2d, slot_index_live = apply_entry_slots_from_arc_offsets(
        preset.get("paths_chord_3d", {}),
        path_electrodes,
        path_terminals,
        electrodes_2d,
        terminals_2d,
        terminal_zones,
        slot_index_preset,
    )

    uv_grid_raw = new2d.create_uv_grid(mesh, cz_pos, resolution=100)
    uv_context = recon.UVReconstructionContext(
        uv_grid_for_context(uv_grid_raw), mesh
    )

    paths_2d: list[np.ndarray] = []
    output_paths: list[dict] = []

    print(
        "v4 apply: rigid landmarks + arc-offset entries + target 10-20 + 3D chord replay"
    )
    print(
        f"Terminals: LEFT={terminals_3d['TERMINAL_LEFT'].round(1).tolist()}, "
        f"RIGHT={terminals_3d['TERMINAL_RIGHT'].round(1).tolist()}"
    )

    for electrode, entry in preset.get("paths_chord_3d", {}).items():
        terminal = entry.get("terminal") or preset["terminal_assignments"].get(electrode)
        if not terminal or electrode not in electrodes:
            continue

        e3d = np.asarray(electrodes[electrode], dtype=float)
        t3d = np.asarray(terminals_3d[terminal], dtype=float)
        e2d = electrodes_2d[electrode]

        if electrode in entry_points_2d:
            end2d = entry_points_2d[electrode]
            end3d = entry_3d_for_strip(
                end2d, uv_context, mesh, e3d=e3d, terminal_3d=t3d, e2d=e2d, cz_pos=cz_pos
            )
        elif entry.get("entry_position_3d") is not None:
            end3d = transform_point(
                np.asarray(entry["entry_position_3d"], dtype=float), r, t
            )
            end2d = new2d.polar_projection(np.array([end3d]), cz_pos)[0]
        else:
            end3d = t3d
            end2d = terminals_2d[terminal]

        src_len = float(entry.get("chord_length", 0.0))
        _, _, _, tgt_len = chord_basis_3d(e3d, end3d, up_ref)
        chord_scale = (tgt_len / src_len) if src_len > 1e-12 else 1.0

        path_3d = denormalize_path_3d(entry, e3d, end3d, up_ref, chord_scale=chord_scale)
        if snap_interior:
            path_3d = snap_path_to_mesh(path_3d, mesh, pin_endpoints=True)

        path_2d = np.stack(
            [new2d.polar_projection(np.array([p]), cz_pos)[0] for p in path_3d],
            axis=0,
        )
        path_2d = new2d.pin_path_endpoints_2d(path_2d, e2d, end2d)

        out: dict[str, Any] = {
            "electrode": electrode,
            "terminal": terminal,
            "modified_path_2d": path_2d.tolist(),
            "path_points": path_3d.tolist(),
        }
        if electrode in entry_points_2d or entry.get("entry_position_3d") is not None:
            out["entry_point_2d"] = end2d.tolist()
            out["entry_position_3d"] = end3d.tolist()
        si = slot_index_live.get(electrode) or entry.get("slot_index")
        if si is not None:
            out["slot_index"] = int(si)

        paths_2d.append(path_2d)
        output_paths.append(out)

    try:
        analysis = validate_preset_on_subject(
            target_subject_id,
            paths_2d,
            path_electrodes,
            path_terminals,
            metrics_mode=metrics_mode,
            electrodes_2d=electrodes_2d,
            terminals_2d=terminals_2d,
        )
    except RecursionError:
        print("WARNING: collision analysis skipped (RecursionError).")
        analysis = {
            "collision_score": None,
            "crossing_count": None,
            "electrode_violations": None,
            "layout_collision_free": None,
        }

    if output_path is None:
        output_path = (
            f"data/output/applied_v4_{target_subject_id}_"
            f"{preset.get('preset_id', 'preset')}.json"
        )

    result = {
        "metadata": {
            "target_subject_id": target_subject_id,
            "preset_version": PRESET_VERSION_V4,
            "preset_id": preset.get("preset_id"),
            "source_subject_id": preset.get("source_subject_id"),
            "path_lift": "landmark_rigid_chord_3d",
            "terminal_mode": (
                "rigid_landmarks_slot_fit"
                if fit_terminals
                else ENTRY_MODE_ARC_OFFSET
            ),
            "terminal_positions_3d": {
                k: v.tolist() for k, v in terminals_3d.items()
            },
            "timestamp": datetime.now().isoformat(),
        },
        "preset_path": preset_path.replace("\\", "/"),
        "collision_metrics": _metrics_dict(analysis),
        "paths": output_paths,
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    cm = result["collision_metrics"]
    print(
        f"Applied v4 → subject {target_subject_id}: "
        f"crossings={cm.get('crossing_count')}, "
        f"electrode_violations={cm.get('electrode_violations')}, "
        f"score={cm.get('collision_score')}"
    )
    print(f"Wrote {output_path}")
    return result


def _pairs_with_geometric_crossings(
    paths: list[np.ndarray],
    path_terminals: list[str],
    terminal_zones: dict,
    electrode_zones: dict,
) -> list[tuple[int, int, float]]:
    """Conflict pairs that actually intersect in 2D."""
    dense = new2d._build_crossing_detection_path_cache(
        paths, path_terminals, electrode_zones
    )
    hits: list[tuple[int, int, float]] = []
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            inter, _, crossing_points = new2d._compute_path_pair_crossing_geometry(
                paths[i],
                paths[j],
                path_terminals[i],
                path_terminals[j],
                terminal_zones,
                electrode_zones,
                dense_path_cache=dense,
                path_idx_a=i,
                path_idx_b=j,
            )
            has_cross = bool(crossing_points) or (
                inter is not None and not inter.is_empty
            )
            if not has_cross:
                continue
            pen = new2d.analyze_pair_path_penalty(
                paths[i],
                paths[j],
                terminal_zones,
                terminal_a=path_terminals[i],
                terminal_b=path_terminals[j],
                electrode_zones=electrode_zones,
                dense_path_cache=dense,
                path_idx_a=i,
                path_idx_b=j,
            )
            hits.append((i, j, float(pen)))
    return sorted(hits, key=lambda x: -x[2])


def uncross_applied_layout(
    applied_path: str,
    output_path: str | None = None,
    uv_resolution: int = 100,
    max_rounds: int = 40,
) -> dict[str, Any]:
    """
    Target remaining crossings by rerouting one trace at a time when global
    crossing count drops (works on near-straight paths like F7 that greedy locks).
    """
    from PYTHON.tools.layoutPreset import (
        _child_from_applied_data,
        _package_applied_result,
        ensure_init_connection_paths,
        resolve_layout_fiducials,
    )

    with open(applied_path, "r") as f:
        data = json.load(f)

    target_id = int(data["metadata"]["target_subject_id"])
    electrodes, _ = load_subject_data(target_id)
    fiducials, _ = resolve_layout_fiducials(target_id, data)
    original_paths = ensure_init_connection_paths(target_id, electrodes, fiducials)
    if hasattr(new2d, "_SUBJECT_LAYOUT_CACHE"):
        new2d._SUBJECT_LAYOUT_CACHE.pop(target_id, None)
    ctx = new2d.get_subject_layout(target_id, electrodes, fiducials, original_paths)

    child = _child_from_applied_data(data)
    paths = [np.asarray(p["modified_path_2d"], dtype=float) for p in child["paths"]]
    path_electrodes = [p["electrode"] for p in child["paths"]]
    path_terminals = [p["terminal"] for p in child["paths"]]
    entry_points, slot_index, _ = new2d.slot_metadata_from_child_paths(child["paths"])

    electrodes_2d = ctx["electrodes_2d"]
    terminals_2d = ctx["terminals_2d"]
    electrode_zones = ctx["electrode_zones"]
    terminal_zones = ctx["terminal_zones"]
    x_bounds = ctx["x_bounds"]

    analysis = new2d.analyze_path_collisions(
        paths,
        terminal_zones,
        electrode_zones=electrode_zones,
        path_electrodes=path_electrodes,
        path_terminals=path_terminals,
    )
    print(
        f"Uncross start: crossings={analysis['crossing_count']}, "
        f"min_sep={analysis.get('min_trace_separation', 0):.2f}"
    )

    for round_idx in range(max_rounds):
        if int(analysis.get("crossing_count", 0)) == 0:
            break
        cross_pairs = _pairs_with_geometric_crossings(
            paths, path_terminals, terminal_zones, electrode_zones
        )
        if not cross_pairs:
            break

        improved = False
        before_cross = int(analysis["crossing_count"])
        before_sep = float(analysis.get("min_trace_separation") or 0.0)

        for i, j, _ in cross_pairs[:6]:
            for path_idx in (i, j):
                electrode = path_electrodes[path_idx]
                terminal = path_terminals[path_idx]
                start = electrodes_2d[electrode]
                end = entry_points.get(electrode, terminals_2d[terminal])
                partner_idx = j if path_idx == i else i

                spacing_trial = new2d._greedy_spacing_outside_terminal_zone(
                    paths[path_idx],
                    terminal,
                    terminal_zones,
                    electrode,
                    electrode_zones,
                    start,
                    end,
                    paths[partner_idx],
                )
                candidates = []
                if spacing_trial is not None:
                    candidates.append(spacing_trial)
                for _ in range(32):
                    candidates.append(
                        new2d.randomly_modify_path(
                            paths[path_idx].copy(),
                            electrode,
                            electrode_zones,
                            terminal_zones,
                            x_bounds=x_bounds,
                            target_electrode_pos=start,
                            target_terminal_pos=end,
                            target_terminal_name=terminal,
                        )
                    )

                for trial in candidates:
                    trial_paths = [p.copy() for p in paths]
                    trial_paths[path_idx] = new2d.pin_path_endpoints_2d(
                        trial, start, end
                    )
                    trial_analysis = new2d.analyze_path_collisions(
                        trial_paths,
                        terminal_zones,
                        electrode_zones=electrode_zones,
                        path_electrodes=path_electrodes,
                        path_terminals=path_terminals,
                    )
                    cross = int(trial_analysis["crossing_count"])
                    sep = float(trial_analysis.get("min_trace_separation") or 0.0)
                    better = cross < before_cross or (
                        cross == before_cross and sep > before_sep + 0.05
                    )
                    if better:
                        paths = trial_paths
                        analysis = trial_analysis
                        improved = True
                        print(
                            f"  Round {round_idx + 1}: {electrode} "
                            f"crossings {before_cross}->{cross}, "
                            f"min_sep {before_sep:.2f}->{sep:.2f}"
                        )
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            print(f"Uncross stopped (no improvement at round {round_idx + 1})")
            break

    for conn, path in zip(child["paths"], paths):
        conn["modified_path_2d"] = path.tolist()
        if conn["electrode"] in entry_points:
            conn["entry_point_2d"] = entry_points[conn["electrode"]].tolist()

    meta_extra = {
        k: v
        for k, v in data.get("metadata", {}).items()
        if k not in ("timestamp", "grid_resolution", "grid_bounds")
    }
    meta_extra["uncross_from"] = applied_path.replace("\\", "/")
    result = _package_applied_result(
        target_id, child, meta_extra, uv_resolution=uv_resolution, fiducials=fiducials
    )
    if data.get("preset_path"):
        result["preset_path"] = data["preset_path"]

    if output_path is None:
        stem = Path(applied_path).stem
        output_path = f"data/output/{stem}_uncross.json"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    cm = result["collision_metrics"]
    print(
        f"Uncross done: crossings={cm.get('crossing_count')}, "
        f"electrode_violations={cm.get('electrode_violations')}, "
        f"min_sep={analysis.get('min_trace_separation')}"
    )
    print(f"Wrote {output_path}")
    return result


def refine_applied_v4(
    applied_path: str,
    output_path: str | None = None,
    uv_resolution: int = 100,
) -> dict[str, Any]:
    """
    Strong deterministic polish for v4 layouts: repair + uncross loop (no GA).
    """
    from PYTHON.tools.layoutPreset import repair_applied_preset

    if output_path is None:
        stem = Path(applied_path).stem.replace("_refined", "").replace("_repair", "")
        output_path = f"data/output/{stem}_refined.json"

    interim = output_path.replace(".json", "_pre_uncross.json")
    repair_applied_preset(
        applied_path,
        output_path=interim,
        uv_resolution=uv_resolution,
        phase2_max_rounds=16,
        aggressive_pass=False,
    )
    return uncross_applied_layout(
        interim,
        output_path=output_path,
        uv_resolution=uv_resolution,
        max_rounds=80,
    )
