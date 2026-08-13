"""
Export and apply subject-agnostic EEG interconnect layout presets.

See .cursor/skills/layout-preset-transfer/ in the repo root for workflow documentation.

Run from genetic_SHAPE/app/:
  python PYTHON/tools/layoutPreset.py export --subject 2 --individual 45-7 --out data/presets/s2.json
  python PYTHON/tools/layoutPreset.py apply --preset data/presets/s2.json --target 5
  python PYTHON/tools/layoutPreset.py visualize --applied data/output/applied_preset_5.json
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

_APP_ROOT = Path(__file__).resolve().parents[2]
if str(_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT))

import PYTHON.GA.geneticOperators as genetics
import PYTHON.tools.initiate3DConnections as init_connections
import PYTHON.tools.new2dAlterations as new2d
import PYTHON.tools.reconstructUsingUVmesh as recon
from PYTHON.tools.helper import load_electrode_positions_and_fiducials

PRESET_VERSION = 2
DEFAULT_LOG_DIR = "data/output/logs"
UV_GRID_RESOLUTION = getattr(new2d, "UV_GRID_RESOLUTION", 100)

# Head registration uses anatomical landmarks only; wire terminals come from the preset.
ANATOMICAL_FIDUCIAL_ALIASES: dict[str, tuple[str, ...]] = {
    "nasion": ("nasion", "Nasion", "NASION"),
    "lpa": ("lpa", "LPA"),
    "rpa": ("rpa", "RPA"),
    "inion": ("inion", "Inion", "INION"),
}
TERMINAL_FIDUCIAL_KEYS = ("TERMINAL_LEFT", "TERMINAL_RIGHT")


def _pyvista_read_stl(subject_id: int):
    return new2d._pyvista().read(f"data/cleaned_scans/{subject_id}.stl")


def resolve_ga_log_path(
    subject_id: int,
    individual_key: str,
    log_dir: str | None = None,
) -> str:
    """Resolve GA_{subject}_{individual}_mod_connection_paths.json."""
    base = log_dir or DEFAULT_LOG_DIR
    path = os.path.join(
        base, f"GA_{subject_id}_{individual_key}_mod_connection_paths.json"
    )
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"GA log not found: {path}. Run the GA or pass --log-dir (e.g. records/RUN_ID)."
        )
    return path


def load_subject_data(subject_id: int) -> tuple[dict, dict]:
    """Load electrode positions and fiducials for a subject."""
    with open(f"data/json/electrode_positions_{subject_id}.json") as f:
        electrodes = {k: np.asarray(v, dtype=float) for k, v in json.load(f).items()}
    with open(f"data/json/fiducials_{subject_id}.json") as f:
        fiducials = {k: np.asarray(v, dtype=float) for k, v in json.load(f).items()}
    return electrodes, fiducials


def extract_anatomical_fiducials(fiducials: dict) -> dict[str, np.ndarray]:
    """Return nasion, lpa, rpa, inion (canonical lowercase keys)."""
    out: dict[str, np.ndarray] = {}
    for canonical, aliases in ANATOMICAL_FIDUCIAL_ALIASES.items():
        for alias in aliases:
            if alias in fiducials:
                out[canonical] = np.asarray(fiducials[alias], dtype=float)
                break
        if canonical not in out:
            raise KeyError(
                f"Missing anatomical fiducial '{canonical}' (tried {aliases}). "
                "Run 1_selectFiducials for nasion, LPA, RPA, inion."
            )
    return out


def anatomical_fiducials_to_json(anatomical: dict[str, np.ndarray]) -> dict[str, list]:
    return {k: np.asarray(v, dtype=float).tolist() for k, v in anatomical.items()}


def extract_terminal_positions(fiducials: dict) -> dict[str, np.ndarray]:
    """TERMINAL_LEFT / TERMINAL_RIGHT 3D positions from a fiducial dict."""
    out: dict[str, np.ndarray] = {}
    for term in TERMINAL_FIDUCIAL_KEYS:
        if term in fiducials:
            out[term] = np.asarray(fiducials[term], dtype=float)
    if len(out) != 2:
        missing = [t for t in TERMINAL_FIDUCIAL_KEYS if t not in out]
        raise KeyError(f"Missing terminal fiducials: {missing}")
    return out


def rigid_transform_from_landmarks(
    source: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Kabsch rigid transform mapping source landmark 3D coordinates → target.

    Returns R (3×3) and t (3,) with: target ≈ R @ source + t
    """
    keys = sorted(source.keys())
    if set(keys) != set(target.keys()):
        raise ValueError("Source and target landmark keys must match")

    src = np.stack([source[k] for k in keys], dtype=float)
    tgt = np.stack([target[k] for k in keys], dtype=float)
    src_centroid = src.mean(axis=0)
    tgt_centroid = tgt.mean(axis=0)
    src_c = src - src_centroid
    tgt_c = tgt - tgt_centroid
    h = src_c.T @ tgt_c
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt = vt.copy()
        vt[-1, :] *= -1
        r = vt.T @ u.T
    t = tgt_centroid - r @ src_centroid
    return r, t


def transform_point(point: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return rotation @ np.asarray(point, dtype=float) + translation


def resolve_layout_fiducials(
    subject_id: int,
    applied_data: dict | None = None,
    *,
    prefer_applied_terminals: bool = True,
    preset_path: str | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Fiducials for layout / GA / repair consistent with how the preset was applied.

    Priority when prefer_applied_terminals is True:
      1. metadata.terminal_positions_3d on the applied JSON (incl. fiducial_uv apply)
      2. Recompute inherited terminals from preset if terminal_mode is inherited
      3. TERMINAL_LEFT/RIGHT from fiducials_{subject}.json (target clicks)

    terminal_mode ``fiducial_uv`` always uses stored terminal_positions_3d from apply.
    """
    _, fiducials_file = load_subject_data(subject_id)
    fiducials: dict[str, np.ndarray] = {
        k: np.asarray(v, dtype=float) for k, v in fiducials_file.items()
    }

    meta = (applied_data or {}).get("metadata", {})
    stored = meta.get("terminal_positions_3d")
    if prefer_applied_terminals and stored:
        for term, pos in stored.items():
            if term in TERMINAL_FIDUCIAL_KEYS:
                fiducials[term] = np.asarray(pos, dtype=float)
        terminals_3d = extract_terminal_positions(fiducials)
        return fiducials, terminals_3d

    mode = meta.get("terminal_mode")
    ppath = preset_path or (applied_data or {}).get("preset_path")
    if (
        prefer_applied_terminals
        and mode == "inherited"
        and ppath
        and os.path.isfile(ppath)
    ):
        with open(ppath, "r") as f:
            preset = json.load(f)
        if int(preset.get("source_subject_id", -1)) >= 0:
            _, terminals_3d, _ = map_preset_terminals_to_target(preset, fiducials_file)
            for term, pos in terminals_3d.items():
                fiducials[term] = np.asarray(pos, dtype=float)
            return fiducials, terminals_3d

    return fiducials, extract_terminal_positions(fiducials)


def sync_terminal_fiducials_json(
    subject_id: int,
    terminals_3d: dict[str, np.ndarray],
    *,
    backup: bool = True,
) -> str:
    """
    Write TERMINAL_LEFT/RIGHT into fiducials_{subject}.json (keeps anatomical landmarks).
    """
    path = f"data/json/fiducials_{subject_id}.json"
    with open(path, "r") as f:
        data = json.load(f)
    if backup:
        bak = f"data/json/fiducials_{subject_id}.json.bak"
        with open(bak, "w") as f:
            json.dump(data, f, indent=2)
    for term, pos in terminals_3d.items():
        data[term] = np.asarray(pos, dtype=float).tolist()
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    if hasattr(new2d, "_SUBJECT_LAYOUT_CACHE"):
        new2d.clear_subject_layout_cache(subject_id)
    print(f"Updated {path} terminal positions (backup: {backup})")
    return path


def map_preset_terminals_to_target(
    preset: dict,
    target_fiducials_file: dict,
) -> tuple[dict, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Build fiducials for apply: target anatomy + preset terminals rigid-mapped to target.

    Returns (effective_fiducials, terminals_3d_on_target, target_anatomical).
    """
    if "terminal_positions_3d" not in preset or "source_anatomical_fiducials" not in preset:
        raise ValueError(
            "Preset must include terminal_positions_3d and source_anatomical_fiducials. "
            "Re-export with: layoutPreset.py export ..."
        )

    source_anatomical = {
        k: np.asarray(v, dtype=float)
        for k, v in preset["source_anatomical_fiducials"].items()
    }
    target_anatomical = extract_anatomical_fiducials(target_fiducials_file)
    rotation, translation = rigid_transform_from_landmarks(source_anatomical, target_anatomical)

    terminals_3d: dict[str, np.ndarray] = {}
    for term, pos in preset["terminal_positions_3d"].items():
        terminals_3d[term] = transform_point(pos, rotation, translation)

    effective_fiducials = dict(target_fiducials_file)
    for term, pos in terminals_3d.items():
        effective_fiducials[term] = pos

    return effective_fiducials, terminals_3d, target_anatomical


def build_layout_2d(
    electrodes: dict,
    fiducials: dict,
    *,
    terminal_2d_mode: str = "inflated",
) -> tuple[dict, dict, np.ndarray]:
    """Electrodes 2D, terminals 2D, and Cz position."""
    cz_pos = electrodes["Cz"]
    electrodes_2d = {
        k: new2d.polar_projection(np.array([v]), cz_pos)[0] for k, v in electrodes.items()
    }
    terminals_2d = new2d.build_terminals_2d(
        electrodes_2d,
        fiducials,
        cz_pos,
        mode=new2d.normalize_terminal_2d_mode(terminal_2d_mode),
    )
    return electrodes_2d, terminals_2d, cz_pos


def _chord_frame(e2d: np.ndarray, t2d: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    delta = np.asarray(t2d, dtype=float) - np.asarray(e2d, dtype=float)
    length = float(np.linalg.norm(delta))
    if length < 1e-12:
        u = np.array([1.0, 0.0], dtype=float)
    else:
        u = delta / length
    n = np.array([-u[1], u[0]], dtype=float)
    return u, n, length


def _arc_length_params(path: np.ndarray) -> np.ndarray:
    path = np.asarray(path, dtype=float)
    if len(path) == 0:
        return np.array([], dtype=float)
    if len(path) == 1:
        return np.array([0.0], dtype=float)
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if total < 1e-12:
        return np.linspace(0.0, 1.0, len(path))
    t = s / total
    t[0] = 0.0
    t[-1] = 1.0
    return t


def normalize_path_2d(
    path_2d: np.ndarray,
    e2d: np.ndarray,
    t2d: np.ndarray,
) -> dict[str, list]:
    """Arc-length t and chord-frame offsets for one 2D path."""
    path = new2d.pin_path_endpoints_2d(
        np.asarray(path_2d, dtype=float), e2d, t2d
    )
    t_values = _arc_length_params(path)
    u, n, _ = _chord_frame(e2d, t2d)
    delta = np.asarray(t2d, dtype=float) - np.asarray(e2d, dtype=float)
    offsets = []
    for p, t in zip(path, t_values):
        on_chord = np.asarray(e2d, dtype=float) + float(t) * delta
        residual = np.asarray(p, dtype=float) - on_chord
        offsets.append(
            [float(np.dot(residual, u)), float(np.dot(residual, n))]
        )
    return {
        "t_values": t_values.tolist(),
        "offsets": offsets,
    }


def denormalize_path_2d(
    preset_path: dict,
    e2d: np.ndarray,
    t2d: np.ndarray,
) -> np.ndarray:
    """Rebuild 2D path on a target subject from normalized preset entry."""
    t_values = np.asarray(preset_path["t_values"], dtype=float)
    offsets = np.asarray(preset_path["offsets"], dtype=float)
    u, n, _ = _chord_frame(e2d, t2d)
    delta = np.asarray(t2d, dtype=float) - np.asarray(e2d, dtype=float)
    points = []
    for t, off in zip(t_values, offsets):
        on_chord = np.asarray(e2d, dtype=float) + float(t) * delta
        points.append(on_chord + float(off[0]) * u + float(off[1]) * n)
    return new2d.pin_path_endpoints_2d(np.asarray(points, dtype=float), e2d, t2d)


def uv_grid_for_context(uv_grid: dict) -> dict:
    """Format from create_uv_grid / get_cached_uv_grid for UVReconstructionContext."""
    if "points_2d" in uv_grid:
        return uv_grid
    return {
        "points_2d": np.asarray(uv_grid["grid_2d"], dtype=float).tolist(),
        "points_3d": np.asarray(uv_grid["grid_3d"], dtype=float).tolist(),
    }


def export_layout_preset(
    source_subject_id: int,
    individual_key: str,
    preset_path: str,
    log_dir: str | None = None,
    preset_id: str | None = None,
) -> dict:
    """
    Read a GA individual log, normalize 2D paths, and write a layout preset JSON.
    """
    ga_path = resolve_ga_log_path(source_subject_id, individual_key, log_dir)
    with open(ga_path, "r") as f:
        ga_data = json.load(f)

    electrodes, fiducials = load_subject_data(source_subject_id)
    electrodes_2d, terminals_2d, _ = build_layout_2d(electrodes, fiducials)

    terminal_assignments: dict[str, str] = {}
    paths_normalized: dict[str, dict] = {}

    for conn in ga_data.get("paths", []):
        electrode = conn.get("electrode")
        terminal = conn.get("terminal")
        mod_2d = conn.get("modified_path_2d")
        if not electrode or not terminal or mod_2d is None:
            continue
        if electrode not in electrodes_2d or terminal not in terminals_2d:
            continue

        terminal_assignments[electrode] = terminal
        e2d = electrodes_2d[electrode]
        t2d = terminals_2d[terminal]
        norm = normalize_path_2d(np.asarray(mod_2d), e2d, t2d)
        paths_normalized[electrode] = {
            "terminal": terminal,
            **norm,
        }

    if not paths_normalized:
        raise ValueError(f"No paths with modified_path_2d found in {ga_path}")

    source_anatomical = extract_anatomical_fiducials(fiducials)
    terminal_positions_3d = extract_terminal_positions(fiducials)

    preset = {
        "preset_version": PRESET_VERSION,
        "preset_id": preset_id or f"subject{source_subject_id}_{individual_key}",
        "electrode_layout": "standard_10-20",
        "source_subject_id": source_subject_id,
        "source_individual": individual_key,
        "source_ga_log": ga_path.replace("\\", "/"),
        "source_anatomical_fiducials": anatomical_fiducials_to_json(source_anatomical),
        "terminal_positions_3d": {
            k: v.tolist() for k, v in terminal_positions_3d.items()
        },
        "terminal_inheritance": "rigid_anatomical_landmarks",
        "terminal_assignments": terminal_assignments,
        "paths_normalized": paths_normalized,
    }

    os.makedirs(os.path.dirname(preset_path) or ".", exist_ok=True)
    with open(preset_path, "w") as f:
        json.dump(preset, f, indent=2)

    print(
        f"Exported preset '{preset['preset_id']}' "
        f"({len(paths_normalized)} paths) → {preset_path}"
    )
    return preset


def validate_preset_on_subject(
    target_subject_id: int,
    paths_2d: list[np.ndarray],
    path_electrodes: list[str],
    path_terminals: list[str],
    metrics_mode: str = "clearance",
    electrodes_2d: dict | None = None,
    terminals_2d: dict | None = None,
) -> dict[str, Any]:
    """Run analyze_path_collisions on target 2D paths."""
    if electrodes_2d is None or terminals_2d is None:
        electrodes, fiducials = load_subject_data(target_subject_id)
        electrodes_2d, terminals_2d, _ = build_layout_2d(electrodes, fiducials)
    electrode_zones, terminal_zones = new2d.create_zones(electrodes_2d, terminals_2d)
    analysis = new2d.analyze_path_collisions(
        paths_2d,
        terminal_zones,
        electrode_zones=electrode_zones,
        path_electrodes=path_electrodes,
        path_terminals=path_terminals,
        metrics_mode=metrics_mode,
    )
    analysis["layout_collision_free"] = new2d.is_layout_collision_free(analysis)
    return analysis


def apply_layout_preset(
    preset_path: str,
    target_subject_id: int,
    output_path: str | None = None,
    uv_resolution: int = UV_GRID_RESOLUTION,
    metrics_mode: str = "clearance",
) -> dict[str, Any]:
    """
    Denormalize preset paths on the target head, reconstruct 3D, validate collisions.

    TERMINAL_LEFT/RIGHT 3D positions come from the preset, rigidly registered onto the
    target using only nasion/LPA/RPA/inion on the target (terminal clicks are ignored).
    """
    with open(preset_path, "r") as f:
        preset = json.load(f)

    if preset.get("preset_version") != PRESET_VERSION:
        raise ValueError(
            f"Unsupported preset_version {preset.get('preset_version')}; "
            f"re-export with layoutPreset export (expected {PRESET_VERSION})."
        )

    mesh_path = f"data/cleaned_scans/{target_subject_id}.stl"
    if not os.path.isfile(mesh_path):
        raise FileNotFoundError(
            f"Target mesh missing: {mesh_path}. Run 0_PREP for subject {target_subject_id}."
        )

    electrodes, fiducials_file = load_subject_data(target_subject_id)
    layout_fiducials, terminals_3d, _ = map_preset_terminals_to_target(
        preset, fiducials_file
    )
    print(
        "Terminals: inherited from preset (rigid fit on target nasion/LPA/RPA/inion)."
    )
    electrodes_2d, terminals_2d, cz_pos = build_layout_2d(electrodes, layout_fiducials)
    mesh = _pyvista_read_stl(target_subject_id)
    uv_grid_raw = new2d.create_uv_grid(mesh, cz_pos, resolution=uv_resolution)
    uv_grid_ctx = uv_grid_for_context(uv_grid_raw)
    uv_context = recon.UVReconstructionContext(uv_grid_ctx, mesh)

    terminal_assignments = preset.get("terminal_assignments", {})
    paths_normalized = preset.get("paths_normalized", {})

    paths_2d: list[np.ndarray] = []
    path_electrodes: list[str] = []
    path_terminals: list[str] = []
    output_paths: list[dict] = []

    for electrode, norm_entry in paths_normalized.items():
        terminal = norm_entry.get("terminal") or terminal_assignments.get(electrode)
        if not terminal:
            raise ValueError(f"No terminal assignment for electrode {electrode}")
        if terminal not in terminals_3d:
            raise ValueError(f"Terminal {terminal} not available in layout fiducials")
        if electrode not in electrodes_2d:
            raise ValueError(
                f"Electrode {electrode} missing from electrode_positions_{target_subject_id}.json"
            )
        if terminal not in terminals_2d:
            raise ValueError(f"Terminal {terminal} not in build_terminals_2d for target")

        e2d = electrodes_2d[electrode]
        t2d = terminals_2d[terminal]
        path_2d = denormalize_path_2d(norm_entry, e2d, t2d)
        e3d = electrodes[electrode]
        t3d = terminals_3d[terminal]
        path_3d = uv_context.reconstruct(e3d, t3d, path_2d)

        paths_2d.append(path_2d)
        path_electrodes.append(electrode)
        path_terminals.append(terminal)
        output_paths.append(
            {
                "electrode": electrode,
                "terminal": terminal,
                "modified_path_2d": np.asarray(path_2d, dtype=float).tolist(),
                "path_points": np.asarray(path_3d, dtype=float).tolist(),
            }
        )

    analysis = validate_preset_on_subject(
        target_subject_id,
        paths_2d,
        path_electrodes,
        path_terminals,
        metrics_mode=metrics_mode,
        electrodes_2d=electrodes_2d,
        terminals_2d=terminals_2d,
    )

    if output_path is None:
        preset_id = preset.get("preset_id", "preset")
        output_path = (
            f"data/output/applied_preset_{target_subject_id}_{preset_id}.json"
        )

    result = {
        "metadata": {
            "target_subject_id": target_subject_id,
            "preset_id": preset.get("preset_id"),
            "source_subject_id": preset.get("source_subject_id"),
            "source_individual": preset.get("source_individual"),
            "terminal_mode": "inherited",
            "terminal_positions_3d": {
                k: np.asarray(v, dtype=float).tolist() for k, v in terminals_3d.items()
            },
            "timestamp": datetime.now().isoformat(),
            "grid_resolution": uv_resolution,
            "grid_bounds": uv_grid_raw.get("bounds"),
        },
        "preset_path": preset_path.replace("\\", "/"),
        "collision_metrics": {
            "collision_score": analysis.get("collision_score"),
            "crossing_count": analysis.get("crossing_count"),
            "electrode_violations": analysis.get("electrode_violations"),
            "layout_collision_free": analysis.get("layout_collision_free"),
        },
        "uv_grid": uv_grid_ctx,
        "paths": output_paths,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    score = analysis.get("collision_score", "?")
    free = analysis.get("layout_collision_free", False)
    print(
        f"Applied preset to subject {target_subject_id}: "
        f"collision_score={score}, layout_collision_free={free}"
    )
    print(
        "GA/repair: use layoutPreset.py ga without --file-terminals so terminals match apply."
    )
    print(f"Wrote {output_path}")
    return result


def _electrodes_fiducials_json_ready(
    electrodes: dict, fiducials: dict
) -> tuple[dict, dict]:
    """Lists for JSON/connectivity helpers; arrays are fine for geodesic code."""
    def _to_list(v):
        return v.tolist() if isinstance(v, np.ndarray) else v

    return (
        {k: _to_list(v) for k, v in electrodes.items()},
        {k: _to_list(v) for k, v in fiducials.items()},
    )


def ensure_init_connection_paths(
    subject_id: int,
    electrodes: dict | None = None,
    fiducials: dict | None = None,
) -> list:
    """
    Load geodesic seed paths for layout/repair context.

    If missing, creates them via initiate3DConnections (no full GA required).
    """
    path = f"data/json/init_connection_paths_{subject_id}.json"
    if os.path.isfile(path):
        with open(path, "r") as f:
            return json.load(f)

    if electrodes is None or fiducials is None:
        electrodes, fiducials = load_subject_data(subject_id)

    mesh_path = f"data/cleaned_scans/{subject_id}.stl"
    if not os.path.isfile(mesh_path):
        raise FileNotFoundError(
            f"{mesh_path} missing. Complete 0_PREP (through cleaned scan) for subject {subject_id}."
        )

    el_json, fid_json = _electrodes_fiducials_json_ready(electrodes, fiducials)
    print(
        f"Creating {path} (geodesic seeds for repair — one-time, may take a minute)..."
    )
    init_connections.createAndSaveInitConnections(
        SUBJECT_ID=subject_id,
        electrodes=el_json,
        fiducials=fid_json,
    )
    with open(path, "r") as f:
        return json.load(f)


def _child_from_applied_data(data: dict) -> dict:
    """GA-style child dict for apply_smart_collision_resolution."""
    child_paths = []
    for p in data["paths"]:
        entry = {
            "electrode": p["electrode"],
            "terminal": p["terminal"],
            "modified_path_2d": p["modified_path_2d"],
        }
        if p.get("entry_point_2d") is not None:
            entry["entry_point_2d"] = p["entry_point_2d"]
        if p.get("entry_position_3d") is not None:
            entry["entry_position_3d"] = p["entry_position_3d"]
        if p.get("path_end_2d") is not None:
            entry["path_end_2d"] = p["path_end_2d"]
        if p.get("path_end_3d") is not None:
            entry["path_end_3d"] = p["path_end_3d"]
        if p.get("slot_index") is not None:
            entry["slot_index"] = p["slot_index"]
        child_paths.append(entry)
    return {"paths": child_paths}


def _package_applied_result(
    target_subject_id: int,
    child: dict,
    metadata_extra: dict,
    uv_resolution: int = UV_GRID_RESOLUTION,
    fiducials: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Rebuild applied JSON (2D paths, 3D lift, collision metrics) from a child genome."""
    electrodes, _ = load_subject_data(target_subject_id)
    if fiducials is None:
        _, fiducials = load_subject_data(target_subject_id)
    cz_pos = electrodes["Cz"]
    mesh = _pyvista_read_stl(target_subject_id)
    uv_grid_raw = new2d.create_uv_grid(mesh, cz_pos, resolution=uv_resolution)
    uv_grid_ctx = uv_grid_for_context(uv_grid_raw)

    paths_2d: list[np.ndarray] = []
    path_electrodes: list[str] = []
    path_terminals: list[str] = []
    output_paths: list[dict] = []

    for conn in child["paths"]:
        path_2d = np.asarray(conn["modified_path_2d"], dtype=float)
        e3d = electrodes[conn["electrode"]]
        if conn.get("path_end_3d") is not None:
            end3d = np.asarray(conn["path_end_3d"], dtype=float)
        else:
            end3d = fiducials[conn["terminal"]]
        path_3d = recon.reconstruct_with_uv_grid(
            e3d, end3d, path_2d, uv_grid_ctx, mesh
        )
        out = {
            "electrode": conn["electrode"],
            "terminal": conn["terminal"],
            "modified_path_2d": path_2d.tolist(),
            "path_points": np.asarray(path_3d, dtype=float).tolist(),
        }
        if conn.get("entry_point_2d") is not None:
            out["entry_point_2d"] = conn["entry_point_2d"]
        if conn.get("entry_position_3d") is not None:
            out["entry_position_3d"] = conn["entry_position_3d"]
        if conn.get("path_end_2d") is not None:
            out["path_end_2d"] = conn["path_end_2d"]
        if conn.get("path_end_3d") is not None:
            out["path_end_3d"] = conn["path_end_3d"]
        if conn.get("slot_index") is not None:
            out["slot_index"] = conn["slot_index"]
        output_paths.append(out)
        paths_2d.append(path_2d)
        path_electrodes.append(conn["electrode"])
        path_terminals.append(conn["terminal"])

    analysis = validate_preset_on_subject(
        target_subject_id,
        paths_2d,
        path_electrodes,
        path_terminals,
        metrics_mode="clearance",
    )
    meta = {
        "target_subject_id": target_subject_id,
        "timestamp": datetime.now().isoformat(),
        "grid_resolution": uv_resolution,
        "grid_bounds": uv_grid_raw.get("bounds"),
        **metadata_extra,
    }
    return {
        "metadata": meta,
        "collision_metrics": {
            "collision_score": analysis.get("collision_score"),
            "crossing_count": analysis.get("crossing_count"),
            "electrode_violations": analysis.get("electrode_violations"),
            "layout_collision_free": analysis.get("layout_collision_free"),
        },
        "uv_grid": uv_grid_ctx,
        "paths": output_paths,
    }


def repair_applied_preset(
    applied_path: str,
    output_path: str | None = None,
    electrodes_only: bool = False,
    uv_resolution: int = UV_GRID_RESOLUTION,
    *,
    phase2_max_rounds: int = 12,
    aggressive_pass: bool = False,
    focus: str = "separation",
    skip_phase1_when_electrode_free: bool = True,
    fixed_endpoints: bool = True,
    profile_phase2: bool = False,
    min_trace_separation: float | None = None,
) -> dict[str, Any]:
    """
  Polish layout: fixed electrode + truncated wire ends; improve trace separation
  without increasing crossing count. Skips phase-1 electrode rerouting by default.
    """
    with open(applied_path, "r") as f:
        data = json.load(f)

    target_id = int(data["metadata"]["target_subject_id"])
    electrodes, _ = load_subject_data(target_id)
    fiducials, terminals_3d = resolve_layout_fiducials(target_id, data)
    print(
        "Repair/GA fiducials: using applied/inherited terminals "
        f"(LEFT={terminals_3d['TERMINAL_LEFT'].round(1).tolist()}, "
        f"RIGHT={terminals_3d['TERMINAL_RIGHT'].round(1).tolist()})"
    )
    original_paths = ensure_init_connection_paths(target_id, electrodes, fiducials)
    terminal_2d_mode = data.get("metadata", {}).get(
        "terminal_2d_mode", "inflated_legacy"
    )
    if hasattr(new2d, "clear_subject_layout_cache"):
        new2d.clear_subject_layout_cache(target_id)
    elif hasattr(new2d, "_SUBJECT_LAYOUT_CACHE"):
        new2d._SUBJECT_LAYOUT_CACHE.pop(target_id, None)
    new2d.warm_subject_caches(
        target_id,
        electrodes,
        fiducials,
        original_paths,
        terminal_2d_mode=terminal_2d_mode,
    )

    child = _child_from_applied_data(data)
    before = data.get("collision_metrics", {})
    focus_separation = focus == "separation"
    baseline_crossings = int(before.get("crossing_count") or 0)
    if min_trace_separation is None:
        min_trace_separation = float(new2d.PHASE2_INNER_TRACE_SEPARATION)
    else:
        min_trace_separation = float(min_trace_separation)
    print(
        f"Before polish: collision_score={before.get('collision_score')}, "
        f"crossings={before.get('crossing_count')}, "
        f"electrode_violations={before.get('electrode_violations')}, "
        f"min_trace_sep={before.get('min_trace_separation')}"
    )
    if focus_separation:
        print(
            f"Polish: fixed endpoints, separation-only "
            f"(phase2 rounds={phase2_max_rounds}, max_crossings={baseline_crossings}, "
            f"min_sep={min_trace_separation:.2f}mm)"
        )

    skip_phase1 = (
        fixed_endpoints
        or (
            focus_separation
            and skip_phase1_when_electrode_free
            and int(before.get("electrode_violations") or 0) == 0
        )
    )
    if skip_phase1:
        print("Phase 1 skipped (fixed endpoints / electrode-free layout)")
    else:
        child = new2d.apply_smart_collision_resolution(
            child,
            target_id,
            electrodes,
            fiducials,
            original_paths,
            greedy_electrodes_only=True,
            terminal_2d_mode=terminal_2d_mode,
        )

    if not electrodes_only:
        print(
            f"Phase 2 separation polish (up to {phase2_max_rounds} pair rounds)..."
        )
        if profile_phase2:
            from app.polish.phase2_profile import start_phase2_profile, stop_phase2_profile

            start_phase2_profile()
        try:
            child = new2d.apply_smart_collision_resolution(
                child,
                target_id,
                electrodes,
                fiducials,
                original_paths,
                greedy_electrodes_only=False,
                phase2_max_pair_rounds=phase2_max_rounds,
                force_trace_resolution=True,
                focus_separation=focus_separation,
                fixed_endpoints=fixed_endpoints,
                max_crossing_count=baseline_crossings,
                min_separation=min_trace_separation,
                terminal_2d_mode=terminal_2d_mode,
            )
        finally:
            if profile_phase2:
                from app.polish.phase2_profile import stop_phase2_profile

                stop_phase2_profile()
        if aggressive_pass:
            print("Phase 2 aggressive greedy pass (legacy; may add crossings)...")
            child = new2d.apply_smart_collision_resolution(
                child,
                target_id,
                electrodes,
                fiducials,
                original_paths,
                greedy_electrodes_only=False,
                use_greedy_aggressive=True,
                use_gentle_resolution=False,
                force_trace_resolution=True,
                focus_separation=focus_separation,
                min_separation=min_trace_separation,
                terminal_2d_mode=terminal_2d_mode,
            )

    metadata_extra = {
        k: v
        for k, v in data.get("metadata", {}).items()
        if k not in ("timestamp", "grid_resolution", "grid_bounds")
    }
    metadata_extra["repaired_from"] = applied_path.replace("\\", "/")
    if focus_separation:
        metadata_extra["polish_focus"] = "separation"
    if fixed_endpoints:
        metadata_extra["polish_fixed_endpoints"] = True
    metadata_extra["polish_min_trace_separation_mm"] = min_trace_separation
    result = _package_applied_result(
        target_id, child, metadata_extra, uv_resolution=uv_resolution, fiducials=fiducials
    )
    result["metadata"]["terminal_positions_3d"] = {
        k: v.tolist() for k, v in terminals_3d.items()
    }
    if data.get("preset_path"):
        result["preset_path"] = data["preset_path"]

    if output_path is None:
        stem = Path(applied_path).stem
        output_path = f"data/output/{stem}_repaired.json"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    cm = result["collision_metrics"]
    print(
        f"After repair: collision_score={cm.get('collision_score')}, "
        f"crossings={cm.get('crossing_count')}, "
        f"electrode_violations={cm.get('electrode_violations')}, "
        f"min_trace_sep={cm.get('min_trace_separation')}, "
        f"layout_collision_free={cm.get('layout_collision_free')}"
    )
    print(f"Wrote {output_path}")
    return result


def _sync_terminal_assignments_from_applied(
    subject_id: int, applied_path_entries: list[dict]
) -> dict[str, str]:
    """Align initial_terminal_assignments with the applied layout."""
    assignments = {
        p["electrode"]: p["terminal"] for p in applied_path_entries
    }
    out_path = f"data/json/initial_terminal_assignments_{subject_id}.json"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(assignments, f, indent=2)
    if hasattr(new2d, "_SUBJECT_LAYOUT_CACHE"):
        new2d.clear_subject_layout_cache(subject_id)
    print(f"Synced terminal assignments → {out_path}")
    return assignments


def build_ga_child_from_applied(
    subject_id: int,
    applied_data: dict,
    electrodes: dict,
    fiducials: dict,
    original_paths: list,
) -> dict:
    """Build a GA child genome ordered like get_subject_layout()['optimized']."""
    _sync_terminal_assignments_from_applied(subject_id, applied_data["paths"])
    ctx = new2d.get_subject_layout(subject_id, electrodes, fiducials, original_paths)
    electrodes_2d = ctx["electrodes_2d"]
    terminals_2d = ctx["terminals_2d"]
    path_by_electrode = {p["electrode"]: p for p in applied_data["paths"]}

    child_paths: list[dict] = []
    for conn in ctx["optimized"]:
        electrode = conn["electrode"]
        if electrode not in path_by_electrode:
            raise ValueError(
                f"Applied layout missing electrode '{electrode}' required for GA."
            )
        ap = path_by_electrode[electrode]
        entry = {
            "electrode": electrode,
            "terminal": ap["terminal"],
            "modified_path_2d": ap["modified_path_2d"],
        }
        if ap.get("entry_point_2d") is not None:
            entry["entry_point_2d"] = ap["entry_point_2d"]
        if ap.get("slot_index") is not None:
            entry["slot_index"] = ap["slot_index"]
        child_paths.append(entry)

    entry_points, slot_index, _ = new2d.slot_metadata_from_child_paths(child_paths)
    if not entry_points:
        electrodes_2d = ctx["electrodes_2d"]
        terminals_2d = ctx["terminals_2d"]
        paths_2d = [np.asarray(p["modified_path_2d"], dtype=float) for p in child_paths]
        path_electrodes = [p["electrode"] for p in child_paths]
        path_terminals = [p["terminal"] for p in child_paths]
        chord_to_terminal = new2d.straighten_paths_to_chords(
            paths_2d,
            path_electrodes,
            path_terminals,
            electrodes_2d,
            terminals_2d,
        )
        entry_points, slot_index, _ = new2d.assign_terminal_entry_slots(
            path_electrodes,
            path_terminals,
            chord_to_terminal,
            ctx["terminal_zones"],
            terminals_2d=terminals_2d,
            spacing=new2d.TERMINAL_ENTRY_SLOT_SPACING,
        )
        for entry in child_paths:
            name = entry["electrode"]
            entry["entry_point_2d"] = np.asarray(
                entry_points[name], dtype=float
            ).tolist()
            entry["slot_index"] = int(slot_index[name])

    child = {"paths": child_paths}
    new2d.pin_child_paths_2d(child, electrodes_2d, terminals_2d)
    return child


def seed_ga_generation_zero(
    subject_id: int,
    applied_path: str,
    population_size: int = 20,
    log_dir: str = DEFAULT_LOG_DIR,
    mutate_siblings: bool = True,
    mutation_fraction: float = 0.4,
    repair_on_seed: bool = True,
    prefer_applied_terminals: bool = True,
    sync_terminals_to_fiducials_json: bool = False,
) -> dict[str, float]:
    """
    Seed generation 0 from an applied (or repaired) layout.

    Individual ``0-0`` is the applied layout; ``0-1`` … ``0-{N-1}`` are mutations of it
    (same strategy as GA conception, phase 1).
    """
    with open(applied_path, "r") as f:
        applied_data = json.load(f)

    meta_subject = int(applied_data["metadata"]["target_subject_id"])
    if meta_subject != subject_id:
        raise ValueError(
            f"Applied JSON target_subject_id={meta_subject} does not match --subject {subject_id}"
        )

    electrodes, _ = load_subject_data(subject_id)
    fiducials, terminals_3d = resolve_layout_fiducials(
        subject_id, applied_data, prefer_applied_terminals=prefer_applied_terminals
    )
    meta_mode = applied_data.get("metadata", {}).get("terminal_mode", "file")
    print(
        f"GA seed terminals ({meta_mode}): "
        f"LEFT={terminals_3d['TERMINAL_LEFT'].round(1).tolist()}, "
        f"RIGHT={terminals_3d['TERMINAL_RIGHT'].round(1).tolist()}"
    )
    if sync_terminals_to_fiducials_json:
        sync_terminal_fiducials_json(subject_id, terminals_3d)

    original_paths = ensure_init_connection_paths(subject_id, electrodes, fiducials)
    if hasattr(new2d, "_SUBJECT_LAYOUT_CACHE"):
        new2d.clear_subject_layout_cache(subject_id)
    new2d.warm_subject_caches(subject_id, electrodes, fiducials, original_paths)

    genetics.set_ga_optimization_phase(1)
    seed_child = build_ga_child_from_applied(
        subject_id, applied_data, electrodes, fiducials, original_paths
    )

    os.makedirs(log_dir, exist_ok=True)

    def _save_individual(child: dict, individual_id: str) -> float:
        if repair_on_seed:
            child = new2d.apply_smart_collision_resolution(
                child,
                subject_id,
                electrodes,
                fiducials,
                original_paths,
                greedy_electrodes_only=True,
            )
        new2d.only_save_new_2D_alteration(
            child=copy.deepcopy(child),
            SUBJECT_ID=subject_id,
            electrodes=electrodes,
            fiducials=fiducials,
            INDIVIDUAL_ID=individual_id,
            metrics_mode="electrodes_only",
            ga_phase=1,
        )
        return round(
            genetics.getIndividual2DFitnessScoreFromFileLogs(
                INDIVIDUAL_ID=individual_id,
                SUBJECT_ID=subject_id,
                verbose=False,
            ),
            4,
        )

    fitness: dict[str, float] = {}
    print(f"Seeding {log_dir}/GA_{subject_id}_0-0 from applied layout...")
    fitness["0-0"] = _save_individual(copy.deepcopy(seed_child), "0-0")

    if population_size <= 1 or not mutate_siblings:
        genetics.saveFitnessTrackerToFile(data=fitness, SUBJECT_ID=subject_id)
        return fitness

    with open(
        os.path.join(log_dir, f"GA_{subject_id}_0-0_mod_connection_paths.json"),
        "r",
    ) as f:
        seed_saved = json.load(f)

    for i in range(1, population_size):
        individual_id = f"0-{i}"
        child = copy.deepcopy(seed_saved)
        child = new2d.mutateRandomElectrodePathsForSelectedChild(
            child=child,
            original_paths=original_paths,
            electrodes=electrodes,
            fiducials=fiducials,
            MUTATE_N_ELECTRODES_PERCENTAGE=mutation_fraction,
            SUBJECT_ID=subject_id,
            ga_phase=1,
        )
        print(f"Seeding {log_dir}/GA_{subject_id}_{individual_id} (mutated from 0-0)...")
        fitness[individual_id] = _save_individual(child, individual_id)

    genetics.saveFitnessTrackerToFile(data=fitness, SUBJECT_ID=subject_id)
    genetics.maybe_transition_to_phase2_after_generation(
        subject_id,
        [f"0-{i}" for i in range(population_size)],
    )
    print(f"Generation 0 seeded ({population_size} individuals). Fitness: {fitness}")
    return fitness


def run_ga_from_applied_preset(
    applied_path: str,
    subject_id: int | None = None,
    n_generations: int = 100,
    population_size: int = 20,
    clear_logs: bool = False,
    mutate_gen0_siblings: bool = True,
    repair_on_seed: bool = True,
    prefer_applied_terminals: bool = True,
    sync_terminals_to_fiducials_json: bool = False,
) -> Any:
    """
    Full GA on a subject with generation 0 warm-started from an applied-preset JSON.
    """
    import PYTHON.GA.GA as ga_module

    with open(applied_path, "r") as f:
        applied_data = json.load(f)
    if subject_id is None:
        subject_id = int(applied_data["metadata"]["target_subject_id"])

    electrodes, fiducials = load_electrode_positions_and_fiducials(scanID=subject_id)
    ensure_init_connection_paths(subject_id, electrodes, fiducials)

    if clear_logs:
        pattern = f"data/output/logs/GA_{subject_id}_*"
        removed = 0
        for path in glob.glob(pattern):
            os.remove(path)
            removed += 1
        print(f"Cleared {removed} files matching {pattern}")

    gen0_fitness = seed_ga_generation_zero(
        subject_id,
        applied_path,
        population_size=population_size,
        mutate_siblings=mutate_gen0_siblings,
        repair_on_seed=repair_on_seed,
        prefer_applied_terminals=prefer_applied_terminals,
        sync_terminals_to_fiducials_json=sync_terminals_to_fiducials_json,
    )

    print(
        f"\nStarting GA generations 1–{n_generations - 1} on subject {subject_id} "
        f"(warm-started from {applied_path})...\n"
    )
    return ga_module.run(
        electrodes=electrodes,
        fiducials=fiducials,
        SUBJECT_ID=subject_id,
        start_generation=1,
        initial_fitness_tracker=gen0_fitness,
        n_generations=n_generations,
        population_size=population_size,
    )


def _collision_highlights_for_visualize(
    paths_2d: list,
    terminal_zones: dict,
    electrode_zones: dict,
    path_electrodes: list[str],
    path_terminals: list[str],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return collision marker arrays; None on Shapely RecursionError (dense crossings)."""
    try:
        path_collisions = new2d.find_path_collisions(
            paths_2d,
            terminal_zones,
            electrode_zones=electrode_zones,
            path_electrodes=path_electrodes,
            path_terminals=path_terminals,
        )
        electrode_collisions = new2d.find_electrode_collisions(
            paths_2d, electrode_zones, path_electrodes
        )
        return path_collisions, electrode_collisions
    except RecursionError:
        print(
            "WARNING: 2D collision analysis skipped (Shapely recursion on dense crossings). "
            "Paths and zones are still plotted."
        )
        return None, None


def visualize_applied_preset(
    applied_path: str,
    save_path: str | None = None,
    show_3d: bool = False,
    show_plot: bool = True,
    only_3d: bool = False,
    skip_collisions: bool = False,
    save_3d_path: str | None = None,
) -> None:
    """
    Plot 2D layout (and optional 3D on mesh) from an apply() output JSON.
    """
    with open(applied_path, "r") as f:
        data = json.load(f)

    meta = data.get("metadata", {})
    target_id = int(meta["target_subject_id"])
    preset_id = meta.get("preset_id", "preset")
    collision_metrics = data.get("collision_metrics", {})

    electrodes, fiducials = load_subject_data(target_id)
    layout_fiducials = fiducials
    stored_terminals = meta.get("terminal_positions_3d")
    if stored_terminals:
        layout_fiducials = dict(fiducials)
        for term, pos in stored_terminals.items():
            layout_fiducials[term] = np.asarray(pos, dtype=float)
    electrodes_2d, terminals_2d, _ = build_layout_2d(
        electrodes,
        layout_fiducials,
        terminal_2d_mode=meta.get("terminal_2d_mode", "inflated"),
    )
    electrode_zones, terminal_zones = new2d.create_zones(electrodes_2d, terminals_2d)

    path_specs = data.get("paths", [])
    paths_2d = [np.asarray(p["modified_path_2d"], dtype=float) for p in path_specs]
    path_electrodes = [p["electrode"] for p in path_specs]
    path_terminals = [p["terminal"] for p in path_specs]
    entry_points_2d = {
        p["electrode"]: np.asarray(p["entry_point_2d"], dtype=float)
        for p in path_specs
        if p.get("entry_point_2d") is not None
    }

    if not only_3d:
        if skip_collisions:
            path_collisions, electrode_collisions = None, None
        else:
            path_collisions, electrode_collisions = _collision_highlights_for_visualize(
                paths_2d,
                terminal_zones,
                electrode_zones,
                path_electrodes,
                path_terminals,
            )

        if save_path is None:
            stem = Path(applied_path).stem
            save_path = f"data/output/pics/{stem}_2d.png"

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        title = (
            f"Applied preset — subject {target_id} ({preset_id})\n"
            f"collision_score={collision_metrics.get('collision_score', '?')}, "
            f"crossings={collision_metrics.get('crossing_count', '?')}, "
            f"electrode_violations={collision_metrics.get('electrode_violations', '?')}"
        )
        new2d.plot_single_version(
            ax=None,
            paths=paths_2d,
            electrodes=electrodes_2d,
            terminals=terminals_2d,
            electrode_zones=electrode_zones,
            terminal_zones=terminal_zones,
            path_collisions=path_collisions,
            electrode_collisions=electrode_collisions,
            title=title,
            dpi=300,
            show_plot=show_plot and not show_3d,
            save_path=save_path,
            entry_points_by_electrode=entry_points_2d or None,
        )
        print(f"2D layout saved: {save_path}")

    if not show_3d:
        return

    pv = new2d._pyvista()
    mesh = _pyvista_read_stl(target_id)
    off_screen = bool(save_3d_path)
    plotter = pv.Plotter(
        window_size=(1200, 900),
        off_screen=off_screen,
    )
    plotter.add_mesh(mesh, color="white", opacity=0.75)
    for name, pos in electrodes.items():
        plotter.add_mesh(pv.Sphere(radius=mesh.length * 0.008, center=pos), color="red")
        plotter.add_point_labels([pos], [name], font_size=10)
    for term in ("TERMINAL_LEFT", "TERMINAL_RIGHT"):
        if term in layout_fiducials:
            pos = layout_fiducials[term]
            plotter.add_mesh(pv.Sphere(radius=mesh.length * 0.01, center=pos), color="gray")
            plotter.add_point_labels([pos], [term.split("_")[-1]], font_size=10)
    for conn in path_specs:
        entry_3d = conn.get("entry_position_3d")
        if entry_3d is not None:
            ep = np.asarray(entry_3d, dtype=float)
            plotter.add_mesh(
                pv.Sphere(radius=mesh.length * 0.006, center=ep), color="lime"
            )
    for conn in path_specs:
        path_3d = np.asarray(conn.get("path_points"), dtype=float)
        if len(path_3d) >= 2:
            plotter.add_mesh(pv.Spline(path_3d), color="cyan", line_width=4)
    plotter.add_title(f"Subject {target_id} — applied preset {preset_id}", font_size=14)
    if save_3d_path:
        os.makedirs(os.path.dirname(save_3d_path) or ".", exist_ok=True)
        plotter.show(auto_close=False)
        plotter.screenshot(save_3d_path)
        plotter.close()
        print(f"3D layout saved: {save_3d_path}")
    else:
        plotter.show()


def _roundtrip_self_test() -> None:
    """Sanity-check normalize → denormalize without GA files."""
    e2d = np.array([0.0, 0.0])
    t2d = np.array([10.0, 0.0])
    path = np.array(
        [
            e2d,
            [3.0, 1.0],
            [7.0, -0.5],
            t2d,
        ],
        dtype=float,
    )
    norm = normalize_path_2d(path, e2d, t2d)
    restored = denormalize_path_2d(norm, e2d, t2d)
    err = float(np.max(np.linalg.norm(restored - path, axis=1)))
    if err > 1e-9:
        raise AssertionError(f"Round-trip error {err} exceeds tolerance")
    print(f"Round-trip self-test passed (max error {err:.2e})")


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export and apply EEG layout presets across subjects."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    repair_p = sub.add_parser(
        "repair",
        help="2D collision repair on applied preset output (GA smart_collision_resolution)",
    )
    repair_p.add_argument(
        "--applied",
        required=True,
        help="Input applied_preset_*.json from apply",
    )
    repair_p.add_argument("--out", default=None, help="Output JSON path")
    repair_p.add_argument(
        "--electrodes-only",
        action="store_true",
        help="Only phase-1 electrode clearance (skip crossing/separation pass)",
    )
    repair_p.add_argument("--uv-resolution", type=int, default=UV_GRID_RESOLUTION)

    viz_p = sub.add_parser(
        "visualize", help="2D/3D plot of an apply() output JSON"
    )
    viz_p.add_argument(
        "--applied",
        required=True,
        help="Path to applied_preset_*.json from apply",
    )
    viz_p.add_argument(
        "--save",
        default=None,
        help="2D PNG path (default: data/output/pics/<stem>_2d.png)",
    )
    viz_p.add_argument(
        "--3d",
        dest="show_3d",
        action="store_true",
        help="Open PyVista 3D viewer on the target mesh",
    )
    viz_p.add_argument(
        "--3d-only",
        dest="only_3d",
        action="store_true",
        help="Skip 2D plot; open PyVista only (avoids collision analysis)",
    )
    viz_p.add_argument(
        "--skip-collisions",
        action="store_true",
        help="Skip 2D Shapely collision markers (faster; use if analysis recurses)",
    )
    viz_p.add_argument(
        "--no-show",
        action="store_true",
        help="Save 2D PNG only, do not open matplotlib window",
    )
    viz_p.add_argument(
        "--save-3d",
        default=None,
        help="PNG path for PyVista screenshot (use with --3d or --3d-only)",
    )

    ga_p = sub.add_parser(
        "ga",
        help="Full GA on a subject warm-started from applied/repaired preset JSON",
    )
    ga_p.add_argument(
        "--applied",
        required=True,
        help="applied_preset_*.json (target_subject_id must match --subject)",
    )
    ga_p.add_argument(
        "--subject",
        type=int,
        default=None,
        help="Override subject ID (default: from applied JSON metadata)",
    )
    ga_p.add_argument("--generations", type=int, default=100)
    ga_p.add_argument("--population", type=int, default=20)
    ga_p.add_argument(
        "--clear-logs",
        action="store_true",
        help="Delete existing data/output/logs/GA_{subject}_* before seeding",
    )
    ga_p.add_argument(
        "--no-mutate-gen0",
        action="store_true",
        help="Only seed individual 0-0; skip mutated 0-1..0-N-1",
    )
    ga_p.add_argument(
        "--raw-seed",
        action="store_true",
        help="Do not run electrode repair when seeding generation 0 (preserve apply layout)",
    )
    ga_p.add_argument(
        "--file-terminals",
        action="store_true",
        help="Use fiducials_*.json terminal clicks instead of applied/inherited metadata",
    )
    ga_p.add_argument(
        "--sync-terminals",
        action="store_true",
        help="Write applied/inherited TERMINAL_LEFT/RIGHT into fiducials_{subject}.json before GA",
    )

    test_p = sub.add_parser("self-test", help="Run normalize/denormalize round-trip")
    test_p.set_defaults(command="self-test")

    ev4 = sub.add_parser(
        "export-v4",
        help="Export v4 preset (3D chord shape + rigid landmark terminals)",
    )
    ev4.add_argument("--subject", type=int, required=True)
    ev4.add_argument("--individual", required=True)
    ev4.add_argument("--out", required=True)
    ev4.add_argument("--log-dir", default=None)
    ev4.add_argument("--preset-id", default=None)

    av4 = sub.add_parser(
        "apply-v4",
        help="Apply v4 preset (no GA): 4 fiducials + target electrodes only",
    )
    av4.add_argument("--preset", required=True)
    av4.add_argument("--target", type=int, required=True)
    av4.add_argument("--out", default=None)
    av4.add_argument(
        "--fit-terminals",
        action="store_true",
        help="Legacy: rotate hub angle to reduce slot-order inversions",
    )
    av4.add_argument(
        "--synthesize",
        action="store_true",
        help="Straight paths + hub/entry optimization (no chord replay, no repair)",
    )
    av4.add_argument(
        "--preserve-entry-order",
        action="store_true",
        help="With --synthesize: keep source strip slot order (default: target-native slots)",
    )
    av4.add_argument(
        "--tail-swap",
        action="store_true",
        help="With --synthesize --preserve-entry-order: pairwise tail swap at crossings",
    )
    av4.add_argument(
        "--use-target-terminals",
        action="store_true",
        help="Synthesize: hubs from target fiducials_*.json; preset assignments only",
    )
    av4.add_argument(
        "--rotate",
        action="store_true",
        help="With --synthesize: ±36° hub angle search around fiducial clicks",
    )

    rf4 = sub.add_parser(
        "refine-v4",
        help="Polish v4 apply output: more phase-2 + aggressive crossing/separation fix",
    )
    rf4.add_argument("--applied", required=True)
    rf4.add_argument("--out", default=None)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_cli()
    args = parser.parse_args(argv)

    if args.command == "self-test":
        _roundtrip_self_test()
        return

    if args.command == "repair":
        repair_applied_preset(
            args.applied,
            output_path=args.out,
            electrodes_only=args.electrodes_only,
            uv_resolution=args.uv_resolution,
        )
        return

    if args.command == "ga":
        run_ga_from_applied_preset(
            args.applied,
            subject_id=args.subject,
            n_generations=args.generations,
            population_size=args.population,
            clear_logs=args.clear_logs,
            mutate_gen0_siblings=not args.no_mutate_gen0,
            repair_on_seed=not args.raw_seed,
            prefer_applied_terminals=not args.file_terminals,
            sync_terminals_to_fiducials_json=args.sync_terminals,
        )
        return

    if args.command == "visualize":
        show_3d = args.show_3d or args.only_3d
        visualize_applied_preset(
            args.applied,
            save_path=args.save,
            show_3d=show_3d,
            show_plot=not args.no_show,
            only_3d=args.only_3d,
            skip_collisions=args.skip_collisions,
            save_3d_path=args.save_3d,
        )
        return

    if args.command == "export-v4":
        import PYTHON.tools.layoutPresetV4 as v4

        v4.export_layout_preset_v4(
            args.subject,
            args.out,
            args.individual,
            log_dir=args.log_dir,
            preset_id=args.preset_id,
        )
        return

    if args.command == "apply-v4":
        import PYTHON.tools.layoutPresetV4 as v4

        if args.synthesize:
            v4.apply_layout_preset_v4_synthesize(
                args.preset,
                args.target,
                output_path=args.out,
                preserve_entry_order=args.preserve_entry_order,
                use_tail_swap=args.tail_swap,
                use_target_terminals=args.use_target_terminals,
                optimize_terminals=args.rotate,
            )
        else:
            v4.apply_layout_preset_v4(
                args.preset,
                args.target,
                output_path=args.out,
                fit_terminals=args.fit_terminals,
            )
        return

    if args.command == "refine-v4":
        import PYTHON.tools.layoutPresetV4 as v4

        v4.refine_applied_v4(args.applied, output_path=args.out)
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
