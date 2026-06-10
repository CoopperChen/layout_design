"""2D and 3D layout visualization."""
from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Literal

import numpy as np

from app import paths
from app.runtime import setup_runtime

VisualizeMode = Literal["2d", "3d", "both"]

_LAYOUT_REQUIRED = (
    "Expected a layout JSON from synthesize/polish "
    "(metadata + paths), not a smooth export."
)


def _tag_from_applied(stem: str, subject_id: int | str) -> str:
    suffix = f"_s{subject_id}"
    if stem.endswith(suffix):
        return stem[: -len(suffix)] or "layout"
    return stem


def _resolve_applied(applied: str | Path) -> Path:
    p = Path(applied)
    return p if p.is_absolute() else paths.REPO_ROOT / p


def _infer_subject_id(stem: str, mesh_file: str | None = None) -> int | None:
    if mesh_file:
        m = re.search(r"(\d+)\.stl", mesh_file.replace("\\", "/"), re.I)
        if m:
            return int(m.group(1))
    for pattern in (r"_s(\d+)(?:_|\.|$)", r"smooth_s(\d+)", r"subject_(\d+)"):
        m = re.search(pattern, stem, re.I)
        if m:
            return int(m.group(1))
    return None


def _coerce_layout_document(raw: dict, applied_path: Path) -> dict:
    """
    Normalize synthesize/polish layout JSON for visualize.

    Also accepts smooth-stage JSON (final_paths) by converting to layout shape.
    """
    if raw.get("metadata") and raw.get("paths"):
        return raw

    if raw.get("final_paths") and raw.get("mesh_file"):
        subject_id = _infer_subject_id(applied_path.stem, raw.get("mesh_file"))
        if subject_id is None:
            raise ValueError(
                f"Cannot infer subject id from smooth file {applied_path.name}. "
                "Pass a layout under data/output/layouts/ (e.g. synth_s2.json)."
            )
        setup_runtime()
        import PYTHON.tools.new2dAlterations as new2d
        from PYTHON.tools.layoutPreset import load_subject_data

        electrodes, fiducials = load_subject_data(subject_id)
        cz_pos = np.asarray(electrodes["Cz"], dtype=float)
        terminals_3d = {
            k: np.asarray(v, dtype=float).tolist()
            for k, v in (raw.get("terminal_positions") or {}).items()
            if "TERMINAL" in k
        }
        if not terminals_3d:
            terminals_3d = {
                k: np.asarray(fiducials[k], dtype=float).tolist()
                for k in fiducials
                if "TERMINAL" in k
            }

        path_entries = []
        for fp in raw["final_paths"]:
            conn: dict = {
                "electrode": fp["electrode"],
                "terminal": fp["terminal"],
                "path_points": fp.get("path_3d") or fp.get("path_points"),
            }
            if conn["path_points"] is None:
                continue
            path_3d = np.asarray(conn["path_points"], dtype=float)
            conn["modified_path_2d"] = [
                new2d.polar_projection(np.array([p]), cz_pos)[0].tolist()
                for p in path_3d
            ]
            path_entries.append(conn)

        return {
            "metadata": {
                "target_subject_id": subject_id,
                "preset_id": "smoothed",
                "path_lift": "smooth",
                "terminal_positions_3d": terminals_3d,
            },
            "paths": path_entries,
            "collision_metrics": {},
        }

    raise ValueError(
        f"{_LAYOUT_REQUIRED}\n"
        f"  File: {applied_path}\n"
        f"  Keys found: {sorted(raw.keys())}\n"
        f"  Use: python -m app visualize --applied data/output/layouts/synth_s2.json"
    )


def _surface_paths_3d(data: dict, subject_id: int) -> dict:
    """
    Lift modified_path_2d onto the head mesh (UV grid + closest-point snap).

    Replaces straight 3D chords from older synthesize runs (path_lift=straight_synthesize).
    """
    setup_runtime()
    import PYTHON.tools.new2dAlterations as new2d
    import PYTHON.tools.reconstructUsingUVmesh as recon
    import pyvista as pv
    from PYTHON.tools.layoutPreset import build_layout_2d, load_subject_data, uv_grid_for_context
    from PYTHON.tools.layoutPresetV4 import (
        entry_3d_for_strip,
        pin_path_endpoints_3d,
        snap_path_to_mesh,
    )

    electrodes, fiducials = load_subject_data(subject_id)
    mesh = pv.read(str(paths.cleaned_scan(subject_id)))
    cz_pos = np.asarray(electrodes["Cz"], dtype=float)

    layout_fiducials = dict(fiducials)
    stored = data.get("metadata", {}).get("terminal_positions_3d")
    if stored:
        for term, pos in stored.items():
            layout_fiducials[term] = np.asarray(pos, dtype=float)

    if data.get("uv_grid"):
        uv_ctx = recon.UVReconstructionContext(data["uv_grid"], mesh)
    else:
        uv_raw = new2d.create_uv_grid(mesh, cz_pos, resolution=100)
        uv_ctx = recon.UVReconstructionContext(uv_grid_for_context(uv_raw), mesh)

    terminal_2d_mode = data.get("metadata", {}).get("terminal_2d_mode", "inflated_legacy")
    if terminal_2d_mode == "fiducial_native":
        t2d_mode = "fiducial"
    else:
        t2d_mode = "inflated"
    electrodes_2d, terminals_2d, _ = build_layout_2d(
        electrodes, layout_fiducials, terminal_2d_mode=t2d_mode
    )
    terminal_zone_size = None
    if t2d_mode == "fiducial":
        ez, _ = new2d.create_zones(electrodes_2d, terminals_2d)
        terminal_zone_size = float(ez["metadata"].get("terminal_zone_size", 0.0))

    out = deepcopy(data)
    lift = data.get("metadata", {}).get("path_lift")
    for conn in out.get("paths", []):
        if conn.get("path_points") and lift in ("smooth",):
            continue
        if conn.get("path_points") and not conn.get("modified_path_2d"):
            continue
        electrode = conn["electrode"]
        terminal = conn["terminal"]
        e3d = np.asarray(electrodes[electrode], dtype=float)
        t3d = np.asarray(layout_fiducials[terminal], dtype=float)
        path_2d = np.asarray(conn["modified_path_2d"], dtype=float)
        e2d = electrodes_2d[electrode]
        if conn.get("entry_point_2d") is not None:
            end2d = np.asarray(conn["entry_point_2d"], dtype=float)
        else:
            end2d = new2d.polar_projection(np.array([t3d]), cz_pos)[0]
        end3d = entry_3d_for_strip(
            end2d,
            uv_ctx,
            mesh,
            e3d=e3d,
            terminal_3d=t3d,
            e2d=e2d,
            cz_pos=cz_pos,
            terminal_2d_mode=terminal_2d_mode,
            terminal_zone_size=terminal_zone_size,
        )

        path_3d = uv_ctx.reconstruct(e3d, end3d, path_2d)
        path_3d = snap_path_to_mesh(path_3d, mesh, pin_endpoints=True)
        path_3d = pin_path_endpoints_3d(path_3d, e3d, end3d, mesh)
        conn["path_points"] = path_3d.tolist()
        if conn.get("entry_point_2d") is not None:
            conn["entry_position_3d"] = end3d.tolist()
    return out


def visualize_layout(
    applied: str | Path,
    *,
    mode: VisualizeMode = "both",
    save_2d: str | Path | None = None,
    save_3d: str | Path | None = None,
    show: bool = False,
    show_3d: bool = True,
    skip_collisions: bool = False,
) -> tuple[Path | None, Path | None]:
    """
    Render layout JSON as 2D polar plot and/or 3D head mesh view.

    Default (mode=both):
      - Saves 2D PNG under data/output/pics/{tag}_s{id}_2d.png
      - Opens interactive PyVista 3D window (wires on mesh surface)
    """
    setup_runtime()
    from PYTHON.tools.layoutPreset import visualize_applied_preset

    applied_path = _resolve_applied(applied)
    if not applied_path.exists():
        raise FileNotFoundError(applied_path)

    raw = json.loads(applied_path.read_text(encoding="utf-8"))
    raw = _coerce_layout_document(raw, applied_path)
    subject_id = int(raw["metadata"]["target_subject_id"])
    tag = _tag_from_applied(applied_path.stem, subject_id)
    preset_id = raw.get("metadata", {}).get("preset_id", "layout")

    enriched = _surface_paths_3d(raw, subject_id)
    work_path = applied_path
    if enriched is not raw:
        work_path = paths.DATA_DIR / "output" / "pics" / f".{tag}_s{subject_id}_viz_tmp.json"
        work_path.parent.mkdir(parents=True, exist_ok=True)
        work_path.write_text(json.dumps(enriched, indent=2), encoding="utf-8")

    out_2d: Path | None = None
    out_3d: Path | None = None

    do_2d = mode in ("2d", "both")
    do_3d = mode in ("3d", "both") and show_3d

    if do_2d:
        out_2d = Path(save_2d) if save_2d else paths.layout_pic(subject_id, tag, "2d")
        if not out_2d.is_absolute():
            out_2d = paths.REPO_ROOT / out_2d
        out_2d.parent.mkdir(parents=True, exist_ok=True)

    if save_3d is not None:
        out_3d = Path(save_3d)
        if not out_3d.is_absolute():
            out_3d = paths.REPO_ROOT / out_3d
        out_3d.parent.mkdir(parents=True, exist_ok=True)

    if do_2d:
        visualize_applied_preset(
            str(work_path),
            save_path=str(out_2d),
            show_3d=False,
            show_plot=show,
            only_3d=False,
            skip_collisions=skip_collisions,
            save_3d_path=None,
        )

    if do_3d:
        _show_3d_interactive(
            enriched,
            subject_id=subject_id,
            preset_id=str(preset_id),
            save_3d_path=str(out_3d) if out_3d else None,
        )

    if work_path != applied_path and work_path.name.startswith("."):
        try:
            work_path.unlink()
        except OSError:
            pass

    if out_2d:
        print(f"2D: {out_2d}")
    if do_3d and out_3d is None:
        print("3D: interactive PyVista window (wires on scalp mesh; close to exit)")
    elif out_3d:
        print(f"3D screenshot: {out_3d}")
    elif mode in ("3d", "both") and not show_3d:
        print("3D: skipped (omit --no-show to open interactive view)")

    return out_2d, out_3d


def _show_3d_interactive(
    data: dict,
    *,
    subject_id: int,
    preset_id: str,
    save_3d_path: str | None = None,
) -> None:
    """PyVista mesh + surface-snapped wires."""
    setup_runtime()
    import pyvista as pv
    from PYTHON.tools.layoutPreset import _pyvista_read_stl, load_subject_data

    electrodes, fiducials = load_subject_data(subject_id)
    layout_fiducials = dict(fiducials)
    stored = data.get("metadata", {}).get("terminal_positions_3d")
    if stored:
        for term, pos in stored.items():
            layout_fiducials[term] = np.asarray(pos, dtype=float)

    mesh = _pyvista_read_stl(subject_id)
    off_screen = bool(save_3d_path)
    plotter = pv.Plotter(window_size=(1200, 900), off_screen=off_screen)
    plotter.add_mesh(mesh, color="white", opacity=0.75)
    for name, pos in electrodes.items():
        plotter.add_mesh(pv.Sphere(radius=mesh.length * 0.008, center=pos), color="red")
        plotter.add_point_labels([pos], [name], font_size=10)
    for term in ("TERMINAL_LEFT", "TERMINAL_RIGHT"):
        if term in layout_fiducials:
            pos = layout_fiducials[term]
            plotter.add_mesh(pv.Sphere(radius=mesh.length * 0.01, center=pos), color="gray")
            plotter.add_point_labels([pos], [term.split("_")[-1]], font_size=10)
    for conn in data.get("paths", []):
        path_3d = np.asarray(conn.get("path_points"), dtype=float)
        if len(path_3d) >= 2:
            plotter.add_mesh(pv.Spline(path_3d), color="cyan", line_width=4)
    plotter.add_title(f"Subject {subject_id} — {preset_id}", font_size=14)

    if save_3d_path:
        plotter.show(auto_close=False)
        plotter.screenshot(save_3d_path)
        plotter.close()
    else:
        plotter.show()
