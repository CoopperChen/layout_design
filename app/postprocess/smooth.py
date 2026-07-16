"""Stage D — smooth 3D paths from applied/repaired layout JSON."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app import paths
from app.config_loader import load_defaults
from app.runtime import setup_runtime

_TARGET_SEGMENT_MM = 3.0
_MAX_SMOOTH_POINTS = 300


def _resampled_count(path: np.ndarray) -> int:
    """Point count for ~_TARGET_SEGMENT_MM spacing (denser toolpath, gentler axis steps)."""
    total = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    n = int(np.ceil(total / _TARGET_SEGMENT_MM)) + 1
    return int(np.clip(n, len(path), _MAX_SMOOTH_POINTS))


def _smooth_3d_path(path: np.ndarray, smoothing_factor: float) -> np.ndarray:

    from scipy.interpolate import splev, splprep



    if len(path) < 3:

        return path.copy()

    try:

        if np.any(np.isnan(path)) or np.any(np.isinf(path)):

            return path.copy()

        unique_mask = np.ones(len(path), dtype=bool)

        for i in range(1, len(path)):

            if np.allclose(path[i], path[i - 1], atol=1e-10):

                unique_mask[i] = False

        cleaned_path = path[unique_mask]

        if len(cleaned_path) < 4:

            return path.copy()

        s_param = smoothing_factor * len(cleaned_path) * 10

        n_out = _resampled_count(cleaned_path)

        strategies = [

            (3, s_param),

            (2, s_param),

            (1, s_param),

            (3, s_param * 0.1),

            (3, 0),

            (1, 0),

        ]

        for k, s in strategies:

            if len(cleaned_path) <= k:

                continue

            try:

                tck, _u = splprep(cleaned_path.T, s=s, k=k)

                u_new = np.linspace(0, 1, n_out)

                smoothed_path = np.column_stack(splev(u_new, tck))

                smoothed_path[0] = path[0]

                smoothed_path[-1] = path[-1]

                return smoothed_path

            except Exception:

                continue

        return path.copy()

    except Exception:

        return path.copy()





def smooth_from_applied(

    applied: str | Path,

    output: str | Path | None = None,

    *,

    tag: str = "final",

    smoothing_strength: float | None = None,

) -> Path:

    setup_runtime()

    import PYTHON.tools.reconstructUsingUVmesh as recon
    import pyvista as pv



    applied_parts = paths.split_concatenated_paths(applied)

    applied_path = paths.resolve_json_path(applied_parts[0], role="Applied layout")

    if output is None and len(applied_parts) > 1:

        output = applied_parts[1]

    data = json.loads(applied_path.read_text(encoding="utf-8"))

    subject_id = int(data["metadata"]["target_subject_id"])



    from PYTHON.tools.helper import load_electrode_positions_and_fiducials



    electrodes, fiducials = load_electrode_positions_and_fiducials(scanID=subject_id)

    mesh = pv.read(str(paths.cleaned_scan(subject_id)))



    defaults = load_defaults().get("postprocess", {})

    strength = smoothing_strength if smoothing_strength is not None else float(

        defaults.get("smoothing_strength", 0.1)

    )



    uv_context = None

    if data.get("uv_grid"):

        uv_context = recon.UVReconstructionContext(data["uv_grid"], mesh)



    from app.layout.visualize import _surface_paths_3d



    if data.get("metadata", {}).get("path_lift") == "straight_synthesize":

        data = _surface_paths_3d(data, subject_id)



    paths_3d = []

    for conn in data["paths"]:

        if conn.get("path_points"):

            path_3d = np.asarray(conn["path_points"], dtype=float)

        elif uv_context is not None:

            end3d = conn.get("path_end_3d") or conn.get("entry_position_3d")

            if end3d is None:

                end3d = fiducials[conn["terminal"]]

            path_3d = uv_context.reconstruct(

                np.array(electrodes[conn["electrode"]]),

                np.array(end3d),

                np.array(conn["modified_path_2d"]),

            )

        else:

            raise ValueError(

                f"Path {conn.get('electrode')} missing path_points and no uv_grid in {applied_path}"

            )

        paths_3d.append(_smooth_3d_path(path_3d, strength))



    terminal_positions = {

        k: fiducials[k] for k in fiducials if "TERMINAL" in k

    }

    if output:

        out_parts = paths.split_concatenated_paths(output)

        out_path = Path(out_parts[0].strip().strip('"'))

        if not out_path.is_absolute():

            out_path = paths.REPO_ROOT / out_path

    else:

        out_path = paths.smooth_json(subject_id, tag)

    out_path.parent.mkdir(parents=True, exist_ok=True)



    mesh_rel = paths.cleaned_scan(subject_id).relative_to(paths.REPO_ROOT)

    synth_stop = data.get("metadata", {}).get("terminal_stop_mm")

    final_model = {

        "mesh_file": mesh_rel.as_posix(),

        "electrode_positions": {

            name: pos.tolist() if isinstance(pos, np.ndarray) else pos

            for name, pos in electrodes.items()

        },

        "terminal_positions": {

            name: pos.tolist() if isinstance(pos, np.ndarray) else pos

            for name, pos in terminal_positions.items()

        },

        "final_paths": [],

        "source_applied": applied_path.as_posix(),

        "smoothing_strength": strength,

    }

    if synth_stop is not None:

        final_model["terminal_stop_mm"] = synth_stop

    if data.get("collision_metrics"):

        final_model["collision_metrics"] = data["collision_metrics"]

    for i, path in enumerate(paths_3d):

        conn = data["paths"][i]

        final_model["final_paths"].append(

            {

                "electrode": conn["electrode"],

                "terminal": conn["terminal"],

                "path_3d": path.tolist(),

            }

        )

    out_path.write_text(json.dumps(final_model, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {out_path} ({len(paths_3d)} paths, strength={strength})")

    return out_path

