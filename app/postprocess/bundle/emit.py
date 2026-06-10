"""Emit eeg_subject_bundle/1.0.0 from smoothed layout JSON."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app import paths
from app.postprocess.bundle.schema import (
    ANATOMICAL_KEYS,
    CALIBRATION_LANDMARK_KEYS,
    CALIBRATION_LANDMARK_NAMES,
    ELECTRODE_CIRCLE_RESOLUTION,
    ELECTRODE_DIAMETER_MM,
    SCHEMA_VERSION,
)
from app.postprocess.export_matlab_legacy import (
    create_matlab_data_structure,
    load_final_paths,
    resolve_mesh_file,
)
from app.postprocess.mesh_export import load_mesh_context
from app.preprocess.fiducials_io import load_picks, matlab_landmarks_from_picks


class CalibrationLandmarksMissingError(ValueError):
    """Raised when fiducials JSON lacks calibration picks required for export."""


def _subject_id_from_mesh(mesh_file: str) -> int | None:
    stem = Path(mesh_file).stem
    digits = "".join(c for c in stem if c.isdigit())
    return int(digits) if digits else None


def require_calibration_landmarks(subject_id: int) -> tuple[np.ndarray, list[str]]:
    picks = load_picks(subject_id)
    missing = [k for k in CALIBRATION_LANDMARK_KEYS if k not in picks]
    if missing:
        raise CalibrationLandmarksMissingError(
            f"fiducials_{subject_id}.json missing calibration landmarks: {missing}. "
            f"Run: python -m app preprocess --subject {subject_id} --step fiducials "
            f"and complete picks 7–9 ({', '.join(CALIBRATION_LANDMARK_KEYS)})."
        )
    parsed = matlab_landmarks_from_picks(picks)
    if parsed is None:
        raise CalibrationLandmarksMissingError(
            f"Could not parse calibration landmarks from fiducials_{subject_id}.json"
        )
    return parsed


def _anatomical_xyz(picks: dict) -> np.ndarray | None:
    if not all(k in picks for k in ANATOMICAL_KEYS):
        return None
    return np.asarray([[float(c) for c in picks[k]] for k in ANATOMICAL_KEYS], dtype=float)


def _electrode_centers(final_paths_data: dict) -> np.ndarray:
    positions = final_paths_data.get("electrode_positions", {})
    order = [p["electrode"] for p in final_paths_data["final_paths"]]
    return np.asarray([positions[name] for name in order], dtype=float)


def _terminals(final_paths_data: dict) -> list[str]:
    return [str(p.get("terminal", "")) for p in final_paths_data["final_paths"]]


def _build_manifest(
    *,
    subject_id: int | str,
    bundle_dir: Path,
    smooth_json: Path,
    mesh_file: Path,
    channel_count: int,
) -> dict:
    rel = lambda p: str(p.relative_to(paths.REPO_ROOT)) if p.is_relative_to(paths.REPO_ROOT) else str(p)
    return {
        "schema_version": SCHEMA_VERSION,
        "subject_id": int(subject_id) if str(subject_id).isdigit() else subject_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "coordinate_frame": {
            "space": "head_scan",
            "units": "mm",
            "origin": "landmark_index_0",
            "origin_note": (
                "Postprocessor subtracts landmarks[0] before registration; "
                "bundle stores pre-origin coordinates."
            ),
            "normal_convention": "outward_from_head_surface",
            "mesh_face_indexing": "zero_based_in_npz",
            "matlab_face_indexing": "one_based_in_legacy_mat",
        },
        "landmarks": {
            "calibration": {
                "names": list(CALIBRATION_LANDMARK_NAMES),
                "keys": list(CALIBRATION_LANDMARK_KEYS),
                "count": 3,
                "array_ref": "geometry.npz:landmarks_xyz",
            },
            "anatomical": {
                "keys": list(ANATOMICAL_KEYS),
                "array_ref": "geometry.npz:anatomical_xyz",
                "purpose": "layout_synthesis_only",
            },
        },
        "electrode_layout": "standard_10-20",
        "channel_count": channel_count,
        "arrays": {
            "mesh_points": "geometry.npz:mesh_points",
            "mesh_faces": "geometry.npz:mesh_faces",
            "trace_names": "traces.npz:channel_names",
            "interconnect_xyzn": "traces.npz:interconnect_xyzn",
            "electrode_xyzn": "traces.npz:electrode_xyzn",
        },
        "electrode_circle": {
            "diameter_mm": ELECTRODE_DIAMETER_MM,
            "resolution": ELECTRODE_CIRCLE_RESOLUTION,
            "generation": "mesh_projected_circle",
        },
        "sources": {
            "smooth_json": rel(smooth_json),
            "mesh_stl": rel(mesh_file),
            "fiducials_json": rel(paths.fiducials_json(subject_id)),
            "bundle_dir": str(bundle_dir),
        },
    }


def export_bundle(
    smooth_json: str | Path,
    output_folder: str | Path | None = None,
    *,
    strict_landmarks: bool = True,
    skip_validation: bool = False,
    verbose: bool = True,
) -> Path:
    """
    Export canonical subject bundle: manifest.json, geometry.npz, traces.npz.

    Requires calibration landmarks in fiducials JSON when strict_landmarks=True.
    """
    from app.runtime import setup_runtime

    setup_runtime()

    smooth_path = paths.resolve_json_path(smooth_json, role="Smooth JSON")
    final_paths_data = load_final_paths(str(smooth_path), verbose=verbose)
    if not skip_validation:
        from app.postprocess.validate_export import validate_smooth_for_export

        validate_smooth_for_export(
            final_paths_data,
            smooth_path=smooth_path,
            require_collision_free=True,
        )

    mesh_file = Path(resolve_mesh_file(str(smooth_path), final_paths_data["mesh_file"]))

    subject_id = _subject_id_from_mesh(str(mesh_file))
    if subject_id is None:
        raise ValueError(f"Could not infer subject id from mesh file: {mesh_file}")

    if output_folder is None:
        out = paths.bundle_export_dir(subject_id)
    else:
        out = Path(output_folder)
        if not out.is_absolute():
            out = paths.REPO_ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    if strict_landmarks:
        landmarks_xyz, landmark_names = require_calibration_landmarks(subject_id)
    else:
        from app.postprocess.export_matlab_legacy import create_landmarks_data

        landmarks_xyz, landmark_names = create_landmarks_data(
            final_paths_data,
            subject_id=subject_id,
            strict_landmarks=False,
        )

    picks = load_picks(subject_id)
    anatomical = _anatomical_xyz(picks)

    if verbose:
        print(f" Loading mesh from: {mesh_file}")
    ctx = load_mesh_context(mesh_file)
    interconnects, electrodes, path_names = create_matlab_data_structure(
        final_paths_data, ctx, verbose=verbose
    )

    mesh_points = np.asarray(ctx.mesh.points, dtype=np.float64)
    mesh_faces = np.asarray(ctx.mesh.faces.reshape(-1, 4)[:, 1:4], dtype=np.int32)

    geometry_kwargs: dict = {
        "mesh_points": mesh_points,
        "mesh_faces": mesh_faces,
        "landmarks_xyz": np.asarray(landmarks_xyz, dtype=np.float64),
        "electrode_centers": _electrode_centers(final_paths_data),
    }
    if anatomical is not None:
        geometry_kwargs["anatomical_xyz"] = anatomical

    np.savez_compressed(out / "geometry.npz", **geometry_kwargs)

    ic_obj = np.empty(len(interconnects), dtype=object)
    el_obj = np.empty(len(electrodes), dtype=object)
    for i, (ic, el) in enumerate(zip(interconnects, electrodes)):
        ic_obj[i] = np.asarray(ic, dtype=np.float64)
        el_obj[i] = np.asarray(el, dtype=np.float64)

    np.savez_compressed(
        out / "traces.npz",
        channel_names=np.asarray(path_names, dtype=object),
        terminals=np.asarray(_terminals(final_paths_data), dtype=object),
        interconnect_xyzn=ic_obj,
        electrode_xyzn=el_obj,
    )

    manifest = _build_manifest(
        subject_id=subject_id,
        bundle_dir=out,
        smooth_json=smooth_path,
        mesh_file=mesh_file,
        channel_count=len(path_names),
    )
    manifest["landmarks"]["calibration"]["names"] = list(landmark_names)
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    if verbose:
        print(f" Bundle export complete: {out}")
        print("   manifest.json")
        print(f"   geometry.npz ({len(mesh_points)} vertices, {len(mesh_faces)} faces)")
        print(f"   traces.npz ({len(path_names)} channels)")
        pm_hint = paths.postprocessor_subject_pm(subject_id)
        if not pm_hint.is_file():
            print(
                f" Next: python -m app init-print-config --subject {subject_id} "
                f"(then convert-gcode)"
            )
    return out
