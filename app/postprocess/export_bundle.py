"""Stage D — export smoothed JSON to eeg_subject_bundle/1.0.0."""
from __future__ import annotations

from pathlib import Path

from app import paths
from app.runtime import setup_runtime


def export_bundle(
    smooth_json: str | Path,
    output_folder: str | Path | None = None,
    *,
    strict_landmarks: bool = True,
    skip_validation: bool = False,
    quiet: bool = False,
) -> Path:
    setup_runtime()
    from app.postprocess.bundle.emit import export_bundle as _export

    smooth_path = paths.resolve_json_path(smooth_json, role="Smooth JSON")
    if output_folder is None:
        data = __import__("json").loads(smooth_path.read_text(encoding="utf-8"))
        mesh_entry = data.get("mesh_file", "")
        sid = "".join(c for c in Path(mesh_entry).stem if c.isdigit()) or None
        out = paths.bundle_export_dir(int(sid)) if sid else paths.bundle_export_dir("unknown")
    else:
        out = Path(output_folder)
        if not out.is_absolute():
            out = paths.REPO_ROOT / out

    return _export(
        smooth_path,
        out,
        strict_landmarks=strict_landmarks,
        skip_validation=skip_validation,
        verbose=not quiet,
    )
