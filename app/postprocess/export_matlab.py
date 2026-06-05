"""Stage D — export smoothed JSON to MATLAB .mat bundle."""
from __future__ import annotations

from pathlib import Path

from app import paths
from app.runtime import setup_runtime


def export_matlab(
    smooth_json: str | Path,
    output_folder: str | Path | None = None,
) -> Path:
    setup_runtime()
    from app.postprocess.export_matlab_legacy import export_to_matlab_format

    import json

    smooth_path = Path(smooth_json)
    if not smooth_path.is_absolute():
        smooth_path = paths.REPO_ROOT / smooth_path
    if not smooth_path.exists():
        raise FileNotFoundError(smooth_path)

    data = json.loads(smooth_path.read_text(encoding="utf-8"))
    mesh_entry = data.get("mesh_file", "")
    mesh_candidate = Path(mesh_entry)
    if mesh_entry and not mesh_candidate.is_absolute():
        resolved = (paths.REPO_ROOT / mesh_candidate).resolve()
        if resolved.exists():
            data["mesh_file"] = str(resolved)
            smooth_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    if output_folder is None:
        data = __import__("json").loads(smooth_path.read_text(encoding="utf-8"))
        mesh_file = data.get("mesh_file", "")
        sid = "".join(c for c in Path(mesh_file).stem if c.isdigit()) or "0"
        out = paths.matlab_export_dir(int(sid) if sid else None)
    else:
        out = Path(output_folder)
        if not out.is_absolute():
            out = paths.REPO_ROOT / out

    export_to_matlab_format(str(smooth_path), str(out))
    return out
