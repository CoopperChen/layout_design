"""
Canonical filesystem paths for the layout_design pipeline.

All stages resolve paths through this module so scripts stay cwd-independent
when run from the repository root or from app/.

Working directory convention:
  - Preferred: repository root (layout_design/)
  - Legacy compat: app/ (set APP_ROOT to parent automatically)
"""
from __future__ import annotations

import os
from pathlib import Path

# Repository root: parent of app/ if this file lives in app/paths.py
_APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get("LAYOUT_DESIGN_ROOT", _APP_DIR.parent)).resolve()
APP_DIR = REPO_ROOT / "app"
DATA_DIR = REPO_ROOT / "data"
DOCS_DIR = REPO_ROOT / "docs"
LEGACY_GCODE_DIR = REPO_ROOT / "legacy_gcode_examples"
CONFIG_DIR = REPO_ROOT / "config"


def _subject_stem(subject_id: int | str) -> str:
    return str(int(subject_id))


# --- A: Preprocess inputs / outputs ---

def raw_scan(subject_id: int | str, ext: str = "stl") -> Path:
    return DATA_DIR / "raw" / f"{_subject_stem(subject_id)}.{ext}"


def raw_point_cloud(subject_id: int | str) -> Path:
    """Input scan PLY for Poisson reconstruction (Stage A, reconstruct step)."""
    return DATA_DIR / "raw" / f"{_subject_stem(subject_id)}.ply"


def cleaned_scan(subject_id: int | str, ext: str = "stl") -> Path:
    return DATA_DIR / "cleaned_scans" / f"{_subject_stem(subject_id)}.{ext}"


def textured_head_obj(subject_id: int | str) -> Path:
    """
    Textured OBJ for interactive fiducial picking only.

    Same geometry as ``cleaned_scan(subject_id)`` (.stl); layout, geodesics,
    smooth, and MATLAB mesh export all use the STL.
    """
    sid = _subject_stem(subject_id)
    candidates = [
        DATA_DIR / "raw" / f"{sid}.obj",
        DATA_DIR / "cleaned_scans" / f"{sid}.obj",
    ]
    for path in candidates:
        if path.is_file():
            return path
    stl = cleaned_scan(sid)
    raise FileNotFoundError(
        f"No textured OBJ for subject {sid}. Place {sid}.obj in data/raw/ "
        f"(or data/cleaned_scans/) alongside {stl.name} — same geometry, "
        f"OBJ for fiducial picking, STL for all other steps."
    )


def head_mesh_for_fiducials(subject_id: int | str) -> Path:
    """Alias for :func:`textured_head_obj`."""
    return textured_head_obj(subject_id)


def fiducials_json(subject_id: int | str) -> Path:
    return DATA_DIR / "json" / f"fiducials_{_subject_stem(subject_id)}.json"


def cz_json(subject_id: int | str) -> Path:
    return DATA_DIR / "json" / f"Cz_{_subject_stem(subject_id)}.json"


def electrode_positions_json(subject_id: int | str) -> Path:
    return DATA_DIR / "json" / f"electrode_positions_{_subject_stem(subject_id)}.json"


def terminal_assignments_json(subject_id: int | str) -> Path:
    return DATA_DIR / "json" / f"initial_terminal_assignments_{_subject_stem(subject_id)}.json"


def init_connection_paths_json(subject_id: int | str) -> Path:
    """Geodesic seeds; created for assignment/polish, not required for synthesize-only."""
    return DATA_DIR / "json" / f"init_connection_paths_{_subject_stem(subject_id)}.json"


# --- B: Presets & synthesized layouts ---

def preset_path(name: str) -> Path:
    """Name with or without .json."""
    p = Path(name)
    if p.suffix.lower() != ".json":
        p = p.with_suffix(".json")
    if p.is_absolute():
        return p
    return DATA_DIR / "presets" / p.name


def synth_layout(subject_id: int | str, tag: str = "synth") -> Path:
    return layout_json(subject_id, tag)


def layout_json(subject_id: int | str, tag: str) -> Path:
    sid = _subject_stem(subject_id)
    return DATA_DIR / "output" / "layouts" / f"{tag}_s{sid}.json"


def layout_pic(subject_id: int | str, tag: str, suffix: str = "2d") -> Path:
    sid = _subject_stem(subject_id)
    return DATA_DIR / "output" / "pics" / f"{tag}_s{sid}_{suffix}.png"


# --- C: Polish logs (optional short GA) ---

def polish_log_dir(subject_id: int | str) -> Path:
    return DATA_DIR / "output" / "logs" / f"subject_{_subject_stem(subject_id)}"


# --- D: Postprocess ---

def smooth_json(subject_id: int | str, tag: str = "final") -> Path:
    sid = _subject_stem(subject_id)
    return DATA_DIR / "output" / "smooth" / f"smooth_s{sid}_{tag}.json"


def matlab_export_dir(subject_id: int | str | None = None) -> Path:
    if subject_id is None:
        return DATA_DIR / "output" / "matlab" / "subject_optimized"
    return DATA_DIR / "output" / "matlab" / f"subject_{_subject_stem(subject_id)}"


def bundle_export_dir(subject_id: int | str) -> Path:
    """Canonical eeg_subject_bundle/1.0.0 output for Postprocessor."""
    return DATA_DIR / "output" / "bundles" / f"subject_{_subject_stem(subject_id)}"


def postprocessor_config_dir() -> Path:
    return CONFIG_DIR / "postprocessor"


def postprocessor_machine_config() -> Path:
    return postprocessor_config_dir() / "machine_default.yaml"


def postprocessor_subject_pm(subject_id: int | str) -> Path:
    """Per-subject physical landmarks (pm) for G-code registration."""
    sid = str(subject_id)
    stem = _subject_stem(subject_id) if sid.isdigit() else sid
    return postprocessor_config_dir() / "subjects" / f"subject_{stem}.yaml"


def gcode_output_dir(subject_id: int | str | None = None) -> Path:
    base = DATA_DIR / "output" / "gcode"
    if subject_id is None:
        return base
    return base / f"subject_{_subject_stem(subject_id)}_post"


# --- Legacy / reference archives ---

def archive_run(run_id: str) -> Path:
    return DATA_DIR / "archive" / run_id


def split_concatenated_paths(raw: str | Path) -> list[str]:
    """
    Split accidental glued paths (common on Windows copy/paste).

    Example: ``a.jsonD:\\b.json`` → ``[a.json, D:\\b.json]``.
    """
    s = str(raw).strip().strip('"')
    if ".json" not in s.lower():
        return [s]
    parts: list[str] = []
    i = 0
    lower = s.lower()
    while i < len(s):
        j = lower.find(".json", i)
        if j < 0:
            tail = s[i:].strip()
            if tail:
                parts.append(tail)
            break
        parts.append(s[i : j + 5])
        i = j + 5
    return parts


def resolve_json_path(
    raw: str | Path,
    *,
    must_exist: bool = True,
    role: str = "JSON path",
) -> Path:
    """Resolve repo-relative path; recover from concatenated multi-path strings."""
    last_err: Exception | None = None
    for part in split_concatenated_paths(raw):
        part = part.strip().strip('"')
        if not part:
            continue
        p = Path(part)
        if not p.is_absolute():
            p = REPO_ROOT / p
        try:
            p = p.resolve()
        except OSError as e:
            last_err = e
            continue
        if must_exist and not p.is_file():
            continue
        return p
    hint = ""
    if ".json" in str(raw).lower() and len(split_concatenated_paths(raw)) > 1:
        hint = " (looks like two paths were pasted without a space — quote each path separately)"
    msg = f"{role} not found: {raw}{hint}"
    if last_err is not None:
        raise FileNotFoundError(msg) from last_err
    raise FileNotFoundError(msg)


def ensure_data_tree() -> None:
    """Create standard data directories if missing."""
    dirs = [
        DATA_DIR / "raw",
        DATA_DIR / "cleaned_scans",
        DATA_DIR / "json",
        DATA_DIR / "presets",
        DATA_DIR / "output" / "layouts",
        DATA_DIR / "output" / "pics",
        DATA_DIR / "output" / "smooth",
        DATA_DIR / "output" / "matlab",
        DATA_DIR / "output" / "bundles",
        DATA_DIR / "output" / "gcode",
        CONFIG_DIR / "postprocessor" / "subjects",
        DATA_DIR / "output" / "logs",
        DATA_DIR / "archive",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
