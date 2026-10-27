"""Canonical paths. One subject lives under data/subjects/{id}/."""

from __future__ import annotations

import os
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get("LAYOUT_DESIGN_ROOT", _APP_DIR.parent)).resolve()
APP_DIR = REPO_ROOT / "app"
DATA_DIR = REPO_ROOT / "data"
DOCS_DIR = REPO_ROOT / "docs"
CONFIG_DIR = REPO_ROOT / "config"


def _subject_stem(subject_id: int | str) -> str:
    return str(int(subject_id))


def subject_dir(subject_id: int | str) -> Path:
    return DATA_DIR / "subjects" / _subject_stem(subject_id)


def source_dir(subject_id: int | str) -> Path:
    return subject_dir(subject_id) / "source"


def source_ply(subject_id: int | str) -> Path:
    return source_dir(subject_id) / "scan.ply"


def source_obj(subject_id: int | str) -> Path:
    return source_dir(subject_id) / "head.obj"


def source_mtl(subject_id: int | str) -> Path:
    return source_dir(subject_id) / "head.mtl"


def source_jpg(subject_id: int | str) -> Path:
    return source_dir(subject_id) / "head.jpg"


def source_image(subject_id: int | str) -> Path:
    """Texture image next to the source OBJ. Defaults to head.jpg when none is imported yet."""
    folder = source_dir(subject_id)
    for name in ("head.jpg", "head.jpeg", "head.png"):
        path = folder / name
        if path.is_file():
            return path
    return folder / "head.jpg"


def head_stl(subject_id: int | str) -> Path:
    return subject_dir(subject_id) / "head.stl"


def head_obj(subject_id: int | str) -> Path:
    return subject_dir(subject_id) / "head.obj"


def head_mtl(subject_id: int | str) -> Path:
    return subject_dir(subject_id) / "head.mtl"


def head_jpg(subject_id: int | str) -> Path:
    return subject_dir(subject_id) / "head.jpg"


def layout_json(subject_id: int | str) -> Path:
    return subject_dir(subject_id) / "layout.json"


def pm_yaml(subject_id: int | str) -> Path:
    return subject_dir(subject_id) / "pm.yaml"


def electrodes_gcode(subject_id: int | str) -> Path:
    return subject_dir(subject_id) / "electrodes.gcode"


def channel_gcode(subject_id: int | str, name: str) -> Path:
    return subject_dir(subject_id) / f"{name}.gcode"


def machine_config() -> Path:
    return CONFIG_DIR / "machine.yaml"


def mach4_home_bc_gcode() -> Path:
    return REPO_ROOT / "scripts" / "mach4_home_bc.txt"


def mach4_scan_head_gcode() -> Path:
    return REPO_ROOT / "scripts" / "mach4_scan_head.txt"


def defaults_yaml() -> Path:
    return CONFIG_DIR / "defaults.yaml"


def ensure_subject_dir(subject_id: int | str) -> Path:
    root = subject_dir(subject_id)
    (root / "source").mkdir(parents=True, exist_ok=True)
    return root


def ensure_data_tree() -> None:
    (DATA_DIR / "subjects").mkdir(parents=True, exist_ok=True)


def subject_artifacts(subject_id: int | str) -> list[tuple[str, Path]]:
    """Label and path for every file the pipeline reads or writes."""
    return [
        ("source ply", source_ply(subject_id)),
        ("source obj", source_obj(subject_id)),
        ("source mtl", source_mtl(subject_id)),
        ("source jpg", source_image(subject_id)),
        ("head stl", head_stl(subject_id)),
        ("head obj", head_obj(subject_id)),
        ("layout", layout_json(subject_id)),
        ("pm", pm_yaml(subject_id)),
        ("gcode", electrodes_gcode(subject_id)),
    ]
