"""Shared fiducial I/O for preprocess and MATLAB export.

Fiducials are picked on the textured OBJ; 3D coordinates are stored in JSON
and used with the paired STL mesh for all other pipeline steps.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from app import paths

# Interactive pick order (keys in fiducials_{id}.json)
PICK_SEQUENCE: list[tuple[str, str]] = [
    ("nasion", "1) Nasion — bridge of nose"),
    ("lpa", "2) LPA — left pre-auricular point"),
    ("rpa", "3) RPA — right pre-auricular point"),
    ("inion", "4) Inion — bump at back of head"),
    ("TERMINAL_RIGHT", "5) Right terminal"),
    ("TERMINAL_LEFT", "6) Left terminal"),
    ("landmark_central", "7) Landmark (central)"),
    ("landmark_left", "8) Landmark (left)"),
    ("landmark_back", "9) Landmark (back)"),
]

MATLAB_LANDMARK_KEYS = ("landmark_central", "landmark_left", "landmark_back")
MATLAB_LANDMARK_NAMES = (
    "Landmark(central)",
    "Landmark(left)",
    "Landmark(back)",
)

PICK_COLORS = [
    "orange",
    "green",
    "blue",
    "yellow",
    "black",
    "red",
    "cyan",
    "magenta",
    "white",
]


def fiducials_path(subject_id: int | str) -> Path:
    return paths.fiducials_json(subject_id)


def load_picks(subject_id: int | str) -> dict[str, list[float]]:
    path = fiducials_path(subject_id)
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: [float(v) for v in vals] for k, vals in data.items()}


def save_picks(subject_id: int | str, picks: dict[str, Any]) -> Path:
    path = fiducials_path(subject_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {k: [float(c) for c in pt] for k, pt in picks.items()}
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    return path


def matlab_landmarks_from_picks(
    picks: dict[str, Any],
) -> tuple[np.ndarray, list[str]] | None:
    if not all(k in picks for k in MATLAB_LANDMARK_KEYS):
        return None
    rows = [[float(c) for c in picks[k]] for k in MATLAB_LANDMARK_KEYS]
    return np.asarray(rows, dtype=float), list(MATLAB_LANDMARK_NAMES)


def save_landmarks_mat(subject_id: int | str, picks: dict[str, Any]) -> Path | None:
    """Write Landmarks.mat + LandmarkNames.mat for legacy MATLAB postprocessor."""
    from scipy.io import savemat

    parsed = matlab_landmarks_from_picks(picks)
    if parsed is None:
        return None
    landmarks, names = parsed
    out_dir = paths.matlab_export_dir(subject_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    savemat(str(out_dir / "Landmarks.mat"), {"Landmarks": landmarks}, format="5")
    savemat(
        str(out_dir / "LandmarkNames.mat"),
        {"LandmarkNames": np.array(names, dtype=object)},
        format="5",
    )
    return out_dir


def load_head_mesh(mesh_path: Path):
    """Load STL/OBJ head mesh (OBJ via Open3D — VTK-safe, keeps vertex colors)."""
    from app.preprocess.mesh_io import load_head_mesh as _load

    return _load(mesh_path)
