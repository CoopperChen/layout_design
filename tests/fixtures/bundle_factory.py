"""Canonical minimal eeg_subject_bundle/1.0.0 for contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.postprocess.bundle.schema import SCHEMA_VERSION

SYNTHETIC_SUBJECT_ID = 99


def write_synthetic_fiducials(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "landmark_central": [10.0, 20.0, 50.0],
                "landmark_left": [40.0, 20.0, 50.0],
                "landmark_back": [10.0, 50.0, 50.0],
                "TERMINAL_LEFT": [5.0, 45.0, 48.0],
                "TERMINAL_RIGHT": [45.0, 45.0, 48.0],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def write_synthetic_mesh(path: Path) -> None:
    import pyvista as pv

    box = pv.Box(bounds=(0, 50, 0, 50, 40, 60))
    box.save(str(path))


def write_synthetic_smooth(path: Path, mesh_name: str = "99.stl") -> None:
    payload = {
        "mesh_file": mesh_name,
        "electrode_positions": {
            "C3": [15.0, 25.0, 55.0],
            "C4": [35.0, 25.0, 55.0],
        },
        "terminal_positions": {
            "TERMINAL_LEFT": [5.0, 45.0, 48.0],
            "TERMINAL_RIGHT": [45.0, 45.0, 48.0],
        },
        "collision_metrics": {
            "layout_collision_free": True,
            "crossing_count": 0,
            "electrode_violations": 0,
        },
        "final_paths": [
            {
                "electrode": "C3",
                "terminal": "TERMINAL_LEFT",
                "path_3d": [
                    [15.0, 25.0, 55.0],
                    [12.0, 30.0, 53.0],
                    [8.0, 38.0, 51.0],
                ],
            },
            {
                "electrode": "C4",
                "terminal": "TERMINAL_RIGHT",
                "path_3d": [
                    [35.0, 25.0, 55.0],
                    [38.0, 30.0, 53.0],
                    [42.0, 38.0, 51.0],
                ],
            },
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_golden_bundle(bundle_dir: Path) -> None:
    """Write a tiny valid bundle (no mesh export step required)."""
    bundle_dir.mkdir(parents=True, exist_ok=True)

    mesh_points = np.array(
        [
            [0, 0, 0],
            [100, 0, 0],
            [0, 100, 0],
            [0, 0, 100],
        ],
        dtype=float,
    )
    mesh_faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
    landmarks = np.array(
        [
            [10.0, 20.0, 30.0],
            [40.0, 20.0, 30.0],
            [10.0, 50.0, 30.0],
        ],
        dtype=float,
    )
    interconnect = np.array(
        [
            [15, 25, 35, 0, 0, 1],
            [20, 25, 36, 0, 0, 1],
            [25, 25, 37, 0, 0, 1],
        ],
        dtype=float,
    )
    electrode = np.array(
        [
            [15, 25, 35, 0, 0, 1],
            [16, 26, 35, 0, 0, 1],
            [15, 27, 35, 0, 0, 1],
        ],
        dtype=float,
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "subject_id": "synthetic",
        "coordinate_frame": {
            "space": "head_scan",
            "units": "mm",
            "origin": "landmark_index_0",
        },
        "landmarks": {
            "calibration": {
                "names": ["Landmark(central)", "Landmark(left)", "Landmark(back)"],
            }
        },
        "channel_count": 1,
        "sources": {},
    }
    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        bundle_dir / "geometry.npz",
        mesh_points=mesh_points,
        mesh_faces=mesh_faces,
        landmarks_xyz=landmarks,
    )
    np.savez_compressed(
        bundle_dir / "traces.npz",
        channel_names=np.array(["C3"], dtype=object),
        terminals=np.array(["TERMINAL_LEFT"], dtype=object),
        interconnect_xyzn=np.array([interconnect], dtype=object),
        electrode_xyzn=np.array([electrode], dtype=object),
    )
