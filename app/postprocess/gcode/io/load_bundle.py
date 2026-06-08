"""Load eeg_subject_bundle/1.0.0 (manifest.json + NPZ)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..models import SCHEMA_VERSION, SubjectBundle, TraceChannel


def _load_npz_arrays(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def load_bundle(bundle_dir: Path | str) -> SubjectBundle:
    bundle_dir = Path(bundle_dir)
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.json in {bundle_dir}")

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    schema = manifest.get("schema_version", "")
    if schema != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema: {schema} (expected {SCHEMA_VERSION})")

    geometry = _load_npz_arrays(bundle_dir / "geometry.npz")
    traces = _load_npz_arrays(bundle_dir / "traces.npz")

    channel_names = traces["channel_names"]
    if hasattr(channel_names, "tolist"):
        channel_names = [str(x) for x in channel_names.tolist()]
    else:
        channel_names = [str(x) for x in channel_names]

    interconnects = traces["interconnect_xyzn"]
    electrodes = traces["electrode_xyzn"]
    terminals = traces.get("terminals", [""] * len(channel_names))

    channels: list[TraceChannel] = []
    for i, name in enumerate(channel_names):
        ic = np.asarray(interconnects[i], dtype=float)
        el = np.asarray(electrodes[i], dtype=float)
        term = str(terminals[i]) if i < len(terminals) else ""
        channels.append(
            TraceChannel(name=name, interconnect=ic, electrode=el, terminal=term)
        )

    landmark_names = manifest.get("landmarks", {}).get("calibration", {}).get(
        "names",
        ["Landmark(central)", "Landmark(left)", "Landmark(back)"],
    )

    return SubjectBundle(
        schema_version=schema,
        subject_id=manifest.get("subject_id", bundle_dir.name),
        mesh_points=np.asarray(geometry["mesh_points"], dtype=float),
        mesh_faces=np.asarray(geometry["mesh_faces"], dtype=int),
        landmarks_xyz=np.asarray(geometry["landmarks_xyz"], dtype=float),
        landmark_names=list(landmark_names),
        channels=channels,
        anatomical_xyz=(
            np.asarray(geometry["anatomical_xyz"], dtype=float)
            if "anatomical_xyz" in geometry
            else None
        ),
        sources=manifest.get("sources", {}),
    )
