"""Load eeg_subject_bundle/1.0.0 for tests and downstream tools."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.postprocess.bundle.models import SubjectBundle, TraceChannel
from app.postprocess.bundle.schema import SCHEMA_VERSION


def load_bundle(bundle_dir: Path | str) -> SubjectBundle:
    bundle_dir = Path(bundle_dir)
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No manifest.json in {bundle_dir}")

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    schema = manifest.get("schema_version", "")
    if schema != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema: {schema} (expected {SCHEMA_VERSION})")

    with np.load(bundle_dir / "geometry.npz", allow_pickle=True) as geom:
        mesh_points = np.asarray(geom["mesh_points"], dtype=float)
        mesh_faces = np.asarray(geom["mesh_faces"], dtype=int)
        landmarks_xyz = np.asarray(geom["landmarks_xyz"], dtype=float)
        anatomical = (
            np.asarray(geom["anatomical_xyz"], dtype=float)
            if "anatomical_xyz" in geom
            else None
        )

    with np.load(bundle_dir / "traces.npz", allow_pickle=True) as traces:
        names = [str(x) for x in traces["channel_names"].tolist()]
        terminals = traces.get("terminals", [""] * len(names))
        interconnects = traces["interconnect_xyzn"]
        electrodes = traces["electrode_xyzn"]

    landmark_names = manifest.get("landmarks", {}).get("calibration", {}).get(
        "names",
        ["Landmark(central)", "Landmark(left)", "Landmark(back)"],
    )

    channels: list[TraceChannel] = []
    for i, name in enumerate(names):
        channels.append(
            TraceChannel(
                name=name,
                interconnect=np.asarray(interconnects[i], dtype=float),
                electrode=np.asarray(electrodes[i], dtype=float),
                terminal=str(terminals[i]) if i < len(terminals) else "",
            )
        )

    return SubjectBundle(
        schema_version=schema,
        subject_id=manifest.get("subject_id", bundle_dir.name),
        mesh_points=mesh_points,
        mesh_faces=mesh_faces,
        landmarks_xyz=landmarks_xyz,
        landmark_names=list(landmark_names),
        channels=channels,
        anatomical_xyz=anatomical,
        sources=manifest.get("sources", {}),
    )
