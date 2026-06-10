"""Load legacy MATLAB .mat subject bundle."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from ..models import SubjectBundle, TraceChannel

DEFAULT_LANDMARK_NAMES = ["Landmark(central)", "Landmark(left)", "Landmark(back)"]


def _unwrap(item):
    while isinstance(item, np.ndarray) and item.shape == (1, 1):
        item = item[0, 0]
    return item


def _as_str(item) -> str:
    item = _unwrap(item)
    if isinstance(item, np.ndarray):
        flat = item.flat[0]
        if isinstance(flat, bytes):
            return flat.decode("utf-8")
        return str(flat)
    return str(item)


def _as_array(item) -> np.ndarray:
    return np.asarray(_unwrap(item), dtype=float)


def _load_landmark_names(names_mat: dict) -> list[str]:
    if "LandmarkNames" in names_mat:
        raw = names_mat["LandmarkNames"]
    else:
        # Some exports save under 'None' or MATLAB opaque keys
        keys = [k for k in names_mat if not k.startswith("__")]
        if not keys:
            return DEFAULT_LANDMARK_NAMES.copy()
        raw = names_mat[keys[0]]

    if isinstance(raw, np.ndarray):
        if raw.dtype == object:
            return [_as_str(x) for x in raw.ravel()]
        return [str(x) for x in raw.ravel()]
    return [_as_str(raw)]


def _parse_interconnect_electrode_paths(alldata: np.ndarray) -> list[TraceChannel]:
    """
    Support AdjPoints Nx3 layout and layout_design export 3x1 nested cells.
    """
    channels: list[TraceChannel] = []

    if alldata.shape == (3, 1):
        ic_cell = alldata[0, 0]
        el_cell = alldata[1, 0]
        nm_cell = alldata[2, 0]
        n_elect = ic_cell.shape[0]
        for i in range(n_elect):
            interconnect = _as_array(ic_cell[i, 0])
            electrode = _as_array(el_cell[i, 0])
            name = _as_str(nm_cell[i, 0])
            channels.append(
                TraceChannel(name=name, interconnect=interconnect, electrode=electrode)
            )
        return channels

    n_elect = alldata.shape[0]
    for i in range(n_elect):
        interconnect = _as_array(alldata[i, 0])
        electrode = _as_array(alldata[i, 1])
        name = _as_str(alldata[i, 2])
        channels.append(
            TraceChannel(name=name, interconnect=interconnect, electrode=electrode)
        )
    return channels


def _load_head_mesh(mesh_mat: dict) -> tuple[np.ndarray, np.ndarray]:
    if "dataref" in mesh_mat:
        dataref = mesh_mat["dataref"]
    else:
        keys = [k for k in mesh_mat if not k.startswith("__")]
        if not keys:
            raise KeyError("HeadMesh.mat missing dataref")
        dataref = mesh_mat[keys[0]]

    # mat_struct (v7 re-export) or plain struct ndarray
    if hasattr(dataref, "Points"):
        points = np.asarray(dataref.Points, dtype=float)
        faces = np.asarray(dataref.ConnectivityList, dtype=int) - 1
        return points, faces

    if isinstance(dataref, np.ndarray) and dataref.dtype.names:
        points = np.asarray(dataref["Points"], dtype=float)
        faces = np.asarray(dataref["ConnectivityList"], dtype=int) - 1
        if points.ndim == 3:
            points = points[0]
        if faces.ndim == 3:
            faces = faces[0]
        return points, faces

    raise TypeError(f"Unsupported HeadMesh dataref type: {type(dataref)}")


def load_mat_subject(subject_dir: Path | str) -> SubjectBundle:
    subject_dir = Path(subject_dir)
    paths_mat = loadmat(
        subject_dir / "InterconnectElectrodePaths.mat",
        squeeze_me=False,
        struct_as_record=False,
    )
    alldata = paths_mat["InterconnectElectrodePaths"]

    landmarks_mat = loadmat(subject_dir / "Landmarks.mat", squeeze_me=True)
    ps = np.asarray(landmarks_mat["Landmarks"], dtype=float)

    names_mat = loadmat(subject_dir / "LandmarkNames.mat", squeeze_me=True, struct_as_record=False)
    landmark_names = _load_landmark_names(names_mat)

    mesh_mat = loadmat(
        subject_dir / "HeadMesh.mat",
        squeeze_me=True,
        struct_as_record=False,
    )
    points, faces = _load_head_mesh(mesh_mat)

    channels = _parse_interconnect_electrode_paths(alldata)

    return SubjectBundle(
        schema_version="legacy_mat",
        subject_id=subject_dir.name,
        mesh_points=points,
        mesh_faces=faces,
        landmarks_xyz=ps,
        landmark_names=landmark_names,
        channels=channels,
    )
