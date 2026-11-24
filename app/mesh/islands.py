"""Drop small disconnected pieces from a reconstructed head mesh."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv


def remove_islands(mesh, min_size_ratio: float = 0.02, min_absolute_size: int = 200):
    connected = mesh.connectivity(largest=False)
    regions = connected.cell_data["RegionId"]
    unique_labels, counts = np.unique(regions, return_counts=True)
    if len(unique_labels) <= 1:
        return mesh
    max_size = counts.max()
    size_threshold = max(max_size * min_size_ratio, min_absolute_size)
    labels_to_keep = unique_labels[counts >= size_threshold]
    if len(labels_to_keep) == 0:
        return mesh
    keep = np.isin(regions, labels_to_keep)
    cleaned = connected.extract_cells(keep)
    if cleaned.n_cells == 0:
        return mesh
    return _as_polydata(cleaned)


def drop_specks(mesh, *, max_cells: int = 50, max_diagonal_mm: float = 8.0):
    """Drop tiny floating fragments, such as a one-triangle speck above the scalp.

    A fragment is removed only when it has few faces and a short bounding diagonal.
    Larger shells stay, including UV charts that do not share vertices with the rest
    of the scan. ``Cz`` is the highest face vertex, so a speck a few millimetres
    above the crown pulls the whole 10–20 set to one side.
    """
    if int(getattr(mesh, "n_cells", 0)) < 2:
        return mesh
    surface = mesh if bool(getattr(mesh, "is_all_triangles", False)) else mesh.triangulate()
    connected = surface.connectivity(largest=False)
    regions = np.asarray(connected.cell_data["RegionId"])
    labels, counts = np.unique(regions, return_counts=True)
    if len(labels) <= 1:
        return mesh
    small = labels[counts <= max_cells]
    if len(small) == 0:
        return mesh
    faces = np.asarray(connected.faces, dtype=np.int64).reshape(-1, 4)[:, 1:]
    pts = np.asarray(connected.points, dtype=float)
    drop: list[int] = []
    for label in small:
        ids = np.unique(faces[regions == label])
        diagonal = float(np.linalg.norm(pts[ids].max(axis=0) - pts[ids].min(axis=0)))
        if diagonal <= max_diagonal_mm:
            drop.append(int(label))
    if not drop:
        return mesh
    keep = ~np.isin(regions, np.asarray(drop, dtype=regions.dtype))
    if not np.any(keep):
        return mesh
    cleaned = _as_polydata(connected.extract_cells(np.flatnonzero(keep)))
    texture = getattr(mesh, "texture", None)
    if texture is not None and getattr(cleaned, "texture", None) is None:
        pv.set_new_attribute(cleaned, "texture", texture)
    print(
        f"Ignoring {int((~keep).sum())} speck faces in {len(drop)} tiny islands; "
        f"surface has {int(cleaned.n_cells)} faces"
    )
    return cleaned


def _as_polydata(mesh):
    if isinstance(mesh, pv.PolyData):
        return mesh.triangulate()
    return mesh.extract_surface(algorithm="dataset_surface").triangulate()


def clear_islands(
    stl_path: Path,
    *,
    min_size_ratio: float = 0.02,
    min_absolute_size: int = 200,
) -> Path:
    path = Path(stl_path)
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"STL missing or empty: {path}")
    mesh = pv.read(str(path))
    if isinstance(mesh, pv.MultiBlock):
        mesh = mesh.combine()
    if mesh.n_points == 0 or mesh.n_cells == 0:
        raise ValueError(f"STL has no geometry: {path}")
    n_before = int(mesh.n_cells)
    cleaned = remove_islands(
        mesh,
        min_size_ratio=min_size_ratio,
        min_absolute_size=min_absolute_size,
    )
    if not isinstance(cleaned, pv.PolyData):
        cleaned = cleaned.extract_surface(algorithm="dataset_surface").triangulate()
    cleaned.save(str(path), binary=True)
    print(f"Clear-islands: {n_before} → {int(cleaned.n_cells)} cells; saved {path}")
    return path
