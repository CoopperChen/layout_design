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
    return cleaned.extract_surface().triangulate()


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
