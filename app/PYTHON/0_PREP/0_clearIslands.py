"""Remove small mesh islands from reconstructed STL (automated, no GUI)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pyvista as pv

SUBJECT_ID = int(os.environ.get("LAYOUT_SUBJECT_ID", "1"))


def remove_islands(mesh, min_size_ratio=0.01, min_absolute_size=100):
    """
    Remove disconnected components smaller than thresholds.

    Parameters:
        mesh: Input mesh
        min_size_ratio: Minimum relative size (ratio to largest component)
        min_absolute_size: Minimum absolute face count to keep
    """
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

    # Keep only selected region IDs (not the contiguous ID range).
    keep = np.isin(regions, labels_to_keep)
    cleaned = connected.extract_cells(keep)
    if cleaned.n_cells == 0:
        return mesh
    return cleaned.extract_surface().triangulate()


def save_as_stl(mesh, filename, binary=True):
    """Properly save mesh as STL file."""
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface(algorithm="dataset_surface").triangulate()
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    mesh.save(filename, binary=binary)


def main(SUBJECT_ID: int) -> int:
    file_path = f"data/raw/{SUBJECT_ID}.stl"
    print(f"Loading mesh from: {file_path}")
    if not Path(file_path).is_file():
        raise FileNotFoundError(file_path)
    original_mesh = pv.read(file_path)
    n_before = int(original_mesh.n_cells)

    cleaned_mesh = remove_islands(
        original_mesh,
        min_size_ratio=0.02,
        min_absolute_size=200,
    )
    n_after = int(cleaned_mesh.n_cells)

    output_path = f"data/cleaned_scans/{SUBJECT_ID}.stl"
    save_as_stl(cleaned_mesh, output_path)
    print(
        f"Clear-islands: {n_before} → {n_after} cells; "
        f"saved {output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(SUBJECT_ID=SUBJECT_ID))
