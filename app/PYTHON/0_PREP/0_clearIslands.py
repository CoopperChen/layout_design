"""Remove small mesh islands from reconstructed STL."""

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

    p_before = pv.Plotter(window_size=(800, 600), title="Before Cleaning")
    p_before.add_mesh(original_mesh, color="lightgray", opacity=1.0)
    p_before.add_text(
        "Original mesh — close window (or Space/Enter) to continue",
        position="upper_edge",
    )
    p_before.show_bounds()

    def _close_before() -> None:
        p_before.close()

    p_before.add_key_event("space", _close_before)
    p_before.add_key_event("Return", _close_before)
    print(
        "\nClear-islands (1/2): review BEFORE window.\n"
        "  Close window or Space/Enter to continue.\n"
    )
    p_before.show()

    cleaned_mesh = remove_islands(
        original_mesh,
        min_size_ratio=0.02,
        min_absolute_size=200,
    )

    p_after = pv.Plotter(window_size=(800, 600), title="After Cleaning")
    p_after.add_mesh(cleaned_mesh, color="lightgray", opacity=1.0)
    p_after.add_text(
        "Cleaned mesh — close / Space / Enter / S = SAVE · Q = discard",
        position="upper_edge",
    )
    p_after.show_bounds()

    state = {"save": True}

    def _confirm_save() -> None:
        state["save"] = True
        p_after.close()

    def _discard() -> None:
        state["save"] = False
        p_after.close()

    p_after.add_key_event("space", _confirm_save)
    p_after.add_key_event("Return", _confirm_save)
    p_after.add_key_event("s", _confirm_save)
    p_after.add_key_event("q", _discard)

    print(
        "\nClear-islands (2/2): review AFTER window.\n"
        "  Close window / Space / Enter / S → SAVE cleaned mesh\n"
        "  Q → discard (pipeline cannot continue without cleaned STL)\n"
    )
    p_after.show()

    if not state["save"]:
        print("Cleaned mesh NOT saved (Q). Need data/cleaned_scans/ for later stages.")
        return 1

    output_path = f"data/cleaned_scans/{SUBJECT_ID}.stl"
    save_as_stl(cleaned_mesh, output_path)
    print(f"Cleaned mesh saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(SUBJECT_ID=SUBJECT_ID))
