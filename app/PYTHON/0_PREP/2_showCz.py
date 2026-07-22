"""Compute Cz from cleaned mesh + fiducials (automated, no GUI)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pyvista as pv

SUBJECT_ID = int(os.environ.get("LAYOUT_SUBJECT_ID", "1"))


def slice_loop(mesh: pv.PolyData, P, Q, R):
    n = np.cross(Q - P, R - P)
    if float(np.linalg.norm(n)) < 1e-12:
        raise ValueError(
            "Degenerate fiducial geometry for Cz slice "
            "(nasion/inion/LPA/RPA nearly colinear with crown)."
        )
    return mesh.slice(normal=n, origin=P)


def main(SUBJECT_ID: int) -> int:
    cleaned = Path(f"data/cleaned_scans/{SUBJECT_ID}.stl")
    fid_path = Path(f"data/json/fiducials_{SUBJECT_ID}.json")
    if not cleaned.is_file():
        raise FileNotFoundError(
            f"Missing cleaned mesh: {cleaned}\n"
            f"Run clear-islands first (writes data/cleaned_scans/{SUBJECT_ID}.stl)."
        )
    if not fid_path.is_file():
        raise FileNotFoundError(f"Missing fiducials: {fid_path}")

    mesh = pv.read(str(cleaned))
    with fid_path.open(encoding="utf-8") as f:
        fid = json.load(f)

    for key in ("nasion", "lpa", "rpa", "inion"):
        if key not in fid:
            raise ValueError(f"Fiducials missing {key!r} in {fid_path}")

    pts = mesh.points
    crown_idx = int(np.argmax(pts[:, 2]))
    crown = pts[crown_idx]
    print("Auto-detected crown at", crown.tolist())

    loop_fb = slice_loop(
        mesh,
        np.array(fid["nasion"], dtype=float),
        np.array(fid["inion"], dtype=float),
        crown,
    )
    loop_lr = slice_loop(
        mesh,
        np.array(fid["lpa"], dtype=float),
        np.array(fid["rpa"], dtype=float),
        crown,
    )
    if loop_fb.n_points < 1 or loop_lr.n_points < 1:
        raise ValueError("Cz slice loops are empty — check mesh / fiducials.")

    a = loop_fb.points
    b = loop_lr.points
    d2 = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2)
    i, j = np.unravel_index(np.argmin(d2), d2.shape)
    cz = 0.5 * (a[i] + b[j])
    print("Cz = ", cz.tolist())

    out = Path(f"data/json/Cz_{SUBJECT_ID}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump({"Cz": cz.tolist()}, f, indent=2)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(SUBJECT_ID=SUBJECT_ID))
