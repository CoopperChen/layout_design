"""Compute Cz from cleaned mesh + fiducials; preview and save."""

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
            f"Run clear-islands and SAVE the cleaned mesh first "
            f"(close the AFTER window, or Space/Enter/S)."
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

    pl = pv.Plotter(window_size=(800, 600))
    pl.add_mesh(mesh, color="cyan", opacity=0.3)
    pl.add_mesh(loop_fb, color="red", line_width=3)
    pl.add_mesh(loop_lr, color="green", line_width=3)
    pl.add_mesh(
        pv.Sphere(center=cz, radius=mesh.length * 0.01),
        color="yellow",
    )
    pl.add_text(
        "Cz preview — Space / Enter / S / close = SAVE · Q = discard",
        font_size=12,
    )

    state = {"save": True}

    def _confirm_save() -> None:
        state["save"] = True
        pl.close()

    def _discard() -> None:
        state["save"] = False
        pl.close()

    pl.add_key_event("space", _confirm_save)
    pl.add_key_event("Return", _confirm_save)
    pl.add_key_event("s", _confirm_save)
    pl.add_key_event("q", _discard)
    print(
        "\nCz review:\n"
        "  Space / Enter / S (or close) = SAVE Cz\n"
        "  Q = discard (pipeline needs Cz_{id}.json)\n"
    )
    pl.show()

    if not state["save"]:
        print("Cz NOT saved (Q).")
        return 1

    out = Path(f"data/json/Cz_{SUBJECT_ID}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump({"Cz": cz.tolist()}, f, indent=2)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(SUBJECT_ID=SUBJECT_ID))
