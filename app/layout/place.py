"""Cz and standard 10–20 sites from four anatomical landmarks."""

from __future__ import annotations

import numpy as np
import pyvista as pv

from app.layout.template import CHANNELS, template_xyz


def compute_cz(mesh: pv.PolyData, anatomy: dict[str, np.ndarray]) -> np.ndarray:
    pts = np.asarray(mesh.points, dtype=float)
    if pts.shape[0] == 0:
        raise ValueError("Head mesh has no points.")
    crown = pts[int(np.argmax(pts[:, 2]))]
    nasion = np.asarray(anatomy["nasion"], dtype=float)
    inion = np.asarray(anatomy["inion"], dtype=float)
    lpa = np.asarray(anatomy["lpa"], dtype=float)
    rpa = np.asarray(anatomy["rpa"], dtype=float)

    def slice_loop(p, q, r):
        normal = np.cross(q - p, r - p)
        if float(np.linalg.norm(normal)) < 1e-12:
            raise ValueError("Fiducials are degenerate; re-pick nasion, inion, LPA, and RPA.")
        return mesh.slice(normal=normal, origin=p)

    loop_fb = slice_loop(nasion, inion, crown)
    loop_lr = slice_loop(lpa, rpa, crown)
    if loop_fb.n_points < 1 or loop_lr.n_points < 1:
        raise ValueError("Cz slice is empty. Check that the head mesh is upright.")
    a = np.asarray(loop_fb.points, dtype=float)
    b = np.asarray(loop_lr.points, dtype=float)
    d2 = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2)
    i, j = np.unravel_index(int(np.argmin(d2)), d2.shape)
    return 0.5 * (a[i] + b[j])


def _head_axes(anatomy: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ap_axis = np.asarray(anatomy["nasion"], dtype=float) - np.asarray(anatomy["inion"], dtype=float)
    ap_len = float(np.linalg.norm(ap_axis))
    if ap_len < 1e-6:
        raise ValueError("Nasion and inion are coincident.")
    ap_axis = ap_axis / ap_len
    lr_vector = np.asarray(anatomy["rpa"], dtype=float) - np.asarray(anatomy["lpa"], dtype=float)
    si_axis = np.cross(lr_vector, ap_axis)
    si_len = float(np.linalg.norm(si_axis))
    if si_len < 1e-6:
        fallback = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(fallback, ap_axis))) > 0.9:
            fallback = np.array([0.0, 1.0, 0.0])
        si_axis = np.cross(ap_axis, fallback)
        si_len = float(np.linalg.norm(si_axis))
        if si_len < 1e-6:
            raise ValueError("Cannot build a head frame from these fiducials.")
    si_axis = si_axis / si_len
    lr_axis = np.cross(ap_axis, si_axis)
    lr_axis = lr_axis / np.linalg.norm(lr_axis)
    return ap_axis, lr_axis, si_axis


def project_electrodes(
    mesh: pv.PolyData,
    anatomy: dict[str, np.ndarray],
    cz: np.ndarray,
    *,
    adjustment: float = 0.8,
) -> dict[str, np.ndarray]:
    """Snap the 10–20 template onto the mesh. ``adjustment`` scales the template."""
    ap_axis, lr_axis, si_axis = _head_axes(anatomy)
    nasion = np.asarray(anatomy["nasion"], dtype=float)
    inion = np.asarray(anatomy["inion"], dtype=float)
    cz = np.asarray(cz, dtype=float)
    arc = np.vstack([np.linspace(nasion, cz, 50), np.linspace(cz, inion, 50)[1:]])
    arc_length = float(np.sum(np.linalg.norm(np.diff(arc, axis=0), axis=1)))
    template_arc = float(
        np.linalg.norm(template_xyz("Fpz") - template_xyz("Cz"))
        + np.linalg.norm(template_xyz("Cz") - template_xyz("Oz"))
    )
    scale = (arc_length / template_arc) * float(adjustment)
    positions: dict[str, np.ndarray] = {}
    for name in CHANNELS:
        local = (template_xyz(name) - template_xyz("Cz")) * scale
        guess = cz + local[0] * lr_axis + local[1] * ap_axis + local[2] * si_axis
        idx = mesh.find_closest_point(guess)
        positions[name] = np.asarray(mesh.points[idx], dtype=float)
    return positions
