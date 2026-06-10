"""Coordinate alignment: origin shift, rough rotation, scan2phys."""

from __future__ import annotations

import numpy as np

from ..models import JobConfig, SubjectBundle, TraceChannel
from ..transform.rotations import apply_rotation, roty, rotz
from ..transform.scan2phys import scan2phys


def _normalize_normals(trace: np.ndarray) -> np.ndarray:
    out = trace.copy()
    norms = out[:, 3:6]
    for i in range(norms.shape[0]):
        n = norms[i]
        norm = np.linalg.norm(n)
        if norm > 0:
            out[i, 3:6] = n / norm
    return out


def align_subject(
    bundle: SubjectBundle,
    job: JobConfig,
) -> tuple[list[TraceChannel], np.ndarray]:
    """
    Apply origin shift, rough rotation, and scan2phys registration.

    Returns (aligned_channels, registered_mesh_points).
    """
    ps = bundle.landmarks_xyz.copy()
    pm = np.asarray(job.physical_landmarks_mm, dtype=float)

    channels: list[TraceChannel] = []
    for ch in bundle.channels:
        ic = ch.interconnect.copy()
        el = ch.electrode.copy()
        ic[:, :3] -= ps[0]
        el[:, :3] -= ps[0]
        ic = _normalize_normals(ic)
        el = _normalize_normals(el)
        channels.append(
            TraceChannel(
                name=ch.name,
                interconnect=ic,
                electrode=el,
                terminal=ch.terminal,
            )
        )

    mesh = bundle.mesh_points - ps[0]
    ps_aligned = ps - ps[0]

    rot = roty(job.rot0y_deg) @ rotz(job.rot0z_deg)
    mesh = apply_rotation(mesh, rot)
    ps_aligned = apply_rotation(ps_aligned, rot)

    for ch in channels:
        ch.interconnect[:, :3] = apply_rotation(ch.interconnect[:, :3], rot)
        ch.interconnect[:, 3:6] = apply_rotation(ch.interconnect[:, 3:6], rot)
        ch.electrode[:, :3] = apply_rotation(ch.electrode[:, :3], rot)
        ch.electrode[:, 3:6] = apply_rotation(ch.electrode[:, 3:6], rot)

    offset = pm[0] - ps_aligned[0]
    ps_shifted = ps_aligned + offset

    mesh_registered = scan2phys(mesh + offset, ps_shifted, pm)

    for ch in channels:
        ch.interconnect[:, :3] = scan2phys(ch.interconnect[:, :3] + offset, ps_shifted, pm)
        ch.interconnect[:, 3:6] = scan2phys(ch.interconnect[:, 3:6], ps_shifted, pm)
        ch.electrode[:, :3] = scan2phys(ch.electrode[:, :3] + offset, ps_shifted, pm)
        ch.electrode[:, 3:6] = scan2phys(ch.electrode[:, 3:6], ps_shifted, pm)

    return channels, mesh_registered
