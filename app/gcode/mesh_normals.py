"""Outward-oriented surface normals for head meshes and traces."""

from __future__ import annotations

import numpy as np


def head_center_from_points(points: np.ndarray) -> np.ndarray:
    """Approximate head interior reference from vertex/path cloud."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(1, 3)
    return np.mean(pts, axis=0)


def orient_normals_outward(
    points: np.ndarray,
    normals: np.ndarray,
    head_center: np.ndarray,
) -> np.ndarray:
    """
    Flip normals so they point away from ``head_center``.

    For each point, the normal should have a positive dot product with
    ``point - head_center`` (radial outward on a convex head-like surface).
    """
    pts = np.asarray(points, dtype=float)
    nrm = np.asarray(normals, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(1, 3)
    if nrm.ndim == 1:
        nrm = nrm.reshape(1, 3)
    center = np.asarray(head_center, dtype=float).reshape(3)
    out = nrm.copy()
    radial = pts - center
    radial_norm = np.linalg.norm(radial, axis=1, keepdims=True)
    radial_norm = np.maximum(radial_norm, 1e-12)
    radial_unit = radial / radial_norm
    inward = np.sum(out * radial_unit, axis=1) < 0.0
    out[inward] *= -1.0
    lengths = np.linalg.norm(out, axis=1, keepdims=True)
    lengths = np.maximum(lengths, 1e-12)
    return out / lengths


def orient_normals_along_path(
    points: np.ndarray,
    normals: np.ndarray,
    head_center: np.ndarray,
    *,
    radial_threshold: float = 0.15,
) -> np.ndarray:
    """
    Outward normals with continuity along a wire path.

    Crown points have small ``|n·radial|`` so per-point outward tests flip B between
    branches. Inherit neighbors only when the result stays outward — never flip a
    normal inward just to match the previous point.
    """
    out = orient_normals_outward(points, normals, head_center)
    n = len(out)
    if n <= 1:
        return out

    pts = np.asarray(points, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(1, 3)
    center = np.asarray(head_center, dtype=float).reshape(3)
    radial_unit = pts - center
    radial_norm = np.linalg.norm(radial_unit, axis=1, keepdims=True)
    radial_norm = np.maximum(radial_norm, 1e-12)
    radial_unit = radial_unit / radial_norm

    def _outward_flip(normal: np.ndarray, i: int) -> np.ndarray:
        if float(np.dot(normal, radial_unit[i])) < 0.0:
            return -normal
        return normal

    def _enforce_forward(normals: np.ndarray) -> None:
        for i in range(1, len(normals)):
            if float(np.dot(normals[i], normals[i - 1])) >= 0.0:
                continue
            flipped = -normals[i]
            if float(np.dot(flipped, radial_unit[i])) >= float(
                np.dot(normals[i], radial_unit[i])
            ):
                normals[i] = flipped

    def _fix_ambiguous(normals: np.ndarray) -> None:
        radial_dot = np.sum(normals * radial_unit, axis=1)
        for i in range(len(normals)):
            if abs(radial_dot[i]) >= radial_threshold:
                normals[i] = _outward_flip(normals[i], i)
                continue
            if i > 0:
                normals[i] = _outward_flip(normals[i - 1].copy(), i)
            elif i + 1 < len(normals):
                normals[i] = _outward_flip(normals[i + 1].copy(), i)
            else:
                normals[i] = _outward_flip(normals[i], i)

    _enforce_forward(out)
    _fix_ambiguous(out)
    _enforce_forward(out)
    for i in range(n):
        out[i] = _outward_flip(out[i], i)

    lengths = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(lengths, 1e-12)


def smooth_normals_along_path(
    normals: np.ndarray,
    *,
    alpha: float = 0.5,
    passes: int = 1,
) -> np.ndarray:
    """
    Exponential blend of neighboring normals (forward then backward).

    Reduces azimuth jitter when ``n`` is nearly vertical so C does not whip
    between samples and drag the tip in a serpentine.
    """
    out = np.asarray(normals, dtype=float).copy()
    if out.ndim == 1:
        out = out.reshape(1, 3)
    n = len(out)
    if n <= 1 or passes <= 0 or alpha <= 0.0:
        lengths = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(lengths, 1e-12)

    a = float(np.clip(alpha, 0.0, 1.0))
    for _ in range(int(passes)):
        for i in range(1, n):
            blended = (1.0 - a) * out[i] + a * out[i - 1]
            norm = np.linalg.norm(blended)
            if norm > 1e-12:
                out[i] = blended / norm
        for i in range(n - 2, -1, -1):
            blended = (1.0 - a) * out[i] + a * out[i + 1]
            norm = np.linalg.norm(blended)
            if norm > 1e-12:
                out[i] = blended / norm

    lengths = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(lengths, 1e-12)


def stabilize_normals_near_pole(
    normals: np.ndarray,
    *,
    nxy_min: float = 0.08,
) -> np.ndarray:
    """
    Softly inherit XY heading when |n_xy| is small (crown).

    Weight goes to 1 at the pole and to 0 at ``nxy_min``, so leaving the crown
    does not create a single large C cliff.
    """
    out = np.asarray(normals, dtype=float).copy()
    if out.ndim == 1:
        out = out.reshape(1, 3)
    n = len(out)
    if n <= 1 or nxy_min <= 0.0:
        lengths = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(lengths, 1e-12)

    threshold = float(nxy_min)

    def _blend_pass(order: range) -> None:
        for i in order:
            prev = i - 1 if i > 0 else i + 1
            if prev < 0 or prev >= n:
                continue
            nxy = float(np.hypot(out[i, 0], out[i, 1]))
            if nxy >= threshold:
                continue
            prev_xy = out[prev, :2]
            prev_len = float(np.linalg.norm(prev_xy))
            if prev_len < 1e-12:
                continue
            weight = 1.0 - (nxy / threshold)
            curr_dir = out[i, :2]
            curr_len = float(np.linalg.norm(curr_dir))
            if curr_len < 1e-12:
                curr_dir = prev_xy / prev_len
                curr_len = 1.0
            else:
                curr_dir = curr_dir / curr_len
            prev_dir = prev_xy / prev_len
            mixed = (1.0 - weight) * curr_dir + weight * prev_dir
            mixed_len = float(np.linalg.norm(mixed))
            if mixed_len < 1e-12:
                continue
            mixed = mixed / mixed_len
            scale = max(nxy, 1e-6)
            out[i, 0] = mixed[0] * scale
            out[i, 1] = mixed[1] * scale
            norm = np.linalg.norm(out[i])
            if norm > 1e-12:
                out[i] /= norm

    _blend_pass(range(1, n))
    _blend_pass(range(n - 2, -1, -1))

    lengths = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(lengths, 1e-12)


def orient_trace_xyzn(trace: np.ndarray, head_center: np.ndarray) -> np.ndarray:
    """Orient columns 3:6 of an Nx6 trace outward from head center."""
    out = np.asarray(trace, dtype=float).copy()
    out[:, 3:6] = orient_normals_outward(out[:, :3], out[:, 3:6], head_center)
    return out


def orient_electrode_trace_xyzn(trace: np.ndarray, head_center: np.ndarray) -> np.ndarray:
    """
    One outward normal for the whole electrode disk (AdjPoints repmat).

    Uses pad centroid for the outward test — row-0 XYZ is on the circle edge and
    gap-offset, which can wrongly flip normals when tested against head_center.
    """
    out = np.asarray(trace, dtype=float).copy()
    normal = out[0, 3:6]
    norm = np.linalg.norm(normal)
    if norm <= 0.0:
        return out
    normal = normal / norm
    anchor = np.mean(out[:, :3], axis=0)
    radial = anchor - np.asarray(head_center, dtype=float).reshape(3)
    radial_norm = np.linalg.norm(radial)
    if radial_norm > 1e-12 and float(np.dot(normal, radial / radial_norm)) < 0.0:
        normal = -normal
    out[:, 3:6] = normal
    return out
