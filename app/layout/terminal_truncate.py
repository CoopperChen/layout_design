"""Terminal tail truncation — applied at synthesis (before polish/smooth)."""
from __future__ import annotations

import numpy as np

from app.config_loader import load_defaults


def default_terminal_stop_mm() -> float:
    cfg = load_defaults().get("synthesize", {})
    return float(cfg.get("terminal_stop_mm", 10.0))


def default_terminal_min_points() -> int:
    cfg = load_defaults().get("synthesize", {})
    return int(cfg.get("terminal_min_points", 4))


def path_arc_length(path: np.ndarray) -> float:
    pts = np.asarray(path, dtype=float)
    if len(pts) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def truncate_terminal_tail(
    path: np.ndarray,
    *,
    stop_mm: float,
    min_points: int = 4,
) -> np.ndarray:
    """
    Shorten a path by ``stop_mm`` arc length from the terminal (last) end.

    Works on Nx2 or Nx3 polylines. The electrode (first) end is kept.
    """
    pts = np.asarray(path, dtype=float)
    if stop_mm <= 0.0 or len(pts) < 2:
        return pts.copy()

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = float(seg.sum())
    if total <= stop_mm + 1e-9:
        keep = min(len(pts), max(int(min_points), 2))
        return pts[:keep].copy()

    remain = float(stop_mm)
    for i in range(len(pts) - 2, -1, -1):
        a, b = pts[i], pts[i + 1]
        length = float(seg[i])
        if length < 1e-12:
            continue
        if remain <= length + 1e-9:
            new_end = b - (remain / length) * (b - a)
            if np.linalg.norm(new_end - b) <= 1e-6:
                out = pts[: i + 1]
            elif np.linalg.norm(new_end - a) <= 1e-6:
                out = pts[: i + 1]
            else:
                out = np.vstack([pts[: i + 1], new_end])
            if len(out) < min_points:
                return pts[: min(len(pts), min_points)].copy()
            return out
        remain -= length

    keep = min(len(pts), max(int(min_points), 2))
    return pts[:keep].copy()


def apply_wire_truncation(
    path_2d: np.ndarray,
    path_3d: np.ndarray,
    *,
    stop_mm: float,
    min_points: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Truncate using 3D arc length in mm, then shorten 2D by the same fraction.

    Returns (path_2d_trunc, path_3d_trunc, wire_end_3d). Caller should pin the
    2D end to the polar projection of wire_end_3d so path_end_2d matches.
    """
    path_2d = np.asarray(path_2d, dtype=float)
    path_3d = np.asarray(path_3d, dtype=float)
    if stop_mm <= 0.0:
        end3d = path_3d[-1].copy()
        return path_2d.copy(), path_3d.copy(), end3d

    p3 = truncate_terminal_tail(path_3d, stop_mm=stop_mm, min_points=min_points)
    wire_end_3d = np.asarray(p3[-1], dtype=float).copy()

    len3 = path_arc_length(path_3d)
    kept3 = path_arc_length(p3)
    frac_kept = (kept3 / len3) if len3 > 1e-12 else 1.0
    len2 = path_arc_length(path_2d)
    stop_2d = max(0.0, len2 * (1.0 - frac_kept))
    p2 = truncate_terminal_tail(path_2d, stop_mm=stop_2d, min_points=min_points)
    return p2, p3, wire_end_3d
