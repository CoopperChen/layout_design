"""Outward normal orientation for head meshes and traces."""

from __future__ import annotations

import numpy as np

from app.postprocess.mesh_normals import (
    orient_normals_along_path,
    orient_normals_outward,
    orient_trace_xyzn,
    smooth_normals_along_path,
    stabilize_normals_near_pole,
)


def test_orient_normals_outward_flips_inward():
    center = np.array([0.0, 0.0, 0.0])
    point = np.array([10.0, 0.0, 0.0])
    inward = np.array([-1.0, 0.0, 0.0])
    outward = orient_normals_outward(point.reshape(1, 3), inward.reshape(1, 3), center)[0]
    np.testing.assert_allclose(outward, [1.0, 0.0, 0.0], atol=1e-9)


def test_orient_trace_xyzn():
    center = np.zeros(3)
    trace = np.array([[10.0, 0.0, 0.0, -1.0, 0.0, 0.0]])
    out = orient_trace_xyzn(trace, center)
    np.testing.assert_allclose(out[0, 3:6], [1.0, 0.0, 0.0], atol=1e-9)


def test_orient_normals_along_path_stays_outward_on_crown():
    """Path helper keeps outward orientation when inheriting neighbors."""
    center = np.array([0.0, 0.0, 0.0])
    xyz = np.array(
        [
            [0.0, 90.0, -10.0],
            [0.0, 91.0, -12.0],
            [0.0, 92.0, -14.0],
        ],
        dtype=float,
    )
    raw = np.array(
        [
            [0.0, 0.2, 0.98],
            [0.0, -0.2, -0.98],
            [0.0, 0.2, 0.98],
        ],
        dtype=float,
    )
    out = orient_normals_along_path(xyz, raw, center)
    radial = xyz - center
    radial /= np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-12)
    assert np.all(np.sum(out * radial, axis=1) >= 0.0)


def test_smooth_normals_along_path_reduces_xy_jitter():
    jagged = np.array(
        [
            [0.04, -0.04, 0.998],
            [0.02, 0.02, 1.0],
            [0.03, 0.04, 0.999],
            [0.005, 0.10, 0.995],
        ],
        dtype=float,
    )
    jagged /= np.linalg.norm(jagged, axis=1, keepdims=True)

    def _c_from_n(n: np.ndarray) -> float:
        return float(np.degrees(np.arctan2(n[1], n[0])))

    raw_steps = [
        abs(_c_from_n(jagged[i]) - _c_from_n(jagged[i - 1]))
        for i in range(1, len(jagged))
    ]
    smooth = smooth_normals_along_path(jagged, alpha=0.5, passes=1)
    smooth_steps = [
        abs(_c_from_n(smooth[i]) - _c_from_n(smooth[i - 1]))
        for i in range(1, len(smooth))
    ]
    assert max(smooth_steps) < max(raw_steps)


def test_stabilize_normals_near_pole_biases_heading_toward_previous():
    normals = np.array(
        [
            [0.2, 0.0, 0.98],
            [0.01, 0.05, 0.999],  # tiny n_xy, different heading
        ],
        dtype=float,
    )
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    out = stabilize_normals_near_pole(normals, nxy_min=0.08)
    h0 = np.arctan2(normals[0, 1], normals[0, 0])
    h_raw = np.arctan2(normals[1, 1], normals[1, 0])
    h1 = np.arctan2(out[1, 1], out[1, 0])
    # Soft blend: closer to previous heading than the raw crown sample.
    assert abs(h1 - h0) < abs(h_raw - h0)
