"""Outward normal orientation for head meshes and traces."""

from __future__ import annotations

import numpy as np

from app.postprocess.mesh_normals import (
    orient_normals_along_path,
    orient_normals_outward,
    orient_trace_xyzn,
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
