"""Outward normal orientation for head meshes and traces."""

from __future__ import annotations

import numpy as np

from app.postprocess.mesh_normals import orient_normals_outward, orient_trace_xyzn


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
