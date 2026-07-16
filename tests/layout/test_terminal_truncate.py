"""Terminal tail truncation at synthesis."""

from __future__ import annotations

import numpy as np

from app.layout.terminal_truncate import (
    apply_wire_truncation,
    path_arc_length,
    truncate_terminal_tail,
)


def test_truncate_terminal_tail_shortens_from_end():
    path = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ]
    )
    out = truncate_terminal_tail(path, stop_mm=10.0, min_points=2)
    end_dist = float(np.linalg.norm(out[-1] - path[-1]))
    np.testing.assert_allclose(end_dist, 10.0, atol=1e-6)
    np.testing.assert_allclose(out[0], path[0])
    assert len(out) == 3


def test_truncate_terminal_tail_zero_is_noop():
    path = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    out = truncate_terminal_tail(path, stop_mm=0.0)
    np.testing.assert_allclose(out, path)


def test_truncate_terminal_tail_respects_min_points():
    path = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [8.0, 0.0, 0.0]])
    out = truncate_terminal_tail(path, stop_mm=20.0, min_points=4)
    assert len(out) == 3


def test_apply_wire_truncation_matches_3d_fraction_on_2d():
    """2D is shortened by the same arc fraction as the 3D mm stop — not raw mm."""
    path_3d = np.array(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [30.0, 0.0, 0.0]]
    )
    path_2d = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0], [15.0, 0.0]])
    p2, p3, end3 = apply_wire_truncation(
        path_2d, path_3d, stop_mm=10.0, min_points=2
    )
    np.testing.assert_allclose(end3, p3[-1])
    frac3 = path_arc_length(p3) / path_arc_length(path_3d)
    frac2 = path_arc_length(p2) / path_arc_length(path_2d)
    np.testing.assert_allclose(frac2, frac3, atol=1e-6)
    np.testing.assert_allclose(path_arc_length(path_3d) - path_arc_length(p3), 10.0)
