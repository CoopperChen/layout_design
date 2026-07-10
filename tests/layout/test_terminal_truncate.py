"""Terminal tail truncation at synthesis."""

from __future__ import annotations

import numpy as np

from app.layout.terminal_truncate import truncate_terminal_tail


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
