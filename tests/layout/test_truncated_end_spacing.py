"""Truncated wire ends keep strip-fan spacing."""

from __future__ import annotations

import numpy as np

from app.runtime import setup_runtime

setup_runtime()

from PYTHON.tools.layoutPresetV4 import (  # noqa: E402
    _align_end_to_strip_ray,
    _spread_truncated_wire_ends,
    _straight_path_2d,
)


def test_align_end_to_strip_ray_preserves_radius():
    hub = np.array([0.0, 0.0])
    strip = np.array([10.0, 0.0])
    end = np.array([5.0, 3.0])
    out = _align_end_to_strip_ray(end, strip, hub)
    np.testing.assert_allclose(out, [np.linalg.norm(end), 0.0], atol=1e-9)


def test_spread_truncated_ends_separates_same_terminal():
    electrodes_2d = {
        "A": np.array([-20.0, 40.0]),
        "B": np.array([20.0, 40.0]),
    }
    terminals_2d = {"TERMINAL_LEFT": np.array([0.0, 0.0])}
    strip_entries = {
        "A": np.array([-4.0, 2.0]),
        "B": np.array([4.0, 2.0]),
    }
    # Truncated ends deliberately bunched near the midline.
    ends = {
        "A": np.array([-0.2, 8.0]),
        "B": np.array([0.2, 8.0]),
    }
    paths = [
        _straight_path_2d(electrodes_2d["A"], ends["A"]),
        _straight_path_2d(electrodes_2d["B"], ends["B"]),
    ]
    assert np.linalg.norm(paths[0][-1] - paths[1][-1]) < 1.0

    out = _spread_truncated_wire_ends(
        paths,
        ["A", "B"],
        ["TERMINAL_LEFT", "TERMINAL_LEFT"],
        strip_entries,
        terminals_2d,
        electrodes_2d,
        min_separation=3.0,
    )
    gap = float(np.linalg.norm(out[0][-1] - out[1][-1]))
    assert gap >= 3.0 - 1e-6
    # Ends stay on hub→strip rays (y/x ≈ strip y/x).
    for i, name in enumerate(("A", "B")):
        hub = terminals_2d["TERMINAL_LEFT"]
        strip = strip_entries[name]
        end = out[i][-1]
        u_strip = strip - hub
        u_end = end - hub
        cross = abs(u_strip[0] * u_end[1] - u_strip[1] * u_end[0])
        assert cross < 1e-6 * (np.linalg.norm(u_strip) * np.linalg.norm(u_end) + 1.0)
