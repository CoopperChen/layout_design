"""Truncated wire ends keep equal strip-fan spacing."""

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


def test_spread_truncated_ends_equalizes_radii_on_strip_rays():
    electrodes_2d = {
        "A": np.array([-20.0, 40.0]),
        "B": np.array([0.0, 40.0]),
        "C": np.array([20.0, 40.0]),
    }
    terminals_2d = {"TERMINAL_LEFT": np.array([0.0, 0.0])}
    strip_entries = {
        "A": np.array([-6.0, 2.0]),
        "B": np.array([0.0, 2.0]),
        "C": np.array([6.0, 2.0]),
    }
    # Unequal truncation radii on purpose.
    ends = {
        "A": np.array([-3.0, 1.0]),
        "B": np.array([0.0, 8.0]),
        "C": np.array([4.5, 1.5]),
    }
    paths = [
        _straight_path_2d(electrodes_2d[n], ends[n]) for n in ("A", "B", "C")
    ]

    out = _spread_truncated_wire_ends(
        paths,
        ["A", "B", "C"],
        ["TERMINAL_LEFT"] * 3,
        strip_entries,
        terminals_2d,
        electrodes_2d,
        min_separation=3.0,
    )
    hub = terminals_2d["TERMINAL_LEFT"]
    radii = [float(np.linalg.norm(out[i][-1] - hub)) for i in range(3)]
    assert max(radii) - min(radii) < 1e-6

    # Equal angles + equal radius => nearly equal consecutive chord gaps.
    ordered = sorted(range(3), key=lambda i: float(np.arctan2(out[i][-1, 1], out[i][-1, 0])))
    gaps = [
        float(np.linalg.norm(out[b][-1] - out[a][-1]))
        for a, b in zip(ordered, ordered[1:])
    ]
    assert abs(gaps[0] - gaps[1]) < 1e-5

    for i, name in enumerate(("A", "B", "C")):
        strip = strip_entries[name]
        end = out[i][-1]
        u_strip = strip - hub
        u_end = end - hub
        cross = abs(u_strip[0] * u_end[1] - u_strip[1] * u_end[0])
        assert cross < 1e-6 * (np.linalg.norm(u_strip) * np.linalg.norm(u_end) + 1.0)
