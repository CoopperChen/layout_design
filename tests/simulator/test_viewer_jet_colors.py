"""Tip path jet-on vs jet-off coloring in the simulator viewer."""

from __future__ import annotations

import numpy as np

from app.simulator.viewer import (
    _dedupe_path_with_markers,
    _segment_jet_on,
    _tip_segment_colors,
)


def test_segment_jet_off_after_m11_before_next_m10():
    markers = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 11.0, 0.0, 0.0, 10.0, 0.0])
    active = _segment_jet_on(markers)
    assert active == [False, False, True, True, True, False, False, False, True]


def test_tip_segment_colors_match_jet_state():
    markers = np.array([0.0, 10.0, 0.0, 11.0, 0.0])
    colors = _tip_segment_colors(markers)
    assert colors == ["tomato", "lime", "lime", "tomato"]


def test_dedupe_path_with_markers_keeps_alignment():
    path = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )
    markers = np.array([0.0, 10.0, 0.0, 11.0])
    deduped_path, deduped_markers = _dedupe_path_with_markers(path, markers)
    assert len(deduped_path) == len(deduped_markers) == 3
    assert deduped_markers.tolist() == [10.0, 0.0, 11.0]
    assert len(_tip_segment_colors(deduped_markers)) == len(deduped_path) - 1
