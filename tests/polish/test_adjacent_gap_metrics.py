"""Adjacent-slot gap metrics and maximin rank for gentle polish."""

from __future__ import annotations

import numpy as np

from app.runtime import setup_runtime

setup_runtime()

from PYTHON.tools.new2dAlterations import (  # noqa: E402
    _adjacent_gap_rank_key,
    _slot_adjacent_path_index_pairs,
    compute_adjacent_slot_gap_metrics,
)


def _line(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y1]], dtype=float)


def test_slot_adjacent_path_pairs_by_slot_order():
    electrodes = ["A", "B", "C"]
    terminals = ["TERMINAL_LEFT"] * 3
    slots = {"A": 0, "B": 2, "C": 1}  # order A, C, B
    pairs = _slot_adjacent_path_index_pairs(electrodes, terminals, slots)
    assert pairs == [(0, 2), (1, 2)]  # A-C and C-B by slot


def test_adjacent_gap_metrics_min_and_hub_variance():
    # Three parallel routes; B squeezed toward A near the hub (y→0).
    paths = [
        _line(-10.0, 40.0, -10.0, 0.0),
        _line(-2.0, 40.0, -8.0, 0.0),  # drifts toward A near terminal
        _line(10.0, 40.0, 10.0, 0.0),
    ]
    electrodes = ["A", "B", "C"]
    terminals = ["T"] * 3
    slots = {"A": 0, "B": 1, "C": 2}
    zones = {"zones": {}, "metadata": {}}
    m = compute_adjacent_slot_gap_metrics(
        paths, electrodes, terminals, zones, slots
    )
    assert m["n_adjacent_slot_pairs"] == 2
    assert m["min_adjacent_gap"] < 8.0
    # Hub gaps near y=0: A–B ~2mm, B–C ~18mm → high variance / spread
    assert m["hub_gap_spread"] > 5.0


def test_adjacent_gap_rank_prefers_larger_min_gap():
    worse = {"min_adjacent_gap": 1.0, "hub_gap_variance": 0.0, "hub_gap_spread": 0.0}
    better = {"min_adjacent_gap": 3.0, "hub_gap_variance": 10.0, "hub_gap_spread": 5.0}
    assert _adjacent_gap_rank_key(better) < _adjacent_gap_rank_key(worse)


def test_adjacent_gap_rank_prefers_lower_hub_variance_when_min_tied():
    a = {"min_adjacent_gap": 4.0, "hub_gap_variance": 8.0, "hub_gap_spread": 6.0}
    b = {"min_adjacent_gap": 4.0, "hub_gap_variance": 1.0, "hub_gap_spread": 2.0}
    assert _adjacent_gap_rank_key(b) < _adjacent_gap_rank_key(a)
