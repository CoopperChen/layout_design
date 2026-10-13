"""Adjacent-slot gap metrics and maximin rank for gentle polish."""

from __future__ import annotations

import numpy as np

from app.runtime import setup_runtime

setup_runtime()

from PYTHON.tools.new2dAlterations import (  # noqa: E402
    _adjacent_gap_rank_key,
    _format_adjacent_neighbor_gaps,
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
    assert m["max_adjacent_gap"] > m["min_adjacent_gap"]
    report = _format_adjacent_neighbor_gaps(m, electrodes, terminals, slots)
    assert "A-B" in report and "B-C" in report
    assert "uneven" in report
    # Hub gaps near y=0: A–B ~2mm, B–C ~18mm → high variance / spread
    assert m["hub_gap_spread"] > 5.0


def test_neighbor_evenness_uses_gap_difference_not_widest_minus_tightest():
    metrics = {
        "adjacent_pairs": [(0, 1), (1, 2), (2, 3)],
        "adjacent_gaps": [4.0, 4.2, 8.0],
    }
    electrodes = ["A", "B", "C", "D"]
    terminals = ["TERMINAL_LEFT"] * 4
    slots = {"A": 0, "B": 1, "C": 2, "D": 3}
    report = _format_adjacent_neighbor_gaps(metrics, electrodes, terminals, slots)
    lines = report.splitlines()
    assert lines == [
        "LEFT: C B-C 4.20/C-D 8.00 diff=3.80 uneven",
        "LEFT: B A-B 4.00/B-C 4.20 diff=0.20 even",
    ]
    assert "max_adj_gap_diff" not in report


def test_neighbor_log_keeps_four_largest_gap_differences():
    metrics = {
        "adjacent_pairs": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)],
        "adjacent_gaps": [1.0, 2.0, 3.0, 9.0, 10.0, 11.0],
    }
    electrodes = ["A", "B", "C", "D", "E", "F", "G"]
    terminals = ["TERMINAL_LEFT"] * 7
    slots = {name: i for i, name in enumerate(electrodes)}
    report = _format_adjacent_neighbor_gaps(metrics, electrodes, terminals, slots)
    # Diffs: B 1.00, C 1.00, D 6.00, E 1.00, F 1.00. Only D exceeds the next tier.
    assert len(report.splitlines()) == 4
    assert report.count("diff=") == 4
    assert "D C-D 3.00/D-E 9.00 diff=6.00 uneven" in report
    assert report.index("diff=6.00") < report.index("diff=1.00")


def test_adjacent_gap_rank_prefers_larger_min_gap():
    worse = {"min_adjacent_gap": 1.0, "hub_gap_variance": 0.0, "hub_gap_spread": 0.0}
    better = {"min_adjacent_gap": 3.0, "hub_gap_variance": 10.0, "hub_gap_spread": 5.0}
    assert _adjacent_gap_rank_key(better) < _adjacent_gap_rank_key(worse)


def test_adjacent_gap_rank_prefers_lower_hub_variance_when_min_tied():
    a = {"min_adjacent_gap": 4.0, "hub_gap_variance": 8.0, "hub_gap_spread": 6.0}
    b = {"min_adjacent_gap": 4.0, "hub_gap_variance": 1.0, "hub_gap_spread": 2.0}
    assert _adjacent_gap_rank_key(b) < _adjacent_gap_rank_key(a)
