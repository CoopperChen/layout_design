"""Slack transfer: slide a trace toward its loose neighbor to open a tight side."""

from __future__ import annotations

import numpy as np

from app.runtime import setup_runtime

setup_runtime()
from PYTHON.tools.new2dAlterations import (  # noqa: E402
    _adjacent_gap_rank_key,
    _slack_transfer_moves,
    _slack_transfer_trials,
    compute_adjacent_slot_gap_metrics,
)


def _poly(*pts):
    return np.asarray(pts, dtype=float)


def _sandwich_layout():
    """B and C pinch in the middle; C stays far from D."""
    paths = [
        _poly((0.0, -40.0), (50.0, -20.0), (100.0, 0.0)),
        _poly((0.0, 0.0), (50.0, 0.0), (100.0, 6.0)),
        _poly((0.0, 40.0), (50.0, 2.0), (100.0, 12.0)),
        _poly((0.0, 80.0), (50.0, 80.0), (100.0, 40.0)),
    ]
    electrodes = ["A", "B", "C", "D"]
    terminals = ["T"] * 4
    slots = {"A": 0, "B": 1, "C": 2, "D": 3}
    electrodes_2d = {name: paths[i][0].copy() for i, name in enumerate(electrodes)}
    entries = {name: paths[i][-1].copy() for i, name in enumerate(electrodes)}
    zones = {"zones": {}, "metadata": {}}
    return paths, electrodes, terminals, slots, electrodes_2d, entries, zones


def test_slack_move_targets_the_loose_side_of_the_pinch():
    paths, electrodes, terminals, slots, _e2d, _entries, zones = _sandwich_layout()
    moves = _slack_transfer_moves(paths, electrodes, terminals, zones, slots)
    # C (index 2) is tight against B and loose against D.
    toward_c = [m for m in moves if m[0] == 2]
    assert toward_c
    move_idx, toward_idx, away_idx, tight_gap, loose_gap = toward_c[0]
    assert (move_idx, toward_idx, away_idx) == (2, 3, 1)
    assert tight_gap + 1.0 < loose_gap
    loose_gaps = [m[4] for m in moves]
    assert loose_gaps == sorted(loose_gaps, reverse=True)
    assert moves[0][0] == 2


def test_slack_trial_opens_the_pinched_gap():
    paths, electrodes, terminals, slots, electrodes_2d, entries, zones = _sandwich_layout()
    baseline = compute_adjacent_slot_gap_metrics(
        paths, electrodes, terminals, zones, slots, sample_step=1.0
    )
    trials = _slack_transfer_trials(
        paths,
        electrodes,
        terminals,
        zones,
        {},
        electrodes_2d,
        {"T": np.array([100.0, 20.0])},
        entries,
        slots,
        baseline,
        min_separation=4.0,
        max_crossing_count=0,
    )
    assert trials
    best = min(trials, key=lambda item: item[0])
    assert best[0] < (*_adjacent_gap_rank_key(baseline), 0.0)
    assert best[3]["min_adjacent_gap"] > baseline["min_adjacent_gap"] + 0.05


def test_slack_keeps_a_bow_that_evens_a_trace_without_raising_global_min():
    """A-B is the layout minimum and stays put. Bowing C still counts."""
    paths = [
        _poly((0.0, 0.0), (50.0, 0.0), (100.0, 0.0)),
        _poly((0.0, 2.0), (50.0, 2.0), (100.0, 2.0)),
        _poly((0.0, 24.0), (50.0, 6.0), (100.0, 24.0)),
        _poly((0.0, 60.0), (50.0, 60.0), (100.0, 60.0)),
    ]
    electrodes = ["A", "B", "C", "D"]
    terminals = ["T"] * 4
    slots = {name: i for i, name in enumerate(electrodes)}
    electrodes_2d = {name: paths[i][0].copy() for i, name in enumerate(electrodes)}
    entries = {name: paths[i][-1].copy() for i, name in enumerate(electrodes)}
    zones = {"zones": {}, "metadata": {}}
    baseline = compute_adjacent_slot_gap_metrics(
        paths, electrodes, terminals, zones, slots, sample_step=1.0
    )
    trials = _slack_transfer_trials(
        paths,
        electrodes,
        terminals,
        zones,
        {},
        electrodes_2d,
        {"T": np.array([100.0, 30.0])},
        entries,
        slots,
        baseline,
        min_separation=4.0,
        max_crossing_count=0,
    )
    assert trials
    best = min(trials, key=lambda item: item[0])
    assert "diff" in best[2]
    assert best[3]["min_adjacent_gap"] <= baseline["min_adjacent_gap"] + 0.05
    assert best[3]["min_adjacent_gap"] + 0.05 >= baseline["min_adjacent_gap"]


def test_slack_drop_prints_why(capsys):
    paths, electrodes, terminals, slots, electrodes_2d, entries, zones = _sandwich_layout()
    baseline = compute_adjacent_slot_gap_metrics(
        paths, electrodes, terminals, zones, slots, sample_step=1.0
    )
    reported_min = float(baseline["min_adjacent_gap"])
    baseline["min_adjacent_gap"] = reported_min + 50.0
    trials = _slack_transfer_trials(
        paths,
        electrodes,
        terminals,
        zones,
        {},
        electrodes_2d,
        {"T": np.array([100.0, 20.0])},
        entries,
        slots,
        baseline,
        min_separation=4.0,
        max_crossing_count=0,
    )
    assert trials == []
    log = capsys.readouterr().out
    assert "slack drop C toward D (away from B):" in log
    assert "would lower the minimum adjacent gap" in log
