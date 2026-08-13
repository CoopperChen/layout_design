"""Polish separation focus: pair ordering and acceptance rules."""
from __future__ import annotations

import numpy as np

from app.runtime import setup_runtime

setup_runtime()
from PYTHON.tools import new2dAlterations as n2d  # noqa: E402


def _straight_path(x0, y0, x1, y1, n=20):
    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y0, y1, n)
    return np.column_stack([xs, ys])


def test_find_conflict_pairs_sorts_tightest_first_when_focus_separation():
    """Without slot metadata, focus mode lists the closest mid-route pair first."""
    paths = [
        _straight_path(0, 0, 10, 0),
        _straight_path(0, 1.0, 10, 1.0),
        _straight_path(0, 8.0, 10, 8.0),
    ]
    path_terminals = ["TERMINAL_LEFT"] * 3
    terminal_zones = {}
    electrode_zones = {}

    default_order = n2d._find_conflict_path_pairs(
        paths,
        path_terminals,
        terminal_zones,
        electrode_zones,
        min_separation=4.0,
        focus_separation=False,
    )
    focus_order = n2d._find_conflict_path_pairs(
        paths,
        path_terminals,
        terminal_zones,
        electrode_zones,
        min_separation=4.0,
        focus_separation=True,
    )

    assert default_order
    assert focus_order
    # Pair (0,1) is ~1 mm apart; (0,2) is ~8 mm — focus mode should list (0,1) first.
    assert focus_order[0][:2] == (0, 1)
    # Default mode ranks by total penalty; both pairs violate 4 mm separation.
    assert (0, 1) in [item[:2] for item in default_order]


def test_pair_separation_penalizes_long_near_parallel_more_than_brief_pass():
    """Integrated deficit should punish sustained close contact, not only min_dist."""
    long_a = _straight_path(0, 0, 20, 0, n=40)
    long_b = _straight_path(0, 1.0, 20, 1.0, n=40)
    brief_a = _straight_path(0, 0, 20, 0, n=40)
    # Approach then leave: same-ish min gap, shorter close contact.
    xs = np.linspace(0, 20, 40)
    ys = np.abs(xs - 10.0) * 0.4 + 1.0  # min gap ~1 mm at mid, wider elsewhere
    brief_b = np.column_stack([xs, ys])

    long_sep = n2d._pair_separation_metrics(
        long_a, long_b, "TERMINAL_LEFT", "TERMINAL_LEFT", {}, min_separation=4.0
    )
    brief_sep = n2d._pair_separation_metrics(
        brief_a, brief_b, "TERMINAL_LEFT", "TERMINAL_LEFT", {}, min_separation=4.0
    )
    assert long_sep["min_dist"] < 1.5
    assert brief_sep["min_dist"] < 1.5
    assert long_sep["deficit_norm"] > brief_sep["deficit_norm"] * 1.5

    long_pen = n2d._pair_layout_penalty(
        long_a, long_b, "TERMINAL_LEFT", "TERMINAL_LEFT", {}, {}, min_separation=4.0
    )
    brief_pen = n2d._pair_layout_penalty(
        brief_a, brief_b, "TERMINAL_LEFT", "TERMINAL_LEFT", {}, {}, min_separation=4.0
    )
    assert long_pen > brief_pen


def test_pair_separation_severe_term_dominates_near_zero_gap():
    """Gaps much smaller than min_sep should produce a large severe_norm."""
    far = n2d._pair_separation_metrics(
        _straight_path(0, 0, 10, 0),
        _straight_path(0, 2.5, 10, 2.5),
        "TERMINAL_LEFT",
        "TERMINAL_LEFT",
        {},
        min_separation=4.0,
    )
    near = n2d._pair_separation_metrics(
        _straight_path(0, 0, 10, 0),
        _straight_path(0, 0.2, 10, 0.2),
        "TERMINAL_LEFT",
        "TERMINAL_LEFT",
        {},
        min_separation=4.0,
    )
    assert near["severe_norm"] > far["severe_norm"]
    assert near["severe_norm"] > 10.0


def test_sparse_slot_neighbors_are_penalized():
    """Slot-adjacent twins above max_sep get a mid-route sparse penalty."""
    paths = [
        _straight_path(0, 0, 10, 0),
        _straight_path(0, 12.0, 10, 12.0),
    ]
    sparse = n2d._pair_separation_metrics(
        paths[0],
        paths[1],
        "TERMINAL_LEFT",
        "TERMINAL_LEFT",
        {},
        min_separation=4.0,
        punish_sparse=True,
    )
    assert sparse["sparse_norm"] > 0.0
    adj_sparse = n2d._pair_layout_penalty(
        paths[0],
        paths[1],
        "TERMINAL_LEFT",
        "TERMINAL_LEFT",
        {},
        {},
        min_separation=4.0,
        punish_sparse=True,
    )
    no_sparse = n2d._pair_layout_penalty(
        paths[0],
        paths[1],
        "TERMINAL_LEFT",
        "TERMINAL_LEFT",
        {},
        {},
        min_separation=4.0,
        punish_sparse=False,
    )
    assert adj_sparse > no_sparse


def test_near_terminal_equal_gap_preferred_over_min_sep():
    """Near-terminal band may sit below min_sep; unequal fan gaps are punished."""
    # Three slot neighbors: gaps 1 and 5 near the terminal end (y).
    paths = [
        _straight_path(0, 0, 0, 10),
        _straight_path(1, 0, 1, 10),
        _straight_path(6, 0, 6, 10),
    ]
    # Paths run toward +y terminal; near-terminal samples are high-y.
    path_terminals = ["T"] * 3
    path_electrodes = ["E0", "E1", "E2"]
    slot_index = {"E0": 0, "E1": 1, "E2": 2}

    targets = n2d._hub_near_terminal_gap_targets(
        paths, path_terminals, path_electrodes, slot_index
    )
    assert (0, 1) in targets and (1, 2) in targets
    mean_gap = targets[(0, 1)]
    assert abs(mean_gap - 3.0) < 0.5  # mean of ~1 and ~5

    tight = n2d._pair_separation_metrics(
        paths[0],
        paths[1],
        "T",
        "T",
        {},
        min_separation=4.0,
        equalize_terminal=True,
        target_near_terminal_gap=mean_gap,
    )
    assert tight["equal_norm"] > 0.0
    wide = n2d._pair_separation_metrics(
        paths[1],
        paths[2],
        "T",
        "T",
        {},
        min_separation=4.0,
        equalize_terminal=True,
        target_near_terminal_gap=mean_gap,
    )
    assert wide["equal_norm"] > 0.0

    focus_order = n2d._find_conflict_path_pairs(
        paths,
        path_terminals,
        {},
        {},
        min_separation=4.0,
        focus_separation=True,
        path_electrodes=path_electrodes,
        slot_index_by_electrode=slot_index,
    )
    assert focus_order
    # Unequal fan pairs are packing-priority 0 (before pure mid-route pinches).
    assert focus_order[0][:2] in {(0, 1), (1, 2)}


def test_nudge_toward_partner_reduces_gap():
    a = _straight_path(0, 0, 10, 0)
    b = _straight_path(0, 12, 10, 12)
    nudged = n2d._nudge_path_toward_partner(a, b, blend=0.25, toward=True)
    assert nudged is not None
    mid_y = float(nudged[len(nudged) // 2, 1])
    assert mid_y > 1.0
