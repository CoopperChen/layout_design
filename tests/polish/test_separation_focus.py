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
    """Separation focus should resolve the closest pair before distant ones."""
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
