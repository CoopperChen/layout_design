"""Synthesize entry-order swap uncross."""

from __future__ import annotations

import numpy as np

from app.runtime import setup_runtime

setup_runtime()

from PYTHON.tools.layoutPresetV4 import (  # noqa: E402
    _count_pair_crossings,
    _straight_path_2d,
    _uncross_by_entry_order_swap,
)


def test_entry_order_swap_uncrosses_same_terminal_x():
    """Crossed pair to swapped strip slots should uncross after entry swap + replan."""
    # Electrodes left/right; entries initially crossed (A→right slot, B→left slot).
    electrodes_2d = {
        "A": np.array([-20.0, 40.0]),
        "B": np.array([20.0, 40.0]),
    }
    entry_points = {
        "A": np.array([10.0, 0.0]),
        "B": np.array([-10.0, 0.0]),
    }
    paths = [
        _straight_path_2d(electrodes_2d["A"], entry_points["A"]),
        _straight_path_2d(electrodes_2d["B"], entry_points["B"]),
    ]
    assert _count_pair_crossings(paths) >= 1

    new_paths, new_entries, new_slots = _uncross_by_entry_order_swap(
        paths,
        ["A", "B"],
        ["TERMINAL_LEFT", "TERMINAL_LEFT"],
        electrodes_2d,
        entry_points,
        {"zones": {}, "metadata": {}},
        slot_index={"A": 0, "B": 1},
    )

    assert _count_pair_crossings(new_paths) == 0
    np.testing.assert_allclose(new_entries["A"], entry_points["B"])
    np.testing.assert_allclose(new_entries["B"], entry_points["A"])
    assert new_slots["A"] == 1
    assert new_slots["B"] == 0


def test_entry_order_swap_skips_different_terminals():
    electrodes_2d = {
        "A": np.array([-20.0, 40.0]),
        "B": np.array([20.0, 40.0]),
    }
    entry_points = {
        "A": np.array([10.0, 0.0]),
        "B": np.array([-10.0, 0.0]),
    }
    paths = [
        _straight_path_2d(electrodes_2d["A"], entry_points["A"]),
        _straight_path_2d(electrodes_2d["B"], entry_points["B"]),
    ]
    before = _count_pair_crossings(paths)
    assert before >= 1

    new_paths, new_entries, _ = _uncross_by_entry_order_swap(
        paths,
        ["A", "B"],
        ["TERMINAL_LEFT", "TERMINAL_RIGHT"],
        electrodes_2d,
        entry_points,
        {"zones": {}, "metadata": {}},
    )
    # Different hubs: no swap attempted; geometry unchanged.
    assert _count_pair_crossings(new_paths) == before
    np.testing.assert_allclose(new_entries["A"], entry_points["A"])
    np.testing.assert_allclose(new_entries["B"], entry_points["B"])
