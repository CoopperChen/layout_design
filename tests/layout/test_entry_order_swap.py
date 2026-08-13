"""Synthesize entry-order swap uncross."""

from __future__ import annotations

import numpy as np

from app.runtime import setup_runtime

setup_runtime()

from PYTHON.tools.layoutPresetV4 import (  # noqa: E402
    _bent_path_2d,
    _count_pair_crossings,
    _straight_path_2d,
    _uncross_by_entry_order_swap,
    _uncross_with_fixed_ends,
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


def test_entry_order_angular_escape_uncrosses_three_wire_reversal():
    """Fully reversed 3-slot assignment: pairwise may stall; angular reassign clears it."""
    electrodes_2d = {
        "A": np.array([-30.0, 40.0]),
        "B": np.array([0.0, 45.0]),
        "C": np.array([30.0, 40.0]),
    }
    # Slots reversed vs electrode left-to-right order.
    entry_points = {
        "A": np.array([10.0, 0.0]),
        "B": np.array([0.0, 0.0]),
        "C": np.array([-10.0, 0.0]),
    }
    paths = [
        _straight_path_2d(electrodes_2d[n], entry_points[n]) for n in ("A", "B", "C")
    ]
    assert _count_pair_crossings(paths) >= 1

    new_paths, new_entries, _ = _uncross_by_entry_order_swap(
        paths,
        ["A", "B", "C"],
        ["TERMINAL_LEFT", "TERMINAL_LEFT", "TERMINAL_LEFT"],
        electrodes_2d,
        entry_points,
        {"zones": {}, "metadata": {}},
        slot_index={"A": 0, "B": 1, "C": 2},
    )
    assert _count_pair_crossings(new_paths) == 0
    # Left electrode should own left strip slot after escape.
    assert new_entries["A"][0] < new_entries["B"][0] < new_entries["C"][0]


def test_uncross_with_fixed_ends_clears_bowed_crossing():
    """Fixed-end bends clear an X without moving electrode or wire ends."""
    electrodes_2d = {
        "A": np.array([-20.0, 40.0]),
        "B": np.array([20.0, 40.0]),
    }
    ends = {
        "A": np.array([-10.0, 0.0]),
        "B": np.array([10.0, 0.0]),
    }
    # Uncrossed endpoints, but opposite bows that still cross.
    paths = [
        _bent_path_2d(electrodes_2d["A"], ends["A"], perp_sign=1.0, scale=30.0),
        _bent_path_2d(electrodes_2d["B"], ends["B"], perp_sign=-1.0, scale=30.0),
    ]
    assert _count_pair_crossings(paths) >= 1

    out = _uncross_with_fixed_ends(
        paths,
        ["A", "B"],
        electrodes_2d,
        {"zones": {}, "metadata": {}},
        ends,
        idxs=[0, 1],
    )
    assert _count_pair_crossings(out) == 0
    np.testing.assert_allclose(out[0][0], electrodes_2d["A"])
    np.testing.assert_allclose(out[1][0], electrodes_2d["B"])
    np.testing.assert_allclose(out[0][-1], ends["A"])
    np.testing.assert_allclose(out[1][-1], ends["B"])


def test_entry_order_swap_bends_when_straight_replan_still_crosses():
    """
    Slot swap is correct, but a hostile replan_fn returns bowed crossing chords.

    Bend cleanup with fixed ends should accept the swap and clear the X.
    """
    electrodes_2d = {
        "A": np.array([-20.0, 40.0]),
        "B": np.array([20.0, 40.0]),
    }
    # Initially crossed strip assignment.
    entry_points = {
        "A": np.array([10.0, 0.0]),
        "B": np.array([-10.0, 0.0]),
    }
    paths = [
        _straight_path_2d(electrodes_2d["A"], entry_points["A"]),
        _straight_path_2d(electrodes_2d["B"], entry_points["B"]),
    ]
    assert _count_pair_crossings(paths) >= 1

    def _hostile_replan(idx: int, entries: dict) -> np.ndarray:
        name = ("A", "B")[idx]
        start = electrodes_2d[name]
        end = np.asarray(entries[name], dtype=float)
        # After a correct swap ends are uncrossed; force opposite bows that cross.
        sign = 1.0 if name == "A" else -1.0
        return _bent_path_2d(start, end, perp_sign=sign, scale=30.0)

    new_paths, new_entries, new_slots = _uncross_by_entry_order_swap(
        paths,
        ["A", "B"],
        ["TERMINAL_LEFT", "TERMINAL_LEFT"],
        electrodes_2d,
        entry_points,
        {"zones": {}, "metadata": {}},
        slot_index={"A": 0, "B": 1},
        replan_fn=_hostile_replan,
    )

    assert _count_pair_crossings(new_paths) == 0
    # Slots swapped to uncrossed assignment.
    np.testing.assert_allclose(new_entries["A"], entry_points["B"])
    np.testing.assert_allclose(new_entries["B"], entry_points["A"])
    assert new_slots["A"] == 1
    assert new_slots["B"] == 0
    # Ends stay at the (swapped) strip targets from hostile replan.
    np.testing.assert_allclose(new_paths[0][-1], new_entries["A"])
    np.testing.assert_allclose(new_paths[1][-1], new_entries["B"])
