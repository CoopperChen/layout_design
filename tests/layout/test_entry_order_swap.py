"""Synthesize entry-order swap uncross."""

from __future__ import annotations

import numpy as np

from app.runtime import setup_runtime

setup_runtime()

from PYTHON.tools.layoutPresetV4 import (  # noqa: E402
    _bent_path_2d,
    _count_pair_crossings,
    _mutual_crossing_count,
    _straight_path_2d,
    _uncross_by_entry_order_swap,
    _uncross_even_mutual_pairs,
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


def _same_hub_odd_pair_count(
    paths: list[np.ndarray],
    path_terminals: list[str],
) -> int:
    n_odd = 0
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            if path_terminals[i] != path_terminals[j]:
                continue
            if _mutual_crossing_count(paths[i], paths[j], use_dense=False) % 2 == 1:
                n_odd += 1
    return n_odd


def test_entry_order_swap_accepts_even_mutual_when_count_may_not_drop():
    """Accept swap that clears the pair (even mutual=0), even among 3 wires."""
    electrodes_2d = {
        "A": np.array([-30.0, 40.0]),
        "B": np.array([0.0, 45.0]),
        "C": np.array([30.0, 40.0]),
    }
    entry_points = {
        "A": np.array([10.0, 0.0]),
        "B": np.array([0.0, 0.0]),
        "C": np.array([-10.0, 0.0]),
    }
    paths = [
        _straight_path_2d(electrodes_2d[n], entry_points[n]) for n in ("A", "B", "C")
    ]
    assert _count_pair_crossings(paths) >= 1
    terminals = ["TERMINAL_LEFT"] * 3

    new_paths, new_entries, _ = _uncross_by_entry_order_swap(
        paths,
        ["A", "B", "C"],
        terminals,
        electrodes_2d,
        entry_points,
        {"zones": {}, "metadata": {}},
        slot_index={"A": 0, "B": 1, "C": 2},
    )
    assert (
        not np.allclose(new_entries["A"], entry_points["A"])
        or not np.allclose(new_entries["C"], entry_points["C"])
        or not np.allclose(new_entries["B"], entry_points["B"])
    )
    assert _same_hub_odd_pair_count(new_paths, terminals) == 0


def test_entry_order_swap_clears_all_odd_mutual_pairs():
    """Crossed strip assignment must finish with zero odd-mutual same-hub pairs."""
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
    terminals = ["TERMINAL_LEFT", "TERMINAL_LEFT"]
    assert _same_hub_odd_pair_count(paths, terminals) == 1

    new_paths, _, _ = _uncross_by_entry_order_swap(
        paths,
        ["A", "B"],
        terminals,
        electrodes_2d,
        entry_points,
        {"zones": {}, "metadata": {}},
        slot_index={"A": 0, "B": 1},
    )
    assert _same_hub_odd_pair_count(new_paths, terminals) == 0
    assert _count_pair_crossings(new_paths) == 0


def test_mutual_crossing_count_parity():
    a = _straight_path_2d(np.array([-10.0, 10.0]), np.array([10.0, -10.0]))
    b = _straight_path_2d(np.array([-10.0, -10.0]), np.array([10.0, 10.0]))
    assert _mutual_crossing_count(a, b) == 1
    c = _straight_path_2d(np.array([-10.0, 0.0]), np.array([10.0, 0.0]))
    d = _straight_path_2d(np.array([-10.0, 5.0]), np.array([10.0, 5.0]))
    assert _mutual_crossing_count(c, d) == 0


def test_uncross_even_mutual_clears_double_weave():
    """Uncrossed ends with a double weave should uncross via fixed-end bends."""
    electrodes_2d = {
        "A": np.array([-20.0, 40.0]),
        "B": np.array([20.0, 40.0]),
    }
    entry_points = {
        "A": np.array([-10.0, 0.0]),
        "B": np.array([10.0, 0.0]),
    }
    # Opposite bows create an even mutual weave with uncrossed strip ends.
    paths = [
        _bent_path_2d(electrodes_2d["A"], entry_points["A"], perp_sign=1.0, scale=30.0),
        _bent_path_2d(electrodes_2d["B"], entry_points["B"], perp_sign=-1.0, scale=30.0),
    ]
    mutual0 = _mutual_crossing_count(paths[0], paths[1], use_dense=False)
    assert mutual0 >= 2 and mutual0 % 2 == 0
    # Sparse global count must count both points (not MultiPoint-as-1).
    assert _count_pair_crossings(paths) == mutual0

    out = _uncross_even_mutual_pairs(
        paths,
        ["A", "B"],
        ["TERMINAL_LEFT", "TERMINAL_LEFT"],
        electrodes_2d,
        entry_points,
        {"zones": {}, "metadata": {}},
    )
    assert _mutual_crossing_count(out[0], out[1], use_dense=False) < mutual0
    np.testing.assert_allclose(out[0][0], electrodes_2d["A"])
    np.testing.assert_allclose(out[1][0], electrodes_2d["B"])
    np.testing.assert_allclose(out[0][-1], entry_points["A"])
    np.testing.assert_allclose(out[1][-1], entry_points["B"])
