"""Incremental crossing-cap check matches full layout count."""

from __future__ import annotations

import numpy as np

from app.PYTHON.tools import new2dAlterations as n2d


def _simple_layout():
    paths = [
        np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]], dtype=float),
        np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]], dtype=float),
        np.array([[0.5, 2.0], [1.5, 2.0]], dtype=float),
    ]
    path_terminals = ["LEFT", "LEFT", "RIGHT"]
    terminal_zones = {}
    electrode_zones = {}
    return paths, path_terminals, terminal_zones, electrode_zones


def test_layout_crossing_count_if_replaced_matches_full():
    paths, path_terminals, terminal_zones, electrode_zones = _simple_layout()
    path_idx = 0
    trial = paths[path_idx].copy()
    trial[1] = trial[1] + np.array([0.05, -0.05])

    cache = n2d._build_crossing_detection_path_cache(
        paths, path_terminals, electrode_zones
    )
    among = n2d._count_crossings_among_other_paths(
        path_idx,
        paths,
        path_terminals,
        terminal_zones,
        electrode_zones,
        dense_path_cache=cache,
    )
    incremental = n2d._layout_crossing_count_if_replaced(
        path_idx,
        trial,
        paths,
        path_terminals,
        terminal_zones,
        electrode_zones,
        among,
        dense_path_cache=cache,
    )

    trial_paths = [p.copy() for p in paths]
    trial_paths[path_idx] = trial
    full = n2d._count_layout_crossings(
        trial_paths, path_terminals, terminal_zones, electrode_zones
    )
    assert incremental == full


def test_incremental_matches_analyze_crossing_count():
    paths, path_terminals, terminal_zones, electrode_zones = _simple_layout()
    analyze = n2d.analyze_path_collisions(
        paths,
        terminal_zones,
        electrode_zones=electrode_zones,
        path_terminals=path_terminals,
        metrics_mode="clearance",
    )
    layout = n2d._count_layout_crossings(
        paths, path_terminals, terminal_zones, electrode_zones
    )
    assert analyze["crossing_count"] == layout
