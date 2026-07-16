"""min_trace_separation_mm drives both phase-2 penalty and layout deficit."""

from __future__ import annotations

import numpy as np

from app.runtime import setup_runtime

setup_runtime()
from PYTHON.tools import new2dAlterations as n2d  # noqa: E402


def _parallel_paths(sep_mm: float):
    return [
        np.column_stack([np.linspace(0, 10, 20), np.zeros(20)]),
        np.column_stack([np.linspace(0, 10, 20), np.full(20, sep_mm)]),
    ]


def test_analyze_path_collisions_uses_min_separation_for_deficit():
    paths = _parallel_paths(3.0)
    path_terminals = ["LEFT", "LEFT"]
    loose = n2d.analyze_path_collisions(
        paths,
        {},
        electrode_zones={},
        path_terminals=path_terminals,
        min_separation=2.0,
        metrics_mode="full",
    )
    tight = n2d.analyze_path_collisions(
        paths,
        {},
        electrode_zones={},
        path_terminals=path_terminals,
        min_separation=4.0,
        metrics_mode="full",
    )
    assert loose["trace_separation_min_required"] == 2.0
    assert tight["trace_separation_min_required"] == 4.0
    assert loose["trace_separation_deficit_normalized"] < tight[
        "trace_separation_deficit_normalized"
    ]


def test_pair_penalty_scales_with_min_separation():
    paths = _parallel_paths(3.0)
    low = n2d._pair_layout_penalty(
        paths[0],
        paths[1],
        "LEFT",
        "LEFT",
        {},
        {},
        min_separation=2.0,
    )
    high = n2d._pair_layout_penalty(
        paths[0],
        paths[1],
        "LEFT",
        "LEFT",
        {},
        {},
        min_separation=4.0,
    )
    assert low < high
