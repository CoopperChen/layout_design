"""Trace print-order planning for shorter inter-wire air travel."""

from __future__ import annotations

import numpy as np

from app import paths
from app.gcode.config_loader import load_machine_config
from app.gcode.pipeline.trace_order import TraceOrderConfig, plan_trace_order


def _safety_transition_counts(merged: np.ndarray, zsafe: float, a_mm: float) -> dict[str, int]:
    return {
        "zsafe": int(np.sum(np.isclose(merged[:, 2], zsafe))),
        "origin_xy": int(
            np.sum(np.isclose(merged[:, 0], 0.0) & np.isclose(merged[:, 1], -a_mm))
        ),
        "m11": int(np.sum(merged[:, 6] == 11)),
    }


def test_plan_trace_order_disabled_returns_identity():
    gcode_list = [np.zeros((3, 7)), np.ones((3, 7))]
    machine = load_machine_config(paths.machine_config())
    plan = plan_trace_order(
        gcode_list,
        machine,
        50.0,
        mesh_points=None,
        mesh_faces=None,
        config=TraceOrderConfig(enabled=False),
    )
    assert plan.order == [0, 1]
    assert plan.flip == [False, False]
    assert plan.skip_origin == [False]


