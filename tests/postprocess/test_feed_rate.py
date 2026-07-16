"""Constant tip-speed C-pivot feed rates."""

from __future__ import annotations

import numpy as np

from app.postprocess.gcode.kinematics.feed_rate import (
    compute_feed_rates,
    compute_print_feed_rates,
    tip_positions_from_poses,
)
from app.postprocess.gcode.models import MachineConfig


def _machine() -> MachineConfig:
    return MachineConfig(
        a_mm=180.7,
        d_mm=57.59,
        speed_mm_min=750.0,
        max_speed_mm_min=1500.0,
        transition_speed_mm_min=2000.0,
    )


def test_feed_matches_tip_speed_for_pure_translation():
    machine = _machine()
    # Fixed B/C: tip and C-pivot translate together → F ≈ V_tip.
    pivots = np.array(
        [
            [0.0, -180.7, -50.0],
            [10.0, -180.7, -50.0],
            [25.0, -180.7, -50.0],
        ]
    )
    b = np.zeros(3)
    c = np.full(3, 90.0)
    feeds = compute_print_feed_rates(pivots, b, c, machine)
    assert feeds[0] == machine.speed_mm_min
    assert feeds[1] == machine.speed_mm_min
    assert feeds[2] == machine.speed_mm_min


def test_feed_scales_with_c_pivot_over_tip_ratio():
    machine = _machine()
    pivots = np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    tips = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    feeds = compute_feed_rates(pivots, tips, machine)
    # F = 750 * 20/5 = 3000 → clamped to max 1500
    assert feeds[1] == machine.max_speed_mm_min


def test_fk_tip_feed_slows_orientation_swings():
    """C change with small path tip used to inflate F; FK tip keeps F nearer V_tip."""
    machine = _machine()
    pivots = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )
    b = np.array([10.0, 10.0])
    c = np.array([0.0, 12.0])
    tips = tip_positions_from_poses(pivots, b, c, machine)
    d_c = float(np.linalg.norm(pivots[1] - pivots[0]))
    d_tip = float(np.linalg.norm(tips[1] - tips[0]))
    assert d_tip > d_c  # arm swing dominates tip travel
    feeds = compute_print_feed_rates(pivots, b, c, machine)
    expected = machine.speed_mm_min * d_c / d_tip
    assert feeds[1] == round(expected)
    assert feeds[1] < machine.speed_mm_min
