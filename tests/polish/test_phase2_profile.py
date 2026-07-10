"""Tests for phase-2 polish profiler."""

from __future__ import annotations

from app.polish.phase2_profile import (
    profile_step,
    start_phase2_profile,
    stop_phase2_profile,
)


def test_phase2_profile_round_summary(capsys):
    prof = start_phase2_profile()
    prof.set_round(0)
    with profile_step("find_conflict_pairs"):
        pass
    with profile_step("accept_global_crossing"):
        pass
    prof.print_round_summary(0)
    stop_phase2_profile()
    out = capsys.readouterr().out
    assert "Phase 2 profile round 1" in out
    assert "find_conflict_pairs" in out
    assert "accept_global_crossing" in out


def test_profile_step_noop_when_inactive():
    with profile_step("ignored"):
        pass
