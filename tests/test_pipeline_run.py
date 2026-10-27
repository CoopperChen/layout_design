"""Pipeline stage order."""

import pytest

from app.pipeline.run import STAGES, active_stages


def test_default_span_is_mesh_through_gcode():
    assert active_stages("mesh", "gcode") == list(STAGES)


def test_resume_from_gcode():
    assert active_stages("gcode", "gcode") == ["gcode"]


def test_inverted_range_rejected():
    with pytest.raises(ValueError):
        active_stages("gcode", "mesh")
