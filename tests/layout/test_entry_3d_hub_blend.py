"""Truncated wire ends must not hub-blend back onto the terminal."""

from __future__ import annotations

import inspect

from app.runtime import setup_runtime

setup_runtime()
from PYTHON.tools.layoutPresetV4 import entry_3d_for_strip  # noqa: E402


def test_entry_3d_for_strip_exposes_blend_toward_hub_flag():
    params = inspect.signature(entry_3d_for_strip).parameters
    assert "blend_toward_hub" in params
    assert params["blend_toward_hub"].default is True


def test_synthesize_wire_end_lift_disables_hub_blend():
    """Regression guard: synth export must pass blend_toward_hub=False for wire ends."""
    from pathlib import Path

    src = Path("app/PYTHON/tools/layoutPresetV4.py").read_text(encoding="utf-8")
    # The wire-end lift block should disable hub blend; strip lift may keep default.
    assert "blend_toward_hub=False" in src
    assert "Do not hub-blend" in src
