"""Config defaults."""

from app import paths
from app.config_loader import default_assignments, resolve_assignments


def test_default_assignments_from_yaml():
    assert default_assignments() == "subject1_best_v4"


def test_resolve_assignments_uses_default():
    assert resolve_assignments(None) == "subject1_best_v4"


def test_resolve_assignments_override():
    assert resolve_assignments("custom_map") == "custom_map"


def test_default_assignment_preset_file_ships_in_repo():
    """New clones must find the configured default without build-assignments."""
    preset = paths.preset_path(default_assignments())
    assert preset.is_file(), f"Missing shipped preset: {preset}"
    import json

    data = json.loads(preset.read_text(encoding="utf-8"))
    assignments = data.get("terminal_assignments") or {}
    assert len(assignments) >= 19
    assert set(assignments.values()) <= {"TERMINAL_LEFT", "TERMINAL_RIGHT"}
