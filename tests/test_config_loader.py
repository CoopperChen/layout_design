"""Config defaults."""

from app.config_loader import default_assignments, resolve_assignments


def test_default_assignments_from_yaml():
    assert default_assignments() == "subject1_best_v4"


def test_resolve_assignments_uses_default():
    assert resolve_assignments(None) == "subject1_best_v4"


def test_resolve_assignments_override():
    assert resolve_assignments("custom_map") == "custom_map"
