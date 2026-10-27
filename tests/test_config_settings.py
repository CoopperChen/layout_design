"""config/defaults.yaml mesh and layout knobs."""

import pytest

from app.config_loader import layout_config, mesh_config, reset_defaults_cache


def test_shipped_layout_and_mesh_defaults():
    reset_defaults_cache()
    layout = layout_config()
    mesh = mesh_config()
    assert layout == {"scale": 0.8, "scale_min": 0.5, "scale_max": 1.5}
    assert mesh["poisson_depth"] == 12
    assert mesh["scale_to_mm"] == 1000.0
    assert mesh["poisson_density_quantile"] == 0.02
    assert mesh["icp_fitness_min"] == 0.3
    assert mesh["icp_mean_dist_max_mm"] == 3.0


def test_layout_scale_must_sit_inside_the_slider(monkeypatch):
    monkeypatch.setattr(
        "app.config_loader.load_defaults",
        lambda: {"layout": {"scale": 2.0, "scale_min": 0.5, "scale_max": 1.5}},
    )
    with pytest.raises(ValueError, match="scale"):
        layout_config()
