"""Pipeline orchestration helpers."""

from __future__ import annotations

import pytest

from app.pipeline.run import (
    STAGES,
    PipelinePaths,
    _active_stages,
    _layout_input,
    _polished_layout,
)


def test_active_stages_default_skips_polish():
    stages = _active_stages(from_stage="synthesize", to_stage="gcode", polish=False)
    assert stages == [
        "synthesize",
        "smooth",
        "bundle",
        "print-config",
        "gcode",
    ]


def test_active_stages_full_from_ply():
    stages = _active_stages(from_stage="reconstruct", to_stage="gcode", polish=False)
    assert stages[:5] == [
        "reconstruct",
        "clear-islands",
        "fiducials",
        "cz",
        "electrodes",
    ]
    assert stages[-1] == "gcode"
    assert "polish" not in stages


def test_active_stages_with_polish():
    stages = _active_stages(from_stage="synthesize", to_stage="smooth", polish=True)
    assert stages == ["synthesize", "polish", "smooth"]


def test_active_stages_resume_from_smooth():
    stages = _active_stages(from_stage="smooth", to_stage="simulate", polish=False)
    assert stages == ["smooth", "bundle", "print-config", "gcode", "simulate"]


def test_active_stages_rejects_inverted_range():
    with pytest.raises(ValueError, match="after"):
        _active_stages(from_stage="gcode", to_stage="synthesize", polish=False)


def test_pipeline_paths_for_target():
    pp = PipelinePaths.for_target(2)
    assert pp.assignments == "s1_assignments"
    assert pp.ply.name == "2.ply"
    assert pp.cleaned.name == "2.stl"
    assert pp.layout.name == "synth_s2.json"
    assert pp.smooth.name == "smooth_s2_final.json"
    assert pp.bundle.name == "subject_2"
    assert pp.gcode.name == "allinterconnects.txt"


def test_polished_layout_suffix():
    layout = PipelinePaths.for_target(2).layout
    assert _polished_layout(layout).name == "synth_s2_repaired.json"


def test_layout_input_prefers_repaired_when_polished_exists(tmp_path):
    layout = tmp_path / "synth_s2.json"
    repaired = tmp_path / "synth_s2_repaired.json"
    layout.write_text("{}", encoding="utf-8")
    repaired.write_text("{}", encoding="utf-8")
    pp = PipelinePaths(
        target=2,
        assignments="x",
        ply=tmp_path / "2.ply",
        cleaned=tmp_path / "2.stl",
        fiducials=tmp_path / "fid.json",
        cz=tmp_path / "cz.json",
        electrodes=tmp_path / "elec.json",
        layout=layout,
        smooth=tmp_path / "smooth.json",
        bundle=tmp_path / "bundle",
        gcode=tmp_path / "g.txt",
        print_config=tmp_path / "pm.yaml",
    )
    chosen = _layout_input(pp, ["smooth", "bundle"], polish=True)
    assert chosen == repaired


def test_stage_order_starts_with_preprocess():
    assert STAGES[0] == "reconstruct"
    assert STAGES[4] == "electrodes"
    assert "synthesize" in STAGES
    assert STAGES[-1] == "simulate"
