"""Synthetic Stage D integration: smooth JSON → bundle → G-code."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.postprocess.bundle.load import load_bundle
from app.postprocess.convert_gcode import convert_gcode
from app.postprocess.export_matlab_legacy import export_to_matlab_format
from app.postprocess.bundle.emit import export_bundle
from tests.fixtures.bundle_factory import (
    write_synthetic_fiducials,
    write_synthetic_mesh,
    write_synthetic_smooth,
)


@pytest.fixture
def synthetic_stage_d(tmp_path: Path, monkeypatch):
    synth_dir = tmp_path / "synthetic"
    synth_dir.mkdir()
    write_synthetic_mesh(synth_dir / "99.stl")
    write_synthetic_smooth(synth_dir / "smooth_s99_final.json")
    write_synthetic_fiducials(synth_dir / "fiducials.json")

    monkeypatch.setattr(
        "app.preprocess.fiducials_io.fiducials_path",
        lambda sid: synth_dir / "fiducials.json",
    )
    return synth_dir


def test_validate_then_export_bundle(synthetic_stage_d: Path, tmp_path: Path):
    from app.postprocess.validate_export import validate_smooth_file

    smooth = synthetic_stage_d / "smooth_s99_final.json"
    validate_smooth_file(smooth)

    out = export_bundle(smooth, tmp_path / "bundle", verbose=False)
    bundle = load_bundle(out)
    assert len(bundle.channels) == 2


def test_export_bundle_then_convert_gcode(synthetic_stage_d: Path, tmp_path: Path):
    smooth = synthetic_stage_d / "smooth_s99_final.json"
    bundle_dir = tmp_path / "bundle"
    export_bundle(smooth, bundle_dir, verbose=False)

    pm = _REPO_ROOT / "config/postprocessor/subjects/subject_synthetic.yaml"
    gcode_out = tmp_path / "gcode"
    paths = convert_gcode(
        bundle_dir,
        pm,
        output=gcode_out,
        trace="both",
    )
    assert isinstance(paths, list)
    assert len(paths) == 2
    for path in paths:
        assert path.is_file()
        assert path.read_text(encoding="utf-8").startswith("G94")


def test_mesh_context_cache_reused_across_exports(synthetic_stage_d: Path, tmp_path: Path):
    from app.postprocess.mesh_export import clear_mesh_context_cache, load_mesh_context

    smooth = synthetic_stage_d / "smooth_s99_final.json"
    mesh = synthetic_stage_d / "99.stl"

    clear_mesh_context_cache()
    export_bundle(smooth, tmp_path / "bundle", verbose=False)
    export_to_matlab_format(
        str(smooth),
        str(tmp_path / "matlab"),
        verbose=False,
    )

    ctx = load_mesh_context(mesh)
    assert ctx.mesh.n_points > 0
