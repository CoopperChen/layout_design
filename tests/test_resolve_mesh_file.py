"""Mesh path resolution from smooth JSON."""

from __future__ import annotations

from pathlib import Path

import pytest

from app import paths
from app.postprocess.export_matlab_legacy import resolve_mesh_file


def test_resolve_mesh_file_repo_relative(tmp_path):
    mesh = paths.DATA_DIR / "cleaned_scans" / "4.stl"
    if not mesh.is_file():
        pytest.skip("subject 4 cleaned scan not present")

    smooth_json = paths.REPO_ROOT / "data/output/smooth/smooth_s4_final.json"
    if not smooth_json.is_file():
        pytest.skip("smooth_s4_final.json not present")

    resolved = resolve_mesh_file(str(smooth_json), "data/cleaned_scans/4.stl")
    assert Path(resolved).resolve() == mesh.resolve()


def test_resolve_mesh_file_next_to_json(tmp_path):
    mesh = tmp_path / "head.stl"
    mesh.write_bytes(b"solid x\nendsolid\n")
    smooth_json = tmp_path / "smooth.json"
    smooth_json.write_text("{}", encoding="utf-8")

    resolved = resolve_mesh_file(str(smooth_json), "head.stl")
    assert Path(resolved).resolve() == mesh.resolve()
