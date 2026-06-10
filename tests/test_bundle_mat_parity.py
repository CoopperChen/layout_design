"""Bundle export should match legacy .mat export on the same synthetic smooth JSON."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.postprocess.bundle.emit import export_bundle
from app.postprocess.bundle.load import load_bundle
from app.postprocess.export_matlab_legacy import export_to_matlab_format
from app.postprocess.gcode.io.load_mat import load_mat_subject
from tests.fixtures.bundle_factory import (
    write_synthetic_fiducials,
    write_synthetic_mesh,
    write_synthetic_smooth,
)


@pytest.fixture
def synthetic_export_dirs(tmp_path: Path, monkeypatch):
    synth_dir = tmp_path / "synthetic"
    synth_dir.mkdir()
    write_synthetic_mesh(synth_dir / "99.stl")
    write_synthetic_smooth(synth_dir / "smooth_s99_final.json")
    write_synthetic_fiducials(synth_dir / "fiducials.json")

    monkeypatch.setattr(
        "app.preprocess.fiducials_io.fiducials_path",
        lambda sid: synth_dir / "fiducials.json",
    )

    smooth = synth_dir / "smooth_s99_final.json"
    bundle_dir = tmp_path / "bundle"
    mat_dir = tmp_path / "matlab"

    export_bundle(smooth, bundle_dir, verbose=False)
    export_to_matlab_format(str(smooth), str(mat_dir), verbose=False)

    return bundle_dir, mat_dir


def test_bundle_mat_channel_count(synthetic_export_dirs):
    bundle_dir, mat_dir = synthetic_export_dirs
    bundle = load_bundle(bundle_dir)
    mat_bundle = load_mat_subject(mat_dir)
    assert len(bundle.channels) == len(mat_bundle.channels)


def test_bundle_mat_trace_parity(synthetic_export_dirs):
    bundle_dir, mat_dir = synthetic_export_dirs
    bundle = load_bundle(bundle_dir)
    mat_bundle = load_mat_subject(mat_dir)

    bundle_by_name = {ch.name: ch for ch in bundle.channels}
    mat_by_name = {ch.name: ch for ch in mat_bundle.channels}
    assert bundle_by_name.keys() == mat_by_name.keys()

    for name in bundle_by_name:
        b_ch = bundle_by_name[name]
        m_ch = mat_by_name[name]
        assert b_ch.interconnect.shape == m_ch.interconnect.shape
        assert b_ch.electrode.shape == m_ch.electrode.shape
        np.testing.assert_allclose(b_ch.interconnect, m_ch.interconnect, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(b_ch.electrode, m_ch.electrode, rtol=1e-5, atol=1e-5)


def test_bundle_mat_landmarks_parity(synthetic_export_dirs):
    bundle_dir, mat_dir = synthetic_export_dirs
    bundle = load_bundle(bundle_dir)
    mat_bundle = load_mat_subject(mat_dir)
    np.testing.assert_allclose(bundle.landmarks_xyz, mat_bundle.landmarks_xyz, rtol=1e-5, atol=1e-5)
