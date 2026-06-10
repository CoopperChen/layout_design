"""export-bundle: schema, landmarks gate, round-trip load."""

import json

import numpy as np
import pytest

from app import paths
from app.postprocess.bundle.emit import CalibrationLandmarksMissingError, export_bundle
from app.postprocess.bundle.load import load_bundle
from app.postprocess.bundle.schema import SCHEMA_VERSION


def test_export_bundle_rejects_missing_calibration_landmarks(tmp_path):
    smooth = paths.DATA_DIR / "output" / "smooth" / "smooth_s2_final.json"
    if not smooth.is_file():
        pytest.skip("smooth_s2_final.json not available")

    with pytest.raises(CalibrationLandmarksMissingError):
        export_bundle(smooth, tmp_path / "subject_2", strict_landmarks=True)


@pytest.mark.slow
def test_export_bundle_roundtrip_subject_4(tmp_path):
    smooth = paths.DATA_DIR / "output" / "smooth" / "smooth_s4_final.json"
    mesh = paths.DATA_DIR / "cleaned_scans" / "4.stl"
    fiducials = paths.fiducials_json(4)
    if not (smooth.is_file() and mesh.is_file() and fiducials.is_file()):
        pytest.skip("subject 4 smooth/mesh/fiducials not available")

    out = export_bundle(smooth, tmp_path / "subject_4")
    assert (out / "manifest.json").is_file()
    assert (out / "geometry.npz").is_file()
    assert (out / "traces.npz").is_file()

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["subject_id"] == 4
    assert manifest["channel_count"] > 0

    bundle = load_bundle(out)
    assert bundle.schema_version == SCHEMA_VERSION
    assert bundle.subject_id == 4
    assert bundle.landmarks_xyz.shape == (3, 3)
    assert len(bundle.channels) == manifest["channel_count"]
    for ch in bundle.channels:
        assert ch.interconnect.ndim == 2 and ch.interconnect.shape[1] == 6
        assert ch.electrode.ndim == 2 and ch.electrode.shape[1] == 6
        assert np.all(np.isfinite(ch.interconnect))
        assert np.all(np.isfinite(ch.electrode))
