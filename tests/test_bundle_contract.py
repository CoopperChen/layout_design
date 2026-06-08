"""Subject-agnostic bundle schema contract tests."""

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.postprocess.bundle.emit import export_bundle
from app.postprocess.bundle.load import load_bundle
from app.postprocess.bundle.schema import SCHEMA_VERSION
from app.postprocess.gcode.converter import convert_to_gcode, run_conversion
from app.postprocess.gcode.io.load_bundle import load_bundle as load_gcode_bundle
from app.postprocess.gcode.models import JobConfig, MachineConfig
from tests.fixtures.bundle_factory import (
    SYNTHETIC_SUBJECT_ID,
    write_golden_bundle,
    write_synthetic_fiducials,
    write_synthetic_mesh,
    write_synthetic_smooth,
)

FIXTURES = Path(__file__).parent / "fixtures"
GOLDEN_BUNDLE = FIXTURES / "synthetic_bundle"


@pytest.fixture(scope="session", autouse=True)
def ensure_golden_bundle():
    if not (GOLDEN_BUNDLE / "manifest.json").is_file():
        write_golden_bundle(GOLDEN_BUNDLE)


def test_golden_bundle_schema():
    bundle = load_bundle(GOLDEN_BUNDLE)
    assert bundle.schema_version == SCHEMA_VERSION
    assert bundle.landmarks_xyz.shape == (3, 3)
    assert len(bundle.channels) == 1
    ch = bundle.channels[0]
    assert ch.interconnect.shape[1] == 6
    assert ch.electrode.shape[1] == 6
    assert np.all(np.isfinite(ch.interconnect))
    assert np.all(np.isfinite(ch.electrode))


def test_gcode_loader_accepts_golden_bundle():
    bundle = load_gcode_bundle(GOLDEN_BUNDLE)
    assert bundle.schema_version == SCHEMA_VERSION
    assert len(bundle.channels) == 1


def test_gcode_convert_golden_bundle(tmp_path: Path):
    bundle = load_gcode_bundle(GOLDEN_BUNDLE)
    machine = MachineConfig()
    job = JobConfig(
        subject="synthetic",
        physical_landmarks_mm=np.array(
            [[0, 0, 0], [10, 0, 0], [0, 10, 0]],
            dtype=float,
        ),
        trace_type="interconnect",
        print_mode="all",
    )
    merged, names = convert_to_gcode(bundle, machine, job)
    assert merged.shape[1] == 7
    assert len(names) == 1

    out = run_conversion(bundle, machine, job, output_base=tmp_path / "gcode")
    assert out.is_file()
    assert out.read_text(encoding="utf-8").startswith("G94")


def test_export_bundle_from_synthetic_smooth(tmp_path: Path, monkeypatch):
    synth_dir = tmp_path / "synthetic"
    synth_dir.mkdir()
    write_synthetic_mesh(synth_dir / "99.stl")
    write_synthetic_smooth(synth_dir / "smooth_s99_final.json")
    write_synthetic_fiducials(synth_dir / "fiducials.json")

    monkeypatch.setattr(
        "app.preprocess.fiducials_io.fiducials_path",
        lambda sid: synth_dir / "fiducials.json",
    )

    out = export_bundle(
        synth_dir / "smooth_s99_final.json",
        tmp_path / "bundle_out",
        strict_landmarks=True,
    )
    bundle = load_bundle(out)
    assert bundle.schema_version == SCHEMA_VERSION
    assert bundle.subject_id == SYNTHETIC_SUBJECT_ID
    assert len(bundle.channels) == 2
    for ch in bundle.channels:
        assert ch.interconnect.shape[1] == 6
        assert ch.electrode.shape[1] == 6
