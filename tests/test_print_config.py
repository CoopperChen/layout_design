"""pm-only print config and convert-gcode resolution."""

from pathlib import Path

import numpy as np
import pytest

from app import paths
from app.postprocess.convert_gcode import convert_gcode
from app.postprocess.print_config import (
    init_print_config,
    load_physical_landmarks,
    resolve_pm_config,
)

FIXTURES = Path(__file__).parent / "fixtures"
GOLDEN_BUNDLE = FIXTURES / "synthetic_bundle"
SYNTHETIC_PM = paths.postprocessor_config_dir() / "subjects" / "subject_synthetic.yaml"


def test_init_print_config_creates_pm_only_yaml(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        paths,
        "postprocessor_subject_pm",
        lambda sid: tmp_path / f"subject_{sid}.yaml",
    )
    out = init_print_config(42)
    assert out.is_file()
    text = out.read_text(encoding="utf-8")
    assert "physical_landmarks_mm" in text
    assert "process:" not in text
    pm = load_physical_landmarks(out, require_measured=False)
    assert pm.shape == (3, 3)


def test_init_print_config_refuses_overwrite(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        paths,
        "postprocessor_subject_pm",
        lambda sid: tmp_path / f"subject_{sid}.yaml",
    )
    init_print_config(7)
    with pytest.raises(FileExistsError):
        init_print_config(7)


def test_load_scaffold_rejects_when_require_measured():
    from app.postprocess.print_config import validate_landmark_triangle

    with pytest.raises(ValueError, match="empty scaffold"):
        validate_landmark_triangle(np.zeros((3, 3)))


def test_load_physical_landmarks_rejects_scaffold(tmp_path: Path):
    path = tmp_path / "subject_0.yaml"
    path.write_text(
        "physical_landmarks_mm:\n  - [0, 0, 0]\n  - [0, 0, 0]\n  - [0, 0, 0]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="empty scaffold|record-pm"):
        load_physical_landmarks(path)


def test_pm_is_measured(tmp_path: Path):
    from app.postprocess.print_config import pm_is_measured

    missing = tmp_path / "missing.yaml"
    assert pm_is_measured(missing) is False

    scaffold = tmp_path / "scaffold.yaml"
    scaffold.write_text(
        "physical_landmarks_mm:\n  - [0, 0, 0]\n  - [0, 0, 0]\n  - [0, 0, 0]\n",
        encoding="utf-8",
    )
    assert pm_is_measured(scaffold) is False

    assert pm_is_measured(SYNTHETIC_PM) is True


def test_load_synthetic_pm():
    pm = load_physical_landmarks(SYNTHETIC_PM)
    assert pm.shape == (3, 3)
    assert pm[0, 0] == 0


def test_resolve_pm_from_explicit_file():
    p = resolve_pm_config(GOLDEN_BUNDLE, pm_file=SYNTHETIC_PM)
    assert p == SYNTHETIC_PM.resolve()


def test_resolve_pm_auto_from_bundle_subject_id():
    p = resolve_pm_config(GOLDEN_BUNDLE)
    assert p.name == "subject_synthetic.yaml"


def test_convert_gcode_default_writes_both_trace_types(tmp_path: Path):
    out = convert_gcode(GOLDEN_BUNDLE, output=tmp_path / "gcode")
    assert isinstance(out, list)
    names = {p.name for p in out}
    assert names == {"allinterconnects.txt", "allelectrode.txt"}
    for p in out:
        assert p.read_text(encoding="utf-8").startswith("G94")


def test_convert_gcode_single_channel_both_traces(tmp_path: Path):
    out = convert_gcode(
        GOLDEN_BUNDLE,
        output=tmp_path / "gcode",
        electrode="C3",
    )
    assert isinstance(out, list)
    names = {p.name for p in out}
    assert names == {"C3interconnect.txt", "C3electrode.txt"}


def test_convert_gcode_interconnect_only(tmp_path: Path):
    out = convert_gcode(
        GOLDEN_BUNDLE,
        pm_file=SYNTHETIC_PM,
        output=tmp_path / "gcode",
        trace="interconnect",
        electrode="all",
    )
    assert isinstance(out, Path)
    assert out.name == "allinterconnects.txt"
    assert out.read_text(encoding="utf-8").startswith("G94")
