"""Canonical path helpers."""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app import paths


def test_repo_root_is_parent_of_app():
    assert paths.APP_DIR == paths.REPO_ROOT / "app"
    assert paths.DATA_DIR == paths.REPO_ROOT / "data"


def test_subject_paths_use_stem():
    sid = 2
    assert paths.raw_scan(sid).name == "2.stl"
    assert paths.fiducials_json(sid).name == "fiducials_2.json"
    assert paths.synth_layout(sid).name == "synth_s2.json"
    assert paths.smooth_json(sid, "final").name == "smooth_s2_final.json"


def test_preset_relative_name():
    p = paths.preset_path("reference_v4")
    assert p.parent == paths.DATA_DIR / "presets"
    assert p.suffix == ".json"


if __name__ == "__main__":
    test_repo_root_is_parent_of_app()
    test_subject_paths_use_stem()
    test_preset_relative_name()
    print("OK")
