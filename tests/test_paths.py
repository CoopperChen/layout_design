"""Canonical subject paths."""

from app import paths


def test_repo_root_is_parent_of_app():
    assert paths.APP_DIR == paths.REPO_ROOT / "app"
    assert paths.DATA_DIR == paths.REPO_ROOT / "data"


def test_subject_folder_layout():
    sid = 2
    root = paths.subject_dir(sid)
    assert root.name == "2"
    assert paths.source_ply(sid) == root / "source" / "scan.ply"
    assert paths.source_obj(sid) == root / "source" / "head.obj"
    assert paths.head_stl(sid) == root / "head.stl"
    assert paths.head_obj(sid) == root / "head.obj"
    assert paths.layout_json(sid).name == "layout.json"
    assert paths.pm_yaml(sid).name == "pm.yaml"
    assert paths.electrodes_gcode(sid).name == "electrodes.gcode"
    assert paths.machine_config().name == "machine.yaml"


def test_paths_command_lists_every_artifact():
    labels = [label for label, _path in paths.subject_artifacts(3)]
    assert labels == [
        "source ply",
        "source obj",
        "source mtl",
        "source jpg",
        "head stl",
        "head obj",
        "layout",
        "pm",
        "gcode",
    ]
