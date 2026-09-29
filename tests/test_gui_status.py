"""Stage readiness and run argv for the desktop control panel."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from app import paths
from app.gui.status import run_argv, stage_statuses
from app.pipeline.run import STAGES


def _redirect_data(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(paths, "CONFIG_DIR", tmp_path / "config")


def _touch(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_stage_statuses_missing_when_data_tree_empty(tmp_path: Path, monkeypatch):
    _redirect_data(monkeypatch, tmp_path)
    rows = {row.name: row for row in stage_statuses(7)}
    assert list(rows) == list(STAGES)
    assert all(not row.ready for row in rows.values())
    assert rows["reconstruct"].artifact == tmp_path / "data" / "raw" / "7.stl"
    assert rows["clear-islands"].artifact == tmp_path / "data" / "cleaned_scans" / "7.stl"
    assert rows["align-obj"].artifact == tmp_path / "data" / "cleaned_scans" / "7.obj"
    assert rows["fiducials"].artifact == tmp_path / "data" / "json" / "fiducials_7.json"
    assert rows["cz"].artifact == tmp_path / "data" / "json" / "Cz_7.json"
    assert rows["electrodes"].artifact == tmp_path / "data" / "json" / "electrode_positions_7.json"
    assert rows["synthesize"].artifact == tmp_path / "data" / "output" / "layouts" / "synth_s7.json"
    assert rows["polish"].artifact == (
        tmp_path / "data" / "output" / "layouts" / "synth_s7_repaired.json"
    )
    assert (
        rows["smooth"].artifact == tmp_path / "data" / "output" / "smooth" / "smooth_s7_final.json"
    )
    assert rows["bundle"].artifact == tmp_path / "data" / "output" / "bundles" / "subject_7"
    assert rows["gcode"].artifact == (
        tmp_path / "data" / "output" / "gcode" / "subject_7_post" / "allinterconnects.txt"
    )
    assert rows["simulate"].artifact == rows["gcode"].artifact
    assert rows["reconstruct"].interactive
    assert not rows["synthesize"].interactive
    assert rows["polish"].label == "missing"


def test_stage_statuses_ready_when_artifacts_exist(tmp_path: Path, monkeypatch):
    _redirect_data(monkeypatch, tmp_path)
    rows_missing = stage_statuses(7)
    for row in rows_missing:
        if row.name == "bundle":
            row.artifact.mkdir(parents=True)
        elif row.name == "record-pm":
            continue
        elif row.name != "simulate":
            _touch(row.artifact)
    measured = tmp_path / "config" / "postprocessor" / "subjects" / "subject_7.yaml"
    _touch(
        measured,
        "physical_landmarks_mm:\n  - [0, 0, 0]\n  - [80, 0, 0]\n  - [0, 60, 0]\n",
    )

    rows = {row.name: row for row in stage_statuses(7)}
    assert all(row.ready for row in rows.values())
    assert rows["record-pm"].label == "ready"
    assert rows["simulate"].ready


def test_print_config_scaffold_is_not_measured(tmp_path: Path, monkeypatch):
    _redirect_data(monkeypatch, tmp_path)
    scaffold = tmp_path / "config" / "postprocessor" / "subjects" / "subject_7.yaml"
    _touch(
        scaffold,
        "physical_landmarks_mm:\n  - [0, 0, 0]\n  - [0, 0, 0]\n  - [0, 0, 0]\n",
    )
    rows = {row.name: row for row in stage_statuses(7)}
    assert rows["print-config"].ready
    assert not rows["record-pm"].ready
    assert rows["record-pm"].label == "missing"


def test_polish_off_marks_polish_skipped(tmp_path: Path, monkeypatch):
    _redirect_data(monkeypatch, tmp_path)
    repaired = tmp_path / "data" / "output" / "layouts" / "synth_s7_repaired.json"
    _touch(repaired)
    row = {r.name: r for r in stage_statuses(7, polish=False)}["polish"]
    assert row.skipped
    assert not row.ready
    assert row.label == "skipped"


def test_run_argv_from_synthesize_to_gcode():
    argv = run_argv(
        target=2,
        from_stage="synthesize",
        to_stage="gcode",
        polish=True,
        visualize=True,
        rotate=False,
    )
    assert argv[0] == sys.executable
    assert argv[1:6] == ["-m", "app", "run", "--target", "2"]
    assert argv[argv.index("--from") + 1] == "synthesize"
    assert argv[argv.index("--to") + 1] == "gcode"
    assert "--no-polish" not in argv
    assert "--no-visualize" not in argv
    assert "--rotate" not in argv


def test_run_argv_single_stage_with_flags():
    argv = run_argv(
        target=3,
        from_stage="smooth",
        to_stage="smooth",
        polish=False,
        visualize=False,
        rotate=True,
    )
    assert argv[argv.index("--target") + 1] == "3"
    assert argv[argv.index("--from") + 1] == "smooth"
    assert argv[argv.index("--to") + 1] == "smooth"
    assert "--no-polish" in argv
    assert "--no-visualize" in argv
    assert "--rotate" in argv


def test_run_argv_rejects_inverted_range():
    with pytest.raises(ValueError, match="after"):
        run_argv(target=2, from_stage="gcode", to_stage="synthesize")


def test_gui_backend_message_points_at_this_interpreter(monkeypatch):
    from app.cli import gui_backend_message

    monkeypatch.setattr(sys, "executable", r"D:\Research\layout_design\.venv\Scripts\python.exe")
    missing = gui_backend_message(
        ModuleNotFoundError("No module named 'PySide6'"),
        installed_at=None,
    )
    assert r"D:\Research\layout_design\.venv\Scripts\python.exe" in missing
    assert ' -m pip install -e ".[gui]"' in missing
    assert "No module named 'PySide6'" in missing

    loaded = gui_backend_message(
        ImportError("DLL load failed while importing QtWidgets"),
        installed_at=r"D:\Research\layout_design\.venv\Lib\site-packages\PySide6\__init__.py",
    )
    assert "package location" in loaded
    assert "DLL load failed" in loaded
    assert "pip install" not in loaded


def test_gui_parser_accepts_subject():
    from app.cli import build_parser

    args = build_parser().parse_args(["gui", "--subject", "4"])
    assert args.subject == 4
    assert args.func.__name__ == "cmd_gui"


def test_run_argv_rejects_polish_only_when_polish_off():
    with pytest.raises(ValueError, match="polish is off"):
        run_argv(
            target=2,
            from_stage="polish",
            to_stage="polish",
            polish=False,
        )
