"""Import a differently named PLY and a texture zip into the subject folder."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from app import paths
from app.mesh.import_sources import ensure_subject_sources


def _write_zip(path: Path) -> None:
    obj = "mtllib materials.mtl\nv 0 0 0\nvt 0 0\n"
    mtl = "newmtl skin\nKd 1 1 1\nmap_Kd scans/color.JPG\n"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("export/Patient Model.obj", obj)
        archive.writestr("export/materials.mtl", mtl)
        archive.writestr("export/color.JPG", b"\xff\xd8\xff jpeg")


def test_zip_is_renamed_into_source(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path)
    ply = tmp_path / "incoming" / "cloud_01.ply"
    ply.parent.mkdir()
    ply.write_bytes(b"ply\n")
    package = tmp_path / "incoming" / "head_texture.zip"
    _write_zip(package)

    dest_ply, dest_obj = ensure_subject_sources(1, ply_path=ply, zip_path=package)

    assert dest_ply == tmp_path / "subjects" / "1" / "source" / "scan.ply"
    assert dest_ply.read_bytes() == b"ply\n"
    assert dest_obj.read_text(encoding="utf-8").splitlines()[0] == "mtllib head.mtl"
    mtl = (dest_obj.parent / "head.mtl").read_text(encoding="utf-8")
    assert "map_Kd head.jpg" in mtl
    assert (dest_obj.parent / "head.jpg").read_bytes() == b"\xff\xd8\xff jpeg"
    assert not (dest_obj.parent / "color.JPG").exists()


def test_ready_sources_skip_the_file_dialog(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path)
    source = tmp_path / "subjects" / "1" / "source"
    source.mkdir(parents=True)
    (source / "scan.ply").write_bytes(b"ply")
    (source / "head.obj").write_text("mtllib head.mtl\n", encoding="utf-8")
    (source / "head.mtl").write_text("map_Kd head.jpg\n", encoding="utf-8")
    (source / "head.jpg").write_bytes(b"jpg")

    def _refuse():
        raise AssertionError("dialog should not open")

    dest_ply, dest_obj = ensure_subject_sources(1, pick_ply=_refuse, pick_zip=_refuse)
    assert dest_ply.is_file()
    assert dest_obj.is_file()


def test_zip_with_two_objs_is_rejected(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path)
    package = tmp_path / "two.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("a.obj", "mtllib a.mtl\n")
        archive.writestr("b.obj", "mtllib a.mtl\n")
        archive.writestr("a.mtl", "map_Kd a.jpg\n")
        archive.writestr("a.jpg", b"jpg")
    ply = tmp_path / "scan.ply"
    ply.write_bytes(b"ply")
    with pytest.raises(ValueError, match="one .obj"):
        ensure_subject_sources(1, ply_path=ply, zip_path=package)
