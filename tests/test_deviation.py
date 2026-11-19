"""Printed-head overlap and electrode deviation. Outside the pipeline."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from app import paths
from app.cli import build_parser
from app.deviation.import_printed import ensure_printed_sources
from app.deviation.measure import (
    apply_transform,
    build_document,
    electrode_record,
    save_document,
    summary_record,
)
from app.deviation.overlap import Overlap, initial_similarity
from app.pipeline.run import STAGES


def _overlap() -> Overlap:
    return Overlap(
        transform=np.eye(4),
        scale=1.0,
        fitness=0.9,
        mean_dist_mm=0.4,
        init="identity",
        source="printed scan",
    )


def test_apply_transform_translates_and_rotates():
    shift = np.eye(4)
    shift[:3, 3] = [1.0, 2.0, 3.0]
    assert np.allclose(apply_transform([[0.0, 0.0, 0.0]], shift), [[1.0, 2.0, 3.0]])

    yaw = np.eye(4)
    yaw[:3, :3] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    assert np.allclose(apply_transform([[1.0, 0.0, 0.0]], yaw), [[0.0, 1.0, 0.0]])


def test_electrode_record_is_printed_minus_planned():
    row = electrode_record([0.0, 0.0, 0.0], [3.0, 4.0, 0.0])
    assert row["distance_mm"] == 5.0
    assert row["delta_mm"] == [3.0, 4.0, 0.0]


def test_summary_reports_mean_rms_and_max():
    rows = {
        "Cz": {"distance_mm": 3.0},
        "C3": {"distance_mm": 4.0},
    }
    summary = summary_record(rows)
    assert summary["count"] == 2
    assert summary["mean_mm"] == 3.5
    assert summary["rms_mm"] == round((12.5) ** 0.5, 4)
    assert summary["max_name"] == "C3"
    assert summary["max_mm"] == 4.0


def test_document_requires_every_planned_electrode():
    with pytest.raises(ValueError, match="missing"):
        build_document(2, _overlap(), {"Cz": [0, 0, 0], "C3": [1, 0, 0]}, {"Cz": [0, 0, 1]})


def test_save_deviation_writes_the_subject_file(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path)
    document = build_document(4, _overlap(), {"Cz": [0.0, 0.0, 0.0]}, {"Cz": [0.0, 3.0, 4.0]})
    path = save_document(4, document)
    assert path == tmp_path / "subjects" / "4" / "deviation.json"
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["summary"]["max_name"] == "Cz"
    assert loaded["electrodes"]["Cz"]["distance_mm"] == 5.0
    assert loaded["delta_definition"].startswith("printed_mm - planned_mm")


def test_initial_similarity_scales_metres_into_millimetres():
    src = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.1, 0.05]])
    tgt = src * 1000.0
    transform, scale = initial_similarity(src, tgt, match_scale=True)
    assert scale == pytest.approx(1000.0)
    assert np.allclose(apply_transform(src, transform), tgt, atol=1e-4)


def test_initial_similarity_keeps_a_small_size_difference():
    src = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 40.0, 10.0]])
    tgt = src * 1.02 + np.array([5.0, 0.0, 0.0])
    transform, scale = initial_similarity(src, tgt, match_scale=True)
    assert scale == pytest.approx(1.0)
    assert np.allclose(apply_transform(src.mean(axis=0), transform), tgt.mean(axis=0), atol=1e-6)


def test_initial_similarity_can_refuse_scale():
    src = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.1, 0.05]])
    tgt = src * 1000.0
    _transform, scale = initial_similarity(src, tgt, match_scale=False)
    assert scale == pytest.approx(1.0)


def test_printed_import_stays_out_of_the_pipeline_source(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path)
    ply = tmp_path / "incoming" / "print.ply"
    ply.parent.mkdir()
    ply.write_bytes(b"ply\n")
    package = tmp_path / "incoming" / "print.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("scan/model.obj", "mtllib materials.mtl\nv 0 0 0\nvt 0 0\n")
        archive.writestr("scan/materials.mtl", "newmtl skin\nmap_Kd color.JPG\n")
        archive.writestr("scan/color.JPG", b"\xff\xd8\xff jpeg")

    dest_ply, dest_obj = ensure_printed_sources(2, ply_path=ply, zip_path=package)

    assert dest_ply == tmp_path / "subjects" / "2" / "printed" / "source" / "scan.ply"
    assert dest_ply.read_bytes() == b"ply\n"
    assert dest_obj.parent == dest_ply.parent
    assert "mtllib head.mtl" in dest_obj.read_text(encoding="utf-8")
    assert not (tmp_path / "subjects" / "2" / "source" / "scan.ply").exists()


def test_deviation_command_is_outside_the_pipeline():
    args = build_parser().parse_args(["deviation", "--subject", "2", "--no-scale"])
    assert args.func.__name__ == "cmd_deviation"
    assert args.no_scale is True
    assert "deviation" not in STAGES


def test_align_geometry_recovers_a_translation():
    o3d = pytest.importorskip("open3d")
    from app.deviation.overlap import align_geometry

    def tetra(shift):
        mesh = o3d.geometry.TriangleMesh()
        verts = np.array(
            [
                [0.0, 0.0, 0.0],
                [90.0, 0.0, 0.0],
                [12.0, 80.0, 0.0],
                [8.0, 18.0, 70.0],
            ],
            dtype=float,
        ) + np.asarray(shift, dtype=float)
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(
            np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32)
        )
        mesh.compute_vertex_normals()
        return mesh

    target = tetra([0.0, 0.0, 0.0])
    source = tetra([30.0, -12.0, 8.0])
    transform, _fitness, _init, _mean = align_geometry(
        source,
        target,
        match_scale=True,
        n_samples=2000,
        max_correspondence_mm=15.0,
        max_iteration=30,
    )
    moved = apply_transform(np.asarray(source.vertices), transform)
    assert np.allclose(moved, np.asarray(target.vertices), atol=2.0)
