"""Tests for Wavefront OBJ + MTL + image texture loading."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.preprocess.mesh_io import (
    discover_obj_texture_images,
    load_textured_obj_mesh,
    obj_has_texcoords,
    write_transformed_wavefront_obj,
)


def _require_open3d():
    pytest.importorskip("open3d")
    import open3d as o3d

    return o3d


def _write_tiny_textured_obj(tmp: Path, *, map_kd: str = "textured_output.jpg") -> Path:
    """OBJ with UVs + MTL whose map_Kd name does not match the real jpg stem."""
    o3d = _require_open3d()
    obj = tmp / "head.obj"
    mtl = tmp / "head.mtl"
    # Real image on disk uses subject-style name; MTL points at a different name.
    jpg = tmp / "head.jpg"

    img = np.zeros((16, 16, 3), dtype=np.uint8)
    img[..., 0] = 220
    img[..., 1] = 40
    img[..., 2] = 40
    o3d.io.write_image(str(jpg), o3d.geometry.Image(img))

    mtl.write_text(
        "newmtl material_0\n"
        "Kd 1 1 1\n"
        f"map_Kd {map_kd}\n",
        encoding="utf-8",
    )
    # Unit square, two tris, simple UVs
    obj.write_text(
        "mtllib head.mtl\n"
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 1 1 0\n"
        "v 0 1 0\n"
        "vt 0 0\n"
        "vt 1 0\n"
        "vt 1 1\n"
        "vt 0 1\n"
        "usemtl material_0\n"
        "f 1/1 2/2 3/3\n"
        "f 1/1 3/3 4/4\n",
        encoding="utf-8",
    )
    return obj


def test_discover_texture_falls_back_to_stem_jpg(tmp_path: Path):
    obj = _write_tiny_textured_obj(tmp_path, map_kd="textured_output.jpg")
    found = discover_obj_texture_images(obj)
    assert any(p.name == "head.jpg" for p in found)
    assert obj_has_texcoords(obj)


def test_load_textured_obj_bakes_mismatched_map_kd(tmp_path: Path):
    obj = _write_tiny_textured_obj(tmp_path, map_kd="textured_output.jpg")
    mesh = load_textured_obj_mesh(obj)
    assert mesh.has_vertex_colors()
    colors = np.asarray(mesh.vertex_colors)
    # Baked from red-ish texture
    assert float(colors[:, 0].mean()) > float(colors[:, 1].mean())


def test_write_transformed_wavefront_preserves_mtllib(tmp_path: Path):
    obj = _write_tiny_textured_obj(tmp_path)
    dest = tmp_path / "out" / "head.obj"
    T = np.eye(4)
    T[0, 3] = 10.0
    write_transformed_wavefront_obj(obj, T, dest)
    text = dest.read_text(encoding="utf-8")
    assert "mtllib" in text
    assert "vt " in text
    assert (tmp_path / "out" / "head.mtl").is_file()
    assert (tmp_path / "out" / "head.jpg").is_file()
    # First vertex moved by +10 in X
    v0 = [line for line in text.splitlines() if line.startswith("v ")][0].split()
    assert abs(float(v0[1]) - 10.0) < 1e-5
