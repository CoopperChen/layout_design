"""
Side test: load a Wavefront OBJ (+ MTL / JPG) and print texture diagnostics.

Usage (from repo root):
  python scripts/test_read_obj.py
  python scripts/test_read_obj.py --obj data/raw/7.obj
  python scripts/test_read_obj.py --obj data/raw/7.obj --no-view
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Test-load textured OBJ (MTL/JPG)")
    p.add_argument(
        "--obj",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "7.obj",
        help="Path to .obj (default: data/raw/7.obj)",
    )
    p.add_argument(
        "--no-view",
        action="store_true",
        help="Skip Open3D viewer window",
    )
    args = p.parse_args(argv)

    obj_path = args.obj if args.obj.is_absolute() else REPO_ROOT / args.obj
    print(f"OBJ: {obj_path}")
    print(f"  exists: {obj_path.is_file()}")
    if not obj_path.is_file():
        return 1

    from app.preprocess.mesh_io import (
        discover_obj_texture_images,
        load_textured_obj_mesh,
        obj_has_texcoords,
        parse_map_kd_names,
        parse_mtllib_names,
    )

    mtl_names = parse_mtllib_names(obj_path)
    print(f"  mtllib lines: {mtl_names or '(none in OBJ)'}")
    sidecar_mtl = obj_path.with_suffix(".mtl")
    print(f"  sidecar MTL:  {sidecar_mtl.name} exists={sidecar_mtl.is_file()}")
    if sidecar_mtl.is_file():
        print(f"  map_Kd:       {parse_map_kd_names(sidecar_mtl)}")
    print(f"  has vt (UV):  {obj_has_texcoords(obj_path)}")
    textures = discover_obj_texture_images(obj_path)
    print(f"  texture files:{[t.name for t in textures] or '(none found)'}")

    print("\nLoading via load_textured_obj_mesh …")
    try:
        mesh = load_textured_obj_mesh(obj_path)
    except ValueError as exc:
        print(f"\nERROR: {exc}")
        return 1
    n_v = len(mesh.vertices)
    n_t = len(mesh.triangles)
    print(f"  vertices:     {n_v}")
    print(f"  triangles:    {n_t}")
    print(f"  vertex colors:{mesh.has_vertex_colors()}")
    print(f"  triangle UVs: {mesh.has_triangle_uvs()} "
          f"(len={len(mesh.triangle_uvs) if mesh.has_triangle_uvs() else 0})")
    n_tex = len(getattr(mesh, "textures", []) or [])
    print(f"  textures:     {n_tex}")
    if mesh.has_vertex_colors() and n_v:
        c = np.asarray(mesh.vertex_colors, dtype=float)
        print(
            f"  color range:  "
            f"R[{c[:, 0].min():.3f},{c[:, 0].max():.3f}] "
            f"G[{c[:, 1].min():.3f},{c[:, 1].max():.3f}] "
            f"B[{c[:, 2].min():.3f},{c[:, 2].max():.3f}]"
        )
        uniq = len(np.unique(np.round(c, 3), axis=0))
        print(f"  unique colors:{uniq}")
        if uniq < 5:
            print(
                "\nERROR: mesh has essentially no color variation "
                f"(unique≈{uniq}). Texture did not apply."
            )
            return 1

    if args.no_view:
        return 0

    import open3d as o3d

    print("\nOpen3D viewer (close window to exit)…")
    mesh.compute_vertex_normals()
    # Classic draw_geometries often fails on MTL textures; show baked vertex colors.
    from app.preprocess.mesh_io import for_vertex_color_display

    view = for_vertex_color_display(mesh)
    try:
        # Prefer Filament viewer when available (handles textures better).
        o3d.visualization.draw(
            [mesh],
            title=f"test_read_obj — {obj_path.name}",
        )
    except Exception:
        o3d.visualization.draw_geometries(
            [view],
            window_name=f"test_read_obj — {obj_path.name}",
            width=1280,
            height=900,
            mesh_show_back_face=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
