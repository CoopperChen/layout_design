"""Head mesh I/O: Open3D ↔ PyVista, VTK-compatible OBJ export, OBJ+MTL+image load."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np


def _require_open3d():
    try:
        import open3d as o3d
    except ImportError as e:
        raise ImportError(
            "open3d is required for textured OBJ I/O. Install with: pip install open3d"
        ) from e
    return o3d


def vertex_colors_sidecar(obj_path: Path) -> Path:
    return obj_path.with_name(f"{obj_path.stem}_vertex_colors.npy")


def color_ref_path(obj_path: Path) -> Path:
    """Aligned vertex+color point cloud written with the OBJ (mesh coordinates)."""
    return obj_path.with_name(f"{obj_path.stem}_color_ref.ply")


def _colors_look_valid(colors: np.ndarray, n_verts: int) -> bool:
    if len(colors) != n_verts:
        return False
    if n_verts < 100:
        return True
    unique = len(np.unique(np.round(colors, 3), axis=0))
    min_unique = min(5000, max(200, n_verts // 100))
    return unique >= min_unique


_MTLLIB_RE = re.compile(r"^\s*mtllib\s+(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_MAP_KD_RE = re.compile(r"^\s*map_Kd\s+(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def obj_has_texcoords(obj_path: Path) -> bool:
    """True if the Wavefront OBJ declares at least one ``vt`` line."""
    path = Path(obj_path)
    if not path.is_file():
        return False
    with path.open("r", encoding="utf-8", errors="ignore") as fp:
        for line in fp:
            if line.lstrip().startswith("vt "):
                return True
    return False


def parse_mtllib_names(obj_path: Path) -> list[str]:
    text = Path(obj_path).read_text(encoding="utf-8", errors="ignore")
    return [m.group(1).strip().strip("\"'") for m in _MTLLIB_RE.finditer(text)]


def parse_map_kd_names(mtl_path: Path) -> list[str]:
    if not Path(mtl_path).is_file():
        return []
    text = Path(mtl_path).read_text(encoding="utf-8", errors="ignore")
    return [m.group(1).strip().strip("\"'") for m in _MAP_KD_RE.finditer(text)]


def _resolve_existing_image(base_dir: Path, name: str, *, obj_stem: str) -> Path | None:
    """Resolve a map_Kd (or fallback) path next to the OBJ."""
    raw = Path(name.replace("\\", "/"))
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(base_dir / raw)
        candidates.append(base_dir / raw.name)

    # Same basename, alternate extension
    for c in list(candidates):
        if c.suffix:
            for ext in _IMAGE_EXTS:
                candidates.append(c.with_suffix(ext))

    # Common drops: {stem}.jpg / {stem}_texture.png / textured_output.*
    for ext in _IMAGE_EXTS:
        candidates.append(base_dir / f"{obj_stem}{ext}")
        candidates.append(base_dir / f"{obj_stem}_texture{ext}")
        candidates.append(base_dir / f"textured_output{ext}")

    seen: set[Path] = set()
    for c in candidates:
        try:
            key = c.resolve()
        except OSError:
            key = c
        if key in seen:
            continue
        seen.add(key)
        if c.is_file():
            return c
    return None


def discover_obj_texture_images(obj_path: Path) -> list[Path]:
    """
    Find texture image files for an OBJ via ``mtllib`` / ``map_Kd``, with stem fallbacks.

    Example: MTL says ``map_Kd textured_output.jpg`` but the file is ``7.jpg`` —
    still resolves via ``{stem}.jpg``.
    """
    obj_path = Path(obj_path)
    base = obj_path.parent
    stem = obj_path.stem
    found: list[Path] = []
    seen: set[Path] = set()

    def _add(p: Path | None) -> None:
        if p is None or not p.is_file():
            return
        key = p.resolve()
        if key in seen:
            return
        seen.add(key)
        found.append(p)

    mtl_names = parse_mtllib_names(obj_path)
    if not mtl_names:
        # Sidecar default: {stem}.mtl
        default_mtl = base / f"{stem}.mtl"
        if default_mtl.is_file():
            mtl_names = [default_mtl.name]

    for mtl_name in mtl_names:
        mtl_path = base / Path(mtl_name.replace("\\", "/")).name
        if not mtl_path.is_file():
            mtl_path = base / mtl_name
        for map_name in parse_map_kd_names(mtl_path):
            _add(_resolve_existing_image(base, map_name, obj_stem=stem))

    # Always try stem-named images even if MTL is missing/wrong.
    for ext in _IMAGE_EXTS:
        _add(base / f"{stem}{ext}")
        _add(base / f"{stem}_texture{ext}")

    return found


def _image_is_nonempty(img) -> bool:
    """
    True if Open3D Image has nonzero size.

    Do **not** call ``np.asarray`` on empty images — that aborts the process on
    some Open3D builds (0×0 texture slots from the OBJ loader).
    """
    m = re.search(r"size\s+(\d+)x(\d+)", str(img), flags=re.IGNORECASE)
    if m:
        return int(m.group(1)) > 0 and int(m.group(2)) > 0
    return False


def sanitize_open3d_textures(mesh) -> int:
    """
    Drop empty 0×0 Open3D texture slots (common OBJ loader artifact) and remap
    ``triangle_material_ids``. Returns count of kept textures.
    """
    o3d = _require_open3d()
    textures = list(getattr(mesh, "textures", []) or [])
    if not textures:
        return 0

    keep_idx = [i for i, tex in enumerate(textures) if _image_is_nonempty(tex)]
    if not keep_idx:
        mesh.textures = []
        return 0
    if len(keep_idx) == len(textures):
        return len(textures)

    new_textures = [textures[i] for i in keep_idx]
    old_to_new = {old: new for new, old in enumerate(keep_idx)}
    mesh.textures = new_textures

    if mesh.has_triangle_material_ids():
        ids = np.asarray(mesh.triangle_material_ids, dtype=np.int64)
        remapped = np.array([old_to_new.get(int(i), 0) for i in ids], dtype=np.int32)
        mesh.triangle_material_ids = o3d.utility.IntVector(remapped.tolist())
    return len(new_textures)


def attach_disk_textures(mesh, obj_path: Path, *, replace: bool = False) -> int:
    """
    Manually load JPG/PNG textures referenced by MTL (or stem fallbacks) onto
    ``mesh.textures``.

    When ``replace`` is False (default), only runs if Open3D left textures empty.
    When ``replace`` is True, overwrites Open3D MTL textures — needed because
    Open3D's OBJ texture orientation is often V-flipped vs ``o3d.io.read_image``.
    """
    o3d = _require_open3d()
    if not replace:
        if sanitize_open3d_textures(mesh) > 0:
            return len(mesh.textures)

    images = discover_obj_texture_images(obj_path)
    if not images:
        return 0

    loaded = []
    for img_path in images:
        try:
            img = o3d.io.read_image(str(img_path))
        except Exception as exc:  # noqa: BLE001 — try next candidate
            print(f"Warning: failed to read texture {img_path.name}: {exc}")
            continue
        if not _image_is_nonempty(img):
            continue
        loaded.append(img)
        print(f"Loaded texture image → {img_path.name} ({np.asarray(img).shape})")

    if not loaded:
        return 0

    mesh.textures = loaded
    n_tris = len(mesh.triangles)
    if n_tris > 0:
        mesh.triangle_material_ids = o3d.utility.IntVector([0] * n_tris)
    return len(loaded)


def bake_texture_to_vertex_colors(mesh) -> bool:
    """
    Sample material textures onto vertices when vertex colors are missing.

    Handles multi-texture meshes via ``triangle_material_ids`` and skips empty
    texture slots. Tries both UV V orientations (Open3D MTL vs ``read_image``
    disagree) and keeps the one with more color variation.
    Returns True if the mesh ends with vertex colors.
    """
    o3d = _require_open3d()
    if mesh.has_vertex_colors() and _colors_look_valid(
        np.asarray(mesh.vertex_colors, dtype=float), len(mesh.vertices)
    ):
        return True

    sanitize_open3d_textures(mesh)
    textures = list(getattr(mesh, "textures", []) or [])
    if not textures or not mesh.has_triangle_uvs():
        return False

    uvs = np.asarray(mesh.triangle_uvs, dtype=float)
    if uvs.size == 0:
        return False
    if uvs.ndim == 2 and uvs.shape[1] == 2:
        uvs = uvs.reshape(-1, 3, 2)
    elif uvs.ndim != 3:
        return False

    tris = np.asarray(mesh.triangles, dtype=np.int64)
    n_verts = len(mesh.vertices)
    if len(uvs) != len(tris):
        return False

    tex_arrays: list[np.ndarray] = []
    for tex in textures:
        img = np.asarray(tex)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        if img.shape[-1] > 3:
            img = img[..., :3]
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        tex_arrays.append(img)

    if mesh.has_triangle_material_ids():
        mat_ids = np.asarray(mesh.triangle_material_ids, dtype=np.int64)
    else:
        mat_ids = np.zeros(len(tris), dtype=np.int64)

    def _bake(flip_v: bool) -> np.ndarray:
        accum = np.zeros((n_verts, 3), dtype=float)
        counts = np.zeros(n_verts, dtype=float)
        for ti, tri in enumerate(tris):
            mid = int(mat_ids[ti]) if ti < len(mat_ids) else 0
            mid = min(max(mid, 0), len(tex_arrays) - 1)
            img = tex_arrays[mid]
            h, w = img.shape[:2]
            for k in range(3):
                u, v = uvs[ti, k]
                x = int(np.clip(float(u), 0.0, 1.0) * (w - 1))
                vv = (1.0 - float(v)) if flip_v else float(v)
                y = int(np.clip(vv, 0.0, 1.0) * (h - 1))
                accum[tri[k]] += img[y, x] / 255.0
                counts[tri[k]] += 1.0
        missing = counts < 1.0
        counts[missing] = 1.0
        colors = accum / counts[:, None]
        colors[missing] = 0.7
        return np.clip(colors, 0.0, 1.0)

    c_flip = _bake(True)
    c_raw = _bake(False)
    score = lambda c: float(np.std(c) + c.mean())  # noqa: E731
    colors = c_flip if score(c_flip) >= score(c_raw) else c_raw
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    return True


def ensure_obj_material_paths(obj_path: Path) -> None:
    """
    Align Wavefront material filenames with files actually on disk.

    Scanner packages often ship ``7.obj`` + ``7.mtl`` + ``7.jpg`` while the OBJ
    still says ``mtllib textured_output.mtl`` / ``map_Kd textured_output.jpg``.
    Open3D then fails to load the JPG and the mesh looks untextured.
    """
    obj_path = Path(obj_path)
    base = obj_path.parent
    stem = obj_path.stem

    images = discover_obj_texture_images(obj_path)
    if not images:
        return
    img = images[0]

    # Prefer {stem}.mtl; create/update from any existing MTL text.
    dest_mtl = base / f"{stem}.mtl"
    src_mtl = None
    for name in parse_mtllib_names(obj_path):
        cand = base / Path(name.replace("\\", "/")).name
        if cand.is_file():
            src_mtl = cand
            break
    if src_mtl is None and dest_mtl.is_file():
        src_mtl = dest_mtl
    if src_mtl is None:
        # Minimal MTL so Open3D has a map_Kd target.
        dest_mtl.write_text(
            f"newmtl material_0\nKd 1 1 1\nmap_Kd {img.name}\n",
            encoding="utf-8",
        )
        src_mtl = dest_mtl
    else:
        text = src_mtl.read_text(encoding="utf-8", errors="ignore")
        if _MAP_KD_RE.search(text):
            text = _MAP_KD_RE.sub(f"map_Kd {img.name}", text, count=1)
        else:
            text = text.rstrip() + f"\nmap_Kd {img.name}\n"
        if src_mtl.resolve() != dest_mtl.resolve() or "map_Kd" in text:
            dest_mtl.write_text(text, encoding="utf-8")

    # Point OBJ mtllib at {stem}.mtl when the declared MTL is missing/wrong.
    obj_text = obj_path.read_text(encoding="utf-8", errors="ignore")
    declared = parse_mtllib_names(obj_path)
    need_rewrite = not declared or not (base / Path(declared[0].replace("\\", "/")).name).is_file()
    if need_rewrite or (declared and Path(declared[0]).name != dest_mtl.name):
        if _MTLLIB_RE.search(obj_text):
            obj_text = _MTLLIB_RE.sub(f"mtllib {dest_mtl.name}", obj_text, count=1)
        else:
            obj_text = f"mtllib {dest_mtl.name}\n" + obj_text
        obj_path.write_text(obj_text, encoding="utf-8")
        print(f"Updated {obj_path.name} mtllib → {dest_mtl.name}")
    if dest_mtl.is_file():
        print(f"Material map_Kd → {img.name} (via {dest_mtl.name})")


def load_textured_obj_mesh(obj_path: Path, *, prefer_disk_textures: bool = True):
    """
    Load Wavefront OBJ with MTL/JPG(PNG) textures for align-obj / fiducials.

    Uses ``enable_post_processing=True``, drops empty texture slots, and if
    needed loads images from ``map_Kd`` / ``{stem}.jpg`` next to the OBJ.
    Bakes UV textures to vertex colors when possible.
    """
    o3d = _require_open3d()
    obj_path = Path(obj_path)
    if not obj_path.is_file():
        raise FileNotFoundError(f"Imported OBJ not found: {obj_path}")

    # Fix textured_output.* vs {stem}.* naming before Open3D reads the MTL.
    if prefer_disk_textures:
        ensure_obj_material_paths(obj_path)

    mesh = o3d.io.read_triangle_mesh(str(obj_path), enable_post_processing=True)
    if mesh.is_empty():
        raise ValueError(f"Empty mesh: {obj_path}")

    # Open3D MTL textures are often V-flipped vs o3d.io.read_image; prefer disk.
    sanitize_open3d_textures(mesh)
    n_tex = 0
    if prefer_disk_textures and discover_obj_texture_images(obj_path):
        n_tex = attach_disk_textures(mesh, obj_path, replace=True)
    if not n_tex:
        n_tex = sanitize_open3d_textures(mesh)
        if n_tex:
            print(f"Open3D loaded {n_tex} texture image(s) from {obj_path.name}")
    elif n_tex:
        print(f"Using {n_tex} disk texture image(s) for {obj_path.name}")

    has_uv = mesh.has_triangle_uvs() and len(mesh.triangle_uvs) > 0
    companions = discover_obj_texture_images(obj_path)
    if companions and not has_uv:
        names = ", ".join(p.name for p in companions)
        raise ValueError(
            f"{obj_path.name} has no UV coordinates (vt), but texture file(s) "
            f"were found ({names}).\n"
            f"  This usually means the OBJ was overwritten by a geometry-only "
            f"export (look for a first line like "
            f"'# layout_design VTK-compatible OBJ').\n"
            f"  Replace {obj_path.name} with the original textured OBJ from your "
            f"package (the .obj that came with the .mtl and .jpg — it must "
            f"contain 'vt' and usually 'mtllib' lines), then reload.\n"
            f"  Check with:  python scripts/test_read_obj.py --obj {obj_path}"
        )


    had_vc = mesh.has_vertex_colors()
    baked = bake_texture_to_vertex_colors(mesh)
    if baked and mesh.has_vertex_colors():
        if had_vc and n_tex == 0:
            print(f"Source colors: vertex colors ({len(mesh.vertices)} vertices)")
        else:
            print(f"Source colors: texture baked → {len(mesh.vertices)} vertices")
    elif mesh.has_vertex_colors():
        print(f"Source colors: vertex colors ({len(mesh.vertices)} vertices)")
    else:
        print(
            "Warning: imported OBJ has no vertex colors or bakeable texture — "
            "fiducial picking will show a gray surface."
        )
        mesh.paint_uniform_color([0.75, 0.75, 0.75])

    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    return mesh


def backup_textured_obj_if_needed(obj_path: Path) -> Path | None:
    """
    If ``obj_path`` has UVs, copy it to ``{stem}_textured_src.obj`` once.

    Protects against accidental geometry-only overwrite of the import package.
    """
    obj_path = Path(obj_path)
    if not obj_path.is_file() or not obj_has_texcoords(obj_path):
        return None
    backup = obj_path.with_name(f"{obj_path.stem}_textured_src.obj")
    if backup.is_file():
        return backup
    shutil.copy2(obj_path, backup)
    # Also keep mtl/jpg names discoverable next to backup (same folder).
    print(f"Backed up textured OBJ → {backup.name}")
    return backup


def write_transformed_wavefront_obj(
    src_obj: Path,
    transform: np.ndarray,
    dest_obj: Path,
) -> Path:
    """
    Rewrite an OBJ applying a 4×4 transform to ``v`` lines only.

    Preserves ``vt`` / ``vn`` / ``f`` / ``mtllib`` / ``usemtl``. Copies sidecar
    ``.mtl`` and texture images when the destination directory differs.
    """
    src_obj = Path(src_obj)
    dest_obj = Path(dest_obj)
    dest_obj.parent.mkdir(parents=True, exist_ok=True)

    # Read fully before write (src may equal dest).
    lines = src_obj.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
    T = np.asarray(transform, dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f"Expected 4x4 transform, got {T.shape}")

    out_lines: list[str] = []
    n_v = 0
    for line in lines:
        if line.lstrip().startswith("v "):
            parts = line.split()
            if len(parts) >= 4:
                xyz = np.array([float(parts[1]), float(parts[2]), float(parts[3]), 1.0])
                xyz = T @ xyz
                # Keep optional vertex RGB if present (v x y z r g b)
                rest = " ".join(parts[4:])
                if rest:
                    out_lines.append(
                        f"v {xyz[0]:.6g} {xyz[1]:.6g} {xyz[2]:.6g} {rest}\n"
                    )
                else:
                    out_lines.append(f"v {xyz[0]:.6g} {xyz[1]:.6g} {xyz[2]:.6g}\n")
                n_v += 1
                continue
        out_lines.append(line if line.endswith("\n") else line + "\n")

    dest_obj.write_text("".join(out_lines), encoding="utf-8")
    print(f"Wrote transformed Wavefront OBJ ({n_v} vertices) → {dest_obj}")

    if src_obj.resolve() != dest_obj.resolve():
        # Copy MTL + discovered textures into dest folder.
        for mtl_name in parse_mtllib_names(src_obj):
            src_mtl = src_obj.parent / Path(mtl_name.replace("\\", "/")).name
            if src_mtl.is_file():
                shutil.copy2(src_mtl, dest_obj.parent / src_mtl.name)
        for img in discover_obj_texture_images(src_obj):
            dest_img = dest_obj.parent / img.name
            if img.resolve() != dest_img.resolve():
                shutil.copy2(img, dest_img)

    return dest_obj


def transfer_vertex_colors_from_points(
    mesh,
    source_points: np.ndarray,
    source_colors: np.ndarray,
    *,
    knn: int = 8,
    max_distance_mm: float | None = 8.0,
) -> None:
    """
    Transfer colors onto mesh vertices via inverse-distance-weighted k-NN.

    ``knn=1`` recovers classic nearest-neighbor. Neighbors farther than
    ``max_distance_mm`` are ignored when closer samples exist (nearest always kept).
    """
    import scipy.spatial

    o3d = _require_open3d()
    mesh_vertices = np.asarray(mesh.vertices, dtype=float)
    source_points = np.asarray(source_points, dtype=float)
    source_colors = np.asarray(source_colors, dtype=float)
    if len(source_points) == 0 or len(source_colors) == 0:
        raise ValueError("Source point cloud has no points/colors")
    if len(source_points) != len(source_colors):
        raise ValueError(
            f"Source points/colors length mismatch: "
            f"{len(source_points)} vs {len(source_colors)}"
        )

    k = max(1, min(int(knn), len(source_points)))
    tree = scipy.spatial.cKDTree(source_points)
    dists, indices = tree.query(mesh_vertices, k=k)
    if k == 1:
        mesh.vertex_colors = o3d.utility.Vector3dVector(
            np.clip(source_colors[np.asarray(indices, dtype=np.int64)], 0.0, 1.0)
        )
        return

    dists = np.asarray(dists, dtype=float)
    indices = np.asarray(indices, dtype=np.int64)
    eps = 1e-9
    dists = np.maximum(dists, eps)

    mask = np.ones_like(dists, dtype=bool)
    if max_distance_mm is not None:
        max_d = float(max_distance_mm)
        mask = dists <= max_d
        # Always keep the nearest neighbor so every vertex gets a color.
        nearest = np.argmin(dists, axis=1)
        mask[np.arange(mask.shape[0]), nearest] = True

    weights = np.where(mask, 1.0 / dists, 0.0)
    weight_sum = weights.sum(axis=1, keepdims=True)
    weights = weights / np.maximum(weight_sum, eps)
    gathered = source_colors[indices]  # (N, k, 3)
    colors = (gathered * weights[..., None]).sum(axis=1)
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))


def transfer_vertex_colors_from_color_ref(mesh, ref_path: Path) -> bool:
    """Reload colors from mesh-aligned reference written at reconstruct time."""
    o3d = _require_open3d()
    ref_path = Path(ref_path)
    if not ref_path.is_file():
        return False

    pcd = o3d.io.read_point_cloud(str(ref_path))
    if pcd.is_empty() or not pcd.has_colors():
        return False

    ref_points = np.asarray(pcd.points, dtype=float)
    ref_colors = np.asarray(pcd.colors, dtype=float)
    mesh_vertices = np.asarray(mesh.vertices, dtype=float)

    if len(ref_points) == len(mesh_vertices) and np.allclose(ref_points, mesh_vertices, atol=1e-3):
        mesh.vertex_colors = o3d.utility.Vector3dVector(ref_colors)
    else:
        transfer_vertex_colors_from_points(mesh, ref_points, ref_colors)

    return mesh.has_vertex_colors()


def save_color_reference(mesh, obj_path: Path) -> Path:
    """Persist vertex positions + colors in mesh space for reliable reload."""
    o3d = _require_open3d()
    obj_path = Path(obj_path)
    if not mesh.has_vertex_colors():
        raise ValueError(f"Mesh has no vertex colors to save for {obj_path.name}")

    ref_path = color_ref_path(obj_path)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=float))
    pcd.colors = mesh.vertex_colors
    o3d.io.write_point_cloud(str(ref_path), pcd)
    return ref_path


def attach_vertex_colors(mesh, obj_path: Path) -> bool:
    """
    Ensure an Open3D mesh has vertex colors for fiducial picking.

    Order: existing colors → sidecar npy → color_ref.ply → bake MTL/JPG textures.
    """
    o3d = _require_open3d()
    obj_path = Path(obj_path)
    if mesh.has_vertex_colors() and _colors_look_valid(
        np.asarray(mesh.vertex_colors, dtype=float), len(mesh.vertices)
    ):
        return True

    n_verts = len(mesh.vertices)
    sidecar = vertex_colors_sidecar(obj_path)
    if sidecar.is_file():
        colors = np.asarray(np.load(sidecar), dtype=float)
        if _colors_look_valid(colors, n_verts):
            mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
            if mesh.has_vertex_colors():
                return True
        reason = (
            f"count mismatch ({len(colors)} vs {n_verts})"
            if len(colors) != n_verts
            else "colors look washed out (stale sidecar)"
        )
        print(f"Warning: ignoring {sidecar.name} — {reason}.")

    ref_path = color_ref_path(obj_path)
    if transfer_vertex_colors_from_color_ref(mesh, ref_path):
        np.save(sidecar, np.asarray(mesh.vertex_colors, dtype=float))
        print(f"Restored vertex colors from {ref_path.name} → {sidecar.name}")
        return True

    sanitize_open3d_textures(mesh)
    if not (mesh.has_triangle_uvs() and len(getattr(mesh, "textures", []) or [])):
        attach_disk_textures(mesh, obj_path)
    if bake_texture_to_vertex_colors(mesh):
        np.save(sidecar, np.asarray(mesh.vertex_colors, dtype=float))
        print(f"Baked MTL/JPG textures → {sidecar.name}")
        return True

    print(
        f"Could not load vertex colors for {obj_path.name}. "
        f"Place {obj_path.stem}.obj + .mtl + .jpg together, then re-run align-obj:\n"
        f"  python -m app preprocess --subject {obj_path.stem} --step align-obj"
    )
    return False


def open3d_to_pyvista(mesh):
    """Convert Open3D TriangleMesh to PyVista PolyData with optional RGB scalars."""
    import pyvista as pv

    verts = np.asarray(mesh.vertices, dtype=float)
    tris = np.asarray(mesh.triangles, dtype=np.int64)
    if len(tris) == 0:
        return pv.PolyData(verts)

    faces = np.hstack([np.full((len(tris), 1), 3, dtype=np.int64), tris]).ravel()
    poly = pv.PolyData(verts, faces)

    if mesh.has_vertex_colors():
        rgb = (np.asarray(mesh.vertex_colors, dtype=float) * 255.0).clip(0, 255)
        poly["RGB"] = rgb.astype(np.uint8)

    if mesh.has_vertex_normals():
        poly.point_data["Normals"] = np.asarray(mesh.vertex_normals, dtype=float)

    return poly


def for_vertex_color_display(mesh):
    """
    Copy a mesh for Open3D viewers that ignore / mishandle MTL textures.

    Clears ``textures`` so baked ``vertex_colors`` are what the classic
    Visualizer shows (used by align-obj preview and debug viewers).
    """
    o3d = _require_open3d()
    view = o3d.geometry.TriangleMesh(mesh)
    if view.has_vertex_colors():
        view.textures = []
    if not view.has_vertex_normals():
        view.compute_vertex_normals()
    return view


def load_open3d_mesh(mesh_path: Path):
    """
    Load a mesh for pipeline use.

    Textured Wavefront OBJs (``vt`` / ``mtllib``) use post-processing + MTL/JPG
    bake. Geometry-only OBJs (VTK-compatible export) load without post-processing
    so vertex counts stay stable for color sidecars.
    """
    o3d = _require_open3d()
    mesh_path = Path(mesh_path)
    if mesh_path.suffix.lower() == ".obj":
        if obj_has_texcoords(mesh_path) or parse_mtllib_names(mesh_path):
            return load_textured_obj_mesh(mesh_path)
        mesh = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=False)
        if mesh.is_empty():
            raise ValueError(f"Empty mesh: {mesh_path}")
        attach_vertex_colors(mesh, mesh_path)
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        return mesh

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        raise ValueError(f"Empty mesh: {mesh_path}")
    return mesh


def load_head_mesh(mesh_path: Path):
    """
    Load head mesh for fiducial picking.

    Prefer Wavefront OBJ with UV + JPG (sharp PyVista texture). Geometry-only
    OBJs fall back to vertex-color RGB (vague — not for production picking).
    STL: PyVista geometry only.
    """
    import pyvista as pv

    path = Path(mesh_path)
    if path.suffix.lower() == ".obj":
        if obj_has_texcoords(path):
            from app.preprocess.obj_texture import load_pyvista_textured_obj

            print(f"Fiducials OBJ load (real texture): {path}")
            poly, _tex = load_pyvista_textured_obj(path)
            return poly

        # Geometry-only / sidecar colors (tests + legacy)
        print(
            f"Fiducials OBJ load (vertex RGB fallback — no UV/JPG): {path}"
        )
        o3d_mesh = load_open3d_mesh(path)
        return open3d_to_pyvista(o3d_mesh)

    data = pv.read(str(path))
    if isinstance(data, pv.MultiBlock):
        combined = data.combine()
        return combined if combined is not None else data
    return data


def write_vtk_compatible_obj(mesh, obj_path: Path) -> None:
    """
    Write OBJ with ``v x y z`` lines only (VTK/PyVista-safe).

    Vertex colors are stored in ``{stem}_vertex_colors.npy`` for reload.
    """
    _require_open3d()
    obj_path = Path(obj_path)
    obj_path.parent.mkdir(parents=True, exist_ok=True)

    verts = np.asarray(mesh.vertices, dtype=float)
    tris = np.asarray(mesh.triangles, dtype=np.int64)

    with obj_path.open("w", encoding="utf-8") as fp:
        fp.write("# layout_design VTK-compatible OBJ\n")
        for x, y, z in verts:
            fp.write(f"v {x:.6g} {y:.6g} {z:.6g}\n")
        for a, b, c in tris:
            fp.write(f"f {a + 1} {b + 1} {c + 1}\n")

    sidecar = vertex_colors_sidecar(obj_path)
    if mesh.has_vertex_colors():
        colors = np.asarray(mesh.vertex_colors, dtype=float)
        if len(colors) != len(verts):
            raise ValueError(
                f"Vertex/color count mismatch when writing {obj_path.name}: "
                f"{len(verts)} vertices vs {len(colors)} colors"
            )
        np.save(sidecar, colors)
        save_color_reference(mesh, obj_path)
    elif sidecar.is_file():
        sidecar.unlink(missing_ok=True)
        color_ref_path(obj_path).unlink(missing_ok=True)
