"""Copy a scan PLY and a textured-head zip into data/subjects/{id}/source/."""

from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path

from app import paths

_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
_MTLLIB_RE = re.compile(r"^(\s*mtllib\s+)(\S+)(.*)$", re.IGNORECASE | re.MULTILINE)
_MAP_KEYS = ("map_", "bump", "disp", "refl")


def sources_ready(subject_id: int | str) -> bool:
    return (
        paths.source_ply(subject_id).is_file()
        and paths.source_obj(subject_id).is_file()
        and paths.source_mtl(subject_id).is_file()
        and paths.source_image(subject_id).is_file()
    )


def ensure_subject_sources(
    subject_id: int | str,
    *,
    ply_path: Path | None = None,
    obj_path: Path | None = None,
    zip_path: Path | None = None,
    pick_ply=None,
    pick_zip=None,
) -> tuple[Path, Path]:
    """Make source/scan.ply and source/head.obj exist, prompting when they do not.

    A zip is the textured package (obj, mtl, jpg). ``--obj`` still accepts a loose
    OBJ whose mtl and image sit beside it. Existing source files are left in place
    unless a new path is passed.
    """
    paths.ensure_subject_dir(subject_id)
    dest_ply = paths.source_ply(subject_id)
    dest_obj = paths.source_obj(subject_id)

    if ply_path is None and obj_path is None and zip_path is None and sources_ready(subject_id):
        return dest_ply, dest_obj

    ply_path = _resolve_input(ply_path)
    obj_path = _resolve_input(obj_path)
    zip_path = _resolve_input(zip_path)

    if ply_path is None and not dest_ply.is_file():
        print(f"Subject {subject_id}: choose the scan point cloud (.ply).", flush=True)
        ply_path = (pick_ply or _pick_ply)()
        if ply_path is None:
            raise RuntimeError("No PLY selected.")
    if zip_path is None and obj_path is None and not _texture_ready(subject_id):
        print(f"Subject {subject_id}: choose the textured head (.zip of obj, mtl, and jpg).", flush=True)
        zip_path = (pick_zip or _pick_zip)()
        if zip_path is None:
            raise RuntimeError("No texture zip selected.")

    if ply_path is not None:
        _install_ply(ply_path, dest_ply)
    if zip_path is not None:
        _install_zip(zip_path, paths.source_dir(subject_id))
    elif obj_path is not None:
        _install_loose_obj(obj_path, paths.source_dir(subject_id))

    if not dest_ply.is_file() or not dest_obj.is_file():
        raise FileNotFoundError(
            f"Subject {subject_id} is missing scan.ply or head.obj in {paths.source_dir(subject_id)}"
        )
    return dest_ply, dest_obj


def _texture_ready(subject_id: int | str) -> bool:
    return (
        paths.source_obj(subject_id).is_file()
        and paths.source_mtl(subject_id).is_file()
        and paths.source_image(subject_id).is_file()
    )


def _resolve_input(path: Path | None) -> Path | None:
    if path is None:
        return None
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = paths.REPO_ROOT / resolved
    if not resolved.is_file():
        raise FileNotFoundError(f"Input file not found: {resolved}")
    return resolved


def _pick_ply() -> Path | None:
    return _ask_open_file(
        "Select the scan point cloud",
        [("Point cloud", "*.ply"), ("All files", "*.*")],
    )


def _pick_zip() -> Path | None:
    return _ask_open_file(
        "Select the textured head zip",
        [("Textured head", "*.zip"), ("All files", "*.*")],
    )


def _ask_open_file(title: str, filetypes: list[tuple[str, str]]) -> Path | None:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except tk.TclError:
        pass
    root.update()
    chosen = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    if not chosen:
        return None
    return Path(chosen)


def _install_ply(src: Path, dest: Path) -> None:
    if src.suffix.lower() != ".ply":
        raise ValueError(f"Scan must be a .ply file, got {src.name}")
    if src.resolve() == dest.resolve():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    print(f"Imported scan {src.name} → {dest}")


def _install_zip(zip_path: Path, dest_dir: Path) -> None:
    if zip_path.suffix.lower() != ".zip":
        raise ValueError(f"Texture package must be a .zip file, got {zip_path.name}")
    with zipfile.ZipFile(zip_path) as archive:
        payload = _zip_payload(archive)
    _stage_package(payload, dest_dir, origin=zip_path.name)


def _install_loose_obj(obj_path: Path, dest_dir: Path) -> None:
    if obj_path.suffix.lower() != ".obj":
        raise ValueError(f"Texture mesh must be a .obj file, got {obj_path.name}")
    payload: dict[str, bytes] = {obj_path.name: obj_path.read_bytes()}
    for sibling in obj_path.parent.iterdir():
        if not sibling.is_file() or sibling.resolve() == obj_path.resolve():
            continue
        if sibling.suffix.lower() in _IMAGE_EXTS or sibling.suffix.lower() == ".mtl":
            payload[sibling.name] = sibling.read_bytes()
    _stage_package(payload, dest_dir, origin=obj_path.name)


def _zip_payload(archive: zipfile.ZipFile) -> dict[str, bytes]:
    payload: dict[str, bytes] = {}
    for info in archive.infolist():
        if info.is_dir():
            continue
        name = info.filename.replace("\\", "/")
        if name.startswith("__MACOSX/") or "/__MACOSX/" in f"/{name}":
            continue
        base = Path(name).name
        if not base or base.startswith("."):
            continue
        if base in payload:
            raise ValueError(
                f"Zip contains two files named {base}. Keep one obj, one mtl, and one image."
            )
        payload[base] = archive.read(info)
    if not payload:
        raise ValueError("Zip has no obj, mtl, or image files.")
    return payload


def _stage_package(payload: dict[str, bytes], dest_dir: Path, *, origin: str) -> None:
    objs = _by_suffix(payload, {".obj"})
    mtls = _by_suffix(payload, {".mtl"})
    images = _by_suffix(payload, _IMAGE_EXTS)
    if len(objs) != 1:
        raise ValueError(f"{origin} must contain one .obj file, found {sorted(objs) or 'none'}.")
    if not mtls:
        raise ValueError(f"{origin} must contain a .mtl file.")
    if not images:
        raise ValueError(f"{origin} must contain a .jpg or .png texture.")

    obj_name = next(iter(objs))
    obj_text = objs[obj_name].decode("utf-8", errors="ignore")
    mtl_name = _choose_mtl(obj_text, mtls, origin)
    image_name = _choose_image(mtls[mtl_name].decode("utf-8", errors="ignore"), images, origin)
    dest_image = _dest_image_name(image_name)

    obj_out = _rewrite_mtllib(obj_text)
    mtl_out = _rewrite_maps(
        mtls[mtl_name].decode("utf-8", errors="ignore"),
        {Path(name).name.lower() for name in images},
        dest_image,
    )

    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / "head.obj").write_text(obj_out if obj_out.endswith("\n") else obj_out + "\n", encoding="utf-8")
    (dest_dir / "head.mtl").write_text(mtl_out if mtl_out.endswith("\n") else mtl_out + "\n", encoding="utf-8")
    for stale in ("head.jpg", "head.jpeg", "head.png"):
        if stale != dest_image:
            leftover = dest_dir / stale
            if leftover.is_file():
                leftover.unlink()
    (dest_dir / dest_image).write_bytes(images[image_name])
    print(
        f"Imported texture {origin} → {dest_dir / 'head.obj'}, "
        f"head.mtl, {dest_image} (from {obj_name}, {mtl_name}, {image_name})"
    )


def _by_suffix(payload: dict[str, bytes], suffixes: set[str]) -> dict[str, bytes]:
    return {name: data for name, data in payload.items() if Path(name).suffix.lower() in suffixes}


def _choose_mtl(obj_text: str, mtls: dict[str, bytes], origin: str) -> str:
    declared = [_basename(match.group(2)) for match in _MTLLIB_RE.finditer(obj_text)]
    for name in declared:
        match = _lookup(mtls, name)
        if match is not None:
            return match
    if len(mtls) == 1:
        return next(iter(mtls))
    raise ValueError(
        f"{origin} has {len(mtls)} .mtl files and the OBJ does not name one of them: {sorted(mtls)}"
    )


def _choose_image(mtl_text: str, images: dict[str, bytes], origin: str) -> str:
    referenced: list[str] = []
    for line in mtl_text.splitlines():
        parts = line.split()
        if len(parts) < 2 or not parts[0].lower().startswith(_MAP_KEYS):
            continue
        referenced.append(_basename(parts[-1]))
    for name in referenced:
        match = _lookup(images, name)
        if match is not None:
            return match
    if len(images) == 1:
        return next(iter(images))
    raise ValueError(
        f"{origin} has {len(images)} images and the MTL does not name one of them: {sorted(images)}"
    )


def _dest_image_name(src_name: str) -> str:
    ext = Path(src_name).suffix.lower()
    if ext == ".jpeg":
        ext = ".jpg"
    return f"head{ext}"


def _rewrite_mtllib(text: str) -> str:
    lines = text.splitlines()
    rewritten = False
    out: list[str] = []
    for line in lines:
        match = _MTLLIB_RE.match(line)
        if match and not rewritten:
            out.append(f"{match.group(1)}head.mtl{match.group(3)}")
            rewritten = True
        else:
            out.append(line)
    if not rewritten:
        out.insert(0, "mtllib head.mtl")
    return "\n".join(out) + "\n"


def _rewrite_maps(text: str, image_names_lower: set[str], dest_image: str) -> str:
    out: list[str] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].lower().startswith(_MAP_KEYS):
            token = _basename(parts[-1])
            if token.lower() in image_names_lower or Path(token).suffix.lower() in _IMAGE_EXTS:
                parts[-1] = dest_image
                out.append(" ".join(parts))
                continue
        out.append(line)
    return "\n".join(out) + "\n"


def _basename(token: str) -> str:
    return Path(token.strip().strip("\"'").replace("\\", "/")).name


def _lookup(files: dict[str, bytes], basename: str) -> str | None:
    for name in files:
        if name.lower() == basename.lower():
            return name
    return None
