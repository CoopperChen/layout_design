"""Copy a printed-head PLY and textured OBJ into printed/source/. Does not touch the pipeline scan."""

from __future__ import annotations

from pathlib import Path

from app import paths
from app.mesh.import_sources import (
    _ask_open_file,
    _install_loose_obj,
    _install_ply,
    _install_zip,
    _resolve_input,
)


def ensure_printed_sources(
    subject_id: int | str,
    *,
    ply_path: Path | None = None,
    obj_path: Path | None = None,
    zip_path: Path | None = None,
    pick_ply=None,
    pick_zip=None,
) -> tuple[Path, Path]:
    """Make ``printed/source/scan.ply`` and ``printed/source/head.obj`` exist.

    A zip is the textured package (obj, mtl, jpg). ``--obj`` accepts a loose OBJ
    whose mtl and image sit beside it. Files already imported are kept unless a
    new path is passed.
    """
    paths.ensure_subject_dir(subject_id)
    dest_ply = paths.printed_ply(subject_id)
    dest_obj = paths.printed_obj(subject_id)

    if ply_path is None and obj_path is None and zip_path is None and _printed_ready(subject_id):
        return dest_ply, dest_obj

    ply_path = _resolve_input(ply_path)
    obj_path = _resolve_input(obj_path)
    zip_path = _resolve_input(zip_path)

    if ply_path is None and not dest_ply.is_file():
        print(f"Subject {subject_id}: choose the printed head scan (.ply).", flush=True)
        ply_path = (pick_ply or _pick_printed_ply)()
        if ply_path is None:
            raise RuntimeError("No printed PLY selected.")
    if zip_path is None and obj_path is None and not _texture_ready(subject_id):
        print(
            f"Subject {subject_id}: choose the printed head texture (.zip of obj, mtl, and jpg).",
            flush=True,
        )
        zip_path = (pick_zip or _pick_printed_zip)()
        if zip_path is None:
            raise RuntimeError("No printed texture zip selected.")

    if ply_path is not None:
        _install_ply(ply_path, dest_ply)
    if zip_path is not None:
        _install_zip(zip_path, paths.printed_source_dir(subject_id))
    elif obj_path is not None:
        _install_loose_obj(obj_path, paths.printed_source_dir(subject_id))

    if not dest_ply.is_file() or not dest_obj.is_file():
        raise FileNotFoundError(
            f"Subject {subject_id} is missing a printed scan.ply or head.obj in "
            f"{paths.printed_source_dir(subject_id)}"
        )
    return dest_ply, dest_obj


def _printed_ready(subject_id: int | str) -> bool:
    return paths.printed_ply(subject_id).is_file() and _texture_ready(subject_id)


def _texture_ready(subject_id: int | str) -> bool:
    folder = paths.printed_source_dir(subject_id)
    image = any((folder / name).is_file() for name in ("head.jpg", "head.jpeg", "head.png"))
    return paths.printed_obj(subject_id).is_file() and (folder / "head.mtl").is_file() and image


def _pick_printed_ply() -> Path | None:
    return _ask_open_file(
        "Select the printed head scan",
        [("Point cloud", "*.ply"), ("All files", "*.*")],
    )


def _pick_printed_zip() -> Path | None:
    return _ask_open_file(
        "Select the printed head texture zip",
        [("Textured head", "*.zip"), ("All files", "*.*")],
    )
