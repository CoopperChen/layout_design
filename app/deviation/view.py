"""Click printed electrodes and write their deviation from the planned layout."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
import vtk

from app import paths
from app.deviation.import_printed import ensure_printed_sources
from app.deviation.measure import apply_transform, build_document, format_report, save_document
from app.deviation.overlap import overlap_printed
from app.interact import KEYS_HELP, bind_accept_cancel
from app.mesh.texture import load_pyvista_textured_obj
from app.subject import load_layout


def run_deviation(
    subject_id: int,
    *,
    ply_path: Path | None = None,
    obj_path: Path | None = None,
    zip_path: Path | None = None,
    match_scale: bool = True,
) -> Path | None:
    """Overlap the printed head, pick each printed electrode, and write ``deviation.json``."""
    layout = load_layout(subject_id)
    stl = paths.head_stl(subject_id)
    if not stl.is_file():
        raise FileNotFoundError(
            f"No planned head for subject {subject_id}: {stl}\n"
            f"Run: python -m app mesh --subject {subject_id}"
        )
    _ply, obj = ensure_printed_sources(
        subject_id,
        ply_path=ply_path,
        obj_path=obj_path,
        zip_path=zip_path,
    )
    overlap = overlap_printed(obj, _ply, stl, match_scale=match_scale)
    picked = pick_printed_electrodes(obj, overlap.transform, layout.electrodes)
    if picked is None:
        print("Printed electrode picks discarded.")
        return None
    document = build_document(subject_id, overlap, layout.electrodes, picked)
    path = save_document(subject_id, document)
    print(format_report(document))
    print(f"Wrote {path}")
    return path


def pick_printed_electrodes(obj_path: Path, transform, planned: dict) -> dict[str, np.ndarray] | None:
    """Right-click each printed pad. Space / Enter / S places it. Q / Esc discards."""
    order = [str(name) for name in planned]
    if not order:
        raise ValueError("layout has no electrodes")
    planned_xyz = {name: np.asarray(planned[name], dtype=float).reshape(3) for name in order}
    mesh, _texture = load_pyvista_textured_obj(obj_path)
    _transform_polydata(mesh, transform)

    picked: dict[str, np.ndarray] = {}
    state = {"idx": 0, "last_pt": None, "discard": False}
    radius = _marker_radius(mesh)
    plotter = pv.Plotter(window_size=(1600, 1600))
    plotter.set_background("white")
    _add_head(plotter, mesh)
    _draw_planned(plotter, mesh, order, planned_xyz, radius)

    def show_instr(text: str) -> None:
        try:
            plotter.remove_actor("instr")
        except Exception:
            pass
        plotter.add_text(text, name="instr", font_size=13)

    def show_current() -> None:
        name = order[state["idx"]]
        _drop(plotter, "current_plan")
        plotter.add_mesh(
            pv.Sphere(center=planned_xyz[name], radius=radius * 1.35),
            color="orange",
            smooth_shading=True,
            name="current_plan",
        )
        show_instr(_status())

    def _status() -> str:
        name = order[state["idx"]]
        if len(picked) == len(order):
            dist = float(np.linalg.norm(picked[name] - planned_xyz[name]))
            return (
                f"All {len(order)} set. {name} is {dist:.2f} mm. "
                f"{KEYS_HELP}. N / P re-pick."
            )
        saved = ""
        if name in picked:
            dist = float(np.linalg.norm(picked[name] - planned_xyz[name]))
            saved = f" Saved {dist:.2f} mm."
        return (
            f"{state['idx'] + 1}/{len(order)}  {name} — right-click the printed electrode."
            f"{saved}  {KEYS_HELP}. N / P move."
        )

    picker = vtk.vtkCellPicker()
    picker.SetTolerance(0.0005)

    def on_right_click(obj, _event) -> None:
        x, y = obj.GetEventPosition()
        if not picker.Pick(x, y, 0, plotter.renderer):
            print("Right-click the printed head.")
            return
        pos = picker.GetPickPosition()
        pt = (float(pos[0]), float(pos[1]), float(pos[2]))
        state["last_pt"] = pt
        _drop(plotter, "provisional")
        plotter.add_mesh(
            pv.Sphere(center=pt, radius=radius * 0.85),
            color="white",
            opacity=0.7,
            name="provisional",
        )

    def on_confirm() -> None:
        pt = state["last_pt"]
        if pt is None:
            print("Right-click the printed electrode first.")
            return
        name = order[state["idx"]]
        point = np.asarray(pt, dtype=float)
        picked[name] = point
        distance = float(np.linalg.norm(point - planned_xyz[name]))
        _drop(plotter, "provisional")
        _draw_pick(plotter, name, point, planned_xyz[name], radius)
        print(f"  {name}: {distance:.2f} mm")
        state["last_pt"] = None
        if state["idx"] + 1 < len(order):
            state["idx"] += 1
        show_current()

    def on_accept() -> None:
        if state["last_pt"] is not None:
            on_confirm()
            return
        if len(picked) == len(order):
            plotter.close()
            return
        print("Right-click the printed electrode first.")

    def on_discard() -> None:
        state["discard"] = True
        plotter.close()

    def jump(delta: int) -> None:
        state["idx"] = (state["idx"] + delta) % len(order)
        state["last_pt"] = None
        _drop(plotter, "provisional")
        show_current()

    show_current()
    plotter.iren.add_observer("RightButtonPressEvent", on_right_click)
    bind_accept_cancel(plotter, on_accept, on_discard)
    for key, delta in (("n", 1), ("N", 1), ("p", -1), ("P", -1)):
        plotter.add_key_event(key, lambda d=delta: jump(d))
    print(
        "Printed electrodes: orange is the planned site to match. Blue marks are the other planned sites.\n"
        "  Right-click the printed pad, then Space / Enter / S places it.\n"
        f"  When every electrode is set, {KEYS_HELP}.\n"
        "  N / P move between electrodes. Closing the window accepts a complete set."
    )
    plotter.show()
    if state["discard"]:
        return None
    missing = [name for name in order if name not in picked]
    if missing:
        print(f"Incomplete picks, missing {missing}")
        return None
    return picked


def _transform_polydata(mesh, transform) -> None:
    mesh.points = np.ascontiguousarray(apply_transform(np.asarray(mesh.points), transform))
    mesh.compute_normals(cell_normals=False, point_normals=True, inplace=True)


def _marker_radius(mesh) -> float:
    pts = np.asarray(mesh.points, dtype=float)
    extent = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    return max(extent * 0.015, 1.5)


def _add_head(plotter: pv.Plotter, mesh) -> None:
    texture = getattr(mesh, "texture", None)
    if texture is not None and mesh.active_texture_coordinates is not None:
        plotter.add_mesh(mesh, texture=texture, smooth_shading=True, show_edges=False, name="head")
        return
    if "RGB" in mesh.array_names:
        plotter.add_mesh(
            mesh,
            scalars="RGB",
            rgb=True,
            smooth_shading=True,
            lighting=False,
            show_edges=False,
            name="head",
        )
        return
    plotter.add_mesh(mesh, color="lightgray", smooth_shading=True, name="head")


def _draw_planned(plotter, mesh, order, planned_xyz, radius: float) -> None:
    label_pts = []
    center = np.asarray(mesh.center, dtype=float)
    for name in order:
        point = planned_xyz[name]
        plotter.add_mesh(
            pv.Sphere(center=point, radius=radius),
            color="dodgerblue",
            opacity=0.85,
            smooth_shading=True,
            name=f"plan_{name}",
        )
        radial = point - center
        norm = float(np.linalg.norm(radial))
        if norm < 1e-9:
            radial = np.array([0.0, 0.0, 1.0])
            norm = 1.0
        label_pts.append(point + radial / norm * radius * 2.2)
    plotter.add_point_labels(
        np.asarray(label_pts, dtype=float),
        order,
        font_size=14,
        text_color="black",
        point_size=0,
        shape_opacity=0.75,
        always_visible=True,
        name="planned_labels",
    )


def _draw_pick(plotter, name: str, printed, planned, radius: float) -> None:
    _drop(plotter, f"pick_{name}")
    plotter.add_mesh(
        pv.Sphere(center=printed, radius=radius * 0.72),
        color="crimson",
        smooth_shading=True,
        name=f"pick_{name}",
    )
    _drop(plotter, f"line_{name}")
    if float(np.linalg.norm(printed - planned)) < 1e-4:
        return
    plotter.add_mesh(
        pv.Line(planned, printed).tube(radius=max(radius * 0.18, 0.25)),
        color="crimson",
        smooth_shading=True,
        name=f"line_{name}",
    )


def _drop(plotter, name: str) -> None:
    try:
        plotter.remove_actor(name)
    except Exception:
        pass
