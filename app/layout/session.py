"""One picking session: seven landmarks, then a 10–20 review."""

from __future__ import annotations

import colorsys

import numpy as np
import pyvista as pv
import vtk

from app.config_loader import layout_config
from app.interact import KEYS_HELP, bind_accept_cancel
from app.layout.place import compute_cz, project_electrodes
from app.layout.template import CHANNEL_PAIRS, CHANNELS, SINGLE_CHANNELS
from app.mesh.io import load_head_mesh
from app.subject import ANATOMY_KEYS, CALIBRATION_KEYS, Layout

PICK_SEQUENCE: list[tuple[str, str]] = [
    ("nasion", "1) Nasion — bridge of nose"),
    ("lpa", "2) LPA — left pre-auricular point"),
    ("rpa", "3) RPA — right pre-auricular point"),
    ("inion", "4) Inion — bump at back of head"),
    ("central", "5) Calibration landmark (central)"),
    ("left", "6) Calibration landmark (left)"),
    ("back", "7) Calibration landmark (back)"),
]
PICK_COLORS = ["orange", "green", "blue", "yellow", "cyan", "magenta", "white"]


def _sphere_radius(mesh: pv.DataSet) -> float:
    pts = np.asarray(mesh.points, dtype=float)
    if pts.size == 0:
        return 1.0
    extent = float(mesh.length)
    if getattr(mesh, "n_cells", 0) > 0:
        try:
            if bool(getattr(mesh, "is_all_triangles", False)):
                faces = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 4)[:, 1:4]
                used = pts[np.unique(faces.ravel())]
                if len(used) >= 2:
                    extent = float(np.linalg.norm(used.max(axis=0) - used.min(axis=0)))
        except Exception:  # noqa: BLE001
            pass
    return max(extent * 0.0025, 0.4)


def _add_head(plotter: pv.Plotter, mesh: pv.DataSet) -> None:
    texture = getattr(mesh, "texture", None)
    if texture is not None and mesh.active_texture_coordinates is not None:
        plotter.add_mesh(mesh, texture=texture, show_edges=False, smooth_shading=True, name="head")
    elif "RGB" in mesh.array_names:
        plotter.add_mesh(
            mesh, scalars="RGB", rgb=True, show_edges=False, smooth_shading=True, lighting=False, name="head"
        )
    else:
        plotter.add_mesh(mesh, color="lightgray", opacity=0.85, name="head")


def _outward(mesh: pv.PolyData, point: np.ndarray) -> np.ndarray:
    center = np.asarray(mesh.center, dtype=float)
    radial = np.asarray(point, dtype=float) - center
    normals = mesh.point_normals
    if normals is None:
        mesh = mesh.compute_normals(cell_normals=False, point_normals=True, inplace=False)
        normals = mesh.point_normals
    n = np.asarray(normals[mesh.find_closest_point(point)], dtype=float)
    length = float(np.linalg.norm(n))
    if length < 1e-9:
        length = float(np.linalg.norm(radial))
        n = radial if length >= 1e-9 else np.array([0.0, 0.0, 1.0])
        length = float(np.linalg.norm(n))
    n = n / length
    return -n if float(np.dot(n, radial)) < 0.0 else n


def pick_landmarks(mesh: pv.DataSet) -> dict[str, np.ndarray] | None:
    """Return the seven picks, or None if the operator discards them."""
    picked: dict[str, np.ndarray] = {}
    state = {"idx": 0, "last_pt": None, "discard": False}
    plotter = pv.Plotter(window_size=(1600, 1600))
    _add_head(plotter, mesh)

    def show_instr(text: str) -> None:
        try:
            plotter.remove_actor("instr")
        except Exception:
            pass
        plotter.add_text(text, name="instr", font_size=14)

    show_instr(PICK_SEQUENCE[0][1])
    picker = vtk.vtkCellPicker()
    picker.SetTolerance(0.0005)

    def on_right_click(obj, _event) -> None:
        x, y = obj.GetEventPosition()
        picker.Pick(x, y, 0, plotter.renderer)
        pos = picker.GetPickPosition()
        pt = (float(pos[0]), float(pos[1]), float(pos[2]))
        state["last_pt"] = pt
        try:
            plotter.remove_actor("provisional")
        except Exception:
            pass
        plotter.add_mesh(
            pv.Sphere(center=pt, radius=_sphere_radius(mesh)),
            color="white",
            opacity=0.6,
            name="provisional",
        )

    def on_confirm() -> None:
        pt = state["last_pt"]
        if pt is None:
            print("Right-click the head first.")
            return
        idx = state["idx"]
        name, _label = PICK_SEQUENCE[idx]
        picked[name] = np.asarray(pt, dtype=float)
        try:
            plotter.remove_actor("provisional")
        except Exception:
            pass
        plotter.add_mesh(
            pv.Sphere(center=pt, radius=_sphere_radius(mesh)),
            color=PICK_COLORS[idx],
            name=f"confirmed_{name}",
        )
        print(f"  {name}: {pt}")
        state["last_pt"] = None
        if idx + 1 < len(PICK_SEQUENCE):
            state["idx"] = idx + 1
            show_instr(PICK_SEQUENCE[state["idx"]][1])
        else:
            show_instr(f"All points set — {KEYS_HELP}")

    def on_accept() -> None:
        if state["last_pt"] is not None:
            on_confirm()
            return
        if len(picked) == len(PICK_SEQUENCE):
            plotter.close()
            return
        print("Right-click the head first.")

    def on_discard() -> None:
        state["discard"] = True
        plotter.close()

    def jump_to(index: int) -> None:
        state["idx"] = max(0, min(index, len(PICK_SEQUENCE) - 1))
        state["last_pt"] = None
        show_instr(f"Re-pick: {PICK_SEQUENCE[state['idx']][1]}")

    plotter.iren.add_observer("RightButtonPressEvent", on_right_click)
    bind_accept_cancel(plotter, on_accept, on_discard)
    plotter.add_key_event("n", lambda: jump_to(state["idx"] + 1))
    plotter.add_key_event("p", lambda: jump_to(state["idx"] - 1))
    for digit in range(1, 8):
        plotter.add_key_event(str(digit), lambda d=digit: jump_to(d - 1))
    print(
        "Layout picks: right-click, then Space / Enter / S places the point.\n"
        f"  When all 7 are set, {KEYS_HELP}.\n"
        "  1–7 jump to a landmark. Closing the window accepts a complete set."
    )
    plotter.show()
    if state["discard"]:
        return None
    missing = [key for key, _ in PICK_SEQUENCE if key not in picked]
    if missing:
        print(f"Incomplete picks, missing {missing}")
        return None
    return picked


def review_electrodes(
    mesh: pv.PolyData,
    anatomy: dict[str, np.ndarray],
    cz: np.ndarray,
) -> dict[str, np.ndarray] | None:
    """Show projected sites. Slider scales them. Space/Enter/S saves; Q/Esc discards."""
    cfg = layout_config()
    state = {
        "adjustment": cfg["scale"],
        "positions": {},
        "discard": False,
        "saved": False,
    }
    plotter = pv.Plotter(window_size=(1600, 1600))
    plotter.add_mesh(mesh, color="white", opacity=0.88, name="head")
    radius = float(mesh.length) * 0.008

    def color_for(index: int, count: int, sat: float) -> str:
        r, g, b = colorsys.hsv_to_rgb(index / max(count, 1), sat, 0.9)
        return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"

    colors = {}
    for i, pair in enumerate(CHANNEL_PAIRS):
        col = color_for(i, len(CHANNEL_PAIRS), 0.8)
        for name in pair:
            colors[name] = col
    for i, name in enumerate(SINGLE_CHANNELS):
        colors[name] = color_for(i, len(SINGLE_CHANNELS), 0.6)

    def redraw(adjustment: float) -> None:
        positions = project_electrodes(mesh, anatomy, cz, adjustment=adjustment)
        state["positions"] = positions
        for name in CHANNELS:
            if name in plotter.actors:
                plotter.remove_actor(name)
            center = positions[name] + _outward(mesh, positions[name]) * radius
            plotter.add_mesh(pv.Sphere(center=center, radius=radius), color=colors[name], name=name)
        if "size_text" in plotter.actors:
            plotter.remove_actor("size_text")
        plotter.add_text(
            f"10–20 size {adjustment:.2f}×  ·  {KEYS_HELP}",
            name="size_text",
            font_size=12,
        )

    def on_slide(value: float) -> None:
        state["adjustment"] = float(value)
        redraw(state["adjustment"])

    def accept() -> None:
        state["saved"] = True
        plotter.close()

    def discard() -> None:
        state["discard"] = True
        plotter.close()

    plotter.add_slider_widget(
        on_slide,
        [cfg["scale_min"], cfg["scale_max"]],
        value=cfg["scale"],
        title="Size",
    )
    bind_accept_cancel(plotter, accept, discard)
    redraw(cfg["scale"])
    print(f"Electrode review: drag the size slider. {KEYS_HELP}. Closing the window saves.")
    plotter.show()
    if state["discard"]:
        return None
    return state["positions"] or None


def layout_from_picks(mesh: pv.PolyData, picks: dict[str, np.ndarray]) -> Layout | None:
    anatomy = {key: np.asarray(picks[key], dtype=float) for key in ANATOMY_KEYS}
    calibration = {key: np.asarray(picks[key], dtype=float) for key in CALIBRATION_KEYS}
    cz = compute_cz(mesh, anatomy)
    print(f"Cz = {cz.tolist()}")
    electrodes = review_electrodes(mesh, anatomy, cz)
    if electrodes is None:
        return None
    return Layout(anatomy=anatomy, cz=cz, calibration=calibration, electrodes=electrodes)


def run_layout_session(obj_path) -> Layout | None:
    mesh = load_head_mesh(obj_path)
    picks = pick_landmarks(mesh)
    if picks is None:
        return None
    surface = mesh
    if not isinstance(surface, pv.PolyData):
        surface = surface.extract_surface().triangulate()
    return layout_from_picks(surface, picks)
