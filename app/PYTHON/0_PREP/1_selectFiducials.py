"""Interactive fiducial + terminal + calibration landmark picking on textured OBJ head."""
from __future__ import annotations

import os
import sys

import pyvista as pv
import vtk

from app import paths
from app.preprocess.fiducials_io import (
    PICK_COLORS,
    PICK_SEQUENCE,
    load_head_mesh,
    load_picks,
    save_landmarks_mat,
    save_picks,
)

SUBJECT_ID = int(os.environ.get("LAYOUT_SUBJECT_ID", "1"))


def _first_missing_index(picked: dict) -> int:
    for i, (key, _) in enumerate(PICK_SEQUENCE):
        if key not in picked:
            return i
    return 0


def _add_head_mesh(plotter: pv.Plotter, mesh: pv.DataSet) -> None:
    texture = getattr(mesh, "texture", None)
    if texture is not None:
        plotter.add_mesh(
            mesh,
            texture=texture,
            show_edges=False,
            smooth_shading=True,
            name="head",
        )
    elif "RGB" in mesh.array_names:
        plotter.add_mesh(
            mesh,
            scalars="RGB",
            rgb=True,
            show_edges=False,
            smooth_shading=True,
            lighting=False,
            name="head",
        )
    else:
        print(
            "Warning: head mesh has no vertex colors — showing gray surface. "
            "Re-run reconstruct or ensure data/raw/{id}.ply exists beside the OBJ."
        )
        plotter.add_mesh(mesh, color="lightgray", opacity=0.85, name="head")


def _sphere_radius(mesh: pv.DataSet) -> float:
    return float(mesh.length) * 0.005


def _draw_confirmed(
    plotter: pv.Plotter,
    mesh: pv.DataSet,
    name: str,
    pt: tuple[float, float, float],
    color: str,
) -> None:
    try:
        plotter.remove_actor(f"confirmed_{name}")
    except Exception:
        pass
    sphere = pv.Sphere(
        center=pt,
        radius=_sphere_radius(mesh),
        theta_resolution=16,
        phi_resolution=16,
    )
    plotter.add_mesh(sphere, color=color, name=f"confirmed_{name}")


def main() -> int:
    mesh_path = paths.textured_head_obj(SUBJECT_ID)
    stl_path = paths.cleaned_scan(SUBJECT_ID)
    print("Mesh pairing (same geometry):")
    print(f"  OBJ (this step — textured picking): {mesh_path}")
    if stl_path.is_file():
        print(f"  STL (all other pipeline steps):     {stl_path}")
    else:
        print(
            f"  STL (all other pipeline steps):     {stl_path}  "
            "[missing — run clear-islands or copy STL before synthesize]"
        )
    mesh = load_head_mesh(mesh_path)

    picked = load_picks(SUBJECT_ID)
    if picked:
        print(f"Recalled {len(picked)} point(s) from {paths.fiducials_json(SUBJECT_ID)}")

    state: dict = {"idx": _first_missing_index(picked), "last_pt": None}

    print("Controls:")
    print("  Rotate: left-click + drag")
    print("  Right-click: provisional pick on surface")
    print("  Space / Enter: confirm current pick")
    print("  1–9: jump to landmark (re-pick)")
    print("  n / p: next / previous landmark")
    print("  Close window when done to save")

    plotter = pv.Plotter(window_size=(2000, 2000))
    _add_head_mesh(plotter, mesh)

    def show_instr(text: str) -> None:
        try:
            plotter.remove_actor("instr")
        except Exception:
            pass
        plotter.add_text(text, name="instr", font_size=14)

    def current_label() -> str:
        _, label = PICK_SEQUENCE[state["idx"]]
        return label

    show_instr(current_label())

    for i, (key, _) in enumerate(PICK_SEQUENCE):
        if key in picked:
            pt = tuple(float(c) for c in picked[key])
            _draw_confirmed(plotter, mesh, key, pt, PICK_COLORS[i])

    picker = vtk.vtkCellPicker()
    picker.SetTolerance(0.0005)

    def on_right_click(obj, event) -> None:
        x, y = obj.GetEventPosition()
        picker.Pick(x, y, 0, plotter.renderer)
        pos = picker.GetPickPosition()
        pt = (float(pos[0]), float(pos[1]), float(pos[2]))
        state["last_pt"] = pt
        try:
            plotter.remove_actor("provisional")
        except Exception:
            pass
        sphere = pv.Sphere(
            center=pt,
            radius=_sphere_radius(mesh),
            theta_resolution=16,
            phi_resolution=16,
        )
        plotter.add_mesh(sphere, color="white", opacity=0.6, name="provisional")
        print(f"  → Provisional: {pt}  (Space/Enter to confirm)")

    def on_confirm() -> None:
        pt = state["last_pt"]
        if pt is None:
            print("  (!) Right-click a point on the head first.")
            return

        idx = state["idx"]
        name, label = PICK_SEQUENCE[idx]
        picked[name] = pt
        try:
            plotter.remove_actor("provisional")
        except Exception:
            pass
        _draw_confirmed(plotter, mesh, name, pt, PICK_COLORS[idx])
        print(f"  ✔ {name}: {pt}")
        state["last_pt"] = None

        if idx + 1 < len(PICK_SEQUENCE):
            state["idx"] = idx + 1
            show_instr(current_label())
        else:
            show_instr("All points set — close window to save")

    def jump_to(index: int) -> None:
        index = max(0, min(index, len(PICK_SEQUENCE) - 1))
        state["idx"] = index
        state["last_pt"] = None
        try:
            plotter.remove_actor("provisional")
        except Exception:
            pass
        _, label = PICK_SEQUENCE[index]
        show_instr(f"Re-pick: {label}")
        print(f"  → Selected: {label}")

    def on_next() -> None:
        jump_to(state["idx"] + 1)

    def on_prev() -> None:
        jump_to(state["idx"] - 1)

    plotter.iren.add_observer("RightButtonPressEvent", on_right_click)
    plotter.add_key_event("space", on_confirm)
    plotter.add_key_event("Return", on_confirm)
    plotter.add_key_event("n", on_next)
    plotter.add_key_event("p", on_prev)
    for digit in range(1, 10):
        plotter.add_key_event(str(digit), lambda d=digit: jump_to(d - 1))

    plotter.show()

    if not picked:
        print("No points saved.", file=sys.stderr)
        return 1

    json_path = save_picks(SUBJECT_ID, picked)
    print(f"Saved fiducials → {json_path}")

    mat_dir = save_landmarks_mat(SUBJECT_ID, picked)
    if mat_dir is not None:
        print(f"Saved Landmarks.mat + LandmarkNames.mat → {mat_dir}")
    else:
        print(
            "Landmarks.mat not written (need landmark_central, landmark_left, landmark_back)."
        )

    missing = [k for k, _ in PICK_SEQUENCE if k not in picked]
    if missing:
        print(f"Warning: incomplete picks ({len(picked)}/{len(PICK_SEQUENCE)}). Missing: {missing}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
