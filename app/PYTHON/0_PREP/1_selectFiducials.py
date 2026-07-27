"""Interactive fiducial + terminal + calibration landmark picking on textured OBJ head."""
from __future__ import annotations

import os
import sys

import numpy as np
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
    if texture is not None and mesh.active_texture_coordinates is not None:
        plotter.add_mesh(
            mesh,
            texture=texture,
            show_edges=False,
            smooth_shading=True,
            name="head",
        )
        print("Fiducials display: real OBJ texture (UV + JPG)")
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
        print("Fiducials display: vertex RGB fallback (vague) — prefer textured OBJ")
    else:
        print(
            "Warning: head mesh has no texture — showing gray surface. "
            "Place {id}.obj + .mtl + .jpg and run align-obj."
        )
        plotter.add_mesh(mesh, color="lightgray", opacity=0.85, name="head")


def _sphere_radius(mesh: pv.DataSet) -> float:
    """
    Marker size from the *surface* extent, not ``mesh.length``.

    Textured OBJs often keep many unused ``v`` rows; those inflate the
    bounding-box diagonal and make pick spheres huge.
    """
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
    print("Mesh pairing (same coordinate frame after align-obj):")
    print(f"  aligned OBJ (this step — textured picking): {mesh_path}")
    print(f"  import OBJ (unchanged source):              {paths.imported_textured_obj(SUBJECT_ID)}")
    if stl_path.is_file():
        print(f"  STL (all other pipeline steps):             {stl_path}")
    else:
        print(
            f"  STL (all other pipeline steps):             {stl_path}  "
            "[missing — run clear-islands or copy STL before synthesize]"
        )
    mesh = load_head_mesh(mesh_path)

    picked = load_picks(SUBJECT_ID)
    if picked:
        print(f"Recalled {len(picked)} point(s) from {paths.fiducials_json(SUBJECT_ID)}")

    state: dict = {
        "idx": _first_missing_index(picked),
        "last_pt": None,
        "discard": False,
    }

    print("Controls:")
    print("  Rotate: left-click + drag")
    print("  Right-click: provisional pick on surface")
    print("  Space / Enter: confirm current pick")
    print("  1–9: jump to landmark (re-pick)")
    print("  n / p: next / previous landmark")
    print("  S or close window: save and finish")
    print("  Q: discard (nothing written)")

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
            show_instr("All points set — S / close = save · Q = discard")

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

    def on_save_finish() -> None:
        plotter.close()

    def on_discard() -> None:
        state["discard"] = True
        plotter.close()

    plotter.iren.add_observer("RightButtonPressEvent", on_right_click)
    plotter.add_key_event("space", on_confirm)
    plotter.add_key_event("Return", on_confirm)
    plotter.add_key_event("n", on_next)
    plotter.add_key_event("p", on_prev)
    plotter.add_key_event("s", on_save_finish)
    plotter.add_key_event("q", on_discard)
    for digit in range(1, 10):
        plotter.add_key_event(str(digit), lambda d=digit: jump_to(d - 1))

    plotter.show()

    if state["discard"]:
        print("Fiducials discarded (Q) — nothing written.", file=sys.stderr)
        return 1

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
