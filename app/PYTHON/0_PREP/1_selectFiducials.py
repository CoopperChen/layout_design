# # # # # BELOW: ALSO SAVE TERMINAL ROIS FOR INTERCONNECT PATHS
import os
import pyvista as pv
import vtk
import json
from pathlib import Path

SUBJECT_ID = int(os.environ.get("LAYOUT_SUBJECT_ID", "1"))



# 1) Load your scan
stl_path = f"data/cleaned_scans/{SUBJECT_ID}.stl"
mesh = pv.read(stl_path)

# 2) Landmark prompts and sphere colors (including inion)
order = [
    ("nasion", "1) Right-click the NASION (bridge of nose)"),
    ("lpa",    "2) Right-click LEFT pre-auricular point"),
    ("rpa",    "3) Right-click RIGHT pre-auricular point"),
    ("inion",  "4) Right-click the INION (bump at back of head)"),
]
terminal_order = [
    ("TERMINAL_RIGHT", "5) Right-click the RIGHT terminal location"),
    ("TERMINAL_LEFT", "6) Right-click the LEFT terminal location"),
]
colors = ["orange", "green", "blue", "yellow", "black", "red"]

# 3) State & storage
state = {"idx": 0, "last_pt": None, "phase": "initial"}  # phases: initial, terminal
picked = {}

# 4) Console instructions
print("Rotate with left-click + drag.")
print(order[0][1])

# 5) Create the PyVista window
plotter = pv.Plotter(window_size=(2000, 2000))
plotter.add_mesh(mesh, color="lightgray", opacity=0.5)

# 6) Show on-screen instruction
def show_instr(text):
    try:
        plotter.remove_actor("instr")
    except Exception:
        pass
    plotter.add_text(text, name="instr", font_size=14)
show_instr(order[0][1])

# 7) VTK picker setup
picker = vtk.vtkCellPicker()
picker.SetTolerance(0.0005)

# 8) Right-click callback: provisional pick
def on_right_click(obj, event):
    x, y = obj.GetEventPosition()
    picker.Pick(x, y, 0, plotter.renderer)
    pos = picker.GetPickPosition()
    pt = (float(pos[0]), float(pos[1]), float(pos[2]))
    state["last_pt"] = pt

    # remove old provisional
    try:
        plotter.remove_actor("provisional")
    except Exception:
        pass

    # draw provisional marker
    sphere = pv.Sphere(center=pt, radius=mesh.length * 0.005,
                       theta_resolution=16, phi_resolution=16)
    plotter.add_mesh(sphere, color="white", opacity=0.6, name="provisional")
    print(f"  → Picked provisional: {pt}\n    Now press SPACE or ENTER to confirm.")

plotter.iren.add_observer("RightButtonPressEvent", on_right_click)

# 9) Confirmation callback: lock in pick
def on_confirm():
    i = state["idx"]
    pt = state["last_pt"]
    if pt is None:
        print("  (!) No provisional pick yet—right-click first.")
        return
    
    if state["phase"] == "initial":
        name, _ = order[i]
    else:  # terminal phase
        name, _ = terminal_order[i]
        
    picked[name] = pt

    # remove provisional
    try:
        plotter.remove_actor("provisional")
    except Exception:
        pass

    # draw permanent sphere
    sphere = pv.Sphere(center=pt, radius=mesh.length * 0.005,
                       theta_resolution=16, phi_resolution=16)
    plotter.add_mesh(sphere, color=colors[i], name=f"confirmed_{name}")
    print(f"  ✔ Confirmed {name}: {pt}")

    # advance state
    state["last_pt"] = None
    state["idx"] += 1
    
    if state["phase"] == "initial":
        if state["idx"] < len(order):
            show_instr(order[state["idx"]][1])
        else:
            # Switch to terminal phase
            state["phase"] = "terminal"
            state["idx"] = 0
            show_instr(terminal_order[0][1])
    else:  # terminal phase
        if state["idx"] < len(terminal_order):
            show_instr(terminal_order[state["idx"]][1])
        else:
            show_instr("All points confirmed – close window to continue")
            plotter.iren.remove_observers("RightButtonPressEvent")

plotter.add_key_event("space",  on_confirm)
plotter.add_key_event("Return", on_confirm)

# 10) Show the UI (blocks until closed)
plotter.show()

# 11) After closing, verify picks
expected_total = len(order) + len(terminal_order)
if len(picked) != expected_total:
    raise RuntimeError(f"Expected {expected_total} picks, got {len(picked)}")

# 12) Save fiducials.json
filename_data = f"data/json/fiducials_{SUBJECT_ID}.json"
Path(filename_data).parent.mkdir(parents=True, exist_ok=True)
with open(filename_data, "w") as fp:
    json.dump(picked, fp, indent=2)
print(f"Saved to {filename_data}:", picked)

