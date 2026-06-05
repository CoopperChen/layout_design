import os
import pyvista as pv
import numpy as np
import json
from pathlib import Path

SUBJECT_ID = int(os.environ.get("LAYOUT_SUBJECT_ID", "1"))

# 1) Load
mesh = pv.read(f"data/cleaned_scans/{SUBJECT_ID}.stl")
fid  = json.load(open(f"data/json/fiducials_{SUBJECT_ID}.json"))  # has nasion, lpa, rpa, inion

# 2) Compute “vertex” as highest‐Z mesh point
pts = mesh.points
crown_idx = np.argmax(pts[:, 2])
crown     = pts[crown_idx]
print("Auto‐detected crown at", crown.tolist())

# 3) Helper: build a slice loop through three points P,Q,R
def slice_loop(P, Q, R):
    # plane normal = (Q−P)×(R−P)
    n = np.cross(Q - P, R - P)
    return mesh.slice(normal=n, origin=P)

# 4) Get the two loops
loop_fb = slice_loop(np.array(fid['nasion']),
                     np.array(fid['inion']),
                     crown)
loop_lr = slice_loop(np.array(fid['lpa']),
                     np.array(fid['rpa']),
                     crown)

# 5) Find their nearest crossing → Cz
A = loop_fb.points    # (N1,3)
B = loop_lr.points    # (N2,3)
d2 = np.sum((A[:,None,:] - B[None,:,:])**2, axis=2)
i, j = np.unravel_index(np.argmin(d2), d2.shape)
Cz = 0.5*(A[i] + B[j])
print("Cz = ", Cz.tolist())

# 6) Preview
pl = pv.Plotter(window_size=(800,600))
pl.add_mesh(mesh, color="cyan", opacity=0.3)
pl.add_mesh(loop_fb, color="red",   line_width=3)
pl.add_mesh(loop_lr, color="green", line_width=3)
pl.add_mesh(pv.Sphere(center=Cz, radius=mesh.length*0.01),
            color="yellow")
pl.add_text(f"FB loop (red), LR loop (green), Cz (yellow)", font_size=12)
pl.show()  # inspect in 3D

# 7) Save Cz
filename_data = f"data/json/Cz_{SUBJECT_ID}.json"
Path(filename_data).parent.mkdir(parents=True, exist_ok=True)
with open(filename_data, "w") as f:
    json.dump({"Cz": Cz.tolist()}, f, indent=2)
print(f"Wrote {filename_data}")