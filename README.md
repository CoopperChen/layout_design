# layout_design

Place a standard 10–20 electrode set on one subject’s head scan and write 5-axis G-code for the electrode pads.

You bring a point cloud (`.ply`) and a textured head packaged as a `.zip` (`.obj`, `.mtl`, and `.jpg`, under any filenames). On the printer you measure the same three calibration points you picked on the scan. The program registers those two triangles and writes `electrodes.gcode`.

It does not plan wires. Run every command from the repository root.

**Repository:** https://github.com/CoopperChen/layout_design

## Install

Python **3.12** only.

```powershell
cd layout_design
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

On Linux or macOS, create the venv with `python3.12 -m venv .venv` and activate it with `source .venv/bin/activate`.

`layout <command>` is the same entry point as `python -m app <command>`. Later shells need the venv activated again.

Check that the printer file is present:

```powershell
python -m app paths --subject 1
```

`config/machine.yaml` should print as `ok`. Arm length, tool gap, and speeds live there and are shared by every subject. Motion tuning (slew, trace order, engage clearance) is `config/defaults.yaml`.

## What you need before a subject

1. A `.ply` point cloud. Units can be meters; the mesh step scales meters into millimeters.
2. A `.zip` of the textured head: one `.obj`, one `.mtl`, and one `.jpg` or `.png`. Names inside the zip do not matter.
3. Three physical marks on the head that you can both click on the texture and touch with the printer tip. The layout step calls them calibration **central**, **left**, and **back**.
4. For `record-pm` and `cnc` only: Mach4 publishing the work-coordinate pose. Setup is in [docs/RECORD_PM.md](docs/RECORD_PM.md). Mesh and layout do not need the machine.

Pick an integer subject id, for example `1`. Everything for that person goes in `data/subjects/1/`.

## Run a subject

```powershell
python -m app run --subject 1
```

That is `mesh`, then `layout`, then `record-pm`, then `gcode`. The command stops at the first failure. If `source/` is empty, two file dialogs open:

1. Choose the `.ply`.
2. Choose the `.zip`.

They are copied into the subject folder as `scan.ply`, `head.obj`, `head.mtl`, and `head.jpg`. The OBJ `mtllib` line and the MTL texture filename are rewritten to those names. You can skip the dialogs:

```powershell
python -m app run --subject 1 --ply D:\scans\cloud.ply --zip D:\scans\texture.zip
```

A rerun skips Poisson and ICP when `head.stl` and `head.obj` are already newer than the source files, and it skips `record-pm` when `pm.yaml` is already a valid triangle.

### Keys in every window

| Key | Action |
|-----|--------|
| Space, Enter, or S | Accept |
| Q or Esc | Discard |
| Close the window | Accept |

Closing a window keeps the current result. Q and Esc throw that step away.

### 1. Mesh

`mesh` builds one upright head in the scan frame.

1. **Outlier preview.** Gray points are kept, red points are dropped. Space, Enter, S, Q, or Esc only closes this preview.
2. **Normal flip.** `F` flips the point normals. Accept keeps the normals you are looking at. Discard restores the original direction. The crown of the head has to point up, because Cz is taken from the highest vertex.
3. Poisson reconstruction, density trim, and island removal write `head.stl`.
4. **Overlay check.** The textured OBJ is drawn on the gray STL. Accept if they coincide. Discard stops the mesh step.
5. **Upright rotation.** Arrow keys roll and pitch by 2.5°. `A` and `D` yaw. Keys `1` through `6` snap the camera to ±X, ±Y, and ±Z. Accept applies that rotation to both the OBJ and the STL. Discard leaves the ICP pose unchanged.

`--no-preview` skips the overlay. `--no-rotate-head` skips the upright rotation. `--no-scale` skips the scale match before ICP. `--depth` is the Poisson octree depth (default 12).

### 2. Layout

One textured view, seven picks, in this order:

1. Nasion
2. Left pre-auricular point (LPA)
3. Right pre-auricular point (RPA)
4. Inion
5. Calibration central
6. Calibration left
7. Calibration back

Right-click the surface, then Space, Enter, or S places that landmark. Keys `1` through `7` jump back to a landmark to place it again. `N` and `P` move forward and back in the list. After all seven are placed, Space, Enter, or S continues. Q or Esc discards the set.

The next window projects the 10–20 sites. Drag the size slider (default 0.8, range 0.5–1.5). Space, Enter, or S writes `layout.json`. Q or Esc discards it. Cz is the intersection of the nasion–inion and LPA–RPA slices through the crown, and each site is snapped to the nearest mesh vertex.

The three calibration clicks are the digital triangle. Touch those same three marks on the machine in the next step. They are not electrode sites.

### 3. Record the physical landmarks

Jog the tip to central, then left, then back. Space, Enter, or S captures the current landmark and, once all three are stored, writes `pm.yaml`. `N`, `P`, and `1`/`2`/`3` move among the three. Q or Esc aborts and writes nothing.

Central is stored as `[0, 0, 0]`. Left and back are stored relative to that touch, so you do not need to zero the work coordinates on the central mark. Mach4 must already be publishing the work pose on UDP port `62101`. If `pm.yaml` exists, the command refuses to overwrite it unless you pass `--force`, or `--force-record-pm` on `run`.

### 4. G-code

Each electrode becomes a coplanar zigzag disk. Pad area and zigzag row count are `electrode.area_cm2` and `electrode.nlines` in `config/defaults.yaml` (0.5 cm² and 10 rows). The disk sits `gap_size_mm` above the surface (15 mm in `config/machine.yaml`). It is registered from the digital triangle in `layout.json` to the physical triangle in `pm.yaml`, then the 5-axis postprocessor writes `data/subjects/1/electrodes.gcode`.

```powershell
python -m app gcode --subject 1 --electrode Cz
```

`--electrode all` is the default and writes every pad in trace order. A name or a 1-based index writes `{name}.gcode` in the same folder. `--rot0y` and `--rot0z` are optional bed rotations in degrees and must match if you later simulate.

## Settings

Edit `config/defaults.yaml`. Printer geometry stays in `config/machine.yaml`.

| Key | What it changes |
|-----|-----------------|
| `electrode.area_cm2`, `electrode.nlines` | Pad area and zigzag rows |
| `layout.scale`, `scale_min`, `scale_max` | 10–20 size and the slider range |
| `mesh.scale_to_mm` | PLY unit conversion (1000 when the scan is in metres) |
| `mesh.poisson_depth`, `poisson_density_quantile` | Surface detail and floater trim |
| `mesh.outlier_neighbors`, `outlier_std_ratio` | Point-cloud outlier removal |
| `mesh.island_min_size_ratio`, `island_min_cells` | Small disconnected pieces dropped after Poisson |
| `mesh.icp_samples`, `icp_max_correspondence_mm`, `icp_fitness_min`, `icp_mean_dist_max_mm` | Texture-to-mesh alignment |
| `postprocess:` | Slew, trace order, engage clearance |

`--depth` on `mesh` or `run` overrides `mesh.poisson_depth` for that command.

## Subject folder

```text
data/subjects/1/
  source/
    scan.ply            # your point cloud, renamed
    head.obj            # texture package, renamed; align does not overwrite this
    head.mtl
    head.jpg
  head.stl              # cleaned mesh, scan frame
  head.obj              # texture in that same upright frame
  head.mtl
  head.jpg
  layout.json           # anatomy, Cz, three calibration points, electrode xyz
  pm.yaml               # three physical landmarks, central at the origin
  electrodes.gcode
```

`layout.json` coordinates are millimeters in the scan frame. `python -m app paths --subject 1` prints each file and whether it exists.

## Start again from a later step

`--from` and `--to` are `mesh`, `layout`, `record-pm`, or `gcode`.

```powershell
python -m app run --subject 1 --from layout
python -m app run --subject 1 --from gcode
python -m app run --subject 1 --from record-pm --force-record-pm
```

Use `--from layout` when the mesh is already upright and you only want to repick landmarks. Use `--from gcode` after editing `layout.json` or `pm.yaml`. Mesh and layout can be repeated without the CNC. `record-pm` is the step that has to happen at the machine, and it is not repeated when G-code is regenerated.

## Check the toolpath, then send it

`simulate` and `cnc` are not part of `run`.

```powershell
python -m app simulate --subject 1
python -m app simulate --subject 1 --gcode data/subjects/1/Cz.gcode --animate
```

The viewer shows the mesh, landmarks, C-pivot path, tip, and arm in the machine frame. Use the same `--rot0y` and `--rot0z` you used for `gcode`.

```powershell
python -m app cnc status
python -m app cnc load --subject 1
python -m app cnc start --confirm
```

`load` sends that subject’s `electrodes.gcode`. `home` and `scan` load the scripts in `scripts/` and start only when you pass `--confirm`. `hold` and `stop` do not need `--confirm`.

## More detail

| Topic | Document |
|-------|----------|
| Every flag | [docs/CLI.md](docs/CLI.md) |
| What each stage writes | [docs/PIPELINE.md](docs/PIPELINE.md) |
| `layout.json` and `pm.yaml` fields | [docs/DATA_LAYOUT.md](docs/DATA_LAYOUT.md) |
| Mach4 UDP publisher | [docs/RECORD_PM.md](docs/RECORD_PM.md) |
| B/C, tool offset, registration | [docs/MACHINE_KINEMATICS.md](docs/MACHINE_KINEMATICS.md) |
