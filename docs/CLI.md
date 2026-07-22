# CLI reference

All commands run from the repository root:

```bash
python -m app <command> [options]
```

Console entry point (same interface): `layout <command> …`

---

## Overview by stage

| Stage | Commands |
|-------|----------|
| Setup | `init-data`, `paths` |
| **Full pipeline** | **`run`** (PLY → preprocess → synthesize → … → gcode/simulate) |
| A — Preprocess | `preprocess` |
| B — Layout | `build-assignments`, `synthesize`, `visualize` |
| C — Polish (optional) | `polish` |
| D — Postprocess | `smooth`, `export-bundle`, `init-print-config`, `record-pm`, `list-electrodes`, `convert-gcode`, `simulate-gcode`, `export-matlab` (legacy) |

Typical end-to-end (recommended):

```bash
python -m app init-data
python -m app build-assignments --reference 1 --id s1_assignments   # once
# Place data/raw/2.ply, then:
python -m app run --target 2
```

Step-by-step (debugging / partial reruns):

```bash
python -m app preprocess --subject 2 --step fiducials
python -m app synthesize --target 2
python -m app smooth --applied data/output/layouts/synth_s2.json
python -m app export-bundle --input data/output/smooth/smooth_s2_final.json
python -m app record-pm --subject 2                    # preferred: CNC DRO + keyboard
# python -m app init-print-config --subject 2          # empty scaffold only
python -m app convert-gcode --bundle data/output/bundles/subject_2
python -m app simulate-gcode --gcode data/output/gcode/subject_2_post/allinterconnects.txt --bundle data/output/bundles/subject_2
```

**Assignment preset:** default `subject1_best_v4` in `config/defaults.yaml` → `synthesize.assignments`. Not a CLI flag on `synthesize` or `run`.

---

## Full pipeline — `run`

Run preprocess through G-code (and optionally simulation) for one subject. Input is always a **PLY point cloud**; default path `data/raw/{target}.ply`.

```bash
python -m app run --target 2
python -m app run --target 2 --from synthesize          # skip preprocess
python -m app run --target 2 --to simulate
python -m app run --target 2 --no-polish --from synthesize
```

### Stages (in order)

| Stage | Interactive? | Confirm keys | Output / effect |
|-------|--------------|--------------|-----------------|
| `reconstruct` | align / normals | Space/Enter/S = confirm · Esc/Q = skip/cancel · close = confirm | `data/raw/{id}.stl`, `{id}.obj` |
| `clear-islands` | no | — | `data/cleaned_scans/{id}.stl` |
| `fiducials` | **yes** | Space/Enter = confirm pick · S/close = save · Q = discard | `data/json/fiducials_{id}.json` |
| `cz` | **preview** | Space/Enter/S/close = save · Q = discard | `data/json/Cz_{id}.json` |
| `electrodes` | **yes** | Space/Enter/S/close = save · Q = discard | `data/json/electrode_positions_{id}.json` |
| `synthesize` | no | — | `data/output/layouts/synth_s{id}.json` |
| `polish` | no | — | `*_repaired.json` (skip with `--no-polish`) |
| `smooth` | no | — | `data/output/smooth/smooth_s{id}_final.json` |
| `bundle` | no | — | `data/output/bundles/subject_{id}/` |
| `print-config` | no | — | empty pm YAML scaffold if missing |
| `record-pm` | **yes (CNC)** | Enter/Space = capture (save when all 3 done) · `s` = save · `q` = quit | measured `physical_landmarks_mm` in pm YAML |
| `gcode` | no | — | `data/output/gcode/subject_{id}_post/` (requires measured pm) |
| `simulate` | viewer | — | PyVista 3D viewer |

**GUI convention (all save-capable steps):** **Space / Enter / S** confirm or save · **Q** discard · **closing the window** saves (same as confirm), unless you pressed Q first. Fiducials: Space/Enter confirms each pick; S/close finishes and writes the file.

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--target` | — | Subject id (required) |
| `--ply` | `data/raw/{target}.ply` | Input point cloud |
| `--from` | `reconstruct` | First stage |
| `--to` | `gcode` | Last stage (`simulate` opens viewer) |
| `--no-polish` | off | Skip polish between synthesize and smooth |
| `--polish-mode` | `gentle` | `gentle`, `repair`, `refine`, `ga-short` |
| `--no-align-head` | off | Skip head rotation UI in reconstruct |
| `--depth` | config (`12`) | Poisson octree depth |
| `--preserve-entry-order` | off | Synthesize: keep reference entry order |
| `--inherit-preset-terminals` | off | Synthesize: legacy rigid hub map |
| `--rotate` | off | Synthesize: ±36° hub angle search around fiducial clicks |
| `--uv-resolution` | `100` | Synthesize UV grid |
| `--smooth-tag` | `final` | Smooth output tag |
| `--smoothing-strength` | config | B-spline smoothing factor |
| `--allow-terminal-landmarks` | off | Bundle export without calibration landmarks |
| `--skip-validation` | off | Skip export validation |
| `--quiet` | off | Quiet bundle export |
| `--force-print-config` | off | Overwrite existing pm YAML scaffold |
| `--force-record-pm` | off | Re-capture CNC landmarks even if pm already measured |
| `--pm-port` | `62100` | Mach4 work-pose UDP port (`record-pm`) |
| `--pm-bind-ip` | `0.0.0.0` | UDP bind for `record-pm` |
| `--pm-stale-ms` | `500` | Stale pose threshold for `record-pm` |
| `--config` / `--pm-file` | auto | pm YAML for G-code |
| `--machine` | `machine_default.yaml` | Machine config |
| `--gcode-output` | `data/output/gcode/` | G-code base dir |
| `--trace` | `both` | `interconnect`, `electrode`, or `both` |
| `--electrode` | `all` | Channel filter for G-code |
| `--rot0y`, `--rot0z` | `0` | Bed rotation (deg) |
| `--legacy-subject` | — | Legacy `.mat` folder |
| `--layers` | `mesh,landmarks,origin,tip,arm` | Simulator layers |
| `--animate` | off | Simulator: step with `p` |
| `--verbose` | off | Simulator FK diagnostics |

Stops on first failure. Resume with `--from <stage>`.

---

## Setup

### `init-data`

Create the standard `data/` directory tree if missing.

```bash
python -m app init-data
```

**Output:** `data/raw/`, `data/cleaned_scans/`, `data/json/`, `data/output/`, `config/postprocessor/`, etc.

---

### `paths`

Print canonical file paths for a subject (debugging / copy-paste).

```bash
python -m app paths --subject 2
```

| Argument | Required | Description |
|----------|----------|-------------|
| `--subject` | yes | Subject id |

**Shows:** preprocess inputs, default assignment preset, layout output, smooth JSON, bundle dir, gcode dir, pm config, legacy matlab dir.

---

## Stage A — Preprocess

### `preprocess`

Interactive and batch preprocess steps for one subject.

```bash
python -m app preprocess --subject 2 --step fiducials
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--subject` | — | Subject id (required) |
| `--step` | — | Step name (required); see table below |
| `--ply` | — | Input `.ply` for `reconstruct` |
| `--no-align-head` | off | Skip head rotation UI in reconstruct |
| `--depth` | `12` | Poisson octree depth (`reconstruct`) |
| `--spacing` | `4.5` | Electrode spacing (`electrodes`) |
| `--full-circle` | off | Full 10–20 circle (`electrodes`) |

**Steps:**

| Step | Purpose |
|------|---------|
| `reconstruct` | PLY point cloud → `data/raw/{id}.stl` + textured OBJ |
| `clear-islands` | Remove mesh islands → `data/cleaned_scans/{id}.stl` |
| `fiducials` | Pick anatomical points, terminals, calibration landmarks (OBJ UI) |
| `cz` | Place Cz electrode |
| `electrodes` | Place 10–20 electrode positions |
| `assignments` | Initial LEFT/RIGHT terminal assignments |
| `entry-capacity` | Report strip entry capacity for terminal zones |

---

## Stage B — Layout

### `build-assignments`

Build an assignment-only preset from a reference subject's terminal assignments.

```bash
python -m app build-assignments --reference 1 --id s1_assignments
```

| Argument | Required | Description |
|----------|----------|-------------|
| `--reference` | yes | Reference subject id |
| `--id` / `--preset-id` | yes | Output preset name (saved under `data/presets/`) |
| `--out` | no | Override output path |

**Output:** `data/presets/{id}.json` with `terminal_assignments` only.

### `synthesize`

Generate per-subject wire layout on a target head. Uses the assignment preset from **`config/defaults.yaml`** (`synthesize.assignments`, default `s1_assignments`).

```bash
python -m app synthesize --target 2 --visualize
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--target` | — | Target subject id (required) |
| `--out` | auto | Output layout JSON path |
| `--preserve-entry-order` | off | Keep reference strip slot order from full v4 preset |
| `--inherit-preset-terminals` | off | Legacy: rigid-map reference hub positions |
| `--rotate` | off | ±36° hub angle search around fiducial clicks (may reduce crossings) |
| `--uv-resolution` | `100` | UV grid resolution for 3D lift |
| `--visualize` | off | After synth: 2D PNG + interactive 3D |
| `--show` | off | Also open 2D matplotlib window |
| `--no-show` | off | With `--visualize`: save PNG only, no 3D window |
| `--skip-collisions` | off | Skip collision markers in visualize |

**Output:** `data/output/layouts/synth_s{id}.json`

To change the preset, edit `synthesize.assignments` in `config/defaults.yaml` or run `build-assignments` to create `data/presets/{name}.json`.

---

### `visualize`

2D polar PNG and/or interactive 3D view of a layout JSON.

```bash
python -m app visualize --applied data/output/layouts/synth_s2.json
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--applied` | — | Layout JSON path (required) |
| `--mode` | `both` | `2d`, `3d`, or `both` |
| `--save` | auto | Override 2D PNG path |
| `--save-3d` | — | Optional 3D screenshot path |
| `--no-show` | off | Do not open 3D window |
| `--show-2d` | off | Open interactive 2D window |
| `--skip-collisions` | off | Skip 2D collision markers |

**Note:** Use **layout JSON** (`synth_s*.json`), not smooth JSON.

---

## Stage C — Polish (optional)

### `polish`

Separation polish / repair on an existing layout (not layout discovery).

```bash
python -m app polish --applied data/output/layouts/synth_s2.json --mode gentle
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--applied` | — | Input layout JSON (required) |
| `--mode` | `gentle` | `gentle`, `repair`, `refine`, `ga-short` |
| `--out` | auto | Output layout JSON |
| `--subject` | — | Subject id (required for `ga-short`) |
| `--generations` | — | GA generations (`ga-short`) |
| `--population` | — | GA population (`ga-short`) |
| `--no-clear-logs` | off | Keep previous polish logs |
| `--no-mutate-gen0` | off | Disable gen-0 mutation (`ga-short`) |
| `--electrodes-only` | off | Polish electrode zones only |
| `--visualize` | off | Save 2D + 3D PNGs after polish |

**Output:** e.g. `synth_s2_repaired.json`, `synth_s2_refined.json`

---

## Stage D — Postprocess

### `smooth`

B-spline smooth 3D interconnect paths from layout JSON.

```bash
python -m app smooth --applied data/output/layouts/synth_s2.json
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--applied` | — | Layout JSON (required) |
| `--out` | auto | Output smooth JSON (`smooth_s{id}_final.json`) |
| `--tag` | `final` | Output filename tag |
| `--strength` | config | B-spline smoothing factor (e.g. `0.1`) |

**Output:** `data/output/smooth/smooth_s{id}_{tag}.json` (includes `collision_metrics` from layout when present)

---

### `export-bundle`

Export canonical subject bundle for G-code conversion.

```bash
python -m app export-bundle --input data/output/smooth/smooth_s2_final.json
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | — | Smooth JSON (required) |
| `--output` | auto | `data/output/bundles/subject_{id}/` |
| `--allow-terminal-landmarks` | off | Skip calibration landmark requirement (not recommended) |
| `--skip-validation` | off | Skip collision-free / path checks (not recommended) |

**Output:**

```
data/output/bundles/subject_{id}/
  manifest.json
  geometry.npz
  traces.npz
```

**Validation (default):** collision-free layout, valid paths, calibration landmarks in `fiducials_{id}.json`.

**Electrode traces (`traces.npz` → `electrode_xyzn`):** planar disk zigzag at `surface + gap_size_mm` along the outward normal (default gap 15 mm from `config/postprocessor/machine_default.yaml`). Interconnect traces remain on the scalp surface. Manifest field `electrode_matlab.electrode_coords_include_gap: true` records this for `convert-gcode`.

---

### `init-print-config`

Create pm-only YAML scaffold for physical landmark registration at print time.

```bash
python -m app init-print-config --subject 2
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--subject` | — | Subject id (required) |
| `--force` | off | Overwrite existing file |

**Output:** `config/postprocessor/subjects/subject_{id}.yaml`

Prefer **`record-pm`** to fill values from the live CNC. Manual edit of `physical_landmarks_mm` remains supported. Full guide: [config/postprocessor/README.md](../config/postprocessor/README.md).

---

### `record-pm`

Capture `physical_landmarks_mm` from the live CNC **work** DRO (UDP) with keyboard confirmation.

**Prerequisites:** Mach4 publishing JSON work pose via [`scripts/mach4_work_pose_publisher.lua`](../scripts/mach4_work_pose_publisher.lua) (default port `62100`).

```bash
python -m app record-pm --subject 2
python -m app record-pm --subject 2 --force
python -m app record-pm --subject 2 --port 62100 --bind-ip 0.0.0.0
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--subject` | — | Subject id (required) |
| `--port` | `62100` | UDP listen port |
| `--bind-ip` | `0.0.0.0` | Bind address |
| `--stale-ms` | `500` | Ignore packets older than this many ms |
| `--output` | auto | Override YAML path |
| `--force` | off | Overwrite existing file |

**Keys:** Enter/Space = capture (or save when all 3 done) · 1/2/3 = jump · n/p = next/prev · s = save · q = quit.

**Order:** `landmark_central` → `landmark_left` → `landmark_back`. Central becomes `[0,0,0]`; left/back are relative to the central touch DRO.

**Output:** `config/postprocessor/subjects/subject_{id}.yaml` (plus optional `capture:` audit block).

Details (Mach4 setup, packet format, frames): [config/postprocessor/README.md](../config/postprocessor/README.md).

---

### `list-electrodes`

List channel names and indices in a subject bundle.

```bash
python -m app list-electrodes --bundle data/output/bundles/subject_2
```

| Argument | Required | Description |
|----------|----------|-------------|
| `--bundle` | yes | Bundle directory |

---

### `convert-gcode`

Convert subject bundle to 5-axis G-code.

```bash
python -m app convert-gcode --bundle data/output/bundles/subject_2
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--bundle` | — | Bundle directory (required) |
| `--config` | auto | pm YAML; default `config/postprocessor/subjects/subject_{id}.yaml` |
| `--pm-file` | — | Alias for `--config` |
| `--machine` | `config/postprocessor/machine_default.yaml` | Machine geometry and speeds |
| `--output` | `data/output/gcode/` | Output base directory |
| `--trace` | `both` | `both`, `interconnect`, or `electrode` |
| `--electrode` | `all` | `all`, channel name (`C3`), or 1-based index |
| `--rot0y` | `0` | Registration Y rotation (degrees) |
| `--rot0z` | `0` | Registration Z rotation (degrees) |
| `--subject` | — | Legacy `.mat` folder instead of bundle |

**Default output** (`--trace both`, `--electrode all`):

```
data/output/gcode/subject_{id}_post/
  allinterconnects.txt
  allelectrode.txt
```

**Single channel** (`--electrode C3`): `C3interconnect.txt` + `C3electrode.txt`

**Trace override examples:**

```bash
python -m app convert-gcode --bundle ... --trace interconnect   # wires only
python -m app convert-gcode --bundle ... --trace electrode      # pads only
```

---

### `simulate-gcode`

Interactive 3D PyVista viewer: **forward G-code execution** — programmed **X,Y,Z = C pivot**, **B,C** drive rigid arm/tool (Rz(−C), Rx(−B)).

```bash
python -m app simulate-gcode \
  --gcode data/output/gcode/subject_4_post/allinterconnects.txt \
  --bundle data/output/bundles/subject_4 \
  --verbose
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--gcode` | — | G-code `.txt` file (required) |
| `--bundle` | — | Subject bundle directory (required) |
| `--pm-file` | auto | Physical landmarks YAML (`config/postprocessor/subjects/subject_{id}.yaml`) |
| `--machine-config` | `config/postprocessor/machine_default.yaml` | Printer geometry (`a_mm`, `d_mm`, machine zero) |
| `--rot0y` | `0` | Bed Y rotation (degrees), **must match convert-gcode** |
| `--rot0z` | `0` | Bed Z rotation (degrees), **must match convert-gcode** |
| `--layers` | `mesh,landmarks,origin,tip,arm` | Comma-separated: `mesh`, `landmarks`, `origin`, `cnc`, `tip`, `arm`, `programmed` |
| `--animate` | off | `p` key advances one G-code step (slider always available) |
| `--verbose` | off | Rigid FK checks, registration fit, decode vs FK metrics (machine frame) |

**Layer keys (toggle in viewer):**

| Key | Layer | Content |
|-----|-------|---------|
| `m` | mesh | Registered head mesh |
| `l` | landmarks | Calibration landmarks (machine frame) |
| `o` | origin | Machine-zero C pivot, tip, central landmark |
| `c` | cnc | C-axis pivot path (G-code X,Y,Z) |
| `g` | programmed | Programmed XYZ markers |
| `t` | tip | Forward FK nozzle tip (M10/M11 colored) |
| `a` | arm | Rigid arm skeleton (C→B→tip) |

**Coordinate frame:** mesh, landmarks, and G-code paths are all in **controller machine frame** — C pivot zero at `(0,0,0)`. Central landmark sits at `(0, −a, −(d+calgap_z))` in this frame. See [MACHINE_KINEMATICS.md](MACHINE_KINEMATICS.md).

**Registration:** use the same `--pm-file`, `--rot0y`, and `--rot0z` as `convert-gcode`. The simulator shifts scan2phys output into machine frame automatically; G-code is already in machine frame.

**Kinematics:** forward execution — programmed **X,Y,Z = C pivot**; **B,C** drive rigid arm (C=0 → arm +X; C=90 → arm −Y). No postprocessor compensation at runtime.

**Landmarks:** Digital landmarks from bundle (`geometry.npz`); physical `pm` from YAML — see [config/postprocessor/README.md](../config/postprocessor/README.md).

**Viewer HUD:** current step shows X, Y, Z, B, C. `--verbose` prints arm length a, tool length d, arm⊥tool, and tip vs mesh distance.

---

### `export-matlab` (legacy)

Export four `.mat` files for MATLAB `gcodeConverter_final14.m`.

```bash
python -m app export-matlab --input data/output/smooth/smooth_s2_final.json
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | — | Smooth JSON (required) |
| `--output` | auto | `data/output/matlab/subject_{id}/` |
| `--allow-terminal-landmarks` | off | Allow missing calibration landmarks |
| `--skip-validation` | off | Skip export validation gates |

**Output:** `InterconnectElectrodePaths.mat`, `HeadMesh.mat`, `Landmarks.mat`, `LandmarkNames.mat`

Prefer `export-bundle` → `convert-gcode` for the Python path.

---

## Help

```bash
python -m app --help
python -m app run --help
python -m app synthesize --help
python -m app convert-gcode --help
python -m app simulate-gcode --help
```

## Related docs

- [ARCHITECTURE.md](ARCHITECTURE.md) — repository layout
- [GETTING_STARTED.md](GETTING_STARTED.md) — environment and first run
- [PIPELINE.md](PIPELINE.md) — stage overview
- [MACHINE_KINEMATICS.md](MACHINE_KINEMATICS.md) — arm FK, machine frame, simulator
- [DATA_LAYOUT.md](DATA_LAYOUT.md) — file formats
- [config/postprocessor/README.md](../config/postprocessor/README.md) — pm measurement and print config
