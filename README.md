# layout_design

**Generate** per-subject EEG interconnect layouts (2D/3D wire paths on the scalp). Each head uses its own mesh, electrodes, and terminal clicks; the only cross-subject input is **which electrode goes to LEFT vs RIGHT** (the “preset” = terminal assignment map).

**Not** applying a reference GA layout or rigidly mapping reference hub positions (that is legacy / opt-in).

**Repository:** https://github.com/CoopperChen/layout_design

**Run from repository root** (`layout_design/`).

```powershell
python -m app init-data
python -m app paths --subject 2
```

See [docs/GOAL.md](docs/GOAL.md) for scope.

## Pipeline

| Stage | Purpose |
|-------|---------|
| **A. Preprocess** | Mesh, fiducials, terminals, 10–20, optional local assignments |
| **B. Generate layout** | Assignments file + target → `data/output/layouts/synth_s{id}.json` |
| **C. Polish** (optional) | Separation polish only — not layout discovery |
| **D. Postprocess** | Smooth → bundle → G-code → **simulate** (3D viewer) |

| Stage | Directory | Output |
|-------|-----------|--------|
| A | `app/preprocess/` | `data/json/`, `data/cleaned_scans/` |
| B | `app/layout/` | `data/output/layouts/` |
| C | `app/polish/` | `data/output/layouts/*_repaired.json` |
| D | `app/postprocess/` + `app/simulator/` | `data/output/smooth/`, `bundles/`, `gcode/`; PyVista viewer |

Details: [docs/PIPELINE.md](docs/PIPELINE.md) · [docs/DATA_LAYOUT.md](docs/DATA_LAYOUT.md) · **[docs/CLI.md](docs/CLI.md)** (all commands)

## Terminal entries (target-native)

Synthesize uses **fiducial-native** strip zones by default (`terminal_2d_mode: fiducial_native`):

- Terminal safety zones are centered on your **TERMINAL_LEFT / TERMINAL_RIGHT** clicks (polar projection), not on an inflated legacy hub radius.
- Each wire ends at a **strip entry** on that zone boundary (fixed spacing, collision-free fan-in), not at the hub center.
- 3D endpoints use UV surface lift with a light blend toward the hub for entries near the strip center.

**3D visualize markers**

| Color | Meaning |
|-------|---------|
| Gray | Hub clicks (`TERMINAL_LEFT` / `TERMINAL_RIGHT`) |
| Lime | Wire ends (`entry_position_3d`) |
| Cyan | Path splines |

## Repository layout

```text
layout_design/
├── app/
│   ├── preprocess/         # Stage A
│   ├── layout/             # Stage B
│   ├── polish/             # Stage C (optional)
│   ├── postprocess/        # Stage D: smooth, bundle, convert-gcode
│   └── simulator/          # G-code 3D viewer (forward FK + machine-frame mesh)
├── config/                 # defaults.yaml + postprocessor machine/pm configs
├── data/                   # Pipeline I/O (see data/README.md)
├── docs/                   # CLI.md, PIPELINE.md, MACHINE_KINEMATICS.md, …
├── legacy_gcode_examples/  # MATLAB reference (optional)
├── scripts/                # Sample data helpers
└── tests/                  # Contract + simulator tests
```

Full map: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## Quick start

```powershell
# 1. Prep subject 2 — place {id}.ply under data/raw/, then reconstruct → STL + OBJ
python -m app preprocess --subject 2 --step reconstruct      # PLY → raw/{id}.stl + .obj
python -m app preprocess --subject 2 --step clear-islands   # STL → cleaned_scans/
python -m app preprocess --subject 2 --step fiducials       # OBJ only (textured picking)
python -m app preprocess --subject 2 --step cz
python -m app preprocess --subject 2 --step electrodes

# 2. Assignment map (or use bundled subject1_best_v4)
python -m app build-assignments --reference 1 --id s1_assignments
# requires data/json/fiducials_1.json — or skip and use:
#   --assignments subject1_best_v4

# 3. Generate layout on subject 2 (paths created here — not copied from S1)
python -m app synthesize --assignments subject1_best_v4 --target 2 --visualize

# 4. Optional polish → print prep
python -m app polish --applied data/output/layouts/synth_s2.json --mode gentle
python -m app smooth --applied "data/output/layouts/synth_s2.json" --out "data/output/smooth/smooth_s2_final.json"
python -m app export-bundle --input data/output/smooth/smooth_s2_final.json
python -m app init-print-config --subject 2
python -m app convert-gcode --bundle data/output/bundles/subject_2

# 5. Verify toolpath on registered head mesh (same pm + rot0 as convert-gcode)
python -m app simulate-gcode \
  --gcode data/output/gcode/subject_2_post/allinterconnects.txt \
  --bundle data/output/bundles/subject_2
```

**G-code simulator:** forward kinematics — programmed **X,Y,Z = C pivot**; mesh and paths share **controller machine frame** (C at origin). See [docs/MACHINE_KINEMATICS.md](docs/MACHINE_KINEMATICS.md).

Legacy MATLAB export (optional): `python -m app export-matlab --input data/output/smooth/smooth_s2_final.json`

Use **layout JSON** for `visualize` and **smooth JSON** for `smooth` / `export-bundle`:

```powershell
python -m app visualize --applied data/output/layouts/synth_s2.json
```

Use genetic_SHAPE venv if system Python lacks the scientific stack:

`D:\Research\genetic_layout_design\genetic_SHAPE\genetic\Scripts\python.exe -m app …`

## CLI

Full command reference (all stages, arguments, defaults): **[docs/CLI.md](docs/CLI.md)**

```bash
python -m app --help
python -m app convert-gcode --help
```

## Agent skill

`@layout-synth-pipeline`

## Status

Core path: preprocess → **generate** (`synthesize`) → optional polish → postprocess. Target fiducial hubs + fiducial-native strip entries are default.
