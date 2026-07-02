# layout_design

**Generate** per-subject EEG interconnect layouts (2D/3D wire paths on the scalp). Each head uses its own mesh, electrodes, and terminal clicks; the only cross-subject input is **which electrode goes to LEFT vs RIGHT** (the “preset” = terminal assignment map).

**Not** applying a reference GA layout or rigidly mapping reference hub positions (that is legacy / opt-in).

**Repository:** https://github.com/CoopperChen/layout_design

**Run from repository root** (`layout_design/`).

```powershell
python -m app init-data
python -m app paths --subject 2
```

Console alias: `layout <command> …` (same as `python -m app`).

See [docs/GOAL.md](docs/GOAL.md) for scope.

---

## Recommended workflow: `run`

For a new subject, the usual input is a **PLY point cloud**. Use a single command to walk through preprocess → layout → G-code:

```powershell
# 1. Place scan: data/raw/2.ply
# 2. One-time: build assignment map from reference subject 1
python -m app build-assignments --reference 1 --id s1_assignments

# 3. Full pipeline (interactive fiducials + electrodes in the middle)
python -m app run --target 2
```

**Default stages** (`--from reconstruct` → `--to gcode`):

| Stage | Type | What it does |
|-------|------|----------------|
| `reconstruct` | automated | PLY → `data/raw/{id}.stl` + textured `.obj` (Poisson mesh) |
| `clear-islands` | automated | Remove small mesh islands → `data/cleaned_scans/{id}.stl` |
| `fiducials` | **interactive** | Pick nasion, LPA, RPA, inion, terminals, calibration landmarks on OBJ |
| `cz` | automated | Compute Cz from anatomy slices |
| `electrodes` | **interactive** | Place / confirm 10–20 electrode positions |
| `synthesize` | automated | Generate wire layout → `data/output/layouts/synth_s{id}.json` |
| `smooth` | automated | B-spline 3D paths → `data/output/smooth/smooth_s{id}_final.json` |
| `bundle` | automated | Export `data/output/bundles/subject_{id}/` |
| `print-config` | automated | Create pm YAML if missing (`config/postprocessor/subjects/subject_{id}.yaml`) |
| `gcode` | automated | Write `data/output/gcode/subject_{id}_post/allinterconnects.txt` (+ electrodes) |

Optional stages (not in default run):

| Stage | How to include |
|-------|----------------|
| `polish` | Add `--polish` (runs between synthesize and smooth) |
| `simulate` | Add `--to simulate` (opens PyVista G-code viewer at the end) |

The pipeline **stops on first failure** and prints which stage failed. Resume with `--from <stage>`.

### `run` — common examples

```powershell
# Default: PLY at data/raw/2.ply, preset s1_assignments, through G-code
python -m app run --target 2

# Custom PLY path
python -m app run --target 2 --ply D:\scans\subject2.ply

# Skip preprocess when mesh + fiducials + electrodes already exist
python -m app run --target 2 --from synthesize

# Resume after a failed export
python -m app run --target 2 --from bundle

# Polish + open simulator at the end
python -m app run --target 2 --polish --to simulate

# Preprocess only (stop before layout generation)
python -m app run --target 2 --to electrodes
```

### `run` — all options

#### Subject & input

| Option | Default | Description |
|--------|---------|-------------|
| `--target` | *(required)* | Subject id (used for all paths: `data/raw/{id}.ply`, `synth_s{id}.json`, etc.) |
| `--ply` | `data/raw/{target}.ply` | Input point cloud for reconstruct |

#### Stage range

| Option | Default | Description |
|--------|---------|-------------|
| `--from` | `reconstruct` | First stage to run (see stage table above) |
| `--to` | `gcode` | Last stage to run; use `simulate` for 3D viewer |
| `--polish` | off | Insert `polish` between synthesize and smooth |
| `--polish-mode` | `gentle` | `gentle`, `repair`, `refine`, or `ga-short` |

Valid `--from` / `--to` values (in order):

`reconstruct` → `clear-islands` → `fiducials` → `cz` → `electrodes` → `synthesize` → `polish` → `smooth` → `bundle` → `print-config` → `gcode` → `simulate`

#### Preprocess (reconstruct)

| Option | Default | Description |
|--------|---------|-------------|
| `--no-align-head` | off | Skip head rotation UI during Poisson reconstruct |
| `--depth` | `12` (config) | Poisson octree depth; override `preprocess.poisson_depth` in `config/defaults.yaml` |

#### Synthesize (passed through to layout generation)

| Option | Default | Description |
|--------|---------|-------------|
| *(preset)* | `s1_assignments` | Terminal LEFT/RIGHT map — **not a CLI flag**; set in `config/defaults.yaml` → `synthesize.assignments` |
| `--preserve-entry-order` | off | Keep reference strip slot order (full v4 presets only) |
| `--inherit-preset-terminals` | off | **Legacy:** rigid-map reference hub positions onto target |
| `--fix-terminals` | off | Use exact hub clicks (no ±36° hub angle search) |
| `--uv-resolution` | `100` | UV grid resolution for 3D path lift |

#### Smooth & export

| Option | Default | Description |
|--------|---------|-------------|
| `--smooth-tag` | `final` | Output filename tag (`smooth_s{id}_{tag}.json`) |
| `--smoothing-strength` | config | B-spline factor; default from `postprocess.smoothing_strength` |
| `--allow-terminal-landmarks` | off | Export without calibration landmarks (not recommended) |
| `--skip-validation` | off | Skip collision-free / path checks on export (not recommended) |
| `--quiet` | off | Suppress per-channel bundle export progress |

#### Print config & G-code

| Option | Default | Description |
|--------|---------|-------------|
| `--force-print-config` | off | Overwrite existing pm YAML |
| `--config` / `--pm-file` | auto | Physical landmarks YAML for registration |
| `--machine` | `config/postprocessor/machine_default.yaml` | Machine geometry and speeds |
| `--gcode-output` | `data/output/gcode/` | G-code output base directory |
| `--trace` | `both` | `interconnect`, `electrode`, or `both` |
| `--electrode` | `all` | `all`, channel name (`C3`), or 1-based index |
| `--rot0y`, `--rot0z` | `0` | Bed rotation (degrees); must match simulate |
| `--legacy-subject` | — | Legacy `.mat` folder instead of bundle |

#### Simulator (when `--to simulate`)

| Option | Default | Description |
|--------|---------|-------------|
| `--layers` | `mesh,landmarks,origin,tip,arm` | Comma-separated viewer layers |
| `--animate` | off | `p` key advances one G-code step |
| `--verbose` | off | FK diagnostics, registration fit, tip vs mesh metrics |

```powershell
python -m app run --help    # full list on your install
```

---

## Configuration (`config/defaults.yaml`)

Pipeline defaults live in **`config/defaults.yaml`**. Change these once instead of passing flags every run.

```yaml
preprocess:
  terminal_assignment_strategy: balanced   # balanced | shortest
  poisson_depth: 12                      # reconstruct (--depth overrides)
  align_head: true                       # head rotation UI (--no-align-head disables)
  electrode_spacing: 4.5
  full_circle: false

synthesize:
  assignments: s1_assignments              # data/presets/{name}.json — LEFT/RIGHT map
  use_target_terminals: true
  optimize_terminals: true
  preserve_entry_order: false
  uv_resolution: 100

polish:
  mode: gentle
  ga_generations: 20
  ga_population: 20

postprocess:
  smoothing_strength: 0.1
```

**Assignment preset:** `synthesize.assignments` names a file under `data/presets/` (e.g. `s1_assignments.json`). Create it once:

```powershell
python -m app build-assignments --reference 1 --id s1_assignments
```

Requires `data/json/initial_terminal_assignments_1.json` (or run preprocess assignments on subject 1).

`python -m app paths --subject 2` prints canonical paths including the active assignment preset.

| Environment variable | Effect |
|---------------------|--------|
| `LAYOUT_DESIGN_ROOT` | Override repository root in `app/paths.py` |

---

## Pipeline overview

| Stage | Purpose |
|-------|---------|
| **A. Preprocess** | PLY → mesh, fiducials, terminals, 10–20 electrodes |
| **B. Generate layout** | Assignment map + target geometry → `synth_s{id}.json` |
| **C. Polish** (optional) | Separation polish only — not layout discovery |
| **D. Postprocess** | Smooth → bundle → G-code → **simulate** (3D viewer) |

| Stage | Directory | Output |
|-------|-----------|--------|
| A | `app/preprocess/` | `data/json/`, `data/cleaned_scans/` |
| B | `app/layout/` | `data/output/layouts/` |
| C | `app/polish/` | `data/output/layouts/*_repaired.json` |
| D | `app/postprocess/` + `app/simulator/` | `data/output/smooth/`, `bundles/`, `gcode/` |

Details: [docs/PIPELINE.md](docs/PIPELINE.md) · [docs/DATA_LAYOUT.md](docs/DATA_LAYOUT.md) · **[docs/CLI.md](docs/CLI.md)** (per-command reference)

---

## Individual commands

Use step-by-step commands when debugging one stage or scripting partial workflows.

### Setup

```powershell
python -m app init-data              # create data/ tree
python -m app paths --subject 2      # print canonical paths + default preset
```

### Stage A — `preprocess`

```powershell
python -m app preprocess --subject 2 --step reconstruct       # PLY → STL + OBJ
python -m app preprocess --subject 2 --step clear-islands
python -m app preprocess --subject 2 --step fiducials           # interactive
python -m app preprocess --subject 2 --step cz
python -m app preprocess --subject 2 --step electrodes          # interactive
python -m app preprocess --subject 2 --step assignments         # optional local map
python -m app preprocess --subject 2 --step entry-capacity
```

| Option | Default | Description |
|--------|---------|-------------|
| `--subject` | required | Subject id |
| `--step` | required | Step name (see table in [docs/CLI.md](docs/CLI.md)) |
| `--ply` | `data/raw/{id}.ply` | Input for `reconstruct` |
| `--no-align-head` | off | Skip rotation UI |
| `--depth` | `12` | Poisson depth |
| `--spacing` | `4.5` | Electrode spacing (`entry-capacity`) |
| `--full-circle` | off | Full 10–20 circle |

### Stage B — layout

```powershell
python -m app build-assignments --reference 1 --id s1_assignments
python -m app synthesize --target 2                    # uses default preset from config
python -m app synthesize --target 2 --visualize          # + 2D PNG + 3D window
python -m app visualize --applied data/output/layouts/synth_s2.json
```

`synthesize` no longer takes `--assignments` / `--preset`; the preset is read from **`config/defaults.yaml`**.

| `synthesize` option | Default | Description |
|---------------------|---------|-------------|
| `--target` | required | Target subject id |
| `--out` | auto | Output layout JSON |
| `--preserve-entry-order` | off | Keep reference entry order |
| `--inherit-preset-terminals` | off | Legacy rigid hub map |
| `--fix-terminals` | off | Exact hub clicks, no angle search |
| `--uv-resolution` | `100` | UV grid resolution |
| `--visualize` | off | Save 2D + open 3D after synth |
| `--show` / `--no-show` | — | 2D window / skip 3D with `--visualize` |
| `--skip-collisions` | off | Faster visualize |

### Stage C — `polish` (optional)

```powershell
python -m app polish --applied data/output/layouts/synth_s2.json --mode gentle
```

Modes: `gentle` (repair), `repair`, `refine`, `ga-short`. Output: e.g. `synth_s2_repaired.json`.

### Stage D — postprocess

```powershell
python -m app smooth --applied data/output/layouts/synth_s2.json
python -m app export-bundle --input data/output/smooth/smooth_s2_final.json
python -m app init-print-config --subject 2
python -m app convert-gcode --bundle data/output/bundles/subject_2
python -m app list-electrodes --bundle data/output/bundles/subject_2
python -m app simulate-gcode `
  --gcode data/output/gcode/subject_2_post/allinterconnects.txt `
  --bundle data/output/bundles/subject_2
```

Use **layout JSON** (`synth_s*.json`) for `visualize` / `polish` / `smooth` input. Use **smooth JSON** for `export-bundle`.

Legacy MATLAB: `python -m app export-matlab --input data/output/smooth/smooth_s2_final.json`

Full argument tables: [docs/CLI.md](docs/CLI.md)

---

## Terminal entries (target-native)

Synthesize uses **fiducial-native** strip zones by default:

- Terminal safety zones are centered on **TERMINAL_LEFT / TERMINAL_RIGHT** clicks (polar projection).
- Each wire ends at a **strip entry** on the zone boundary, not at the hub center.
- 3D endpoints use UV surface lift with a light blend toward the hub.

**3D visualize markers**

| Color | Meaning |
|-------|---------|
| Gray | Hub clicks (`TERMINAL_LEFT` / `TERMINAL_RIGHT`) |
| Lime | Wire ends (`entry_position_3d`) |
| Cyan | Path splines |

---

## G-code simulator

Forward kinematics: programmed **X,Y,Z = C pivot**; mesh and paths share **controller machine frame**. Use the same `--pm-file`, `--rot0y`, and `--rot0z` as `convert-gcode`.

See [docs/MACHINE_KINEMATICS.md](docs/MACHINE_KINEMATICS.md) and [docs/CLI.md](docs/CLI.md#simulate-gcode) for layer keys and viewer controls.

---

## Repository layout

```text
layout_design/
├── app/
│   ├── preprocess/         # Stage A
│   ├── layout/             # Stage B
│   ├── pipeline/           # `run` orchestration
│   ├── polish/             # Stage C (optional)
│   ├── postprocess/        # Stage D: smooth, bundle, convert-gcode
│   └── simulator/          # G-code 3D viewer
├── config/                 # defaults.yaml + postprocessor machine/pm configs
├── data/                   # Pipeline I/O (see data/README.md)
├── docs/                   # CLI.md, PIPELINE.md, MACHINE_KINEMATICS.md, …
├── legacy_gcode_examples/  # MATLAB reference (optional)
├── scripts/                # Sample data helpers
└── tests/
```

Full map: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

---

## Environment

Use the genetic_SHAPE venv if system Python lacks the scientific stack:

`D:\Research\genetic_layout_design\genetic_SHAPE\genetic\Scripts\python.exe -m app run --target 2`

---

## Agent skill

`@layout-synth-pipeline`

---

## Status

**Primary path:** `run --target {id}` from PLY through G-code, with default assignment preset in config. Individual commands remain for debugging and partial reruns. Target fiducial hubs + fiducial-native strip entries are default.
