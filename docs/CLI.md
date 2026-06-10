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
| A — Preprocess | `preprocess` |
| B — Layout | `build-assignments`, `synthesize`, `visualize` |
| C — Polish (optional) | `polish` |
| D — Postprocess | `smooth`, `export-bundle`, `init-print-config`, `list-electrodes`, `convert-gcode`, `export-matlab` (legacy) |

Typical end-to-end:

```bash
python -m app init-data
python -m app preprocess --subject 2 --step fiducials
python -m app synthesize --assignments subject1_best_v4 --target 2
python -m app smooth --applied data/output/layouts/synth_s2.json
python -m app export-bundle --input data/output/smooth/smooth_s2_final.json
python -m app init-print-config --subject 2
python -m app convert-gcode --bundle data/output/bundles/subject_2
```

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

**Shows:** preprocess inputs, layout output, smooth JSON, bundle dir, gcode dir, pm config, legacy matlab dir.

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

Generate per-subject wire layout on a target head from an assignment map.

```bash
python -m app synthesize --assignments subject1_best_v4 --target 2 --visualize
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--assignments` / `--preset` | — | Preset name in `data/presets/` (required) |
| `--target` | — | Target subject id (required) |
| `--out` | auto | Output layout JSON path |
| `--preserve-entry-order` | off | Keep reference strip slot order from full v4 preset |
| `--inherit-preset-terminals` | off | Legacy: rigid-map reference hub positions |
| `--fix-terminals` | off | Use exact hub clicks (no ±36° hub angle search) |
| `--uv-resolution` | `100` | UV grid resolution for 3D lift |
| `--visualize` | off | After synth: 2D PNG + interactive 3D |
| `--show` | off | Also open 2D matplotlib window |
| `--no-show` | off | With `--visualize`: save PNG only, no 3D window |
| `--skip-collisions` | off | Skip collision markers in visualize |

**Output:** `data/output/layouts/synth_s{id}.json`

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

Edit `physical_landmarks_mm` after measuring with end-effector on printhead. See [config/postprocessor/README.md](../config/postprocessor/README.md).

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
python -m app synthesize --help
python -m app convert-gcode --help
```

## Related docs

- [ARCHITECTURE.md](ARCHITECTURE.md) — repository layout
- [GETTING_STARTED.md](GETTING_STARTED.md) — environment and first run
- [PIPELINE.md](PIPELINE.md) — stage overview
- [DATA_LAYOUT.md](DATA_LAYOUT.md) — file formats
- [config/postprocessor/README.md](../config/postprocessor/README.md) — pm measurement and print config
