# Getting started

## 1. Environment (venv + pyproject.toml)

Python **3.10+**. From the repository root:

```powershell
cd D:\Research\layout_design   # or your clone path
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

Runtime deps are declared in `pyproject.toml` (`numpy`, `scipy`, `open3d`, `pyvista`, `shapely`, `matplotlib`, `mne`, `pyyaml`). `[dev]` adds `pytest` and `ruff`. `.venv/` is gitignored.

Re-activate in later shells: `.\.venv\Scripts\Activate.ps1` (or `source .venv/bin/activate`).

## 2. Initialize data tree

```powershell
python -m app init-data
python -m app paths --subject 2
```

## 3. Copy a test subject (optional)

```powershell
.\scripts\copy_sample_data.ps1 -Subject 2
```

## 4. Pipeline commands

### Stage A — Preprocess (interactive)

```powershell
$env:LAYOUT_SUBJECT_ID = "2"   # set automatically by CLI
python -m app preprocess --subject 2 --step clear-islands
python -m app preprocess --subject 2 --step fiducials
python -m app preprocess --subject 2 --step cz
python -m app preprocess --subject 2 --step electrodes
python -m app preprocess --subject 2 --step assignments
```

### Stage B — Generate layout (synthesize)

```powershell
python -m app build-assignments --reference 1 --id s1_assignments
python -m app synthesize --assignments s1_assignments --target 2
# default: 2D PNG + 3D view after synthesize (--no-visualize to skip)
# --preset is an alias for --assignments
```

```powershell
# Or separately (default: 2D PNG + interactive 3D window):
python -m app visualize --applied data/output/layouts/synth_s2.json
python -m app visualize --applied data/output/layouts/synth_s2.json --mode 2d --no-show
python -m app visualize --applied data/output/layouts/synth_s2.json --mode 3d
python -m app visualize --applied ... --save-3d data/output/pics/synth_s2_3d.png  # optional screenshot
```

Outputs: `data/output/pics/synth_s2_2d.png`; 3D opens in PyVista (close window when done).

### Stage C — Polish (optional)

```powershell
python -m app polish --applied data/output/layouts/synth_s2.json --mode gentle
python -m app polish --applied data/output/layouts/synth_s2_repaired.json --mode refine
python -m app polish --applied data/output/layouts/synth_s2_refined.json --mode ga-short --subject 2 --generations 20
```

### Stage D — Postprocess

```powershell
python -m app smooth --applied data/output/layouts/synth_s2.json
python -m app export-bundle --input data/output/smooth/smooth_s2_final.json
python -m app init-print-config --subject 2
python -m app record-pm --subject 2
python -m app list-electrodes --bundle data/output/bundles/subject_2
python -m app convert-gcode --bundle data/output/bundles/subject_2
```

`record-pm` fills `physical_landmarks_mm` from the CNC work DRO (Enter/Space to capture, `s` to save). Prefer that over hand-editing the YAML. Trace/channel: `--trace interconnect --electrode C3`. See `config/postprocessor/README.md`.

### Verify G-code on head mesh

```powershell
python -m app simulate-gcode `
  --gcode data/output/gcode/subject_2_post/allinterconnects.txt `
  --bundle data/output/bundles/subject_2 `
  --verbose
```

Uses the same `pm` YAML and `--rot0y` / `--rot0z` as `convert-gcode`. Mesh and toolpath are shown in **machine frame** (C pivot at origin). Layer keys: `m` mesh · `l` landmarks · `o` origin · `t` tip · `a` arm.

**Legacy:** `export-matlab` + `legacy_gcode_examples/gcodeConverter_final14.m` if you still need MATLAB G-code.

## 5. CLI reference

All commands, flags, and defaults: [CLI.md](CLI.md)

## 6. Project layout

See [README.md](../README.md) and [DATA_LAYOUT.md](DATA_LAYOUT.md).
