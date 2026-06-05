# layout_design

**Generate** per-subject EEG interconnect layouts (2D/3D wire paths on the scalp). Each head uses its own mesh, electrodes, and terminal clicks; the only cross-subject input is **which electrode goes to LEFT vs RIGHT** (the “preset” = terminal assignment map).

**Not** applying a reference GA layout or rigidly mapping reference hub positions (that is legacy / opt-in).

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
| **D. Postprocess** | Smooth → MATLAB / g-code |

| Stage | Directory | Output |
|-------|-----------|--------|
| A | `app/preprocess/` | `data/json/`, `data/cleaned_scans/` |
| B | `app/layout/` | `data/output/layouts/` |
| C | `app/polish/` | `data/output/layouts/*_repaired.json` |
| D | `app/postprocess/` | `data/output/smooth/`, `data/output/matlab/` |

Details: [docs/PIPELINE.md](docs/PIPELINE.md) · [docs/DATA_LAYOUT.md](docs/DATA_LAYOUT.md)

## Repository layout

```text
layout_design/
├── app/                    # CLI + synthesize / polish / postprocess
├── data/
│   ├── presets/            # Terminal assignment maps only (LEFT/RIGHT per electrode)
│   └── output/layouts/     # Generated layouts (primary deliverable)
├── docs/GOAL.md
└── .cursor/skills/layout-synth-pipeline/
```

## Quick start

```powershell
# 1. Prep subject 2 (mesh, fiducials incl. TERMINAL_*, electrodes)
python -m app preprocess --subject 2 --step clear-islands
python -m app preprocess --subject 2 --step fiducials
python -m app preprocess --subject 2 --step cz
python -m app preprocess --subject 2 --step electrodes

# 2. Assignment map from reference subject 1 (preset = assignments only)
python -m app build-assignments --reference 1 --id s1_assignments

# 3. Generate layout on subject 2 (paths created here — not copied from S1)
python -m app synthesize --assignments s1_assignments --target 2 --visualize

# 4. Optional polish → print prep
python -m app polish --applied data/output/layouts/synth_s2.json --mode gentle
python -m app smooth --applied data/output/layouts/synth_s2.json
python -m app export-matlab --input data/output/smooth/smooth_s2_final.json
```

Use genetic_SHAPE venv if system Python lacks scientific stack:

`D:\Research\genetic_layout_design\genetic_SHAPE\genetic\Scripts\python.exe -m app …`

## Agent skill

`@layout-synth-pipeline`

## Status

Core path: preprocess → **generate** (`synthesize`) → optional polish → postprocess. Assignment-only presets + target terminal hubs are default.
