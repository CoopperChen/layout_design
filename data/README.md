# Data layout

All pipeline I/O lives under `data/`. Paths are defined in `app/paths.py` — do not hardcode strings in scripts.

## Tree

```text
data/
├── raw/                    # {id}.ply (input), {id}.stl + {id}.obj (reconstruct output)
├── cleaned_scans/          # After island removal — {id}.stl for pipeline; optional {id}.obj
├── json/                   # Per-subject prep artifacts
├── presets/                # Terminal assignment maps only (folder name legacy)
├── output/
│   ├── layouts/            # Synthesized / polished layout JSON
│   ├── pics/               # 2D/3D visualization PNGs
│   ├── smooth/             # B-spline smoothed paths (MATLAB input)
│   ├── matlab/             # Exported .mat per subject or batch
│   └── logs/               # Optional short-GA polish logs only
└── archive/                # Legacy GA run copies (reference curation)
```

## Per-subject files (`json/`)

| File | Stage | Description |
|------|-------|-------------|
| `fiducials_{id}.json` | A | Nasion, LPA, RPA, inion, terminals, three calibration landmarks (picked on OBJ; coordinates apply to STL) |
| `Cz_{id}.json` | A | Cz position for 10–20 placement |
| `electrode_positions_{id}.json` | A | Standard 10–20 electrode coordinates |
| `initial_terminal_assignments_{id}.json` | A | Electrode → TERMINAL_LEFT / TERMINAL_RIGHT |
| `init_connection_paths_{id}.json` | A/C | Geodesic seeds (polish/GA only; not required for synthesize) |

## Assignment map (`presets/`)

Cross-subject input: **`terminal_assignments`** (electrode → `TERMINAL_LEFT` | `TERMINAL_RIGHT`). Not wire shapes or reference hub positions.

Create: `python -m app build-assignments --reference 1 --id s1_assignments`

## Naming conventions

| Pattern | Example | Stage |
|---------|---------|-------|
| `{tag}_s{id}.json` in `output/layouts/` | `synth_s2.json` | B — **generated** layout |
| `smooth_s{id}_{tag}.json` in `output/smooth/` | `smooth_s2_final.json` | D |
| `subject_{id}/` under `output/matlab/` | Interconnect + mesh `.mat` | D |

## Git

Large meshes and generated JSON are gitignored. Directory structure is tracked via `.gitkeep` files. Commit **preset templates** and **example JSON schemas** only when they contain no PHI.

See [docs/DATA_LAYOUT.md](../docs/DATA_LAYOUT.md) for field-level contracts.
