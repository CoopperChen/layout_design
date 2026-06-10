# Data layout

All pipeline I/O lives under `data/`. Paths are defined in `app/paths.py` — do not hardcode strings in scripts.

## Tree

```text
data/
├── raw/                    # {id}.ply (input), {id}.stl + {id}.obj (reconstruct output)
├── cleaned_scans/          # After island removal — {id}.stl for pipeline
├── json/                   # Per-subject prep artifacts (fiducials, electrodes, …)
├── presets/                # Terminal assignment maps only
├── output/
│   ├── layouts/            # Synthesized / polished layout JSON
│   ├── pics/               # 2D/3D visualization PNGs
│   ├── smooth/             # B-spline smoothed paths
│   ├── bundles/            # eeg_subject_bundle/1.0.0 (canonical export)
│   ├── gcode/              # 5-axis G-code (.txt)
│   ├── matlab/             # Legacy .mat export (optional)
│   └── logs/               # Optional polish GA logs
└── archive/                # Legacy GA run copies (reference curation)
```

Print registration (`pm`) lives in `config/postprocessor/subjects/`, not under `data/`.

## Per-subject files (`json/`)

| File | Stage | Description |
|------|-------|-------------|
| `fiducials_{id}.json` | A | Nasion, LPA, RPA, inion, terminals, calibration landmarks |
| `Cz_{id}.json` | A | Cz position for 10–20 placement |
| `electrode_positions_{id}.json` | A | Standard 10–20 electrode coordinates |
| `initial_terminal_assignments_{id}.json` | A | Electrode → TERMINAL_LEFT / TERMINAL_RIGHT |

## Assignment map (`presets/`)

Cross-subject input: **`terminal_assignments`** only. Create:

```bash
python -m app build-assignments --reference 1 --id s1_assignments
```

## Naming conventions

| Pattern | Example | Stage |
|---------|---------|-------|
| `synth_s{id}.json` | `synth_s2.json` | B — generated layout |
| `smooth_s{id}_{tag}.json` | `smooth_s2_final.json` | D — smoothed paths |
| `subject_{id}/` under `output/bundles/` | manifest + NPZ | D — canonical bundle |
| `subject_{id}_post/` under `output/gcode/` | `allinterconnects.txt` | D — G-code |

## Git

Large meshes and generated artifacts are gitignored. Empty directories are tracked via `.gitkeep`. Commit preset templates and example schemas only (no PHI).

**Never commit:**

- Textured scans (`*.obj`, `*.mtl`, texture PNGs) — fiducial picking only
- Per-subject print registration (`config/postprocessor/subjects/subject_*.yaml`) — measured `physical_landmarks_mm` at print time
- Generated bundles, G-code, or smooth JSON under `data/output/`

See [docs/DATA_LAYOUT.md](../docs/DATA_LAYOUT.md) for field-level contracts.
