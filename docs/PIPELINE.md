# Pipeline stages

Run all commands from the **repository root** unless noted.

```bash
python -m app init-data          # create data/ tree
python -m app paths --subject 2  # print canonical paths
```

## Stage map

| Stage | Module | Primary outputs |
|-------|--------|-----------------|
| A | `app/preprocess/` | `data/cleaned_scans/`, `data/json/*` |
| B | `app/layout/` | `data/output/layouts/synth_s*.json` |
| C | `app/polish/` | `data/output/layouts/*_repaired.json`, optional `output/logs/` |
| D | `app/postprocess/` | `data/output/smooth/`, `data/output/bundles/`, `data/output/gcode/` |

## Stage D (postprocess)

```bash
# Smooth B-spline paths
python -m app smooth --applied data/output/layouts/synth_s4.json

# Validate before expensive mesh export
python -m app validate --input data/output/smooth/smooth_s4_final.json

# Canonical bundle export
python -m app export-bundle --input data/output/smooth/smooth_s4_final.json

# Print registration (measured at print time; not in data/)
python -m app init-print-config --subject 4

# 5-axis G-code (default: both interconnect + electrode traces)
python -m app convert-gcode --bundle data/output/bundles/subject_4/
```

Legacy `.mat` export remains available via `export-matlab` for MATLAB tooling; prefer `export-bundle` + `convert-gcode`.

See [CLI.md](CLI.md) for flags (`--quiet`, `--skip-validation`, `--trace`, etc.).

## Agent workflow

Use Cursor skill `@layout-synth-pipeline` for checklists, CLI flags, and anti-patterns.
