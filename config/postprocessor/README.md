# Print registration (`pm`)

Per-subject YAML files store **physical landmarks only** — measured at print time on the real head.

Trace type and channel selection are **CLI flags** on `convert-gcode`, not stored here.

## Setup per subject

```bash
python -m app init-print-config --subject 4
# → config/postprocessor/subjects/subject_4.yaml
```

Edit `physical_landmarks_mm`:

1. Mount end-effector on printhead.
2. Touch **landmark_central** → `[0, 0, 0]`
3. Touch **landmark_left** and **landmark_back** → record machine XYZ.

## Convert to G-code

```bash
# Auto-loads config/postprocessor/subjects/subject_4.yaml from bundle subject id
python -m app convert-gcode --bundle data/output/bundles/subject_4

# Single channel, electrode pads
python -m app convert-gcode --bundle data/output/bundles/subject_4 \
  --trace electrode --electrode Fp1

# Explicit pm file
python -m app convert-gcode --bundle ... --pm-file config/postprocessor/subjects/subject_4.yaml
```

| CLI flag | Default | Purpose |
|----------|---------|---------|
| `--trace` | `both` | `both`, `interconnect`, or `electrode` |
| `--electrode` | `all` | `all`, channel name, or 1-based index |
| `--rot0y`, `--rot0z` | `0` | Optional head rotation on bed |

## Files

| File | Purpose |
|------|---------|
| `machine_default.yaml` | Printer geometry and speeds (shared) |
| `subjects/subject_{id}.yaml` | **pm** for that subject |
| `subjects/example.yaml` | Empty template |
| `subjects/synthetic.yaml` | Contract tests only |
