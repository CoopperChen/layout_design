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
2. Touch **landmark_central** → record as `[0, 0, 0]` (measurement origin).
3. Touch **landmark_left** and **landmark_back** → record machine XYZ.

These values anchor `scan2phys` when converting bundle geometry to print coordinates.

## Coordinate frames

| Frame | Central landmark | C pivot @ machine zero |
|-------|------------------|------------------------|
| **pm measurement** (convert-gcode offline) | `(0, 0, 0)` | not at origin |
| **Controller machine** (runtime G-code, simulate-gcode) | `(0, −a, −(d+calgap_z))` | `(0, 0, 0)` |

`convert-gcode` writes G-code in machine frame. `simulate-gcode` registers the mesh into the same frame so paths and head align. See [docs/MACHINE_KINEMATICS.md](../../docs/MACHINE_KINEMATICS.md).

## Machine geometry

Physical layout and B/C conventions follow **`gcodeConverter_final14.m`** (see [docs/MACHINE_KINEMATICS.md](../../docs/MACHINE_KINEMATICS.md)): `findCaxisAngle` / `findBaxisAngle`, sin/cos(C) arm offset for **offline** G-code writing, rigid FK for **runtime** simulation.

Shared config: `machine_default.yaml` (`a_mm`, `d_mm`, `calgap_z_mm`, `c0_deg`, `b0_deg`).

## Simulate G-code (3D viewer)

Forward kinematics viewer — programmed **X,Y,Z = C pivot**, mesh in machine frame:

```bash
python -m app simulate-gcode \
  --gcode data/output/gcode/subject_4_post/allinterconnects.txt \
  --bundle data/output/bundles/subject_4
```

Use the **same** `pm` YAML and `--rot0y` / `--rot0z` as `convert-gcode`. Digital landmarks (`landmark_central`, `landmark_left`, `landmark_back`) come from the bundle fiducial picks — not re-measured at simulation time.

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
