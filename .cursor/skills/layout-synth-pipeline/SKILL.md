---
name: layout-synth-pipeline
description: >-
  Generates per-subject EEG wire path layouts (not preset path replay). Preprocess
  target heads, synthesize paths from terminal assignment map (LEFT/RIGHT only) plus
  target fiducials/electrodes, optional polish, then bundle/G-code/simulate. Use for
  layout_design generate workflow, assignment-only preset, synthesize, not GA transfer.
---

# EEG Layout Pipeline — Generate (Not Apply)

**Goal:** **Generate** a collision-free interconnect layout on **each** subject. The cross-subject “preset” is **terminal assignment only** (which electrode → `TERMINAL_LEFT` / `TERMINAL_RIGHT`). Paths, entry slots, and hub angles are **computed on the target** by synthesize.

**Not the goal:** Replay reference GA paths, chord shapes, or rigid hub map (legacy: `--inherit-preset-terminals`).

**Working directory:** `layout_design/` repo root. Paths: `app/paths.py`.

---

## Pipeline

```text
A. PREPROCESS (per target subject)
   mesh, fiducials+TERMINAL_*, Cz, 10–20 electrodes

B. GENERATE LAYOUT (synthesize)
   assignment map + target geometry → new 2D paths, slots, surface 3D

C. POLISH — fixed-endpoint separation (default in `run`; `--no-polish` to skip)

D. POSTPROCESS — smooth → export-bundle → convert-gcode → simulate-gcode
```

---

## What “preset” means in layout_design

| In assignment file | On each target (generated / prep) |
|--------------------|-----------------------------------|
| `terminal_assignments` only | Mesh, electrodes, TERMINAL_* clicks |
| Optional: `preset_id`, `source_subject_id` | Synthesized `paths[]`, `collision_metrics` |

**Do not require** in assignment file: `paths_chord_3d`, `paths_normalized`, `terminal_positions_3d` (ignored for default generate).

Build assignment map:

```bash
python -m app build-assignments --reference 1 --id s1_assignments
```

---

## B — Generate layout (primary)

**Default:** target hub clicks + assignment map → **new** layout.

```bash
python -m app synthesize --assignments s1_assignments --target 2
python -m app visualize --applied data/output/layouts/synth_s2.json
```

| Flag | Meaning |
|------|---------|
| *(default)* | Hubs from `fiducials_{target}.json`; optional `--rotate` for ±36° search |
| `--rotate` | Hub angle search for clearance; may have fewer crossings |
| `--inherit-preset-terminals` | **Legacy:** map reference hubs — not generate-first |

**Internal:** target-native entry slots, straight/detour 2D, UV surface `path_points`, terminal tail truncation, validate 0/0.

**Gate:** `crossing_count=0`, `electrode_violations=0` before polish/post.

---

## A — Preprocess

Per subject: `cleaned_scans/`, `fiducials_*` (incl. TERMINAL_LEFT/RIGHT), `electrode_positions_*`.

```bash
python -m app preprocess --subject N --step fiducials
python -m app preprocess --subject N --step electrodes
python -m app preprocess --subject N --step assignments   # optional local balanced map
```

Reference assignments: `build-assignments --reference 1` or copy `initial_terminal_assignments_1.json` into `terminal_assignments` in JSON.

---

## C — Polish (optional)

Only if generated layout needs more **separation** (endpoints fixed at electrode + truncated wire end).

```bash
python -m app polish --applied data/output/layouts/synth_s2.json --mode gentle
```

Requires `PYTHON/GA/greed.py` for spacing adjustments. Does not move electrode or truncated wire endpoints; rejects moves that add crossings.

---

## D — Postprocess

```bash
python -m app smooth --applied data/output/layouts/synth_s2.json
python -m app export-bundle --input data/output/smooth/smooth_s2_final.json
python -m app init-print-config --subject 2   # measure pm at print time
python -m app convert-gcode --bundle data/output/bundles/subject_2
python -m app simulate-gcode \
  --gcode data/output/gcode/subject_2_post/allinterconnects.txt \
  --bundle data/output/bundles/subject_2
```

Legacy: `export-matlab` for MATLAB `gcodeConverter_final14.m`.

---

## Anti-patterns

- Treating full v4 GA export as the normal preset (paths + hubs).
- Using `apply-v4` without `--synthesize` as the main multi-subject path.
- Expecting polish to fix wrong LEFT/RIGHT assignment — edit assignment map and **re-generate**.

---

## Reference

[reference.md](reference.md) · [docs/GOAL.md](../../docs/GOAL.md)
