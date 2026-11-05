---
name: layout-synth-pipeline
description: >-
  Places a 10–20 electrode set on a head scan and writes electrode G-code.
  Inputs are scan.ply plus head.obj/mtl/jpg and CNC landmarks (record-pm).
  Use for layout_design mesh, layout, record-pm, gcode, and the optional
  interconnect view (initial wires and odd slot swaps, no GA).
---

# Electrode G-code

**Goal:** Print electrode pads on each subject. Wire paths are a later view, not part of `run`, and they are not turned into G-code.

**Working directory:** repository root. Paths: `app/paths.py`. One subject folder: `data/subjects/{id}/`.

## Pipeline

```text
mesh → layout → record-pm → gcode
```

```bash
python -m app run --subject 2
python -m app paths --subject 2
python -m app simulate --subject 2
```

| Step | Reads | Writes |
|------|--------|--------|
| mesh | `source/scan.ply`, `source/head.obj` + mtl/jpg. If those are missing, `run` asks for a `.ply` and a `.zip` (obj, mtl, jpg under any names) and copies them into `source/` | `head.stl`, aligned `head.obj` |
| layout | aligned head | `layout.json` |
| record-pm | CNC work pose | `pm.yaml` |
| gcode | layout, mesh, pm, `config/machine.yaml` | `electrodes.gcode` |

`run` skips `record-pm` when `pm.yaml` is already a valid triangle.

## Layout session

Seven picks: nasion, LPA, RPA, inion, calibration central, left, back. Right-click, then Space / Enter / S places the point. When all seven are set, those keys accept and continue. Q / Esc discards. The same keys are used in the mesh viewers and the 10–20 review. On save, Cz is the nasion–inion / LPA–RPA crown intersection and the 10–20 template is snapped to the mesh. A size slider can scale the set before save.

## G-code

Each electrode is a coplanar zigzag disk `gap_size_mm` above the surface. `scan2phys` registers that disk with the digital calibration triangle and `pm.yaml`. Kinematics live in `app/gcode/`.

`--electrode NAME` writes `{name}.gcode` instead of `electrodes.gcode`.

## Interconnect view

After `layout` (and a head mesh) exist:

```bash
python -m app interconnect --subject 2
python -m app interconnect --subject 2 --pick
python -m app interconnect --subject 2 --no-show
```

This writes `interconnect.json`. The first window asks for a left terminal click, then a right one. Wires are the flat synthesize curves lifted onto `head.stl` with the polar UV grid. A later run reuses those clicks. `--pick` clicks again. Leads are balanced across the two hubs, slots sit on a 6 mm pitch (`--spacing`), and odd swaps exchange same-hub slots until same-side wires cross an even number of times. It does not run inside `run`.

## Reference

[docs/GOAL.md](../../docs/GOAL.md) · [docs/CLI.md](../../docs/CLI.md) · [docs/DATA_LAYOUT.md](../../docs/DATA_LAYOUT.md)
