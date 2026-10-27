# Agent instructions (layout_design)

## Goal

Place 10–20 electrodes on a head scan and write electrode G-code. Do not plan interconnects.

## Working directory

Repository root `layout_design/`. Paths via `app.paths`. One subject is `data/subjects/{id}/`.

## Pipeline

```text
mesh → layout → record-pm → gcode
```

```bash
python -m app run --subject 2
python -m app paths --subject 2
python -m app simulate --subject 2
```

- `mesh` reads `source/scan.ply` and `source/head.obj` (+ `.mtl`, `.jpg`) and writes `head.stl` and `head.obj`.
- `layout` writes `layout.json` (anatomy, Cz, three calibration points, electrode xyz).
- `record-pm` writes `pm.yaml`. Skip it when that file is already measured.
- `gcode` writes `electrodes.gcode`. Pads are computed at this step.

Printer constants: `config/machine.yaml`. Motion tuning: `config/defaults.yaml`.

## Key docs

- [docs/GOAL.md](docs/GOAL.md)
- [docs/CLI.md](docs/CLI.md)
- [docs/PIPELINE.md](docs/PIPELINE.md)
- [docs/DATA_LAYOUT.md](docs/DATA_LAYOUT.md)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/MACHINE_KINEMATICS.md](docs/MACHINE_KINEMATICS.md)
