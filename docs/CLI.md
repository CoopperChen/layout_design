# CLI

```powershell
python -m app <command>
```

`layout <command>` is the same entry point.

## Commands

| Command | Role |
|---------|------|
| `paths --subject N` | List the subject folder |
| `mesh --subject N` | PLY + OBJ/MTL/JPG → upright head |
| `layout --subject N` | Pick landmarks and save 10–20 sites |
| `record-pm --subject N` | Measure three landmarks on the CNC |
| `gcode --subject N` | Write `electrodes.gcode` |
| `run --subject N` | mesh → layout → record-pm → gcode |
| `simulate --subject N` | View the G-code |
| `cnc` | Load or run a file on Mach4 |

### mesh

```powershell
python -m app mesh --subject 2
python -m app run --subject 1
python -m app mesh --subject 1 --ply D:\scans\head.ply --zip D:\scans\texture.zip
```

When `source/scan.ply` or the texture package is missing, `mesh` and `run` open a file dialog: first the `.ply`, then a `.zip` that contains the `.obj`, `.mtl`, and `.jpg` under any names. Those files are copied into `data/subjects/{id}/source/` as `scan.ply`, `head.obj`, `head.mtl`, and `head.jpg`. The OBJ `mtllib` line and the MTL texture name are rewritten to match.

`--no-preview` skips the overlay check. `--no-rotate-head` skips the upright rotation. `--no-scale` skips the scale match before ICP. `--depth` overrides `mesh.poisson_depth` in `config/defaults.yaml`.

### layout

One window for seven picks: nasion, LPA, RPA, inion, then calibration central, left, and back. Right-click a point, then Space, Enter, or S places it. When all seven are set, those same keys continue. Q or Esc discards. Closing the window accepts a complete set.

The next window shows the projected 10–20 sites. The slider scales them. Space, Enter, or S saves `layout.json`. Q or Esc discards. Closing the window saves.

### record-pm

Jog the tip to central, left, and back. Enter or Space captures. S saves. The file is `data/subjects/{id}/pm.yaml`. If it already exists, pass `--force`.

Mach4 must publish the work pose on UDP port `62101`. See [RECORD_PM.md](RECORD_PM.md).

### gcode

```powershell
python -m app gcode --subject 2
python -m app gcode --subject 2 --electrode Cz
```

`--electrode all` writes `electrodes.gcode`. A name or 1-based index writes `{name}.gcode` in the same folder. `--rot0y` and `--rot0z` are optional bed rotations in degrees. `--machine` overrides `config/machine.yaml`.

### run

```powershell
python -m app run --subject 2
python -m app run --subject 2 --from layout --to gcode
python -m app run --subject 2 --from gcode --force-record-pm
```

`--from` and `--to` are `mesh`, `layout`, `record-pm`, or `gcode`.

### simulate

```powershell
python -m app simulate --subject 2
python -m app simulate --subject 2 --gcode data/subjects/2/Cz.gcode --animate
```

### cnc

```powershell
python -m app cnc status
python -m app cnc load --subject 2
python -m app cnc start --confirm
```

`load` without `--gcode` sends that subject’s `electrodes.gcode`. `home` and `scan` load bundled scripts and start only with `--confirm`.
