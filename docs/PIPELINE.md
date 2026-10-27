# Pipeline

Run from the repository root.

```text
mesh → layout → record-pm → gcode
```

| Command | What it does | Writes |
|---------|----------------|--------|
| `mesh` | Poisson from the PLY, drop islands, ICP the OBJ into that frame, one upright rotation | `head.stl`, `head.obj` |
| `layout` | Seven picks, then Cz and 10–20 review | `layout.json` |
| `record-pm` | Jog the tip to central, left, and back | `pm.yaml` |
| `gcode` | Disk pads, scan-to-physical registration, 5-axis motion | `electrodes.gcode` |

```powershell
python -m app run --subject 2
python -m app run --subject 2 --from gcode
```

`run` skips `record-pm` when `pm.yaml` already contains a measured triangle. Pass `--force-record-pm` to capture again.

`simulate` and `cnc` are optional. They are not stages of `run`.

Rerunning `mesh` skips Poisson and ICP when `head.stl` and `head.obj` are newer than `source/scan.ply` and `source/head.obj`.
