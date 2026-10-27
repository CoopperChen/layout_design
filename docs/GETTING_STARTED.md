# Getting started

Python **3.12** only. From the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

```powershell
python -m app run --subject 1
```

If the subject folder has no scan yet, choose the `.ply` and the `.zip` that holds the `.obj`, `.mtl`, and `.jpg`. The command copies them to `data/subjects/1/source/` as `scan.ply`, `head.obj`, `head.mtl`, and `head.jpg`.

`run` builds the mesh, opens the landmark picker, records the three CNC landmarks, and writes `data/subjects/2/electrodes.gcode`. If `pm.yaml` is already measured, `record-pm` is skipped.

See [CLI.md](CLI.md) and [DATA_LAYOUT.md](DATA_LAYOUT.md).
