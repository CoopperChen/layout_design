# Subject data

Every subject lives in `data/subjects/{id}/`. `app/paths.py` is the only code that names these files.

```text
data/subjects/{id}/
  source/scan.ply
  source/head.obj
  source/head.mtl
  source/head.jpg
  head.stl
  head.obj
  head.mtl
  head.jpg
  layout.json
  pm.yaml
  electrodes.gcode
```

`mesh` and `run` copy a chosen `.ply` and a texture `.zip` into `source/` and rename them to `scan.ply`, `head.obj`, `head.mtl`, and `head.jpg`. Align does not overwrite `source/`. `pm.yaml` is written by `record-pm`. Printer constants are `config/machine.yaml`.

`python -m app paths --subject N` prints this list and whether each file exists.
