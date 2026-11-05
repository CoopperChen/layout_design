# Data layout

`app/paths.py` is the only place these paths are defined.

```text
data/subjects/{id}/
  source/
    scan.ply
    head.obj          # input texture; align never overwrites this
    head.mtl
    head.jpg
  head.stl            # cleaned geometry, scan frame
  head.obj            # texture in that same frame
  head.mtl
  head.jpg
  layout.json
  pm.yaml
  electrodes.gcode
  interconnect.json   # optional; written by `interconnect`, not by `run`
```

`config/machine.yaml` is the printer, shared by every subject.

## layout.json

```json
{
  "anatomy": {"nasion": [], "lpa": [], "rpa": [], "inion": []},
  "cz": [],
  "calibration": {"central": [], "left": [], "back": []},
  "electrodes": {"Fp1": [], "Cz": []}
}
```

Coordinates are millimetres in the scan frame. `calibration` is the digital triangle. `pm.yaml` is the same three points measured on the machine, with central at `[0, 0, 0]`.

## pm.yaml

```yaml
physical_landmarks_mm:
  - [0, 0, 0]          # landmark_central
  - [80, 0, 0]         # landmark_left
  - [10, 70, 5]        # landmark_back
```

`gcode` refuses an all-zero or degenerate triangle.

## interconnect.json

Written by `python -m app interconnect --subject N` after the left and right terminals are clicked. `picked` is true only for those clicks. `terminals` are the two harness points on the scalp. `assignments` maps each electrode to `TERMINAL_LEFT` or `TERMINAL_RIGHT`. `slots` is the 0-based order along that strip after odd swaps. `routes` are polylines that follow `head.stl` from the electrode to its slot. `odd_pairs` is how many same-hub pairs still cross an odd number of times.

`python -m app paths --subject N` prints each file and whether it exists.
