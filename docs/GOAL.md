# Project goal

Place a 10–20 electrode set on each subject’s head and write G-code that prints those pads.

## Inputs

- `scan.ply` — metric point cloud
- `head.obj`, `head.mtl`, `head.jpg` — textured head, never overwritten
- Three physical landmarks, measured on the CNC with `record-pm`

## Output

`data/subjects/{id}/electrodes.gcode`

## Steps

`mesh` builds one upright head from the point cloud and the texture. `layout` is one picking session (four anatomical points and three calibration points); Cz and the 10–20 sites are computed on save. `record-pm` stores the same three calibration points in the machine frame. `gcode` builds a coplanar disk at each electrode and runs the 5-axis postprocessor.

Wire paths, terminal assignment, and interconnect G-code are not part of this pipeline. The previous interconnect version is `origin/master` at `93cff25`.
