# Project status

This branch places 10–20 electrodes and writes `electrodes.gcode`.

```text
mesh → layout → record-pm → gcode
```

One subject folder: `data/subjects/{id}/`. Printer constants: `config/machine.yaml`.

The interconnect planner (synthesize, polish, smooth, MATLAB export, `allinterconnects.txt`) is not in this tree. Recover it from `origin/master` at `93cff25` on https://github.com/CoopperChen/layout_design.
