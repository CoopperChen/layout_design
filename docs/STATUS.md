# Project status

This branch places 10–20 electrodes and writes `electrodes.gcode`.

```text
mesh → layout → record-pm → gcode
```

One subject folder: `data/subjects/{id}/`. Printer constants: `config/machine.yaml`.

`python -m app interconnect --subject N` is an optional view after that pipeline: balanced left/right hubs, strip slots, odd swaps, even bends, and a polar UV lift onto the scalp. It does not run a genetic search or write interconnect G-code. The earlier planner (polish, smooth, MATLAB export, `allinterconnects.txt`) is `origin/master` at `93cff25` on https://github.com/CoopperChen/layout_design.
