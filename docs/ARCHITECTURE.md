# Repository architecture

```text
layout_design/
├── app/
│   ├── cli.py              # command registration
│   ├── paths.py            # data/subjects/{id}/ paths
│   ├── subject.py          # layout.json and pm.yaml
│   ├── mesh/               # Poisson, islands, ICP, upright rotation
│   ├── layout/             # seven picks, Cz, 10–20 projection
│   ├── interconnect/       # optional initial wires and odd slot swaps
│   ├── gcode/              # pads, scan2phys, 5-axis kinematics
│   ├── machine/            # record-pm and Mach4 control
│   ├── pipeline/run.py     # mesh → layout → record-pm → gcode
│   └── simulator/          # optional forward-FK viewer
├── config/
│   ├── machine.yaml        # arm lengths, gap, speeds
│   └── defaults.yaml       # motion tuning (slew, trace order, engage)
└── data/subjects/{id}/
```

`gcode/` keeps the printer math: pad disks, landmark registration, B/C planning, tool offset, feed, trace order, and safe moves. It takes pad arrays, a mesh, and the two landmark triangles.

`layout/` is electrode placement. `interconnect/` is the later harness view: balanced left/right assignment, strip slots, odd swaps, even bends, and a polar UV lift. It is not a `run` stage and it does not emit G-code.

Every command takes `--subject N` and reads paths from `app/paths.py`. `--ply`, `--obj`, and `--machine` are the only extra path flags.
