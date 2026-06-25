# Repository architecture

```text
layout_design/
├── app/                          # Application code
│   ├── cli.py                    # Unified CLI (python -m app)
│   ├── paths.py                  # Canonical filesystem paths
│   ├── config_loader.py          # config/defaults.yaml (synth/smooth defaults)
│   ├── runtime.py                # PYTHON/ import path setup
│   ├── preprocess/               # Stage A
│   ├── layout/                   # Stage B (synthesize, visualize)
│   ├── polish/                   # Stage C (optional)
│   ├── postprocess/              # Stage D — smooth, bundle, G-code
│   │   ├── smooth.py
│   │   ├── export_bundle.py      # → eeg_subject_bundle/1.0.0
│   │   ├── convert_gcode.py      # CLI facade for G-code
│   │   ├── print_config.py       # pm-only YAML scaffold
│   │   ├── validate_export.py    # Pre-export gates
│   │   ├── mesh_export.py        # Shared mesh/normal export
│   │   ├── export_matlab.py      # Legacy .mat (optional)
│   │   ├── bundle/               # Bundle schema, emit, load
│   │   └── gcode/                # 5-axis postprocessor engine
│   │       └── kinematics/
│   │           └── machine_fk.py # Rigid arm FK (shared with simulator)
│   ├── simulator/                # G-code 3D viewer
│   │   ├── cli.py                # simulate-gcode command
│   │   ├── parser.py             # G-code line parser
│   │   ├── viewer.py             # PyVista layers + step HUD
│   │   ├── kinematics/
│   │   │   ├── machine_execute.py  # forward_states_from_gcode
│   │   │   └── inverse.py        # Postprocessor decode (verbose diagnostics)
│   │   └── registration/
│   │       ├── subject.py        # scan2phys → machine frame shift
│   │       └── mesh.py           # register_mesh_full
│   └── PYTHON/                   # Vendored layout/GA core (genetic_SHAPE)
│       ├── 0_PREP/               # Mesh + fiducial tools
│       ├── GA/                   # Short GA polish
│       └── tools/                # layoutPresetV4, geodesics, UV lift
├── config/
│   ├── defaults.yaml             # Pipeline defaults
│   └── postprocessor/            # Machine + per-subject pm
├── data/                         # All I/O (see data/README.md)
├── docs/                         # CLI, pipeline, kinematics, data contracts
├── legacy_gcode_examples/        # MATLAB gcodeConverter (copy .m from genetic_SHAPE)
├── scripts/                      # Sample data helpers
└── tests/                        # Contract + simulator tests
```

## Entry points

| Command | Module |
|---------|--------|
| `python -m app <cmd>` | `app/cli.py` |
| `layout <cmd>` | Same (pyproject console script) |

G-code conversion is **only** via `python -m app convert-gcode` (not a separate package CLI).

## Data flow

```
preprocess → synthesize → [polish] → smooth → export-bundle → convert-gcode → simulate-gcode
                                              ↘ export-matlab (legacy)
```

## Coordinate frames

| Frame | Origin | Used by |
|-------|--------|---------|
| **Landmark / pm measurement** | Central landmark at `[0,0,0]` when touching head | `convert-gcode` offline (`scan2phys`), pm YAML |
| **Controller machine** | C pivot at X0 Y0 Z0 | G-code X,Y,Z at runtime; **simulate-gcode** mesh + paths |

`simulate-gcode` applies `registration_to_machine_frame()` after `scan2phys` so mesh, landmarks, and G-code FK share the machine frame. Details: [MACHINE_KINEMATICS.md](MACHINE_KINEMATICS.md).

## What not to add here

- Per-subject meshes under version control (`data/raw/`, `cleaned_scans/`) — gitignored
- Generated pipeline outputs — gitignored, regeneratable
- Duplicate postprocessor repos — canonical G-code is `app/postprocess/gcode/`
