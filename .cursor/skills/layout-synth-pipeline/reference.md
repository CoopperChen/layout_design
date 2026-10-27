# Module map

| Piece | Path |
|-------|------|
| Subject folder | `app/paths.py`, `app/subject.py` |
| Poisson, islands, ICP | `app/mesh/` |
| Seven picks, Cz, 10–20 | `app/layout/` |
| Disk pads, scan2phys, kinematics | `app/gcode/` |
| CNC landmarks and load/run | `app/machine/` |
| Optional viewer | `app/simulator/` |

Printer geometry is `config/machine.yaml`. Motion tuning is `config/defaults.yaml` under `postprocess:`.
