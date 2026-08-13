# Agent instructions (layout_design)

## Goal

**Generate** wire path layouts per subject. **Preset** in this repo means **terminal assignment only** (electrode → LEFT/RIGHT), not path transfer from a reference head.

## Working directory

Repository root `layout_design/`. Paths via `app.paths`.

## Pipeline

1. **A** Preprocess target subject (STL mesh, align textured OBJ, fiducials with TERMINAL_*, electrodes)
2. **B** `synthesize` — **create** paths from assignment map + target geometry (default)
3. **C** Polish optional (paths only, not hub discovery)
4. **D** Smooth → export-bundle → convert-gcode → simulate-gcode

**Entry-order swap rule:** Swap strip slots only for same-hub pairs with an **odd** mutual crossing count (1, 3, …). Even mutual weaves (2, 4, …) are left for the post-swap **even-mutual uncross** pass (fixed-end bends). After a trial swap+replan, accept if total crossings decrease, **or** if the pair's mutual count is even (0, 2, …); reject odd leftovers. Each pair is tried at most once. Crossing counts use the same densified terminal-tail metric as full layout analysis.

Do not treat `export-v4` / chord replay / `--inherit-preset-terminals` as the primary workflow.

## Commands

```bash
python -m app build-assignments --reference 1 --id s1_assignments
python -m app synthesize --assignments s1_assignments --target 2
python -m app polish --applied data/output/layouts/synth_s2.json --mode gentle
python -m app smooth --applied data/output/layouts/synth_s2.json
python -m app export-bundle --input data/output/smooth/smooth_s2_final.json
python -m app convert-gcode --bundle data/output/bundles/subject_2
python -m app simulate-gcode --gcode data/output/gcode/subject_2_post/allinterconnects.txt --bundle data/output/bundles/subject_2
```

`--preset` is an alias for `--assignments`. `--inherit-preset-terminals` is legacy hub map only.

**Simulator:** `app/simulator/` — forward FK (X,Y,Z = C pivot), mesh in machine frame. Kinematics: [docs/MACHINE_KINEMATICS.md](docs/MACHINE_KINEMATICS.md).

## Key docs

- [docs/GOAL.md](docs/GOAL.md)
- [docs/CLI.md](docs/CLI.md)
- [docs/MACHINE_KINEMATICS.md](docs/MACHINE_KINEMATICS.md)
- [docs/DATA_LAYOUT.md](docs/DATA_LAYOUT.md)
- `.cursor/skills/layout-synth-pipeline/SKILL.md`
