# Agent instructions (layout_design)

## Goal

**Generate** wire path layouts per subject. **Preset** in this repo means **terminal assignment only** (electrode → LEFT/RIGHT), not path transfer from a reference head.

## Working directory

Repository root `layout_design/`. Paths via `app.paths`.

## Pipeline

1. **A** Preprocess target subject (mesh, fiducials with TERMINAL_*, electrodes)
2. **B** `synthesize` — **create** paths from assignment map + target geometry (default)
3. **C** Polish optional (paths only, not hub discovery)
4. **D** Smooth → MATLAB

Do not treat `export-v4` / chord replay / `--inherit-preset-terminals` as the primary workflow.

## Commands

```bash
python -m app build-assignments --reference 1 --id s1_assignments
python -m app synthesize --assignments s1_assignments --target 2
python -m app polish --applied data/output/layouts/synth_s2.json --mode gentle
python -m app smooth --applied data/output/layouts/synth_s2.json
```

`--preset` is an alias for `--assignments`. `--inherit-preset-terminals` is legacy hub map only.

## Key docs

- [docs/GOAL.md](docs/GOAL.md)
- [docs/DATA_LAYOUT.md](docs/DATA_LAYOUT.md)
- `.cursor/skills/layout-synth-pipeline/SKILL.md`
