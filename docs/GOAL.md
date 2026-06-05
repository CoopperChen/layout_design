# Project goal

**Generate a collision-free wire layout on each subject’s head** — not replay or transfer a reference subject’s path shapes or hub map.

## What we produce (per subject)

| Generated on target | Source |
|---------------------|--------|
| 2D/3D wire paths, entry slots, collision metrics | **Synthesize** (straight/detour + UV surface lift) |
| Hub 3D positions | Target `fiducials_{id}.json` (TERMINAL_LEFT / TERMINAL_RIGHT clicks) |
| 10–20 electrodes, mesh, anatomy | Target prep |

## What “preset” means here

**Preset = terminal assignment only:** which electrode connects to `TERMINAL_LEFT` vs `TERMINAL_RIGHT`.

Typically copied once from a reference subject (e.g. subject 1):

```json
{
  "preset_version": 4,
  "preset_id": "s1_assignments",
  "terminal_assignments": { "Fp1": "TERMINAL_LEFT", "Fp2": "TERMINAL_RIGHT" }
}
```

Stored under `data/presets/` (folder name is legacy; content is assignment-only).

## What preset is NOT

- Not source path curvature or `paths_chord_3d`
- Not reference terminal 3D positions (unless you opt into `--inherit-preset-terminals`)
- Not a substitute for running synthesize on the target mesh

## Pipeline in one line

Prep target → **synthesize** (assignments + target hubs → new paths) → optional polish → smooth → MATLAB.
