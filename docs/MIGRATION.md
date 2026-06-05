# Migration from genetic_SHAPE

## Path mapping

| genetic_SHAPE (`app/`) | layout_design |
|------------------------|---------------|
| `data/raw/` | `data/raw/` |
| `data/cleaned_scans/` | `data/cleaned_scans/` |
| `data/json/` | `data/json/` |
| `data/presets/` | `data/presets/` |
| `data/output/*.json` (flat) | `data/output/layouts/` |
| `data/output/pics/` | `data/output/pics/` |
| `data/output/logs/` | `data/output/logs/subject_{id}/` |
| `records/{RUN_ID}/` | `data/archive/{RUN_ID}/` |
| `NEW_SMOOTH_FINAL_PATHS_*.json` (app root) | `data/output/smooth/smooth_s{id}_{tag}.json` |
| `subject_optimized/` | `data/output/matlab/subject_{id}/` |

## Copy a subject from genetic_SHAPE

```powershell
$src = "D:\Research\genetic_layout_design\genetic_SHAPE\app"
$dst = "D:\Research\layout_design\data"

Copy-Item "$src\data\raw\2.stl" "$dst\raw\" -ErrorAction SilentlyContinue
Copy-Item "$src\data\cleaned_scans\2.stl" "$dst\cleaned_scans\"
Copy-Item "$src\data\json\*_2.json" "$dst\json\"
Copy-Item "$src\data\presets\*.json" "$dst\presets\"
Copy-Item "$src\data\output\applied_v4_s2_synth_slots.json" "$dst\output\layouts\synth_s2.json"
```

## Code port order

1. `app/paths.py` (done)
2. `app/preprocess/` — four `0_PREP` scripts
3. `app/layout/` — `layoutPresetV4` synthesize + CLI
4. `app/polish/` — repair, refine-v4, short GA wrapper
5. `app/postprocess/` — smooth from applied JSON + MATLAB export
6. Wire `app/cli.py` subcommands

## Environment

Reuse the genetic_SHAPE venv until `layout_design` has its own:

```powershell
cd D:\Research\layout_design
D:\Research\genetic_layout_design\genetic_SHAPE\genetic\Scripts\pip install -e .
```
