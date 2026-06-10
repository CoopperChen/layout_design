# Migration from genetic_SHAPE

> **Historical note:** Stages A–D are now implemented in `layout_design`. This document records path mapping from the legacy `genetic_SHAPE` repo for subjects already processed there.

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
| `subject_optimized/` | `data/output/matlab/subject_{id}/` (legacy) |
| — | `data/output/bundles/subject_{id}/` (canonical) |
| — | `data/output/gcode/subject_{id}_post/` |

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

## Port status (complete)

1. `app/paths.py` — done
2. `app/preprocess/` — done
3. `app/layout/` — done
4. `app/polish/` — done
5. `app/postprocess/` — smooth, bundle export, G-code — done
6. `app/cli.py` — unified CLI — done

New subjects should run entirely in `layout_design`. See [PIPELINE.md](PIPELINE.md) and [GETTING_STARTED.md](GETTING_STARTED.md).

## Environment

```powershell
cd D:\Research\layout_design
pip install -e ".[dev]"
```
