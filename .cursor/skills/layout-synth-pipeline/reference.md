# layout-synth-pipeline — port reference

Source repo: `genetic_layout_design/genetic_SHAPE/app/`. Target: `layout_design/` (root cwd), code in `app/`, data in `data/`.

Path API: `app/paths.py`. Full tree: `data/README.md`, `docs/DATA_LAYOUT.md`.

## Module map

| Stage | genetic_SHAPE path | Role in new pipeline |
|-------|-------------------|----------------------|
| Preprocess | `PYTHON/0_PREP/0_clearIslands.py` | Mesh cleaning |
| Preprocess | `PYTHON/0_PREP/1_selectFiducials.py` | Fiducials + terminals |
| Preprocess | `PYTHON/0_PREP/2_showCz.py` | Cz for 10–20 |
| Preprocess | `PYTHON/0_PREP/3_placeElectrodes.py` | Electrode JSON |
| Assignments | `PYTHON/tools/initiate3DConnections.py` | `balanced` assignments only (no full GA) |
| Synthesize | `PYTHON/tools/layoutPresetV4.py` | `apply_layout_preset_v4_synthesize` |
| Synthesize CLI | `PYTHON/tools/layoutPreset.py` | `apply-v4`, `export-v4`, `visualize` |
| Geometry | `PYTHON/tools/new2dAlterations.py` | Zones, collisions, repair, polar 2D |
| 3D lift | `PYTHON/tools/reconstructUsingUVmesh.py` | UV grid reconstruction |
| Polish | `layoutPreset.py` | `repair_applied_preset`, `run_ga_from_applied_preset` |
| Polish | `layoutPresetV4.py` | `refine_applied_v4`, `uncross_applied_layout` |
| Post | `Z_PREP_RESULTS.py` | B-spline `smooth_3d_path` |
| Post | `EXPORT_TO_MATLAB.py` | `.mat` export |
| Post | `legacy_gcode_examples/gcodeConverter_final14.m` | G-code |

## Removed from layout_design (see docs/REMOVED_MODULES.md)

- `layoutPresetV3.py`, `layoutPresetV3_search.py`, `fiducialUV.py`
- v2/v3 `layoutPreset` CLI subcommands

## Optional / demoted

- `PYTHON/GA/GA.py` — only via `python -m app polish --mode ga-short`
- `greed.py` — used by repair (keep)

## Applied JSON fields (postprocess adapter)

Read from B/C output (`data/output/synth_*.json`):

- `metadata.target_subject_id`
- `paths[].electrode`, `paths[].terminal`
- `paths[].modified_path_2d` — for UV reconstruct if `path_points` missing
- `paths[].path_points` — preferred 3D polyline input to smoothing
- `collision_metrics` — gate before postprocess

Output `data/output/smooth/smooth_s{id}_{tag}.json` must match what `EXPORT_TO_MATLAB.load_final_paths` expects (`final_paths`, mesh reference, electrode metadata). Mirror the structure produced by current `Z_PREP_RESULTS.py` SAVE block.

## genetic_SHAPE → layout_design path map

| genetic_SHAPE | layout_design |
|---------------|---------------|
| `data/output/applied_*.json` | `data/output/layouts/synth_s*.json` |
| `data/output/pics/` | `data/output/pics/` |
| `records/{RUN_ID}/` | `data/archive/{RUN_ID}/` |
| `NEW_SMOOTH_*.json` (app root) | `data/output/smooth/smooth_s*_*.json` |
| `subject_optimized/` | `data/output/matlab/subject_{id}/` |

## Preset v4 (synthesize)

Required keys: see `layout-synthesize` skill. Export from reference GA:

```bash
python -m PYTHON.tools.layoutPreset export-v4 --subject 1 --individual 99-2 \
  --log-dir records/0602_run1 --out data/presets/subject1_best_v4.json
```

Assignments-only preset: copy `terminal_assignments` + `source_anatomical_fiducials` from reference `fiducials_{id}.json` and `initial_terminal_assignments_{id}.json`.

## Gentle polish parameters

| API | Key params |
|-----|------------|
| `repair_applied_preset` | `phase2_max_rounds=12`, `aggressive_pass=False` |
| `refine_applied_v4` | repair + `uncross_applied_layout` max_rounds=80 |
| `run_ga_from_applied_preset` | `n_generations=15–30`, `repair_on_seed=True` |
