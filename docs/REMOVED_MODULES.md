# Removed legacy modules

Stripped from `app/PYTHON/` for the synthesize-first pipeline (not used by `python -m app`).

## Deleted files

| File | Was used for |
|------|----------------|
| `tools/layoutPresetV3.py` | Fiducial-UV / polar transfer apply |
| `tools/layoutPresetV3_search.py` | v3 search benchmarks |
| `tools/fiducialUV.py` | v3 UV warp helpers |

Do **not** delete `GA/greed.py` — required by `repair`, `refine`, and `new2dAlterations` spacing greedy.

## Trimmed CLI (`layoutPreset.py` main)

Removed subcommands: `export`, `apply` (v2 chord), `visualize-uv`, `export-v3`, `apply-v3`, `compare-transfer`, `benchmark`.

**Still available** via vendored `layoutPreset` if invoked directly: `repair`, `ga`, `visualize`, `export-v4`, `apply-v4`, `refine-v4`, `self-test`.

Prefer **`python -m app`** wrappers: `synthesize`, `polish`, `smooth`, `export-matlab`.

## Kept (required)

| File | Why |
|------|-----|
| `tools/new2dAlterations.py` | Zones, collisions, repair, synthesize geometry |
| `tools/layoutPresetV4.py` | Synthesize + refine-v4 |
| `tools/layoutPreset.py` | Repair, GA seed, visualize |
| `tools/reconstructUsingUVmesh.py` | 3D path lift |
| `tools/helper.py`, `initiate3DConnections.py` | Data load, assignments |
| `GA/greed.py` | Electrode/spacing greedy inside repair (name only — not the removed v3 stack) |
| `GA/GA.py`, `GA/geneticOperators.py` | Optional `polish --mode ga-short` |

## v2 API functions

`export_layout_preset` / `apply_layout_preset` remain in `layoutPreset.py` for internal reference but have **no CLI** in this repo. Use **v4** + `--synthesize` instead.
