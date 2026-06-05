# Data contracts

Canonical paths: `app/paths.py`. This document describes **what** each artifact contains.

## Stage A — Preprocess

### `data/raw/{id}.stl`

Raw head scan from acquisition. Not modified in place.

### `data/cleaned_scans/{id}.stl`

Single connected head mesh after `0_clearIslands` (small islands removed).

### `data/json/fiducials_{id}.json`

Keys (3D coordinates as lists or objects):

- `nasion`, `lpa`, `rpa`, `inion` — anatomical registration
- `TERMINAL_LEFT`, `TERMINAL_RIGHT` — rear harness terminal clicks

Synthesize uses the **four anatomical** points for hub registration; terminal 3D positions for layout come from the preset / synthesize hub pose unless polish syncs fiducials.

### `data/json/electrode_positions_{id}.json`

Map electrode name → `[x, y, z]` (includes `Cz`).

### `data/json/initial_terminal_assignments_{id}.json`

Map electrode name → `"TERMINAL_LEFT"` | `"TERMINAL_RIGHT"`.

Produced by balanced geodesic strategy or copied from a reference preset.

## Stage B — Assignment map & generated layouts

### `data/presets/*.json` — assignment map only (canonical)

Minimum for **generate** (`synthesize`):

```json
{
  "preset_version": 4,
  "preset_id": "s1_assignments",
  "assignment_only": true,
  "terminal_assignments": { "Fp1": "TERMINAL_LEFT", "Fp2": "TERMINAL_RIGHT" }
}
```

Full v4 GA exports may include `paths_chord_3d`, `terminal_positions_3d`, anatomy — **not used** by default generate (only `terminal_assignments` is read). Hubs come from target `fiducials_{id}.json`.

### `data/output/layouts/{tag}_s{id}.json` — **generated** layout

Produced by synthesize on the target (not copied from reference). Top-level keys:

| Key | Purpose |
|-----|---------|
| `metadata.target_subject_id` | Subject integer |
| `metadata.preset_path` | Source preset |
| `paths[]` | Per-wire: `electrode`, `terminal`, `modified_path_2d`, `path_points` |
| `collision_metrics` | `crossing_count`, `electrode_violations`, `layout_collision_free` |
| `uv_grid` | For 3D reconstruction if `path_points` rebuilt |

**Gate before polish/postprocess:** `crossing_count == 0` and `electrode_violations == 0`.

## Stage C — Polish logs (optional)

### `data/output/logs/subject_{id}/`

Short GA or repair intermediates only. **Not** the default postprocess input.

## Stage D — Print export

### `data/output/smooth/smooth_s{id}_{tag}.json`

Same schema expected by `EXPORT_TO_MATLAB` / legacy g-code:

- `final_paths` — list of smoothed 3D polylines with electrode metadata
- Mesh reference for normal computation

### `data/output/matlab/subject_{id}/`

| File | Contents |
|------|----------|
| `InterconnectElectrodePaths.mat` | Wires + electrode circles |
| `HeadMesh.mat` | Triangulated head |
| `Landmarks.mat` | Terminal positions |
| `LandmarkNames.mat` | Labels |

## Archive (legacy)

### `data/archive/{run_id}/`

Optional copy of `GA_{subject}_*` logs from reference GA runs used only to **export** presets (`export-v4`). Default pipeline does not read archive for print prep.
