# Data contracts

Canonical paths: `app/paths.py`. This document describes **what** each artifact contains.

## Stage A — Preprocess

### `data/raw/{id}.ply`

Raw structured-light / photogrammetry point cloud. Input to the **reconstruct** preprocess step.

### `data/raw/{id}.stl` and `data/raw/{id}.obj`

Per subject you need **both** meshes with the **same geometry** (from reconstruct or external export):

- **STL** — used by clear-islands, electrodes, geodesics, synthesize, smooth, and MATLAB `HeadMesh.mat`
- **OBJ** (textured) — used **only** by `select_fiducials` for interactive picking on the color scan

OBJ may also live under `data/cleaned_scans/{id}.obj`; pipeline steps never read it.

### `data/cleaned_scans/{id}.stl`

Single connected head mesh after `0_clearIslands` (small islands removed). Canonical mesh for all steps after preprocess A.

### `data/json/fiducials_{id}.json`

Keys (3D coordinates as lists; picked on OBJ, valid on STL):

- `nasion`, `lpa`, `rpa`, `inion` — anatomical registration
- `TERMINAL_LEFT`, `TERMINAL_RIGHT` — rear harness terminal clicks
- `landmark_central`, `landmark_left`, `landmark_back` — calibration landmarks → `Landmarks.mat` / `LandmarkNames.mat`

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

### `data/output/bundles/subject_{id}/` — **canonical**

Schema `eeg_subject_bundle/1.0.0`:

| File | Contents |
|------|----------|
| `manifest.json` | Schema version, frame conventions, array refs |
| `geometry.npz` | `mesh_points`, `mesh_faces` (0-based), `landmarks_xyz` |
| `traces.npz` | `channel_names`, `interconnect_xyzn`, `electrode_xyzn` (N×6) |

CLI: `python -m app export-bundle --input data/output/smooth/smooth_s{id}_final.json`

### `data/output/gcode/`

5-axis G-code from `python -m app convert-gcode`. Print session YAML in `config/postprocessor/subjects/`.

### `data/output/matlab/subject_{id}/` — **legacy**

| File | Contents |
|------|----------|
| `InterconnectElectrodePaths.mat` | Wires + electrode circles |
| `HeadMesh.mat` | Triangulated head |
| `Landmarks.mat` | Calibration landmarks |
| `LandmarkNames.mat` | Labels |

## Archive (legacy)

### `data/archive/{run_id}/`

Optional copy of `GA_{subject}_*` logs from reference GA runs used only to **export** presets (`export-v4`). Default pipeline does not read archive for print prep.
