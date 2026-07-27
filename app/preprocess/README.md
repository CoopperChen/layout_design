# Stage A — Preprocess

Interactive / automated steps:

| Script | Output |
|--------|--------|
| `reconstruct` | `data/raw/{id}.ply` → Poisson mesh → `data/raw/{id}.stl` only (no textured OBJ). Density trim via `poisson_density_quantile`. |
| `clear_islands` | `data/cleaned_scans/{id}.stl` |
| `align-obj` | Import textured OBJ (`data/raw/{id}.obj` or `--obj`) → ICP onto cleaned STL → overwrite synced `data/raw/{id}.obj` (+ vertex-color sidecars). Preview overlay: Space accept / Q reject. |
| `select_fiducials` | `data/json/fiducials_{id}.json`, `Landmarks.mat`, `LandmarkNames.mat` — reads **synced** `data/raw/{id}.obj` only; STL (`cleaned_scans/{id}.stl`) is used by all other steps |
| `show_cz` | `data/json/Cz_{id}.json` |
| `place_electrodes` | `data/json/electrode_positions_{id}.json` |
| `assign_terminals` | `data/json/initial_terminal_assignments_{id}.json` |
