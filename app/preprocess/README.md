# Stage A — Preprocess

Interactive / automated steps:

| Script | Output |
|--------|--------|
| `reconstruct` | `data/raw/{id}.ply` → Poisson mesh → `data/raw/{id}.stl` only (no textured OBJ). Density trim via `poisson_density_quantile`. |
| `clear_islands` | `data/cleaned_scans/{id}.stl` |
| `align-obj` | Import `data/raw/{id}.obj` (unchanged) → ICP onto cleaned STL → write synced `data/cleaned_scans/{id}.obj` (+ MTL/JPG copy, optional head rotation). Preview: Space accept / Q reject. |
| `select_fiducials` | `data/json/fiducials_{id}.json`, `Landmarks.mat`, `LandmarkNames.mat` — reads **aligned** `data/cleaned_scans/{id}.obj`; STL (`cleaned_scans/{id}.stl`) is used by all other steps |
| `show_cz` | `data/json/Cz_{id}.json` |
| `place_electrodes` | `data/json/electrode_positions_{id}.json` |
| `assign_terminals` | `data/json/initial_terminal_assignments_{id}.json` |
