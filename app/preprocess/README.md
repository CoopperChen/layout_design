# Stage A — Preprocess

Interactive steps (port from `genetic_SHAPE/app/PYTHON/0_PREP/`):

| Script | Output |
|--------|--------|
| `reconstruct` | `data/raw/{id}.ply` → Poisson mesh → `data/raw/{id}.stl` + `data/raw/{id}.obj` (interactive normal flip) |
| `clear_islands` | `data/cleaned_scans/{id}.stl` |
| `select_fiducials` | `data/json/fiducials_{id}.json`, `Landmarks.mat`, `LandmarkNames.mat` — reads **textured** `data/raw/{id}.obj` only; STL (`cleaned_scans/{id}.stl`) is unchanged and used by all other steps |
| `show_cz` | `data/json/Cz_{id}.json` |
| `place_electrodes` | `data/json/electrode_positions_{id}.json` |
| `assign_terminals` | `data/json/initial_terminal_assignments_{id}.json` |
