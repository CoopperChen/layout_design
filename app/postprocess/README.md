# Stage D — Postprocess

1. **Smooth** — B-spline 3D paths from final layout JSON → `data/output/smooth/`
2. **export-matlab** — `.mat` bundle → `data/output/matlab/subject_{id}/`
3. **MATLAB** — `legacy_gcode_examples/gcodeConverter_final14.m` → g-code

Port from `Z_PREP_RESULTS.py` and `EXPORT_TO_MATLAB.py` (applied-JSON input, not GA archives).
