# Stage D — Postprocess

1. **Smooth** — B-spline 3D paths → `data/output/smooth/`
2. **export-bundle** — canonical `eeg_subject_bundle/1.0.0` → `data/output/bundles/subject_{id}/`
3. **convert-gcode** — bundle + print config → `data/output/gcode/` (`app/postprocess/gcode/`)
4. **simulate-gcode** — 3D PyVista viewer (`app/simulator/`) — forward FK, machine-frame mesh
5. **export-matlab** *(legacy)* — `.mat` bundle → `data/output/matlab/subject_{id}/`

```bash
python -m app smooth --applied data/output/layouts/synth_s{id}.json
python -m app export-bundle --input data/output/smooth/smooth_s{id}_final.json
python -m app init-print-config --subject {id}
python -m app convert-gcode --bundle data/output/bundles/subject_{id}
python -m app simulate-gcode \
  --gcode data/output/gcode/subject_{id}_post/allinterconnects.txt \
  --bundle data/output/bundles/subject_{id}
```

Full CLI docs: [docs/CLI.md](../../docs/CLI.md) (Stage D section).

**Export gates** (fail fast unless `--skip-validation`):
- `collision_metrics.layout_collision_free` in smooth JSON or `source_applied` layout
- Calibration landmarks in `data/json/fiducials_{id}.json`
- Each interconnect ≥ 2 finite `path_3d` points

Print registration: `config/postprocessor/subjects/subject_{id}.yaml` — **pm only**; see `config/postprocessor/README.md`.
