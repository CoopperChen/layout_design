# Project status

Last updated: 2026-06-11

## Done

| Area | Status |
|------|--------|
| Stages A–D in `layout_design` | Implemented |
| Canonical export | `eeg_subject_bundle/1.0.0` (`export-bundle`) |
| G-code postprocessor | Merged into `app/postprocess/gcode/` |
| G-code simulator | `simulate-gcode` — forward FK, PyVista viewer, machine-frame mesh |
| Print registration | pm-only YAML + `init-print-config` |
| Pre-export gates | `validate` CLI + default checks in export |
| Fast contract tests | Synthetic bundle, validation, pm, gcode, simulator (~53 tests) |
| CI | Fast tests + ruff on all branches |

## Canonical Stage D workflow

```bash
python -m app smooth --applied data/output/layouts/synth_s{id}.json
python -m app validate --input data/output/smooth/smooth_s{id}_final.json
python -m app export-bundle --input data/output/smooth/smooth_s{id}_final.json
python -m app init-print-config --subject {id}   # once per subject, after measuring pm
python -m app convert-gcode --bundle data/output/bundles/subject_{id}/
python -m app simulate-gcode \
  --gcode data/output/gcode/subject_{id}_post/allinterconnects.txt \
  --bundle data/output/bundles/subject_{id}/
```

## Legacy (supported, not preferred)

- `export-matlab` — writes four `.mat` files for MATLAB `gcodeConverter_final14.m`
- `convert-gcode --subject` — load legacy `.mat` folder instead of bundle

## Next (Phase 4+)

- Incremental extraction from `app/PYTHON/` vendored code
- Preprocess / synthesize / polish integration tests beyond synthetic mesh
- Optional pm sanity bounds before G-code
- Archive or sync standalone Postprocessor repo

## Known limitations

- Real-mesh `export-bundle` can take several minutes (mesh normals + electrode circles)
- Subject export requires calibration landmarks in `fiducials_{id}.json` (picks 7–9)
- Per-subject `pm` YAML is local-only (gitignored); must be measured at print time
- Rigid FK tip at arbitrary B,C may differ from postprocessor decode (offline compensation); use `--verbose` to compare
