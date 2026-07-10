# Configuration

| File | Purpose |
|------|---------|
| `defaults.yaml` | Global pipeline defaults: preprocess, assignment preset, polish, smoothing |

Key fields in `defaults.yaml`:

| Key | Default | Used by |
|-----|---------|---------|
| `preprocess.poisson_depth` | `12` | `run`, `preprocess reconstruct` |
| `preprocess.align_head` | `true` | `run` reconstruct (disable with `--no-align-head`) |
| `synthesize.assignments` | `s1_assignments` | `run`, `synthesize` |
| `postprocess.smoothing_strength` | `0.1` | `run`, `smooth` |

Per-subject overrides can be added later as `config/subjects/{id}.yaml`.

Environment:

| Variable | Effect |
|----------|--------|
| `LAYOUT_DESIGN_ROOT` | Override repository root detection in `app/paths.py` |
