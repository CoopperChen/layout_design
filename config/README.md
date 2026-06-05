# Configuration

| File | Purpose |
|------|---------|
| `defaults.yaml` | Global pipeline defaults (polish GA length, smoothing, etc.) |

Per-subject overrides can be added later as `config/subjects/{id}.yaml`.

Environment:

| Variable | Effect |
|----------|--------|
| `LAYOUT_DESIGN_ROOT` | Override repository root detection in `app/paths.py` |
