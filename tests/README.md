# Tests

Run from repository root:

```bash
pip install -e ".[dev]"
pytest                    # fast tests only (default: -m "not slow")
pytest -m slow -v         # real-mesh export (minutes)
pytest --markers          # list markers
```

## Layout

| File | Coverage |
|------|----------|
| `test_paths.py` | Canonical path helpers |
| `test_validate_export.py` | Pre-export validation gates |
| `test_bundle_contract.py` | Synthetic bundle schema + gcode smoke |
| `test_export_bundle.py` | Export landmarks gate + slow subject_4 round-trip |
| `test_print_config.py` | pm-only YAML resolution |
| `test_integration_stage_d.py` | Synthetic smooth → bundle → gcode |
| `test_bundle_mat_parity.py` | Bundle vs `.mat` on same synthetic fixture |

## Fixtures

`tests/fixtures/bundle_factory.py` — tiny mesh, smooth JSON, fiducials, golden bundle.

`tests/fixtures/synthetic_bundle/` — pre-built golden bundle for loader tests.

## CI

- **Fast** (`.github/workflows/test.yml`): every push/PR, `pytest -m "not slow"` + ruff
- **Slow** (`.github/workflows/slow.yml`): nightly, real-mesh tests when data present

## Markers

- `slow` — real head mesh export; excluded from default pytest and fast CI
