# Tests

Activate the project venv, then from repository root:

```bash
# once: python -m venv .venv && source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
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
| `postprocess/test_mesh_normals.py` | Mesh normal orientation |

### Simulator (`tests/simulator/`)

| File | Coverage |
|------|----------|
| `test_machine_fk.py` | Rigid arm FK, machine zero, frame transforms |
| `test_machine_execute.py` | Forward G-code execution, geometry checks |
| `test_mesh_registration.py` | scan2phys + machine-frame shift |
| `test_inverse_roundtrip.py` | Undo machine-zero / tool-offset |
| `test_tool_offset_roundtrip.py` | Postprocessor offset parity |
| `test_parser.py` | G-code line parser |

## Fixtures

`tests/fixtures/bundle_factory.py` — tiny mesh, smooth JSON, fiducials, golden bundle.

`tests/fixtures/synthetic_bundle/` — pre-built golden bundle for loader tests.

## CI

- **Fast** (`.github/workflows/test.yml`): every push/PR, `pytest -m "not slow"` + ruff
- **Slow** (`.github/workflows/slow.yml`): nightly, real-mesh tests when data present

## Markers

- `slow` — real head mesh export; excluded from default pytest and fast CI
