# Pipeline stages

Run all commands from the **repository root** unless noted.

```bash
python -m app init-data          # create data/ tree
python -m app paths --subject 2  # print canonical paths
```

## Stage map

| Stage | Module (target) | Primary outputs |
|-------|-----------------|-----------------|
| A | `app/preprocess/` | `data/cleaned_scans/`, `data/json/*` |
| B | `app/layout/` | **Generated** `data/output/layouts/synth_s*.json` |
| C | `app/polish/` | `data/output/layouts/*_repaired.json`, optional `output/logs/` |
| D | `app/postprocess/` | `data/output/smooth/`, `data/output/matlab/` |

## Migration

Until modules are ported, run B–D from `genetic_layout_design/genetic_SHAPE/app/` and copy artifacts into this `data/` tree. See [MIGRATION.md](MIGRATION.md).

## Agent workflow

Use Cursor skill `@layout-synth-pipeline` for checklists, CLI flags, and anti-patterns.
