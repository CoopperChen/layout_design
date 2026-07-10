# Stage C — Polish (optional)

Fixed-endpoint separation polish (`gentle` / `repair`): electrode and truncated wire ends
stay pinned; phase 2 improves trace spacing without increasing crossing count.

Modes:

1. `gentle` / `repair` — separation-only phase 2 (default)
2. `refine` — repair + uncross (no GA)
3. `ga_short` — warm-start GA (~20 generations; may move endpoints)

**Output:** `data/output/layouts/{tag}_s{id}.json` (e.g. `synth_s2_repaired.json`)

### Phase-2 timing profile

Print per-round breakdown (which step dominates: `find_conflict_pairs`, `accept_global_crossing`, etc.):

```bash
python -m app polish --applied data/output/layouts/synth_s2.json --mode gentle --profile
python -m app run --target 2 --from polish --to polish --polish-profile
```

Or set `polish.profile: true` in `config/defaults.yaml`.

Optional logs: `data/output/logs/subject_{id}/`
