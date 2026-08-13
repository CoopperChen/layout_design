# Stage C — Polish (optional)

Fixed-endpoint separation polish (`gentle` / `repair`): electrode and truncated wire ends
stay pinned; phase 2 improves **slot-adjacent** spacing (maximin min gap + near-hub
fairness) without increasing crossing count.

Modes:

1. `gentle` / `repair` — separation-only phase 2 (default)
2. `refine` — repair + uncross (no GA)
3. `ga_short` — warm-start GA (~20 generations; may move endpoints)

**Output:** `data/output/layouts/{tag}_s{id}.json` (e.g. `synth_s2_repaired.json`)

### Phase-2 objective (gentle)

- Conflict pairs = consecutive strip slots on the same hub
- One accept per round: best `(min_adjacent_gap ↑, hub_gap_variance ↓)`
- No credit for pairs already wider than `min_trace_separation_mm`
- Metrics logged: `min_adjacent_gap`, `hub_gap_variance`, `hub_gap_spread`

### Phase-2 timing profile

Print per-round breakdown (which step dominates: `find_conflict_pairs`, `accept_global_crossing`, etc.):

```bash
python -m app polish --applied data/output/layouts/synth_s2.json --mode gentle --profile
python -m app run --target 2 --from polish --to polish --polish-profile
```

Or set `polish.profile: true` in `config/defaults.yaml`.

Trace clearance target: `polish.min_trace_separation_mm` (default `4.0`) controls both
phase-2 pair spacing penalties and the layout separation-deficit metric.

Optional logs: `data/output/logs/subject_{id}/`
