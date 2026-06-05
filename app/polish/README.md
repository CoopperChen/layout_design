# Stage C — Polish (optional)

Separation-only refinement. Order:

1. `repair` — gentle phase-2 trace resolution
2. `refine` — repair + uncross (no GA)
3. `ga_short` — warm-start GA, ~20 generations

**Output:** `data/output/layouts/{tag}_s{id}.json` (e.g. `synth_s2_repaired.json`)

Optional logs: `data/output/logs/subject_{id}/`
