# Stage B — Synthesize & visualize

| Module | Role |
|--------|------|
| `synthesize.py` | `apply-v4 --synthesize` |
| `visualize.py` | 2D polar + 3D head mesh PNGs |

## Visualize

```powershell
python -m app visualize --applied data/output/layouts/synth_s2.json --no-show
python -m app visualize --applied ... --mode 2d    # polar layout only
python -m app visualize --applied ... --mode 3d    # PyVista screenshot only
```

Default `--mode both`:

- Saves `data/output/pics/{tag}_s{id}_2d.png` (polar layout)
- Opens **interactive PyVista 3D** (mesh, electrodes, wires) — close window to exit

Use `--save-3d path` to also capture a screenshot. `--no-show` skips the 3D window. `--show-2d` opens matplotlib interactively.

Synthesize with auto-viz: `python -m app synthesize ... --visualize`
