"""Stage C — optional repair, refine-v4, short GA."""
from __future__ import annotations

from pathlib import Path

from app import paths
from app.config_loader import load_defaults
from app.runtime import setup_runtime


def _resolve_applied(applied: str | Path) -> Path:
    p = Path(applied)
    return p if p.is_absolute() else paths.REPO_ROOT / p


def run_repair(
    applied: str | Path,
    output: str | Path | None = None,
    *,
    electrodes_only: bool = False,
) -> dict:
    setup_runtime()
    from PYTHON.tools.layoutPreset import repair_applied_preset

    applied_path = _resolve_applied(applied)
    if output is None:
        stem = applied_path.stem
        if not stem.endswith("_repaired"):
            stem = f"{stem}_repaired"
        output = applied_path.parent / f"{stem}.json"
    return repair_applied_preset(
        str(applied_path),
        output_path=str(output),
        electrodes_only=electrodes_only,
    )


def run_refine(applied: str | Path, output: str | Path | None = None) -> dict:
    setup_runtime()
    from PYTHON.tools.layoutPresetV4 import refine_applied_v4

    applied_path = _resolve_applied(applied)
    if output is None:
        stem = applied_path.stem.replace("_refined", "").replace("_repaired", "")
        output = applied_path.parent / f"{stem}_refined.json"
    return refine_applied_v4(str(applied_path), output_path=str(output))


def run_ga_short(
    applied: str | Path,
    subject: int | None = None,
    *,
    generations: int | None = None,
    population: int | None = None,
    clear_logs: bool = True,
    no_mutate_gen0: bool = False,
) -> None:
    setup_runtime()
    from PYTHON.tools.layoutPreset import run_ga_from_applied_preset

    defaults = load_defaults().get("polish", {})
    applied_path = _resolve_applied(applied)
    run_ga_from_applied_preset(
        str(applied_path),
        subject_id=subject,
        n_generations=generations or int(defaults.get("ga_generations", 20)),
        population_size=population or int(defaults.get("ga_population", 20)),
        clear_logs=clear_logs,
        mutate_gen0_siblings=not no_mutate_gen0,
    )


def run_polish_mode(
    applied: str | Path,
    mode: str,
    output: str | Path | None = None,
    *,
    visualize: bool = False,
    **kwargs,
) -> dict | None:
    result: dict | None = None
    if mode == "repair" or mode == "gentle":
        result = run_repair(applied, output, electrodes_only=kwargs.get("electrodes_only", False))
    elif mode == "refine":
        result = run_refine(applied, output)
    elif mode in ("ga", "ga-short"):
        run_ga_short(applied, kwargs.get("subject"), **{k: v for k, v in kwargs.items() if k in (
            "generations", "population", "clear_logs", "no_mutate_gen0"
        )})
    else:
        raise ValueError(f"Unknown polish mode {mode!r}. Use: gentle, repair, refine, ga-short")

    if visualize:
        from app.layout.visualize import visualize_layout

        viz_path = output or applied
        if result and output is None:
            stem = _resolve_applied(applied).stem
            if mode in ("repair", "gentle") and not stem.endswith("_repaired"):
                viz_path = _resolve_applied(applied).parent / f"{stem}_repaired.json"
            elif mode == "refine":
                viz_path = _resolve_applied(applied).parent / (
                    stem.replace("_refined", "").replace("_repaired", "") + "_refined.json"
                )
        visualize_layout(viz_path, mode="both", show=False, show_3d=True)
    return result
