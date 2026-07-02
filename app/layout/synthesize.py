"""Stage B — deterministic layout synthesis (apply-v4 --synthesize)."""
from __future__ import annotations

import json
from pathlib import Path

from app import paths
from app.runtime import setup_runtime


def run_synthesize(
    preset: str | Path,
    target: int,
    output: str | Path | None = None,
    *,
    preserve_entry_order: bool = False,
    use_target_terminals: bool = True,
    optimize_terminals: bool = True,
    uv_resolution: int = 100,  # reserved for future UV packaging
) -> dict:
    setup_runtime()
    from PYTHON.tools.layoutPresetV4 import apply_layout_preset_v4_synthesize

    preset_path = paths.preset_path(preset)
    if not preset_path.exists():
        raise FileNotFoundError(
            f"Assignment/preset not found: {preset_path}\n"
            f"  Put {preset_path.name} in data/presets/, or run:\n"
            f"    python -m app build-assignments --reference 1 --id {preset_path.stem}\n"
            f"  Default preset is set in config/defaults.yaml (synthesize.assignments)."
        )

    sid = int(target)
    missing = []
    for label, p in (
        ("cleaned mesh", paths.cleaned_scan(sid)),
        ("electrodes", paths.electrode_positions_json(sid)),
        ("fiducials", paths.fiducials_json(sid)),
    ):
        if not p.exists():
            missing.append(f"  - {label}: {p}")
    if missing:
        raise FileNotFoundError(
            f"Missing preprocess inputs for subject {sid}:\n" + "\n".join(missing)
        )

    out = Path(output) if output else paths.synth_layout(target)
    out.parent.mkdir(parents=True, exist_ok=True)

    result = apply_layout_preset_v4_synthesize(
        str(preset_path),
        target_subject_id=target,
        output_path=str(out),
        preserve_entry_order=preserve_entry_order,
        use_target_terminals=use_target_terminals,
        optimize_terminals=optimize_terminals,
    )
    cm = result.get("collision_metrics", {})
    print(
        f"Wrote {out} | crossings={cm.get('crossing_count')} "
        f"electrode_violations={cm.get('electrode_violations')} "
        f"collision_free={cm.get('layout_collision_free')}"
    )
    return result


def run_visualize(
    applied: str | Path,
    *,
    save_2d: str | Path | None = None,
    save_3d: str | Path | None = None,
    show: bool = False,
    only_3d: bool = False,
    mode: str | None = None,
    skip_collisions: bool = False,
    show_3d: bool = True,
) -> tuple[Path | None, Path | None]:
    from app.layout.visualize import visualize_layout

    if mode is None:
        mode = "3d" if only_3d else "both"
    return visualize_layout(
        applied,
        mode=mode,  # type: ignore[arg-type]
        save_2d=save_2d,
        save_3d=save_3d,
        show=show,
        show_3d=show_3d,
        skip_collisions=skip_collisions,
    )


def run_export_preset_v4(
    subject: int,
    out: str | Path,
    *,
    individual: str | None = None,
    log_dir: str | Path | None = None,
) -> None:
    setup_runtime()
    from PYTHON.tools.layoutPresetV4 import export_layout_preset_v4

    export_layout_preset_v4(
        subject,
        str(paths.preset_path(out)),
        individual_key=individual,
        log_dir=str(log_dir) if log_dir else None,
    )


def build_assignment_map(
    reference_subject: int,
    assignment_id: str,
    out: str | Path | None = None,
) -> Path:
    """Terminal assignment map only (LEFT/RIGHT); no paths or hub positions."""
    setup_runtime()
    fid_path = paths.fiducials_json(reference_subject)
    if not fid_path.exists():
        raise FileNotFoundError(
            f"Reference fiducials missing: {fid_path}\n"
            f"  Run preprocess --subject {reference_subject} --step fiducials, or copy "
            f"fiducials_{reference_subject}.json from genetic_SHAPE into data/json/."
        )
    assign_path = paths.terminal_assignments_json(reference_subject)
    if not assign_path.exists():
        raise FileNotFoundError(
            f"Reference terminal assignments missing: {assign_path}\n"
            f"  Run preprocess --subject {reference_subject} --step assignments, or copy "
            f"initial_terminal_assignments_{reference_subject}.json into data/json/."
        )
    assign = json.loads(assign_path.read_text(encoding="utf-8"))
    doc = {
        "preset_version": 4,
        "preset_id": assignment_id,
        "assignment_only": True,
        "electrode_layout": "standard_10-20",
        "source_subject_id": reference_subject,
        "terminal_assignments": assign,
    }
    out_path = paths.preset_path(out or f"{assignment_id}_assignments")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"Wrote assignment map {out_path} ({len(assign)} electrodes)")
    return out_path

