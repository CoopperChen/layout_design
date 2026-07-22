"""Run full pipeline A → B → C? → D from a PLY point cloud."""
from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from app import paths
from app.config_loader import (
    default_assignments,
    load_defaults,
    preprocess_defaults,
    resolve_assignments,
)

STAGES: tuple[str, ...] = (
    "reconstruct",
    "clear-islands",
    "fiducials",
    "cz",
    "electrodes",
    "synthesize",
    "polish",
    "smooth",
    "bundle",
    "print-config",
    "record-pm",
    "gcode",
    "simulate",
)


@dataclass(frozen=True)
class PipelinePaths:
    target: int
    assignments: str
    ply: Path
    cleaned: Path
    fiducials: Path
    cz: Path
    electrodes: Path
    layout: Path
    smooth: Path
    bundle: Path
    gcode: Path
    print_config: Path

    @classmethod
    def for_target(cls, target: int, assignments: str | None = None) -> PipelinePaths:
        preset = resolve_assignments(assignments)
        return cls(
            target=target,
            assignments=preset,
            ply=paths.raw_point_cloud(target),
            cleaned=paths.cleaned_scan(target),
            fiducials=paths.fiducials_json(target),
            cz=paths.cz_json(target),
            electrodes=paths.electrode_positions_json(target),
            layout=paths.synth_layout(target),
            smooth=paths.smooth_json(target),
            bundle=paths.bundle_export_dir(target),
            gcode=paths.gcode_output_dir(target) / "allinterconnects.txt",
            print_config=paths.postprocessor_subject_pm(target),
        )


def _input_ply(args: argparse.Namespace, pp: PipelinePaths) -> Path:
    if args.ply is not None:
        p = Path(args.ply)
        return p if p.is_absolute() else paths.REPO_ROOT / p
    return pp.ply


def _poisson_depth(args: argparse.Namespace) -> int:
    if args.depth is not None:
        return args.depth
    return int(preprocess_defaults().get("poisson_depth", 12))


def _align_head(args: argparse.Namespace) -> bool:
    if args.no_align_head:
        return False
    return bool(preprocess_defaults().get("align_head", True))


def _polish_enabled(args: argparse.Namespace) -> bool:
    if getattr(args, "no_polish", False):
        return False
    return bool(load_defaults().get("polish", {}).get("enabled", True))


def _polished_layout(layout: Path) -> Path:
    stem = layout.stem
    if stem.endswith("_repaired"):
        return layout
    return layout.parent / f"{stem}_repaired.json"


def _stage_index(name: str) -> int:
    try:
        return STAGES.index(name)
    except ValueError as exc:
        raise ValueError(f"Unknown stage {name!r}; choose from: {', '.join(STAGES)}") from exc


def _active_stages(
    *,
    from_stage: str,
    to_stage: str,
    polish: bool,
) -> list[str]:
    start = _stage_index(from_stage)
    end = _stage_index(to_stage)
    if start > end:
        raise ValueError(f"--from {from_stage} is after --to {to_stage}")

    selected = list(STAGES[start : end + 1])
    if not polish:
        selected = [s for s in selected if s != "polish"]
    return selected


def _require_file(path: Path, stage: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Cannot start pipeline at {stage}: missing {path}\n"
            f"Run an earlier stage first, or lower --from."
        )


def _require_textured_obj(pp: PipelinePaths, stage: str) -> None:
    obj = paths.raw_scan(pp.target, ext="obj")
    if obj.is_file():
        return
    try:
        paths.textured_head_obj(pp.target)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Cannot start pipeline at {stage}: missing textured OBJ for subject {pp.target}\n"
            f"Run reconstruct first."
        ) from exc


def _layout_input(pp: PipelinePaths, stages: Sequence[str], *, polish: bool) -> Path:
    repaired = _polished_layout(pp.layout)
    if "polish" in stages or (polish and repaired.is_file()):
        return repaired
    return pp.layout


def _validate_synthesize_inputs(pp: PipelinePaths) -> None:
    if not paths.preset_path(pp.assignments).is_file():
        raise FileNotFoundError(
            f"Assignment preset not found: {paths.preset_path(pp.assignments)}"
        )
    for label, path in (
        ("cleaned mesh", pp.cleaned),
        ("fiducials", pp.fiducials),
        ("electrodes", pp.electrodes),
    ):
        _require_file(path, "synthesize")


def _validate_inputs(
    pp: PipelinePaths,
    stages: Sequence[str],
    *,
    polish: bool,
    args: argparse.Namespace,
) -> None:
    first = stages[0]
    if first == "reconstruct":
        ply = _input_ply(args, pp)
        if not ply.is_file():
            raise FileNotFoundError(
                f"Cannot start pipeline at reconstruct: missing PLY {ply}\n"
                f"Place {pp.ply.name} under data/raw/ or pass --ply PATH"
            )
        return
    if first == "clear-islands":
        _require_file(paths.raw_scan(pp.target), first)
        return
    if first == "fiducials":
        _require_textured_obj(pp, first)
        return
    if first == "cz":
        _require_file(pp.cleaned, first)
        _require_file(pp.fiducials, first)
        return
    if first == "electrodes":
        _require_file(pp.cleaned, first)
        _require_file(pp.fiducials, first)
        _require_file(pp.cz, first)
        return
    if first == "synthesize":
        _validate_synthesize_inputs(pp)
        return

    layout = _layout_input(pp, stages, polish=polish)
    if first in {"polish", "smooth"}:
        _require_file(pp.layout if first == "polish" else layout, first)
    elif first == "bundle":
        _require_file(pp.smooth, first)
    elif first in {"print-config", "record-pm", "gcode", "simulate"}:
        _require_file(pp.smooth, "smooth")
    if first in {"gcode", "simulate"}:
        if not pp.bundle.is_dir():
            raise FileNotFoundError(
                f"Cannot start pipeline at {first}: missing bundle dir {pp.bundle}"
            )
    if first == "gcode":
        from app.postprocess.print_config import pm_is_measured

        pm_path = Path(args.pm_file or args.config or pp.print_config)
        if not pm_path.is_absolute():
            pm_path = paths.REPO_ROOT / pm_path
        if not pm_is_measured(pm_path):
            raise FileNotFoundError(
                f"Cannot start pipeline at gcode: physical landmarks not measured in {pm_path}\n"
                f"Run record-pm (or --from record-pm), or pass a measured --pm-file."
            )
    if first == "simulate":
        _require_file(pp.gcode, "gcode")


def _print_step(label: str, detail: str = "") -> None:
    line = f"\n=== {label} ==="
    if detail:
        line += f"\n{detail}"
    print(line)


def run_pipeline(args: argparse.Namespace) -> int:
    pp = PipelinePaths.for_target(args.target)
    polish = _polish_enabled(args)
    stages = _active_stages(
        from_stage=args.from_stage,
        to_stage=args.to_stage,
        polish=polish,
    )
    _validate_inputs(pp, stages, polish=polish, args=args)

    handlers: dict[str, Callable[[], int]] = {
        "reconstruct": lambda: _run_reconstruct(args, pp),
        "clear-islands": lambda: _run_clear_islands(pp),
        "fiducials": lambda: _run_fiducials(pp),
        "cz": lambda: _run_cz(pp),
        "electrodes": lambda: _run_electrodes(pp),
        "synthesize": lambda: _run_synthesize(args, pp),
        "polish": lambda: _run_polish(args, pp),
        "smooth": lambda: _run_smooth(args, _layout_input(pp, stages, polish=polish), pp),
        "bundle": lambda: _run_bundle(args, pp),
        "print-config": lambda: _run_print_config(args, pp),
        "record-pm": lambda: _run_record_pm(args, pp),
        "gcode": lambda: _run_gcode(args, pp),
        "simulate": lambda: _run_simulate(args, pp),
    }

    ply = _input_ply(args, pp)
    print(
        f"Pipeline subject {pp.target} | ply={ply.name} | "
        f"assignments={pp.assignments} | stages: {' → '.join(stages)}"
    )
    for stage in stages:
        if stage == "synthesize":
            try:
                _validate_synthesize_inputs(pp)
            except FileNotFoundError as exc:
                print(exc)
                return 1
        rc = handlers[stage]()
        if rc != 0:
            print(f"\nPipeline stopped at {stage} (exit {rc})", flush=True)
            return rc
    print("\nPipeline complete.", flush=True)
    return 0


def _run_reconstruct(args: argparse.Namespace, pp: PipelinePaths) -> int:
    from app.preprocess.run import run_reconstruct

    ply = _input_ply(args, pp)
    _print_step(
        "reconstruct",
        f"{ply} → STL/OBJ for subject {pp.target}\n"
        "  Keys: Space/Enter/S = confirm · Esc/Q = skip/cancel · close = confirm",
    )
    try:
        return run_reconstruct(
            pp.target,
            ply_path=ply,
            align_head=_align_head(args),
            poisson_depth=_poisson_depth(args),
        )
    except (FileNotFoundError, ValueError, ImportError, RuntimeError) as exc:
        print(exc)
        return 1


def _run_clear_islands(pp: PipelinePaths) -> int:
    from app.preprocess.run import run_step

    _print_step(
        "clear-islands",
        f"→ {pp.cleaned}\n"
        "  AFTER window: Space/Enter/S/close = SAVE · Q = discard",
    )
    try:
        return run_step("clear-islands", pp.target)
    except FileNotFoundError as exc:
        print(exc)
        return 1


def _run_fiducials(pp: PipelinePaths) -> int:
    from app.preprocess.run import run_step

    _print_step(
        "fiducials",
        "Interactive pick on OBJ\n"
        "  Space/Enter = confirm pick · S/close = save · Q = discard",
    )
    try:
        return run_step("fiducials", pp.target)
    except FileNotFoundError as exc:
        print(exc)
        return 1


def _run_cz(pp: PipelinePaths) -> int:
    from app.preprocess.run import run_step

    _print_step(
        "cz",
        f"→ {pp.cz}\n"
        "  Space/Enter/S/close = SAVE · Q = discard",
    )
    try:
        return run_step("cz", pp.target)
    except FileNotFoundError as exc:
        print(exc)
        return 1


def _run_electrodes(pp: PipelinePaths) -> int:
    from app.preprocess.run import run_step

    _print_step(
        "electrodes",
        f"→ {pp.electrodes}\n"
        "  Space/Enter/S/close = SAVE · Q = discard",
    )
    try:
        return run_step("electrodes", pp.target)
    except FileNotFoundError as exc:
        print(exc)
        return 1


def _run_synthesize(args: argparse.Namespace, pp: PipelinePaths) -> int:
    from app.layout import synthesize as syn

    _print_step("synthesize", f"target={pp.target} → {pp.layout}")
    try:
        syn.run_synthesize(
            pp.assignments,
            pp.target,
            output=str(pp.layout),
            preserve_entry_order=args.preserve_entry_order,
            use_target_terminals=not args.inherit_preset_terminals,
            optimize_terminals=args.rotate,
            uv_resolution=args.uv_resolution,
        )
        return 0
    except Exception as exc:
        print(f"synthesize failed: {exc}")
        return 1


def _run_polish(args: argparse.Namespace, pp: PipelinePaths) -> int:
    from app.polish import run as polish

    out = _polished_layout(pp.layout)
    _print_step("polish", f"{pp.layout} → {out}")
    try:
        polish.run_polish_mode(
            pp.layout,
            args.polish_mode,
            output=out,
            visualize=False,
            subject=pp.target,
            profile_phase2=args.polish_profile,
        )
        return 0
    except (FileNotFoundError, ValueError) as exc:
        print(exc)
        return 1


def _run_smooth(
    args: argparse.Namespace,
    applied: Path,
    pp: PipelinePaths,
) -> int:
    from app.postprocess import smooth as smooth_mod

    _print_step("smooth", f"{applied} → {pp.smooth}")
    try:
        smooth_mod.smooth_from_applied(
            applied,
            output=pp.smooth,
            tag=args.smooth_tag,
            smoothing_strength=args.smoothing_strength,
        )
        return 0
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(exc)
        return 1


def _run_bundle(args: argparse.Namespace, pp: PipelinePaths) -> int:
    from app.postprocess import export_bundle as eb
    from app.postprocess.bundle.emit import CalibrationLandmarksMissingError
    from app.postprocess.validate_export import ExportValidationError

    _print_step("export-bundle", f"{pp.smooth} → {pp.bundle}")
    try:
        eb.export_bundle(
            pp.smooth,
            pp.bundle,
            strict_landmarks=not args.allow_terminal_landmarks,
            skip_validation=args.skip_validation,
            quiet=args.quiet,
        )
        return 0
    except (FileNotFoundError, ExportValidationError, CalibrationLandmarksMissingError) as exc:
        print(exc)
        return 1


def _run_print_config(args: argparse.Namespace, pp: PipelinePaths) -> int:
    from app.postprocess.print_config import init_print_config

    if pp.print_config.is_file() and not args.force_print_config:
        print(f"\n=== print-config ===\nUsing existing {pp.print_config}")
        return 0

    _print_step("init-print-config", str(pp.print_config))
    try:
        out = init_print_config(pp.target, force=args.force_print_config)
        print(f"Wrote print config scaffold: {out}")
        print("Next stage record-pm fills physical_landmarks_mm from the CNC.")
        return 0
    except FileExistsError as exc:
        print(exc)
        return 1


def _run_record_pm(args: argparse.Namespace, pp: PipelinePaths) -> int:
    from app.postprocess.print_config import pm_is_measured
    from app.postprocess.record_pm import record_physical_landmarks

    force = bool(getattr(args, "force_record_pm", False))
    if pm_is_measured(pp.print_config) and not force:
        print(
            f"\n=== record-pm ===\n"
            f"Using measured landmarks in {pp.print_config} "
            f"(pass --force-record-pm to re-capture)"
        )
        return 0

    _print_step(
        "record-pm",
        f"Interactive CNC work-pose capture → {pp.print_config}\n"
        "  Keys: Enter/Space=capture (save when all 3 done) · "
        "1/2/3=jump · n/p=next/prev · s=save · q=quit\n"
        "  Start Mach4 UDP publisher first (see config/postprocessor/README.md).",
    )
    try:
        out = record_physical_landmarks(
            pp.target,
            bind_ip=getattr(args, "pm_bind_ip", "0.0.0.0"),
            port=int(getattr(args, "pm_port", 62100)),
            stale_sec=float(getattr(args, "pm_stale_ms", 500)) / 1000.0,
            force=True,
            output=pp.print_config,
        )
        print(f"Wrote measured pm: {out}")
        return 0
    except SystemExit as exc:
        code = exc.code
        return 0 if code in (None, 0) else (code if isinstance(code, int) else 1)
    except (FileExistsError, FileNotFoundError, ValueError, OSError) as exc:
        print(exc)
        return 1


def _run_gcode(args: argparse.Namespace, pp: PipelinePaths) -> int:
    from app.postprocess import convert_gcode as cg

    _print_step("convert-gcode", str(pp.bundle))
    try:
        out = cg.convert_gcode(
            pp.bundle,
            args.config,
            pm_file=args.pm_file,
            machine=args.machine,
            output=args.gcode_output,
            trace=args.trace,
            electrode=args.electrode,
            rot0y_deg=args.rot0y,
            rot0z_deg=args.rot0z,
            subject=args.legacy_subject,
        )
        paths_out = out if isinstance(out, list) else [out]
        for path in paths_out:
            print(f"Wrote G-code to {path}")
        return 0
    except (FileNotFoundError, ValueError) as exc:
        print(exc)
        return 1


def _run_simulate(args: argparse.Namespace, pp: PipelinePaths) -> int:
    from app.simulator.cli import simulate_gcode

    _print_step("simulate-gcode", str(pp.gcode))
    try:
        simulate_gcode(
            pp.gcode,
            pp.bundle,
            pm_file=args.pm_file,
            machine_config=args.machine,
            rot0y_deg=args.rot0y,
            rot0z_deg=args.rot0z,
            layers=_parse_layers(args.layers),
            animate=args.animate,
            verbose=args.verbose,
        )
        return 0
    except ImportError as exc:
        print(
            f"{exc}\nInstall PyVista and Open3D (see layout_design pyproject.toml)."
        )
        return 1
    except (FileNotFoundError, ValueError) as exc:
        print(exc)
        return 1


def _parse_layers(spec: str) -> set[str]:
    from app.simulator.cli import _parse_layers as parse

    return parse(spec)


def add_run_parser(sub: argparse._SubParsersAction) -> None:
    run = sub.add_parser(
        "run",
        help="Full pipeline from PLY (preprocess → synthesize → … → gcode/simulate)",
        description=(
            "Run the end-to-end workflow for one subject. "
            f"Input PLY defaults to data/raw/{{id}}.ply; assignment preset: "
            f"{default_assignments()} (config/defaults.yaml). "
            "Interactive GUIs use Space/Enter/S to confirm/save and Q to discard. "
            "Use --from synthesize to skip preprocess when already done. "
            "G-code waits for measured physical landmarks (record-pm)."
        ),
    )
    run.add_argument("--target", type=int, required=True, help="Subject id")
    run.add_argument(
        "--ply",
        type=Path,
        default=None,
        help="Input point cloud (default: data/raw/{target}.ply)",
    )
    run.add_argument(
        "--no-align-head",
        action="store_true",
        help="Skip head rotation UI during reconstruct",
    )
    run.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Poisson octree depth (default: preprocess.poisson_depth in config)",
    )
    run.add_argument(
        "--from",
        dest="from_stage",
        default="reconstruct",
        choices=STAGES,
        help="First stage (default: reconstruct)",
    )
    run.add_argument(
        "--to",
        dest="to_stage",
        default="gcode",
        choices=STAGES,
        help="Last stage (default: gcode; use simulate for 3D viewer)",
    )
    run.add_argument(
        "--no-polish",
        action="store_true",
        help="Skip polish between synthesize and smooth (default: polish runs)",
    )
    run.add_argument(
        "--polish-mode",
        default="gentle",
        choices=["gentle", "repair", "refine", "ga-short"],
    )
    run.add_argument(
        "--polish-profile",
        action="store_true",
        help="Print per-round phase-2 timing breakdown during polish",
    )
    run.add_argument("--preserve-entry-order", action="store_true")
    run.add_argument("--inherit-preset-terminals", action="store_true")
    run.add_argument(
        "--rotate",
        action="store_true",
        help="Synthesize: ±36° hub angle search around fiducial clicks",
    )
    run.add_argument("--uv-resolution", type=int, default=100)
    run.add_argument("--smooth-tag", default="final")
    run.add_argument("--smoothing-strength", type=float, default=None)
    run.add_argument("--allow-terminal-landmarks", action="store_true")
    run.add_argument("--skip-validation", action="store_true")
    run.add_argument("--quiet", action="store_true")
    run.add_argument(
        "--force-print-config",
        action="store_true",
        help="Overwrite pm YAML scaffold even if it already exists",
    )
    run.add_argument(
        "--force-record-pm",
        action="store_true",
        help="Re-capture CNC landmarks even if pm YAML is already measured",
    )
    run.add_argument(
        "--pm-port",
        type=int,
        default=62100,
        help="UDP port for Mach4 work-pose publisher (record-pm)",
    )
    run.add_argument(
        "--pm-bind-ip",
        default="0.0.0.0",
        help="UDP bind address for record-pm (default: 0.0.0.0)",
    )
    run.add_argument(
        "--pm-stale-ms",
        type=float,
        default=500.0,
        help="Treat CNC pose older than this many ms as stale (record-pm)",
    )
    run.add_argument("--config", help="pm YAML for convert-gcode")
    run.add_argument("--pm-file", help="Alias for --config")
    run.add_argument("--machine", help="Machine YAML")
    run.add_argument("--gcode-output", help="G-code output base directory")
    run.add_argument(
        "--trace",
        choices=["interconnect", "electrode", "both"],
        default="both",
    )
    run.add_argument("--electrode", default="all")
    run.add_argument("--rot0y", type=float, default=0.0)
    run.add_argument("--rot0z", type=float, default=0.0)
    run.add_argument("--legacy-subject", dest="legacy_subject", help="Legacy .mat folder")
    run.add_argument(
        "--layers",
        default="mesh,landmarks,origin,tip,arm",
        help="simulate-gcode layers (comma-separated)",
    )
    run.add_argument("--animate", action="store_true", help="simulate-gcode: step with p key")
    run.add_argument("--verbose", action="store_true", help="simulate-gcode: FK diagnostics")
    run.set_defaults(func=run_pipeline)
