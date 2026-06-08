"""
Unified pipeline CLI.

  python -m app init-data
  python -m app paths --subject 2
  python -m app preprocess --subject 1 --step clear-islands
  python -m app build-assignments --reference 1 --id s1_assignments
  python -m app synthesize --assignments s1_assignments --target 2
  # --preset is alias for --assignments (terminal map only; paths are generated)
  python -m app polish --applied data/output/layouts/synth_s2.json --mode gentle
  python -m app smooth --applied data/output/layouts/synth_s2.json
  python -m app export-bundle --input data/output/smooth/smooth_s{id}_final.json
  python -m app convert-gcode --bundle data/output/bundles/subject_{id} --config config/postprocessor/subjects/example.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app import paths
from app.preprocess import run as preprocess_run


def cmd_init_data(_: argparse.Namespace) -> int:
    paths.ensure_data_tree()
    print(f"Data tree ready under {paths.DATA_DIR}")
    return 0


def cmd_paths(args: argparse.Namespace) -> int:
    sid = args.subject
    print("Repository root:", paths.REPO_ROOT)
    print()
    print("A — Preprocess")
    print("  raw PLY:       ", paths.raw_point_cloud(sid))
    print("  raw STL:       ", paths.raw_scan(sid))
    obj = paths.DATA_DIR / "raw" / f"{sid}.obj"
    try:
        obj = paths.textured_head_obj(sid)
    except FileNotFoundError:
        pass
    print("  raw OBJ:       ", obj, "" if obj.is_file() else "(not found — fiducials only)")
    print("  cleaned STL:   ", paths.cleaned_scan(sid))
    print("  fiducials:     ", paths.fiducials_json(sid))
    print("  electrodes:    ", paths.electrode_positions_json(sid))
    print("  assignments:   ", paths.terminal_assignments_json(sid))
    print()
    print("B — Synthesize")
    print("  layout out:    ", paths.synth_layout(sid))
    print()
    print("D — Postprocess")
    print("  smooth:        ", paths.smooth_json(sid))
    print("  bundle:        ", paths.bundle_export_dir(sid))
    print("  gcode:         ", paths.gcode_output_dir(sid))
    print("  matlab:        ", paths.matlab_export_dir(sid), "(legacy)")
    return 0


def cmd_preprocess(args: argparse.Namespace) -> int:
    extra: list[str] = []
    if args.spacing != 4.5:
        extra.extend(["--spacing", str(args.spacing)])
    if args.full_circle:
        extra.append("--full-circle")
    if getattr(args, "ply", None):
        extra.extend(["--ply", str(args.ply)])
    if getattr(args, "no_align_head", False):
        extra.append("--no-align-head")
    if getattr(args, "depth", 12) != 12:
        extra.extend(["--depth", str(args.depth)])
    return preprocess_run.main(
        ["--subject", str(args.subject), "--step", args.step, *extra]
    )


def _assignments_arg(args: argparse.Namespace) -> str:
    return getattr(args, "assignments", None) or args.preset


def cmd_synthesize(args: argparse.Namespace) -> int:
    from app.layout import synthesize as syn

    try:
        syn.run_synthesize(
            _assignments_arg(args),
            args.target,
            output=args.out,
            preserve_entry_order=args.preserve_entry_order,
            use_target_terminals=not args.inherit_preset_terminals,
            optimize_terminals=not args.fix_terminals,
            uv_resolution=args.uv_resolution,
        )
        if args.visualize:
            out = args.out or paths.synth_layout(args.target)
            syn.run_visualize(
                out,
                mode="both",
                show=args.show,
                show_3d=not args.no_show,
                skip_collisions=args.skip_collisions,
            )
        return 0
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1
    except Exception as e:
        print(f"synthesize failed: {e}", file=sys.stderr)
        return 1


def cmd_build_assignments(args: argparse.Namespace) -> int:
    from app.layout import synthesize as syn

    try:
        syn.build_assignment_map(
            args.reference,
            args.id or args.preset_id,
            args.out,
        )
        return 0
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1


def cmd_visualize(args: argparse.Namespace) -> int:
    from app.layout.visualize import visualize_layout

    try:
        visualize_layout(
            args.applied,
            mode=args.mode,
            save_2d=args.save,
            save_3d=args.save_3d,
            show=args.show_2d,
            show_3d=not args.no_show,
            skip_collisions=args.skip_collisions,
        )
        return 0
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1
    except (KeyError, ValueError) as e:
        print(e, file=sys.stderr)
        return 1


def cmd_polish(args: argparse.Namespace) -> int:
    from app.polish import run as polish

    try:
        polish.run_polish_mode(
            args.applied,
            args.mode,
            output=args.out,
            visualize=args.visualize,
            subject=args.subject,
            generations=args.generations,
            population=args.population,
            clear_logs=not args.no_clear_logs,
            no_mutate_gen0=args.no_mutate_gen0,
            electrodes_only=args.electrodes_only,
        )
        return 0
    except (FileNotFoundError, ValueError) as e:
        print(e, file=sys.stderr)
        return 1


def cmd_smooth(args: argparse.Namespace) -> int:
    from app.postprocess import smooth as smooth_mod

    try:
        smooth_mod.smooth_from_applied(
            args.applied,
            output=args.out,
            tag=args.tag,
            smoothing_strength=args.strength,
        )
        return 0
    except (FileNotFoundError, ValueError, OSError) as e:
        print(e, file=sys.stderr)
        if "Invalid argument" in str(e) and getattr(args, "applied", None):
            print(
                "Hint: pass --applied and --out as separate quoted arguments, e.g.\n"
                '  python -m app smooth --applied "data/output/layouts/synth_s2.json" '
                '--out "data/output/smooth/smooth_s2_final.json"',
                file=sys.stderr,
            )
        return 1


def cmd_export_matlab(args: argparse.Namespace) -> int:
    from app.postprocess import export_matlab as em
    from app.postprocess.bundle.emit import CalibrationLandmarksMissingError
    from app.postprocess.validate_export import ExportValidationError

    try:
        em.export_matlab(
            args.input,
            args.output,
            strict_landmarks=not args.allow_terminal_landmarks,
            skip_validation=args.skip_validation,
        )
        return 0
    except (FileNotFoundError, ExportValidationError, CalibrationLandmarksMissingError) as e:
        print(e, file=sys.stderr)
        return 1


def cmd_export_bundle(args: argparse.Namespace) -> int:
    from app.postprocess import export_bundle as eb
    from app.postprocess.bundle.emit import CalibrationLandmarksMissingError
    from app.postprocess.validate_export import ExportValidationError

    try:
        eb.export_bundle(
            args.input,
            args.output,
            strict_landmarks=not args.allow_terminal_landmarks,
            skip_validation=args.skip_validation,
        )
        return 0
    except (FileNotFoundError, ExportValidationError, CalibrationLandmarksMissingError) as e:
        print(e, file=sys.stderr)
        return 1


def cmd_convert_gcode(args: argparse.Namespace) -> int:
    from app.postprocess import convert_gcode as cg

    try:
        out = cg.convert_gcode(
            args.bundle,
            args.config,
            machine=args.machine,
            output=args.output,
            trace=args.trace,
            electrode=args.electrode,
            subject=args.subject,
        )
        print(f"Wrote G-code to {out}")
        return 0
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1


def cmd_list_electrodes(args: argparse.Namespace) -> int:
    from app.postprocess import convert_gcode as cg

    try:
        names = cg.list_electrodes(args.bundle)
        for i, name in enumerate(names, 1):
            print(f"{i:3d}  {name}")
        return 0
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="layout",
        description=(
            "Generate per-subject wire layouts. "
            "Assignment map = LEFT/RIGHT per electrode only; paths are synthesized on each target."
        ),
    )
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("init-data").set_defaults(func=cmd_init_data)

    pp = sub.add_parser("paths", help="Print canonical paths for a subject")
    pp.add_argument("--subject", type=int, required=True)
    pp.set_defaults(func=cmd_paths)

    pr = sub.add_parser("preprocess", help="Stage A")
    pr.add_argument("--subject", type=int, required=True)
    pr.add_argument(
        "--step",
        required=True,
        choices=[
            "reconstruct",
            "clear-islands",
            "fiducials",
            "cz",
            "electrodes",
            "assignments",
            "entry-capacity",
        ],
    )
    pr.add_argument("--ply", type=Path, default=None, help="Input .ply for reconstruct step")
    pr.add_argument("--no-align-head", action="store_true", help="Skip head rotation UI")
    pr.add_argument("--depth", type=int, default=12, help="Poisson octree depth")
    pr.add_argument("--spacing", type=float, default=4.5)
    pr.add_argument("--full-circle", action="store_true")
    pr.set_defaults(func=cmd_preprocess)

    sy = sub.add_parser(
        "synthesize",
        help="Stage B: generate layout on target (paths + slots; not preset path replay)",
    )
    sy.add_argument(
        "--assignments",
        "--preset",
        dest="assignments",
        required=True,
        metavar="NAME",
        help="Terminal assignment map in data/presets/ (LEFT/RIGHT per electrode only)",
    )
    sy.add_argument("--target", type=int, required=True)
    sy.add_argument("--out")
    sy.add_argument("--preserve-entry-order", action="store_true")
    sy.add_argument(
        "--inherit-preset-terminals",
        action="store_true",
        help="Map TERMINAL_LEFT/RIGHT from preset via rigid landmarks (legacy S1→S2)",
    )
    sy.add_argument(
        "--fix-terminals",
        action="store_true",
        help="Use target fiducial hub clicks exactly (no ±36° hub angle search)",
    )
    sy.add_argument("--uv-resolution", type=int, default=100)
    sy.add_argument(
        "--visualize",
        action="store_true",
        help="After synthesize: save 2D PNG and open interactive 3D window",
    )
    sy.add_argument("--show", action="store_true", help="Also open interactive 2D matplotlib window")
    sy.add_argument(
        "--no-show",
        action="store_true",
        help="With --visualize: save 2D PNG only, skip 3D window",
    )
    sy.add_argument(
        "--skip-collisions",
        action="store_true",
        help="Skip 2D collision markers (faster visualize)",
    )
    sy.set_defaults(func=cmd_synthesize)

    ba = sub.add_parser(
        "build-assignments",
        help="Write assignment-only map from reference subject initial_terminal_assignments",
    )
    ba.add_argument("--reference", type=int, required=True)
    ba.add_argument("--id", "--preset-id", dest="id", required=True, metavar="ID")
    ba.add_argument("--out")
    ba.set_defaults(func=cmd_build_assignments)

    bp = sub.add_parser(
        "build-preset",
        help="Alias for build-assignments (deprecated name)",
    )
    bp.add_argument("--reference", type=int, required=True)
    bp.add_argument("--preset-id", dest="id", required=True)
    bp.add_argument("--out")
    bp.set_defaults(func=cmd_build_assignments)

    viz = sub.add_parser("visualize", help="2D PNG and/or interactive 3D layout view")
    viz.add_argument("--applied", required=True)
    viz.add_argument(
        "--mode",
        choices=("2d", "3d", "both"),
        default="both",
        help="2d=polar PNG; 3d=interactive PyVista; both (default)",
    )
    viz.add_argument("--save", help="Override 2D PNG path")
    viz.add_argument(
        "--save-3d",
        default=None,
        help="Optional: also save a 3D screenshot to this path",
    )
    viz.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open 3D window (2d mode still saves PNG when applicable)",
    )
    viz.add_argument(
        "--show-2d",
        action="store_true",
        help="Open interactive 2D matplotlib window in addition to saving PNG",
    )
    viz.add_argument(
        "--skip-collisions",
        action="store_true",
        help="Skip 2D Shapely collision markers (faster)",
    )
    viz.set_defaults(func=cmd_visualize)

    po = sub.add_parser("polish", help="Stage C (optional)")
    po.add_argument("--applied", required=True)
    po.add_argument(
        "--mode",
        default="gentle",
        choices=["gentle", "repair", "refine", "ga-short"],
    )
    po.add_argument("--out")
    po.add_argument("--subject", type=int)
    po.add_argument("--generations", type=int)
    po.add_argument("--population", type=int)
    po.add_argument("--no-clear-logs", action="store_true")
    po.add_argument("--no-mutate-gen0", action="store_true")
    po.add_argument("--electrodes-only", action="store_true")
    po.add_argument(
        "--visualize",
        action="store_true",
        help="After polish, save 2D + 3D PNGs for the output layout",
    )
    po.set_defaults(func=cmd_polish)

    sm = sub.add_parser("smooth", help="Stage D: B-spline smooth")
    sm.add_argument("--applied", required=True)
    sm.add_argument("--out")
    sm.add_argument("--tag", default="final")
    sm.add_argument("--strength", type=float)
    sm.set_defaults(func=cmd_smooth)

    eb = sub.add_parser(
        "export-bundle",
        help="Stage D: write eeg_subject_bundle/1.0.0 (manifest + NPZ)",
    )
    eb.add_argument("--input", required=True, help="Smooth JSON (e.g. smooth_s2_final.json)")
    eb.add_argument("--output", help="Output folder (default: data/output/bundles/subject_{id}/)")
    eb.add_argument(
        "--allow-terminal-landmarks",
        action="store_true",
        help="Allow export without calibration landmarks (not recommended)",
    )
    eb.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip collision-free and path readiness checks (not recommended)",
    )
    eb.set_defaults(func=cmd_export_bundle)

    cg = sub.add_parser("convert-gcode", help="Stage D: bundle → 5-axis G-code")
    cg.add_argument("--bundle", required=True, help="Bundle dir or default export path")
    cg.add_argument(
        "--config",
        required=True,
        help="Print session YAML (config/postprocessor/subjects/*.yaml)",
    )
    cg.add_argument("--machine", help="Machine YAML (default: config/postprocessor/machine_default.yaml)")
    cg.add_argument("--output", help="Output base (default: data/output/gcode/)")
    cg.add_argument(
        "--trace",
        choices=["interconnect", "electrode"],
        help="Override trace type from config",
    )
    cg.add_argument("--electrode", help="Print single channel (name or index)")
    cg.add_argument("--subject", help="Legacy .mat subject folder (if not using bundle)")
    cg.set_defaults(func=cmd_convert_gcode)

    le = sub.add_parser("list-electrodes", help="List channels in a subject bundle")
    le.add_argument("--bundle", required=True)
    le.set_defaults(func=cmd_list_electrodes)

    em = sub.add_parser("export-matlab", help="Stage D (legacy): write .mat files")
    em.add_argument("--input", required=True)
    em.add_argument("--output")
    em.add_argument(
        "--allow-terminal-landmarks",
        action="store_true",
        help="Allow export without calibration landmarks (not recommended)",
    )
    em.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip collision-free and path readiness checks (not recommended)",
    )
    em.set_defaults(func=cmd_export_matlab)

    return p


def main(argv: list[str] | None = None) -> int:
    return build_parser().parse_args(argv).func(
        build_parser().parse_args(argv)
    )


def _main_fixed(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


# fix double parse bug
main = _main_fixed

if __name__ == "__main__":
    raise SystemExit(main())
