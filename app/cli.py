"""Electrode layout commands: mesh, layout, record-pm, gcode."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from app import paths


def _chdir_repo() -> None:
    os.chdir(paths.REPO_ROOT)
    paths.ensure_data_tree()


def cmd_paths(args: argparse.Namespace) -> int:
    print(f"subject {args.subject}")
    for label, path in paths.subject_artifacts(args.subject):
        flag = "ok" if path.is_file() else "missing"
        print(f"  {flag:7}  {label:12}  {path}")
    machine = paths.machine_config()
    flag = "ok" if machine.is_file() else "missing"
    print(f"  {flag:7}  {'machine':12}  {machine}")
    return 0


def cmd_mesh(args: argparse.Namespace) -> int:
    from app.mesh.run import run_mesh

    try:
        return run_mesh(
            args.subject,
            ply_path=args.ply,
            obj_path=args.obj,
            zip_path=getattr(args, "zip_path", None),
            poisson_depth=args.depth,
            preview=not args.no_preview,
            rotate_head=not args.no_rotate_head,
            match_scale=not args.no_scale,
        )
    except (FileNotFoundError, ValueError, ImportError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        return 1


def cmd_layout(args: argparse.Namespace) -> int:
    from app.layout.run import run_layout

    try:
        return run_layout(args.subject)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        return 1


def cmd_record_pm(args: argparse.Namespace) -> int:
    from app.machine.record_pm import record_physical_landmarks

    try:
        out = record_physical_landmarks(
            args.subject,
            bind_ip=args.bind_ip,
            port=args.port,
            stale_sec=args.stale_ms / 1000.0,
            force=args.force,
        )
        print(f"Wrote {out}")
        return 0
    except SystemExit as exc:
        code = exc.code
        return 0 if code in (None, 0) else (code if isinstance(code, int) else 1)
    except (FileExistsError, FileNotFoundError, ValueError, OSError) as exc:
        print(exc, file=sys.stderr)
        return 1


def cmd_gcode(args: argparse.Namespace) -> int:
    from app.gcode.convert import write_electrodes_gcode

    electrode: str | int = args.electrode
    if isinstance(electrode, str) and electrode.isdigit():
        electrode = int(electrode)
    try:
        out = write_electrodes_gcode(
            args.subject,
            machine=args.machine,
            electrode=electrode,
            rot0y_deg=args.rot0y,
            rot0z_deg=args.rot0z,
        )
        print(f"Wrote {out}")
        return 0
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        return 1


def cmd_simulate(args: argparse.Namespace) -> int:
    from app.simulator.cli import _parse_layers, simulate_gcode

    try:
        simulate_gcode(
            args.subject,
            args.gcode,
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
        print(f"{exc}\nInstall PyVista and Open3D.", file=sys.stderr)
        return 1
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1


def cmd_cnc(args: argparse.Namespace) -> int:
    import time

    from app.machine.cnc_control import CncControlClient, CncControlError

    client = CncControlClient(host=args.host, port=args.port)
    try:
        if args.cnc_cmd == "status":
            ack = client.status()
        elif args.cnc_cmd == "load":
            gcode = args.gcode
            if gcode is None:
                if args.subject is None:
                    print("cnc load needs --subject or --gcode", file=sys.stderr)
                    return 1
                gcode = paths.electrodes_gcode(args.subject)
            ack = client.load(gcode)
        elif args.cnc_cmd in ("home", "scan"):
            bundled = {
                "home": (client.home, "B/C home script", "layout cnc home --confirm"),
                "scan": (client.scan, "3D head-scan script", "layout cnc scan --confirm"),
            }
            load_fn, label, hint = bundled[args.cnc_cmd]
            ack = load_fn()
            print(ack.format_line())
            if not ack.ok:
                return 1
            if not getattr(args, "confirm", False):
                print(f"Loaded {label}. Run it with: {hint}")
                return 0
            time.sleep(0.25)
            ack = client.start()
        elif args.cnc_cmd == "start":
            ack = client.start()
        elif args.cnc_cmd == "hold":
            ack = client.hold()
        elif args.cnc_cmd == "stop":
            ack = client.stop()
        else:
            print(f"unknown cnc command {args.cnc_cmd}", file=sys.stderr)
            return 2
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    except CncControlError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(ack.format_line())
    return 0 if ack.ok else 1


def cmd_run(args: argparse.Namespace) -> int:
    from app.pipeline.run import run_pipeline

    return run_pipeline(args)


def _add_cnc(sub: argparse._SubParsersAction) -> None:
    from app.machine.cnc_control import DEFAULT_CNC_HOST, DEFAULT_CNC_PORT

    opts = argparse.ArgumentParser(add_help=False)
    opts.add_argument("--host", default=DEFAULT_CNC_HOST)
    opts.add_argument("--port", type=int, default=DEFAULT_CNC_PORT)
    cnc = sub.add_parser("cnc", help="Load or run G-code on Mach4")
    commands = cnc.add_subparsers(dest="cnc_cmd", required=True)
    commands.add_parser("status", parents=[opts]).set_defaults(func=cmd_cnc)
    load = commands.add_parser("load", parents=[opts])
    load.add_argument("--subject", type=int, default=None)
    load.add_argument("--gcode", type=Path, default=None)
    load.set_defaults(func=cmd_cnc)
    for name, help_text in (
        ("home", "Load the B/C home script"),
        ("scan", "Load the head-scan script"),
    ):
        parser = commands.add_parser(name, parents=[opts], help=help_text)
        parser.add_argument("--confirm", action="store_true")
        parser.set_defaults(func=cmd_cnc)
    start = commands.add_parser("start", parents=[opts])
    start.add_argument("--confirm", action="store_true", required=True)
    start.set_defaults(func=cmd_cnc)
    commands.add_parser("hold", parents=[opts]).set_defaults(func=cmd_cnc)
    commands.add_parser("stop", parents=[opts]).set_defaults(func=cmd_cnc)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="layout",
        description="Place 10–20 electrodes on a scan and write electrode G-code.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    paths_cmd = sub.add_parser("paths", help="Show subject files")
    paths_cmd.add_argument("--subject", type=int, required=True)
    paths_cmd.set_defaults(func=cmd_paths)

    mesh = sub.add_parser("mesh", help="PLY + OBJ/MTL/JPG → upright head mesh")
    mesh.add_argument("--subject", type=int, required=True)
    mesh.add_argument("--ply", type=Path, default=None, help="Scan .ply; copied to source/scan.ply")
    mesh.add_argument("--obj", type=Path, default=None, help="Loose textured .obj (mtl and image beside it)")
    mesh.add_argument("--zip", dest="zip_path", type=Path, default=None, help="Zip of obj, mtl, and jpg")
    mesh.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Poisson octree depth (default: mesh.poisson_depth in config/defaults.yaml)",
    )
    mesh.add_argument("--no-preview", action="store_true")
    mesh.add_argument("--no-rotate-head", action="store_true")
    mesh.add_argument("--no-scale", action="store_true")
    mesh.set_defaults(func=cmd_mesh)

    layout = sub.add_parser("layout", help="Pick landmarks and place 10–20 electrodes")
    layout.add_argument("--subject", type=int, required=True)
    layout.set_defaults(func=cmd_layout)

    record = sub.add_parser("record-pm", help="Measure three landmarks on the CNC")
    record.add_argument("--subject", type=int, required=True)
    record.add_argument("--force", action="store_true")
    record.add_argument("--port", type=int, default=62101)
    record.add_argument("--bind-ip", default="0.0.0.0")
    record.add_argument("--stale-ms", type=float, default=500.0)
    record.set_defaults(func=cmd_record_pm)

    gcode = sub.add_parser("gcode", help="Write electrodes.gcode")
    gcode.add_argument("--subject", type=int, required=True)
    gcode.add_argument("--electrode", default="all")
    gcode.add_argument("--machine", default=None)
    gcode.add_argument("--rot0y", type=float, default=0.0)
    gcode.add_argument("--rot0z", type=float, default=0.0)
    gcode.set_defaults(func=cmd_gcode)

    run = sub.add_parser("run", help="mesh → layout → record-pm → gcode")
    run.add_argument("--subject", type=int, required=True)
    run.add_argument("--ply", type=Path, default=None, help="Scan .ply; copied to source/scan.ply")
    run.add_argument("--obj", type=Path, default=None, help="Loose textured .obj (mtl and image beside it)")
    run.add_argument("--zip", dest="zip_path", type=Path, default=None, help="Zip of obj, mtl, and jpg")
    run.add_argument("--from", dest="from_stage", default="mesh", choices=["mesh", "layout", "record-pm", "gcode"])
    run.add_argument("--to", dest="to_stage", default="gcode", choices=["mesh", "layout", "record-pm", "gcode"])
    run.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Poisson octree depth (default: mesh.poisson_depth in config/defaults.yaml)",
    )
    run.add_argument("--no-preview", action="store_true")
    run.add_argument("--no-rotate-head", action="store_true")
    run.add_argument("--no-scale", action="store_true")
    run.add_argument("--force-record-pm", action="store_true")
    run.add_argument("--port", type=int, default=62101)
    run.add_argument("--bind-ip", default="0.0.0.0")
    run.add_argument("--stale-ms", type=float, default=500.0)
    run.add_argument("--electrode", default="all")
    run.add_argument("--machine", default=None)
    run.add_argument("--rot0y", type=float, default=0.0)
    run.add_argument("--rot0z", type=float, default=0.0)
    run.set_defaults(func=cmd_run)

    simulate = sub.add_parser("simulate", help="View electrodes.gcode on the machine")
    simulate.add_argument("--subject", type=int, required=True)
    simulate.add_argument("--gcode", default=None)
    simulate.add_argument("--pm-file", default=None)
    simulate.add_argument("--machine", default=None)
    simulate.add_argument("--rot0y", type=float, default=0.0)
    simulate.add_argument("--rot0z", type=float, default=0.0)
    simulate.add_argument("--layers", default="mesh,landmarks,origin,tip,arm")
    simulate.add_argument("--animate", action="store_true")
    simulate.add_argument("--verbose", action="store_true")
    simulate.set_defaults(func=cmd_simulate)

    _add_cnc(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    _chdir_repo()
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
