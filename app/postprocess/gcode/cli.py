"""Command-line interface."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config_loader import load_job_config, load_machine_config
from .converter import run_conversion
from .io.load_bundle import load_bundle
from .io.load_mat import load_mat_subject


def _default_machine_config() -> Path:
    from app import paths

    return paths.postprocessor_machine_config()


def _load_subject(bundle: Path | None, subject: Path | None):
    if bundle:
        return load_bundle(bundle)
    if subject:
        subject = Path(subject)
        if (subject / "manifest.json").exists():
            return load_bundle(subject)
        return load_mat_subject(subject)
    raise SystemExit("Provide --bundle or --subject")


def cmd_convert(args: argparse.Namespace) -> None:
    machine_path = args.machine or _default_machine_config()
    machine = load_machine_config(machine_path)
    job = load_job_config(args.config, machine_path)

    if args.bundle:
        bundle = load_bundle(args.bundle)
    else:
        bundle = _load_subject(None, args.subject)

    if not job.subject:
        job.subject = str(bundle.subject_id)

    if args.trace:
        job.trace_type = args.trace
    if args.electrode:
        job.print_mode = args.electrode

    output_base = Path(args.output) if args.output else Path("output/gcode")
    out = run_conversion(bundle, machine, job, output_base)
    print(f"Wrote G-code to {out}")


def cmd_list_electrodes(args: argparse.Namespace) -> None:
    bundle = _load_subject(
        Path(args.bundle) if args.bundle else None,
        Path(args.subject) if args.subject else None,
    )
    for i, ch in enumerate(bundle.channels, 1):
        print(f"{i:3d}  {ch.name}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="EEG 5-axis G-code postprocessor")
    sub = parser.add_subparsers(dest="command", required=True)

    p_convert = sub.add_parser("convert", help="Convert subject bundle to G-code")
    p_convert.add_argument("--bundle", type=Path, help="Path to eeg_subject_bundle directory")
    p_convert.add_argument("--subject", type=Path, help="Path to legacy .mat subject folder")
    p_convert.add_argument("--config", type=Path, required=True, help="Job config YAML")
    p_convert.add_argument("--machine", type=Path, help="Machine config YAML")
    p_convert.add_argument("--output", type=Path, help="Output base directory")
    p_convert.add_argument(
        "--trace",
        choices=["interconnect", "electrode"],
        help="Override trace type from config",
    )
    p_convert.add_argument("--electrode", help="Print single channel (name or index)")
    p_convert.set_defaults(func=cmd_convert)

    p_list = sub.add_parser("list-electrodes", help="List channels in subject bundle")
    p_list.add_argument("--bundle", type=Path)
    p_list.add_argument("--subject", type=Path)
    p_list.set_defaults(func=cmd_list_electrodes)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
