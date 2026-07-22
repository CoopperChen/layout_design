"""Stage A — interactive preprocess steps."""
from __future__ import annotations

import argparse
import os
import runpy
import sys
from collections.abc import Callable
from pathlib import Path

from app import paths
from app.runtime import setup_runtime

_PREP_SCRIPTS = {
    "reconstruct": None,
    "clear-islands": "PYTHON/0_PREP/0_clearIslands.py",
    "fiducials": "PYTHON/0_PREP/1_selectFiducials.py",
    "cz": "PYTHON/0_PREP/2_showCz.py",
    "electrodes": "PYTHON/0_PREP/3_placeElectrodes.py",
}

_EXPECTED_OUTPUTS: dict[str, Callable[[int], Path]] = {
    "clear-islands": paths.cleaned_scan,
    "fiducials": paths.fiducials_json,
    "cz": paths.cz_json,
    "electrodes": paths.electrode_positions_json,
}


def _exit_code(exc: SystemExit) -> int:
    code = exc.code
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    return 1


def _run_script(relative: str, subject_id: int) -> int:
    setup_runtime()
    os.environ["LAYOUT_SUBJECT_ID"] = str(subject_id)
    script = paths.APP_DIR / relative
    if not script.exists():
        raise FileNotFoundError(script)
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as exc:
        return _exit_code(exc)
    return 0


def run_reconstruct(
    subject_id: int,
    *,
    ply_path: Path | None = None,
    align_head: bool = True,
    poisson_depth: int = 12,
) -> int:
    from app.preprocess.reconstruct import run_reconstruct as _run

    return _run(
        subject_id,
        ply_path=ply_path,
        align_head=align_head,
        poisson_depth=poisson_depth,
    )


_EXPECTED_HINTS: dict[str, str] = {
    "clear-islands": "Expected automated write of cleaned STL failed.",
    "fiducials": "S or close window to save picks (Q discards).",
    "cz": "Space/Enter/S or close to save Cz (Q discards).",
    "electrodes": "Space/Enter/S or close to save (Q discards).",
}


def run_step(step: str, subject_id: int) -> int:
    if step not in _PREP_SCRIPTS:
        raise ValueError(f"Unknown step {step!r}. Choose from: {', '.join(_PREP_SCRIPTS)}")
    script = _PREP_SCRIPTS[step]
    if script is None:
        return run_reconstruct(subject_id)
    rc = _run_script(script, subject_id)
    if rc != 0:
        return rc
    expected_fn = _EXPECTED_OUTPUTS.get(step)
    if expected_fn is not None:
        out = expected_fn(subject_id)
        if not Path(out).is_file():
            hint = _EXPECTED_HINTS.get(step, "Confirm/save in the GUI.")
            print(
                f"Stage {step!r} finished without writing {out}.\n  {hint}",
                file=sys.stderr,
            )
            return 1
    return 0


def run_assignments(subject_id: int) -> int:
    """Create geodesic seeds + balanced terminal assignments (no GA)."""
    setup_runtime()
    from PYTHON.tools.helper import load_electrode_positions_and_fiducials
    from PYTHON.tools.initiate3DConnections import createAndSaveInitConnections

    if not paths.cleaned_scan(subject_id).exists():
        raise FileNotFoundError(
            f"Missing cleaned mesh: {paths.cleaned_scan(subject_id)}. Run clear-islands first."
        )
    electrodes, fiducials = load_electrode_positions_and_fiducials(scanID=subject_id)
    createAndSaveInitConnections(subject_id, electrodes, fiducials)
    print(f"Wrote {paths.terminal_assignments_json(subject_id)}")
    return 0


def run_entry_capacity(subject_id: int, spacing: float = 4.5, full_circle: bool = False) -> int:
    setup_runtime()
    from PYTHON.tools import terminal_entry_capacity as tec

    argv = [str(subject_id)]
    if spacing != 4.5:
        argv.extend(["--spacing", str(spacing)])
    if full_circle:
        argv.append("--full-circle")
    return tec.main(argv) or 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage A: preprocess")
    p.add_argument("--subject", type=int, required=True)
    p.add_argument(
        "--step",
        choices=[*_PREP_SCRIPTS, "assignments", "entry-capacity"],
        required=True,
    )
    p.add_argument("--spacing", type=float, default=4.5)
    p.add_argument("--full-circle", action="store_true")
    p.add_argument("--ply", type=Path, default=None, help="Input .ply (reconstruct step)")
    p.add_argument("--no-align-head", action="store_true", help="Skip head rotation (reconstruct)")
    p.add_argument("--depth", type=int, default=12, help="Poisson depth (reconstruct)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.step == "assignments":
            return run_assignments(args.subject)
        if args.step == "entry-capacity":
            return run_entry_capacity(args.subject, args.spacing, args.full_circle)
        if args.step == "reconstruct":
            return run_reconstruct(
                args.subject,
                ply_path=args.ply,
                align_head=not args.no_align_head,
                poisson_depth=args.depth,
            )
        return run_step(args.step, args.subject)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1
