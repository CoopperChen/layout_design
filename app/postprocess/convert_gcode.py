"""Stage D — convert eeg_subject_bundle to 5-axis G-code."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from app import paths
from app.runtime import setup_runtime


def convert_gcode(
    bundle_dir: str | Path,
    config: str | Path | None = None,
    *,
    pm_file: str | Path | None = None,
    machine: str | Path | None = None,
    output: str | Path | None = None,
    trace: Literal["interconnect", "electrode", "both"] = "both",
    electrode: str | int = "all",
    rot0y_deg: float = 0.0,
    rot0z_deg: float = 0.0,
    subject: str | Path | None = None,
) -> Path | list[Path]:
    setup_runtime()
    from app.postprocess.gcode.converter import run_conversion
    from app.postprocess.gcode.config_loader import load_machine_config
    from app.postprocess.gcode.io.load_bundle import load_bundle
    from app.postprocess.gcode.io.load_mat import load_mat_subject
    from app.postprocess.gcode.models import JobConfig
    from app.postprocess.print_config import load_physical_landmarks, resolve_pm_config

    bundle_path = Path(bundle_dir)
    if not bundle_path.is_absolute():
        bundle_path = paths.REPO_ROOT / bundle_path

    if (bundle_path / "manifest.json").exists():
        bundle = load_bundle(bundle_path)
    elif subject is not None:
        bundle = load_mat_subject(Path(subject))
    else:
        raise FileNotFoundError(f"No manifest.json in {bundle_path}")

    pm_path = resolve_pm_config(bundle_path, config=config, pm_file=pm_file)
    pm = load_physical_landmarks(pm_path)

    machine_path = Path(machine) if machine else paths.postprocessor_machine_config()
    if not machine_path.is_absolute():
        machine_path = paths.REPO_ROOT / machine_path

    machine_cfg = load_machine_config(machine_path)
    job = JobConfig(
        subject=str(bundle.subject_id),
        physical_landmarks_mm=pm,
        rot0y_deg=rot0y_deg,
        rot0z_deg=rot0z_deg,
        trace_type=trace,
        print_mode=electrode,
    )

    if output is None:
        output_base = paths.gcode_output_dir()
    else:
        output_base = Path(output)
        if not output_base.is_absolute():
            output_base = paths.REPO_ROOT / output_base

    return run_conversion(bundle, machine_cfg, job, output_base)


def list_electrodes(bundle_dir: str | Path) -> list[str]:
    setup_runtime()
    from app.postprocess.gcode.io.load_bundle import load_bundle

    bundle_path = Path(bundle_dir)
    if not bundle_path.is_absolute():
        bundle_path = paths.REPO_ROOT / bundle_path
    bundle = load_bundle(bundle_path)
    return [ch.name for ch in bundle.channels]
