"""Orchestrate full G-code conversion pipeline."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from .io.write_gcode import write_gcode_file
from .models import JobConfig, MachineConfig, SubjectBundle
from .pipeline.align import align_subject
from .pipeline.merge_traces import merge_traces
from .pipeline.process_traces import process_all_traces


def convert_to_gcode(
    bundle: SubjectBundle,
    machine: MachineConfig,
    job: JobConfig,
) -> tuple[np.ndarray, list[str]]:
    channels, mesh_registered = align_subject(bundle, job)
    choose_print = job.resolve_print_index([ch.name for ch in channels])

    gcode_list = process_all_traces(
        channels,
        machine,
        choose_trace=job.choose_trace,
        choose_print=choose_print,
    )

    mesh_z_max = float(np.max(mesh_registered[:, 2]))
    merged = merge_traces(gcode_list, mesh_z_max, machine, choose_print)
    names = [ch.name for ch in channels]
    return merged, names


def _gcode_output_subdir(subject: str) -> str:
    """Match paths.gcode_output_dir(): subject_{id}_post."""
    s = str(subject) if subject else "unknown"
    stem = s if s.startswith("subject_") else f"subject_{s}"
    return f"{stem}_post"


def output_path_for_job(
    job: JobConfig,
    names: list[str],
    base_output: Path,
) -> Path:
    out_dir = base_output / _gcode_output_subdir(job.subject or "unknown")
    choose_print = job.resolve_print_index(names)

    if job.choose_trace == 1:
        if choose_print == 0:
            return out_dir / "allinterconnects.txt"
        return out_dir / f"{names[choose_print - 1]}interconnect.txt"
    if choose_print == 0:
        return out_dir / "allelectrode.txt"
    return out_dir / f"{names[choose_print - 1]}electrode.txt"


def _write_one_trace(
    bundle: SubjectBundle,
    machine: MachineConfig,
    job: JobConfig,
    base: Path,
) -> Path:
    merged, names = convert_to_gcode(bundle, machine, job)
    out_path = output_path_for_job(job, names, base)
    write_gcode_file(out_path, merged)
    return out_path


def run_conversion(
    bundle: SubjectBundle,
    machine: MachineConfig,
    job: JobConfig,
    output_base: Path | None = None,
) -> Path | list[Path]:
    base = output_base or Path("output/gcode")

    if job.trace_type == "both":
        outputs: list[Path] = []
        for trace_type in ("interconnect", "electrode"):
            sub_job = replace(job, trace_type=trace_type)
            outputs.append(_write_one_trace(bundle, machine, sub_job, base))
        return outputs

    return _write_one_trace(bundle, machine, job, base)
