"""Orchestrate full G-code conversion pipeline."""

from __future__ import annotations

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


def output_path_for_job(
    job: JobConfig,
    names: list[str],
    base_output: Path,
) -> Path:
    subject = job.subject or "subject"
    out_dir = base_output / f"{subject}_post"
    choose_print = job.resolve_print_index(names)

    if job.choose_trace == 1:
        if choose_print == 0:
            return out_dir / "allinterconnects.txt"
        return out_dir / f"{names[choose_print - 1]}interconnect.txt"
    if choose_print == 0:
        return out_dir / "allelectrode.txt"
    return out_dir / f"{names[choose_print - 1]}electrode.txt"


def run_conversion(
    bundle: SubjectBundle,
    machine: MachineConfig,
    job: JobConfig,
    output_base: Path | None = None,
) -> Path:
    merged, names = convert_to_gcode(bundle, machine, job)
    base = output_base or Path("output/gcode")
    out_path = output_path_for_job(job, names, base)
    write_gcode_file(out_path, merged)
    return out_path
