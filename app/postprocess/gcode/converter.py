"""Orchestrate full G-code conversion pipeline."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from .io.write_gcode import write_gcode_file
from .kinematics.machine_fk import registration_to_machine_frame
from .models import JobConfig, MachineConfig, SubjectBundle
from .pipeline.align import align_subject
from .pipeline.merge_traces import merge_traces
from .pipeline.process_traces import process_all_traces
from .pipeline.trace_order import (
    apply_trace_order,
    load_trace_order_config,
    plan_trace_order,
)


def convert_to_gcode(
    bundle: SubjectBundle,
    machine: MachineConfig,
    job: JobConfig,
) -> tuple[np.ndarray, list[str]]:
    channels, mesh_registered = align_subject(bundle, job)
    choose_print = job.resolve_print_index([ch.name for ch in channels])

    mesh_machine = registration_to_machine_frame(
        mesh_registered,
        a_mm=machine.a_mm,
        d_mm=machine.d_mm,
        calgap_z_mm=machine.calgap_z_mm,
    )

    gcode_list = process_all_traces(
        channels,
        machine,
        choose_trace=job.choose_trace,
        choose_print=choose_print,
        mesh_points=mesh_machine,
        mesh_faces=bundle.mesh_faces,
    )

    mesh_z_max = float(np.max(mesh_registered[:, 2]))
    zsafe = round(mesh_z_max + machine.zsafe_margin_mm)
    trace_order_cfg = load_trace_order_config()
    plan = plan_trace_order(
        gcode_list,
        machine,
        zsafe,
        mesh_points=mesh_machine,
        mesh_faces=bundle.mesh_faces,
        config=trace_order_cfg,
    )
    use_planned_order = trace_order_cfg.enabled and choose_print == 0 and len(gcode_list) > 1
    if use_planned_order:
        gcode_list = apply_trace_order(gcode_list, plan.order, plan.flip)

    merged = merge_traces(
        gcode_list,
        mesh_z_max,
        machine,
        choose_print,
        mesh_points=mesh_machine,
        mesh_faces=bundle.mesh_faces,
        alternate_flip=not use_planned_order,
        skip_origin_between=plan.skip_origin if use_planned_order else None,
    )
    if (
        use_planned_order
        and trace_order_cfg.skip_origin_when_bc_close
        and plan.skip_origin
    ):
        label = "interconnect" if job.choose_trace == 1 else "electrode"
        n_skip = sum(plan.skip_origin)
        print(
            f"{label}: {n_skip}/{len(plan.skip_origin)} inter-trace hops skip "
            f"origin detour (dC <= {trace_order_cfg.c_short_transfer_max_delta_deg} deg)",
            flush=True,
        )
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
