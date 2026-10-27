"""Build electrode pads and write 5-axis G-code for one subject."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv

from app import paths
from app.gcode.config_loader import load_machine_config
from app.gcode.io.write_gcode import write_gcode_file
from app.gcode.kinematics.machine_fk import registration_to_machine_frame
from app.gcode.models import MachineConfig, PadTrace
from app.gcode.pads import (
    build_electrode_disk,
    diameter_mm_from_area_cm2,
    electrode_pad_settings,
    export_electrode_xyzn,
)
from app.gcode.pipeline.align import align_pads
from app.gcode.pipeline.merge_traces import merge_traces
from app.gcode.pipeline.process_traces import process_all_traces
from app.gcode.pipeline.trace_order import (
    apply_trace_order,
    load_trace_order_config,
    plan_trace_order,
)
from app.gcode.surface import load_mesh_context
from app.subject import Layout, load_layout, load_pm


def build_pads(layout: Layout, mesh_path: Path, *, gap_mm: float) -> list[PadTrace]:
    mesh = pv.read(str(mesh_path))
    if isinstance(mesh, pv.MultiBlock):
        mesh = mesh.combine()
    ctx = load_mesh_context(mesh_path)
    area_cm2, nlines = electrode_pad_settings()
    diameter = diameter_mm_from_area_cm2(area_cm2)
    print(
        f"Electrode pads: {area_cm2:g} cm² ({diameter:.2f} mm diameter), "
        f"{nlines} zigzag lines, gap {gap_mm:g} mm"
    )
    pads: list[PadTrace] = []
    for name, center in layout.electrodes.items():
        xyz, _origin, _surface, normal = build_electrode_disk(
            center,
            mesh,
            ctx,
            diameter_mm=diameter,
            gap_size_mm=gap_mm,
            nlines=nlines,
        )
        pads.append(PadTrace(name=name, xyzn=export_electrode_xyzn(xyz, normal)))
    return pads


def convert_subject(
    subject_id: int | str,
    machine: MachineConfig,
    *,
    electrode: str | int = "all",
    rot0y_deg: float = 0.0,
    rot0z_deg: float = 0.0,
    mesh_points: np.ndarray | None = None,
    mesh_faces: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str], Path]:
    layout = load_layout(subject_id)
    pm = load_pm(subject_id)
    if mesh_points is None or mesh_faces is None:
        from app.subject import load_head_arrays

        mesh_points, mesh_faces = load_head_arrays(subject_id)
    pads = build_pads(layout, paths.head_stl(subject_id), gap_mm=machine.gap_size_mm)
    names = [pad.name for pad in pads]
    from app.gcode.models import JobConfig

    choose_print = JobConfig(print_mode=electrode).resolve_print_index(names)
    aligned, mesh_registered = align_pads(
        pads,
        mesh_points,
        layout.calibration_xyz(),
        pm,
        rot0y_deg=rot0y_deg,
        rot0z_deg=rot0z_deg,
    )
    mesh_machine = registration_to_machine_frame(
        mesh_registered,
        a_mm=machine.a_mm,
        d_mm=machine.d_mm,
        calgap_z_mm=machine.calgap_z_mm,
    )
    gcode_list = process_all_traces(
        aligned,
        machine,
        choose_print,
        mesh_points=mesh_machine,
        mesh_faces=mesh_faces,
    )
    if not gcode_list:
        raise ValueError(f"No electrode matched {electrode!r}")
    mesh_z_max = float(np.max(mesh_registered[:, 2]))
    zsafe = round(mesh_z_max + machine.zsafe_margin_mm)
    print(
        f"Zsafe={zsafe} mm "
        f"(registration mesh Zmax={mesh_z_max:.2f} + margin={machine.zsafe_margin_mm:g})"
    )
    trace_order_cfg = load_trace_order_config()
    plan = plan_trace_order(
        gcode_list,
        machine,
        zsafe,
        mesh_points=mesh_machine,
        mesh_faces=mesh_faces,
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
        mesh_faces=mesh_faces,
        alternate_flip=not use_planned_order,
        skip_origin_between=plan.skip_origin if use_planned_order else None,
    )
    if (
        use_planned_order
        and trace_order_cfg.skip_origin_when_bc_close
        and plan.skip_origin
    ):
        n_skip = sum(plan.skip_origin)
        print(
            f"electrode: {n_skip}/{len(plan.skip_origin)} inter-trace hops skip "
            f"origin detour (dC <= {trace_order_cfg.c_short_transfer_max_delta_deg} deg)",
            flush=True,
        )
    if choose_print == 0:
        out = paths.electrodes_gcode(subject_id)
    else:
        out = paths.channel_gcode(subject_id, names[choose_print - 1])
    write_gcode_file(out, merged)
    return merged, names, out


def write_electrodes_gcode(
    subject_id: int | str,
    *,
    machine: str | Path | None = None,
    electrode: str | int = "all",
    rot0y_deg: float = 0.0,
    rot0z_deg: float = 0.0,
) -> Path:
    machine_path = Path(machine) if machine else paths.machine_config()
    if not machine_path.is_absolute():
        machine_path = paths.REPO_ROOT / machine_path
    machine_cfg = load_machine_config(machine_path)
    _merged, _names, out = convert_subject(
        subject_id,
        machine_cfg,
        electrode=electrode,
        rot0y_deg=rot0y_deg,
        rot0z_deg=rot0z_deg,
    )
    return out
