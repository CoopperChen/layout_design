"""Load YAML configuration files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from .models import JobConfig, MachineConfig


def load_yaml(path: Path | str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_machine_config(
    path: Path | str | None = None,
    overrides: dict | None = None,
) -> MachineConfig:
    data: dict = {}
    if path:
        data = load_yaml(path)
    if overrides:
        data.update(overrides)

    machine_section = data.get("machine", data)
    return MachineConfig(
        d_mm=float(machine_section.get("d_mm", machine_section.get("d", 57.59))),
        a_mm=float(machine_section.get("a_mm", machine_section.get("a", 180.7))),
        gap_size_mm=float(
            machine_section.get("gap_size_mm", machine_section.get("gap_size", 15))
        ),
        calgap_z_mm=float(
            machine_section.get("calgap_z_mm", machine_section.get("calgap_z", 26.62))
        ),
        c0_deg=float(machine_section.get("c0_deg", machine_section.get("c0", 90))),
        b0_deg=float(machine_section.get("b0_deg", machine_section.get("b0", 0))),
        speed_mm_min=float(
            machine_section.get("speed_mm_min", machine_section.get("speed", 500))
        ),
        max_speed_mm_min=float(
            machine_section.get("max_speed_mm_min", machine_section.get("max_speed", 1500))
        ),
        transition_speed_mm_min=float(
            machine_section.get(
                "transition_speed_mm_min",
                machine_section.get("transition_speed", 1000),
            )
        ),
        jet_freq_hz=float(
            machine_section.get("jet_freq_hz", machine_section.get("jet_freq", 12))
        ),
        zsafe_margin_mm=float(
            machine_section.get("zsafe_margin_mm", machine_section.get("zsafe_margin", 25))
        ),
    )


def load_job_config(path: Path | str, machine_defaults: Path | str | None = None) -> JobConfig:
    data = load_yaml(path)
    reg = data.get("registration", data)
    proc = data.get("process", data)

    pm = data.get("physical_landmarks_mm", reg.get("physical_landmarks_mm", [[0, 0, 0]] * 3))
    pm_arr = np.asarray(pm, dtype=float)

    trace_type = data.get(
        "trace_type", proc.get("trace_type", proc.get("choose_trace", "interconnect"))
    )
    if isinstance(trace_type, int):
        trace_type = "interconnect" if trace_type == 1 else "electrode"
    if trace_type not in ("interconnect", "electrode"):
        trace_type = "interconnect"

    print_mode = data.get("print_mode", proc.get("choose_print", "all"))

    return JobConfig(
        subject=str(data.get("subject", "")),
        physical_landmarks_mm=pm_arr,
        rot0y_deg=float(data.get("rot0y", reg.get("rot0y_deg", 0))),
        rot0z_deg=float(data.get("rot0z", reg.get("rot0z_deg", 0))),
        trace_type=trace_type,
        print_mode=print_mode,
        export_name_version=str(
            data.get("export_name_version", data.get("output", {}).get("export_name_version", "0deg"))
        ),
    )
