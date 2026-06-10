"""Load machine YAML configuration."""

from __future__ import annotations

from pathlib import Path

import yaml

from .models import MachineConfig


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
