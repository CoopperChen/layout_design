"""Electrode G-code configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PadTrace:
    """One electrode pad: rows are x, y, z, nx, ny, nz."""

    name: str
    xyzn: np.ndarray


@dataclass
class MachineConfig:
    d_mm: float = 57.59
    a_mm: float = 180.7
    gap_size_mm: float = 15.0
    calgap_z_mm: float = 26.62
    c0_deg: float = 90.0
    b0_deg: float = 0.0
    speed_mm_min: float = 500.0
    max_speed_mm_min: float = 750.0
    transition_speed_mm_min: float = 1500.0
    retract_speed_mm_min: float = 750.0
    jet_freq_hz: float = 12.0
    zsafe_margin_mm: float = 25.0


@dataclass
class JobConfig:
    subject: str = ""
    physical_landmarks_mm: np.ndarray = field(
        default_factory=lambda: np.zeros((3, 3), dtype=float)
    )
    rot0y_deg: float = 0.0
    rot0z_deg: float = 0.0
    print_mode: str | int = "all"

    def resolve_print_index(self, names: list[str]) -> int:
        if self.print_mode == "all" or self.print_mode == 0:
            return 0
        if isinstance(self.print_mode, int):
            return int(self.print_mode)
        if isinstance(self.print_mode, str) and self.print_mode.isdigit():
            return int(self.print_mode)
        try:
            return names.index(str(self.print_mode)) + 1
        except ValueError as exc:
            raise ValueError(f"Unknown electrode: {self.print_mode}") from exc
