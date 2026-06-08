"""Data models for subject bundles and postprocessor configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

SCHEMA_VERSION = "eeg_subject_bundle/1.0.0"


@dataclass
class TraceChannel:
    name: str
    interconnect: np.ndarray  # (N, 6) x,y,z,nx,ny,nz
    electrode: np.ndarray  # (M, 6)
    terminal: str = ""


@dataclass
class SubjectBundle:
    schema_version: str
    subject_id: str | int
    mesh_points: np.ndarray  # (V, 3)
    mesh_faces: np.ndarray  # (F, 3) 0-based
    landmarks_xyz: np.ndarray  # (3, 3)
    landmark_names: list[str]
    channels: list[TraceChannel]
    anatomical_xyz: np.ndarray | None = None
    sources: dict[str, str] = field(default_factory=dict)


@dataclass
class MachineConfig:
    d_mm: float = 57.59
    a_mm: float = 180.7
    gap_size_mm: float = 15.0
    calgap_z_mm: float = 26.62
    c0_deg: float = 90.0
    b0_deg: float = 0.0
    speed_mm_min: float = 500.0
    max_speed_mm_min: float = 1500.0
    transition_speed_mm_min: float = 1000.0
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
    trace_type: Literal["interconnect", "electrode"] = "interconnect"
    print_mode: str | int = "all"  # "all" or channel name or 1-based index
    export_name_version: str = "0deg"
    output_dir: Path | None = None

    def resolve_print_index(self, names: list[str]) -> int:
        """Return 0 for all channels, else 1-based index matching MATLAB chooseprint."""
        if self.print_mode == "all" or self.print_mode == 0:
            return 0
        if isinstance(self.print_mode, int):
            return int(self.print_mode)
        if isinstance(self.print_mode, str) and self.print_mode.isdigit():
            return int(self.print_mode)
        try:
            return names.index(str(self.print_mode)) + 1
        except ValueError as exc:
            raise ValueError(f"Unknown channel: {self.print_mode}") from exc

    @property
    def choose_trace(self) -> int:
        return 1 if self.trace_type == "interconnect" else 2
