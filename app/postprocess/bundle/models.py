"""Canonical data models for eeg_subject_bundle/1.0.0."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TraceChannel:
    name: str
    interconnect: np.ndarray  # (N, 6) x,y,z,nx,ny,nz
    electrode: np.ndarray  # (M, 6)
    terminal: str = ""


@dataclass
class SubjectBundle:
    schema_version: str
    subject_id: int | str
    mesh_points: np.ndarray
    mesh_faces: np.ndarray
    landmarks_xyz: np.ndarray
    landmark_names: list[str]
    channels: list[TraceChannel]
    anatomical_xyz: np.ndarray | None = None
    sources: dict[str, str] = field(default_factory=dict)
