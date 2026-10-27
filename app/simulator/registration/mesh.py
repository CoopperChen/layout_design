"""Register a head mesh into the controller machine frame."""

from __future__ import annotations

import numpy as np

from app.simulator.registration.subject import SubjectRegistration, register_subject


def register_mesh_full(
    mesh_points: np.ndarray,
    mesh_faces: np.ndarray,
    landmarks_xyz: np.ndarray,
    physical_landmarks_mm: np.ndarray,
    *,
    a_mm: float,
    d_mm: float,
    calgap_z_mm: float,
    rot0y_deg: float = 0.0,
    rot0z_deg: float = 0.0,
    machine_frame: bool = True,
) -> SubjectRegistration:
    return register_subject(
        mesh_points,
        mesh_faces,
        landmarks_xyz,
        physical_landmarks_mm,
        a_mm=a_mm,
        d_mm=d_mm,
        calgap_z_mm=calgap_z_mm,
        rot0y_deg=rot0y_deg,
        rot0z_deg=rot0z_deg,
        machine_frame=machine_frame,
    )
