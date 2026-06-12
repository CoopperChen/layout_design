"""Register head mesh into controller machine frame for simulate-gcode."""

from __future__ import annotations

from app.postprocess.bundle.models import SubjectBundle
from app.simulator.registration.subject import SubjectRegistration, register_subject


def register_mesh_full(
    bundle: SubjectBundle,
    physical_landmarks_mm,
    *,
    a_mm: float,
    d_mm: float,
    calgap_z_mm: float,
    rot0y_deg: float = 0.0,
    rot0z_deg: float = 0.0,
    machine_frame: bool = True,
) -> SubjectRegistration:
    """Register bundle mesh and landmarks for simulate-gcode."""
    return register_subject(
        bundle,
        physical_landmarks_mm,
        a_mm=a_mm,
        d_mm=d_mm,
        calgap_z_mm=calgap_z_mm,
        rot0y_deg=rot0y_deg,
        rot0z_deg=rot0z_deg,
        machine_frame=machine_frame,
    )
