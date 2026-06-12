"""Shared scan2phys registration for simulate-gcode, then shift to machine frame."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.postprocess.bundle.models import SubjectBundle
from app.postprocess.gcode.kinematics.machine_fk import registration_to_machine_frame
from app.postprocess.gcode.pipeline.align import register_mesh_points
from app.postprocess.gcode.transform.rotations import apply_rotation, roty, rotz
from app.postprocess.gcode.transform.scan2phys import scan2phys


@dataclass(frozen=True)
class SubjectRegistration:
    """Registered subject geometry in controller machine frame (C pivot zero)."""

    mesh_points: np.ndarray
    mesh_faces: np.ndarray
    pm: np.ndarray
    calibration_registered: np.ndarray
    rot0y_deg: float
    rot0z_deg: float
    a_mm: float
    d_mm: float
    calgap_z_mm: float

    @property
    def pm_machine(self) -> np.ndarray:
        """Measured landmarks expressed in machine frame."""
        return registration_to_machine_frame(
            self.pm,
            a_mm=self.a_mm,
            d_mm=self.d_mm,
            calgap_z_mm=self.calgap_z_mm,
        )

    @property
    def landmark_fit_error_mm(self) -> float:
        """Max distance between registered digital landmarks and measured pm (machine frame)."""
        return float(
            np.max(
                np.linalg.norm(
                    self.calibration_registered - self.pm_machine,
                    axis=1,
                )
            )
        )


def register_subject(
    bundle: SubjectBundle,
    physical_landmarks_mm: np.ndarray,
    *,
    a_mm: float,
    d_mm: float,
    calgap_z_mm: float,
    rot0y_deg: float = 0.0,
    rot0z_deg: float = 0.0,
    machine_frame: bool = True,
) -> SubjectRegistration:
    """
    Register bundle mesh for simulate-gcode.

    1. ``scan2phys`` (same as ``align_subject`` / ``convert-gcode``) with central at ``pm[0]``.
    2. When ``machine_frame`` (default), shift into controller machine frame:
       C pivot zero at ``(0,0,0)``, central landmark at ``(0, −a, −(d+calgap))``.
    """
    pm = np.asarray(physical_landmarks_mm, dtype=float)
    ps = bundle.landmarks_xyz.copy()
    ps0 = ps - ps[0]
    rot = roty(rot0y_deg) @ rotz(rot0z_deg)
    ps_aligned = apply_rotation(ps0, rot)
    offset = pm[0] - ps_aligned[0]
    ps_shifted = ps_aligned + offset

    mesh = register_mesh_points(
        bundle,
        pm,
        rot0y_deg=rot0y_deg,
        rot0z_deg=rot0z_deg,
    )
    calibration_registered = scan2phys(ps_shifted, ps_shifted, pm)

    if machine_frame:
        fk_kw = {"a_mm": a_mm, "d_mm": d_mm, "calgap_z_mm": calgap_z_mm}
        mesh = registration_to_machine_frame(mesh, **fk_kw)
        calibration_registered = registration_to_machine_frame(
            calibration_registered, **fk_kw
        )

    return SubjectRegistration(
        mesh_points=mesh,
        mesh_faces=bundle.mesh_faces.copy(),
        pm=pm.copy(),
        calibration_registered=calibration_registered,
        rot0y_deg=rot0y_deg,
        rot0z_deg=rot0z_deg,
        a_mm=float(a_mm),
        d_mm=float(d_mm),
        calgap_z_mm=float(calgap_z_mm),
    )
