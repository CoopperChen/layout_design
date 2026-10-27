"""Scan-to-physical registration, then a shift into the machine frame."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.gcode.kinematics.machine_fk import registration_to_machine_frame
from app.gcode.pipeline.align import register_mesh_points
from app.gcode.transform.rotations import apply_rotation, roty, rotz
from app.gcode.transform.scan2phys import scan2phys


@dataclass(frozen=True)
class SubjectRegistration:
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
        return registration_to_machine_frame(
            self.pm,
            a_mm=self.a_mm,
            d_mm=self.d_mm,
            calgap_z_mm=self.calgap_z_mm,
        )

    @property
    def landmark_fit_error_mm(self) -> float:
        return float(
            np.max(np.linalg.norm(self.calibration_registered - self.pm_machine, axis=1))
        )


def register_subject(
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
    pm = np.asarray(physical_landmarks_mm, dtype=float)
    ps = np.asarray(landmarks_xyz, dtype=float).copy()
    ps0 = ps - ps[0]
    rot = roty(rot0y_deg) @ rotz(rot0z_deg)
    ps_aligned = apply_rotation(ps0, rot)
    offset = pm[0] - ps_aligned[0]
    ps_shifted = ps_aligned + offset
    mesh = register_mesh_points(
        mesh_points,
        ps,
        pm,
        rot0y_deg=rot0y_deg,
        rot0z_deg=rot0z_deg,
    )
    calibration = scan2phys(ps_shifted, ps_shifted, pm)
    if machine_frame:
        fk_kw = {"a_mm": a_mm, "d_mm": d_mm, "calgap_z_mm": calgap_z_mm}
        mesh = registration_to_machine_frame(mesh, **fk_kw)
        calibration = registration_to_machine_frame(calibration, **fk_kw)
    return SubjectRegistration(
        mesh_points=mesh,
        mesh_faces=np.asarray(mesh_faces).copy(),
        pm=pm.copy(),
        calibration_registered=calibration,
        rot0y_deg=rot0y_deg,
        rot0z_deg=rot0z_deg,
        a_mm=float(a_mm),
        d_mm=float(d_mm),
        calgap_z_mm=float(calgap_z_mm),
    )
