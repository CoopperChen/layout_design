"""Flip trace normals when rigid arm joints fall inside the head mesh."""

from __future__ import annotations

import numpy as np
import open3d as o3d

from app.postprocess.gcode.kinematics.axis_angles import compute_axis_angles
from app.postprocess.gcode.kinematics.machine_fk import structural_arm_joints
from app.postprocess.gcode.kinematics.machine_zero import apply_machine_zero_offset
from app.postprocess.gcode.kinematics.tool_offset import apply_tool_offset
from app.postprocess.gcode.models import MachineConfig


class HeadMeshInsideChecker:
    """Signed-distance inside test for registered head mesh in machine frame."""

    def __init__(
        self,
        mesh_points: np.ndarray,
        mesh_faces: np.ndarray,
        *,
        inside_margin_mm: float = 0.0,
    ) -> None:
        pts = np.asarray(mesh_points, dtype=np.float64)
        faces = np.asarray(mesh_faces, dtype=np.int32)
        legacy = o3d.geometry.TriangleMesh()
        legacy.vertices = o3d.utility.Vector3dVector(pts)
        legacy.triangles = o3d.utility.Vector3iVector(faces)
        legacy.compute_vertex_normals()
        tmesh = o3d.t.geometry.TriangleMesh.from_legacy(legacy)
        self._scene = o3d.t.geometry.RaycastingScene()
        self._scene.add_triangles(tmesh)
        self._inside_margin_mm = float(inside_margin_mm)

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim == 1:
            pts = pts.reshape(1, 3)
        query = o3d.core.Tensor(pts, dtype=o3d.core.Dtype.Float32)
        return self._scene.compute_signed_distance(query).numpy()

    def is_inside(self, point: np.ndarray) -> bool:
        dist = float(self.signed_distance(point)[0])
        return dist < -self._inside_margin_mm

    def arm_is_inside(self, c_pivot: np.ndarray, b_pivot: np.ndarray, *, samples: int = 5) -> bool:
        c_pivot = np.asarray(c_pivot, dtype=float).reshape(3)
        b_pivot = np.asarray(b_pivot, dtype=float).reshape(3)
        if self.is_inside(c_pivot) or self.is_inside(b_pivot):
            return True
        if samples <= 2:
            return False
        ts = np.linspace(0.0, 1.0, samples)
        seg = c_pivot + (b_pivot - c_pivot) * ts[:, np.newaxis]
        return bool(np.any(self.signed_distance(seg) < -self._inside_margin_mm))


def _unit_normal(normal: np.ndarray) -> np.ndarray:
    n = np.asarray(normal, dtype=float).reshape(3)
    length = np.linalg.norm(n)
    if length <= 1e-12:
        return n
    return n / length


def _gcode_joints_for_normal(
    scalp_xyz: np.ndarray,
    normal: np.ndarray,
    machine: MachineConfig,
    *,
    coords_include_gap: bool,
) -> tuple[np.ndarray, float, float, np.ndarray, np.ndarray]:
    n = _unit_normal(normal)
    g = apply_machine_zero_offset(np.asarray(scalp_xyz, dtype=float).reshape(1, 3), machine)
    b_angles, c_angles = compute_axis_angles(n.reshape(1, 3))
    b = float(b_angles[0])
    c = float(c_angles[0])
    offset_gap = 0.0 if coords_include_gap else None
    g = apply_tool_offset(g, n.reshape(1, 3), c_angles, machine, gap_mm=offset_gap)
    c_pivot = g[0]
    _c, b_pivot, _tip = structural_arm_joints(
        c_pivot, b, c, machine.a_mm, machine.d_mm
    )
    return n, b, c, c_pivot, b_pivot


def resolve_normal_arm_clearance(
    scalp_xyz: np.ndarray,
    normal: np.ndarray,
    machine: MachineConfig,
    checker: HeadMeshInsideChecker,
    *,
    coords_include_gap: bool = False,
) -> np.ndarray:
    """
    Use the synthesized trace normal; flip sign only if the C–B arm lies inside mesh.

    Picks whichever of ``n`` and ``-n`` keeps the arm outside; prefers the trace
    direction when both are clear.
    """
    candidates = [_unit_normal(normal), -_unit_normal(normal)]
    poses: list[tuple[np.ndarray, bool]] = []
    for cand in candidates:
        _n, _b, _c, c_pivot, b_pivot = _gcode_joints_for_normal(
            scalp_xyz, cand, machine, coords_include_gap=coords_include_gap
        )
        poses.append((cand, checker.arm_is_inside(c_pivot, b_pivot, samples=9)))

    if not poses[0][1]:
        return poses[0][0]
    if not poses[1][1]:
        return poses[1][0]
    return poses[0][0]
