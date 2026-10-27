"""Tests for planar electrode disk zigzag."""

from __future__ import annotations

import numpy as np

from app.gcode.models import MachineConfig
from app.gcode.pads import (
    DEFAULT_NLINES,
    build_electrode_disk,
    coplanar_residual_mm,
    diameter_mm_from_area_cm2,
    electrode_pad_settings,
    export_electrode_xyzn,
    perimeter_zigzag_uv,
)
from app.gcode.pipeline.process_traces import process_trace
from app.gcode.surface import closest_points_on_surface, normals_at_points


def test_electrode_pad_settings_read_defaults_yaml():
    area_cm2, nlines = electrode_pad_settings()
    assert area_cm2 == 0.5
    assert nlines == 10


def test_electrode_pad_settings_use_config_override(monkeypatch):
    monkeypatch.setattr(
        "app.gcode.pads.load_defaults",
        lambda: {"electrode": {"area_cm2": 1.25, "nlines": 6}},
    )
    area_cm2, nlines = electrode_pad_settings()
    assert area_cm2 == 1.25
    assert nlines == 6
    assert abs(diameter_mm_from_area_cm2(area_cm2) - 2.0 * np.sqrt(1.25 / np.pi) * 10.0) < 1e-9


def test_diameter_matches_adjpoints_formula():
    d = diameter_mm_from_area_cm2(1.5)
    assert abs(d - 2.0 * np.sqrt(1.5 / np.pi) * 10.0) < 1e-9


def test_perimeter_zigzag_point_count_and_alternating_u():
    uv = perimeter_zigzag_uv(6.9, DEFAULT_NLINES)
    assert uv.shape == (DEFAULT_NLINES + 1, 2)
    for row in range(DEFAULT_NLINES + 1):
        expected_sign = 1.0 if row % 2 == 0 else -1.0
        assert np.sign(uv[row, 0]) == expected_sign or uv[row, 0] == 0.0
    radii = np.linalg.norm(uv, axis=1)
    assert np.allclose(radii, 6.9, atol=1e-6)


def test_build_electrode_disk_on_sphere():
    import pyvista as pv

    from app.gcode.surface import prepare_mesh_context

    mesh = pv.Sphere(radius=50.0, theta_resolution=32, phi_resolution=32)
    ctx = prepare_mesh_context(mesh)
    gap = 15.0
    interconnect = np.zeros((1, 6), dtype=np.float64)
    interconnect[0, :3] = [50.0, 0.0, 0.0]
    interconnect[0, 3:6] = [1.0, 0.0, 0.0]

    xyz, origin, surface, normal = build_electrode_disk(
        np.array([50.0, 0.0, 0.0]),
        mesh,
        ctx,
        diameter_mm=6.0,
        gap_size_mm=gap,
        nlines=DEFAULT_NLINES,
    )
    assert coplanar_residual_mm(xyz, origin, normal) < 1e-6
    assert abs(float((origin - surface) @ normal) - gap) < 1e-4
    closest = closest_points_on_surface(mesh, xyz)
    local_n = normals_at_points(ctx, closest)
    signed = np.sum((xyz - closest) * local_n, axis=1)
    assert np.all(signed >= 0.0)


def test_export_electrode_xyzn_constant_normal():
    normal = np.array([0.0, 3.0, 4.0], dtype=np.float64)
    xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    xyzn = export_electrode_xyzn(xyz, normal)
    expected = np.array([0.0, 0.6, 0.8])
    assert np.allclose(xyzn[:, 3:6], expected)
    assert np.allclose(xyzn[:, :3], xyz)


def test_electrode_offset_excludes_machine_gap():
    machine = MachineConfig(
        d_mm=57.59,
        a_mm=180.7,
        gap_size_mm=15.0,
        calgap_z_mm=26.62,
        c0_deg=90.0,
        b0_deg=0.0,
    )
    normal = np.array([0.0, 0.0, 1.0])
    scalp = np.array([[100.0, 0.0, 0.0]])
    gap_pt = np.array([[100.0, 0.0, 15.0]])
    ic_trace = np.hstack([scalp, normal.reshape(1, 3)])
    el_trace = np.hstack([gap_pt, normal.reshape(1, 3)])

    ic_g = process_trace(ic_trace, machine)[:, :3]
    el_g = process_trace(el_trace, machine, coords_include_gap=True)[:, :3]

    assert np.allclose(ic_g, el_g, atol=0.02)


def test_scan2phys_direction_differs_from_point_transform_for_normals():
    from app.gcode.transform.scan2phys import scan2phys, scan2phys_direction

    ps = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 80.0, 0.0]])
    pm = np.array([[10.0, 5.0, 0.0], [110.0, 5.0, 0.0], [10.0, 85.0, 0.0]])
    n = np.array([0.2, 0.1, 0.97])
    n = n / np.linalg.norm(n)
    as_point = scan2phys(n, ps, pm)
    as_dir = scan2phys_direction(n, ps, pm)
    assert not np.allclose(as_point / np.linalg.norm(as_point), as_dir, atol=1e-6)


def test_electrode_normals_outward_after_scan2phys_direction():
    from app.gcode.mesh_normals import orient_electrode_trace_xyzn
    from app.gcode.transform.scan2phys import scan2phys_direction

    head_center = np.array([50.0, 40.0, -20.0])
    ps = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 80.0, 0.0]])
    pm = np.array([[10.0, 5.0, 0.0], [110.0, 5.0, 0.0], [10.0, 85.0, 0.0]])
    n = np.array([0.2, 0.1, 0.97])
    n = n / np.linalg.norm(n)
    xyz = np.array([[50.0, 40.0, 15.0], [51.0, 40.0, 15.0], [50.0, 41.0, 15.0]])
    trace = np.column_stack([xyz, np.tile(n, (3, 1))])
    trace[:, 3:6] = scan2phys_direction(trace[:, 3:6], ps, pm)
    trace = orient_electrode_trace_xyzn(trace, head_center)
    out_n = trace[0, 3:6]
    anchor = np.mean(trace[:, :3], axis=0)
    assert float(np.dot(out_n, anchor - head_center)) > 0
