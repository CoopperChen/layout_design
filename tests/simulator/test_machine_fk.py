"""MATLAB gcodeConverter kinematics conventions."""

from __future__ import annotations

import numpy as np

from app import paths
from app.postprocess.gcode.config_loader import load_machine_config
from app.postprocess.gcode.kinematics.machine_fk import (
    arm_offset_xy_matlab,
    c_center_from_tip,
    machine_to_registration_frame,
    machine_zero_head_frame,
    registration_to_machine_frame,
    structural_arm_joints,
    structural_arm_offset,
)


def test_structural_tip_at_b0_c0():
    center = np.array([10.0, 20.0, 30.0])
    _c, _b, tip = structural_arm_joints(center, b_deg=0.0, c_deg=0.0, a_mm=180.7, d_mm=57.59)
    np.testing.assert_allclose(tip, center + [180.7, 0.0, -57.59], atol=1e-6)


def test_structural_arm_offset_reference():
    np.testing.assert_allclose(structural_arm_offset(0.0, 180.7), [180.7, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(
        structural_arm_offset(90.0, 180.7), [0.0, -180.7, 0.0], atol=1e-4
    )


def test_structural_tip_at_c0_b0_reference():
    """C=0, B=0 → arm +X, tool −Z."""
    center = np.zeros(3)
    _c, b_pivot, tip = structural_arm_joints(center, 0.0, 0.0, 180.7, 57.59)
    np.testing.assert_allclose(b_pivot, [180.7, 0.0, 0.0], atol=1e-4)
    np.testing.assert_allclose(tip, [180.7, 0.0, -57.59], atol=1e-4)
    arm = b_pivot - center
    tool = tip - b_pivot
    np.testing.assert_allclose(arm / np.linalg.norm(arm), [1.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(tool / np.linalg.norm(tool), [0.0, 0.0, -1.0], atol=1e-6)


def test_structural_tip_at_c0_b90_reference():
    """C=0, B=90 → arm +X, tool −Y."""
    center = np.zeros(3)
    _c, b_pivot, tip = structural_arm_joints(center, 90.0, 0.0, 180.7, 57.59)
    np.testing.assert_allclose(b_pivot, [180.7, 0.0, 0.0], atol=1e-4)
    np.testing.assert_allclose(tip, [180.7, -57.59, 0.0], atol=1e-4)
    arm = b_pivot - center
    tool = tip - b_pivot
    np.testing.assert_allclose(arm / np.linalg.norm(arm), [1.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(tool / np.linalg.norm(tool), [0.0, -1.0, 0.0], atol=1e-6)


def test_structural_tip_at_c90_b0_matlab():
    """C=90, B=0 → arm −Y, tool −Z (machine reference / c0,b0)."""
    center = np.array([0.0, 0.0, 0.0])
    _c, b_pivot, tip = structural_arm_joints(center, 0.0, 90.0, 180.7, 57.59)
    np.testing.assert_allclose(b_pivot, [0.0, -180.7, 0.0], atol=1e-4)
    np.testing.assert_allclose(tip, [0.0, -180.7, -57.59], atol=1e-4)
    arm = b_pivot - center
    tool = tip - b_pivot
    np.testing.assert_allclose(arm / np.linalg.norm(arm), [0.0, -1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(tool / np.linalg.norm(tool), [0.0, 0.0, -1.0], atol=1e-6)


def test_arm_offset_xy_matlab():
    """Postprocessor sin/cos arm term (distinct from rigid FK arm direction)."""
    np.testing.assert_allclose(
        arm_offset_xy_matlab(0.0, 180.7), [-180.7, 0.0, 0.0], atol=1e-6
    )
    np.testing.assert_allclose(
        arm_offset_xy_matlab(90.0, 180.7), [0.0, 180.7, 0.0], atol=1e-4
    )


def test_structural_arm_joints_b0_c0():
    center = np.array([10.0, 20.0, 30.0])
    c, b, tip = structural_arm_joints(center, 0.0, 0.0, 180.7, 57.59)
    np.testing.assert_allclose(c, center)
    np.testing.assert_allclose(b, center + [180.7, 0, 0])
    np.testing.assert_allclose(tip, center + [180.7, 0, -57.59])
    assert abs(np.linalg.norm(b - c) - 180.7) < 1e-6
    assert abs(np.linalg.norm(tip - b) - 57.59) < 1e-6


def test_b_rotates_about_arm_not_arm_direction():
    """B spins tool about C–B axis; B pivot unchanged, tip moves."""
    center = np.zeros(3)
    a, d = 180.7, 57.59
    _c0, b0, tip0 = structural_arm_joints(center, b_deg=0.0, c_deg=0.0, a_mm=a, d_mm=d)
    _c45, b45, tip45 = structural_arm_joints(center, b_deg=45.0, c_deg=0.0, a_mm=a, d_mm=d)
    np.testing.assert_allclose(b0, b45)
    assert not np.allclose(tip0, tip45)
    arm = b0 - center
    tool0 = tip0 - b0
    tool45 = tip45 - b45
    assert abs(np.dot(arm, tool0)) < 1e-6
    assert abs(np.dot(arm, tool45)) < 1e-6
    assert abs(np.linalg.norm(tool0) - d) < 1e-6
    assert abs(np.linalg.norm(tool45) - d) < 1e-6


def test_structural_arm_segment_lengths_constant():
    center = np.array([0.0, 0.0, 0.0])
    a, d = 180.7, 57.59
    for b, c in ((0.0, 0.0), (45.0, 30.0), (90.0, -60.0), (-30.0, 120.0)):
        c_pt, b_pt, tip = structural_arm_joints(center, b, c, a, d)
        assert abs(np.linalg.norm(b_pt - c_pt) - a) < 1e-4
        assert abs(np.linalg.norm(tip - b_pt) - d) < 1e-4
        arm = b_pt - c_pt
        tool = tip - b_pt
        assert abs(np.dot(arm, tool)) < 1e-3


def test_machine_zero_head_frame():
    machine = load_machine_config(paths.postprocessor_machine_config())
    central, tip_mz, c_mz, b_mz = machine_zero_head_frame(
        a_mm=machine.a_mm,
        d_mm=machine.d_mm,
        b0_deg=machine.b0_deg,
        c0_deg=machine.c0_deg,
        calgap_z_mm=machine.calgap_z_mm,
    )
    np.testing.assert_allclose(c_mz, [0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(
        tip_mz, [0.0, -machine.a_mm, -machine.d_mm], atol=1e-6
    )
    np.testing.assert_allclose(
        central,
        [0.0, -machine.a_mm, -(machine.d_mm + machine.calgap_z_mm)],
        atol=1e-6,
    )
    np.testing.assert_allclose(b_mz, [0.0, -machine.a_mm, 0.0], atol=1e-4)


def test_registration_machine_frame_roundtrip():
    machine = load_machine_config(paths.postprocessor_machine_config())
    fk_kw = {
        "a_mm": machine.a_mm,
        "d_mm": machine.d_mm,
        "calgap_z_mm": machine.calgap_z_mm,
    }
    pt = np.array([1.0, 2.0, 3.0])
    back = machine_to_registration_frame(
        registration_to_machine_frame(pt, **fk_kw), **fk_kw
    )
    np.testing.assert_allclose(back, pt, atol=1e-9)


def test_machine_zero_in_registration_frame():
    machine = load_machine_config(paths.postprocessor_machine_config())
    central, tip_mz, c_mz, _b_mz = machine_zero_head_frame(
        a_mm=machine.a_mm,
        d_mm=machine.d_mm,
        b0_deg=machine.b0_deg,
        c0_deg=machine.c0_deg,
        calgap_z_mm=machine.calgap_z_mm,
    )
    np.testing.assert_allclose(
        machine_to_registration_frame(
            central,
            a_mm=machine.a_mm,
            d_mm=machine.d_mm,
            calgap_z_mm=machine.calgap_z_mm,
        ),
        [0.0, 0.0, 0.0],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        machine_to_registration_frame(
            tip_mz,
            a_mm=machine.a_mm,
            d_mm=machine.d_mm,
            calgap_z_mm=machine.calgap_z_mm,
        ),
        [0.0, 0.0, machine.calgap_z_mm],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        machine_to_registration_frame(
            c_mz,
            a_mm=machine.a_mm,
            d_mm=machine.d_mm,
            calgap_z_mm=machine.calgap_z_mm,
        ),
        [0.0, machine.a_mm, machine.d_mm + machine.calgap_z_mm],
        atol=1e-4,
    )


def test_c_pivot_when_tip_on_central_landmark():
    """Tip on central landmark — C pivot reads (0, 0, −calgap) in machine frame."""
    machine = load_machine_config(paths.postprocessor_machine_config())
    _central, _tip_mz, _c_mz, _b_mz = machine_zero_head_frame(
        a_mm=machine.a_mm,
        d_mm=machine.d_mm,
        b0_deg=machine.b0_deg,
        c0_deg=machine.c0_deg,
        calgap_z_mm=machine.calgap_z_mm,
    )
    c = c_center_from_tip(
        _central,
        machine.b0_deg,
        machine.c0_deg,
        machine.a_mm,
        machine.d_mm,
    )
    np.testing.assert_allclose(c, [0.0, 0.0, -machine.calgap_z_mm], atol=1e-6)
    _, _, tip = structural_arm_joints(
        c, machine.b0_deg, machine.c0_deg, machine.a_mm, machine.d_mm
    )
    np.testing.assert_allclose(tip, _central, atol=1e-6)
