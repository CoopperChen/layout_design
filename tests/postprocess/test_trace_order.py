"""Trace print-order planning for shorter inter-wire air travel."""

from __future__ import annotations

import numpy as np
import pytest

from app import paths
from app.postprocess.bundle.load import load_bundle
from app.postprocess.gcode.config_loader import load_machine_config
from app.postprocess.gcode.kinematics.arm_clearance import HeadMeshInsideChecker
from app.postprocess.gcode.kinematics.engage_clearance import load_engage_clearance_config
from app.postprocess.gcode.kinematics.machine_fk import registration_to_machine_frame
from app.postprocess.gcode.models import JobConfig
from app.postprocess.gcode.pipeline.align import align_subject
from app.postprocess.gcode.pipeline.merge_traces import merge_traces
from app.postprocess.gcode.pipeline.process_traces import process_all_traces
from app.postprocess.gcode.pipeline.trace_order import (
    TraceOrderConfig,
    apply_trace_order,
    c_delta_deg,
    plan_trace_order,
    plan_trace_order_c_nearest_neighbor,
    skip_origin_between_traces,
    transition_cost_mm,
)
from app.postprocess.print_config import load_physical_landmarks


def _safety_transition_counts(merged: np.ndarray, zsafe: float, a_mm: float) -> dict[str, int]:
    return {
        "zsafe": int(np.sum(np.isclose(merged[:, 2], zsafe))),
        "origin_xy": int(
            np.sum(np.isclose(merged[:, 0], 0.0) & np.isclose(merged[:, 1], -a_mm))
        ),
        "m11": int(np.sum(merged[:, 6] == 11)),
    }


def test_plan_trace_order_disabled_returns_identity():
    gcode_list = [np.zeros((3, 7)), np.ones((3, 7))]
    machine = load_machine_config(paths.postprocessor_machine_config())
    plan = plan_trace_order(
        gcode_list,
        machine,
        50.0,
        mesh_points=None,
        mesh_faces=None,
        config=TraceOrderConfig(enabled=False),
    )
    assert plan.order == [0, 1]
    assert plan.flip == [False, False]
    assert plan.skip_origin == [False]


def test_c_ordering_reduces_total_c_delta_on_subject_4():
    bundle_dir = paths.REPO_ROOT / "data/output/bundles/subject_4"
    if not (bundle_dir / "manifest.json").is_file():
        pytest.skip("subject_4 bundle not present")
    pm_path = paths.postprocessor_subject_pm(4)
    if not pm_path.is_file():
        pytest.skip("subject_4 pm config not present")

    bundle = load_bundle(bundle_dir)
    pm = load_physical_landmarks(pm_path)
    machine = load_machine_config(paths.postprocessor_machine_config())
    channels, mesh = align_subject(bundle, JobConfig(physical_landmarks_mm=pm))
    mesh_m = registration_to_machine_frame(
        mesh,
        a_mm=machine.a_mm,
        d_mm=machine.d_mm,
        calgap_z_mm=machine.calgap_z_mm,
    )
    gcode_list = process_all_traces(
        channels, machine, 1, 0, mesh_points=mesh_m, mesh_faces=bundle.mesh_faces
    )
    zsafe = round(float(np.max(mesh[:, 2])) + machine.zsafe_margin_mm)
    checker = HeadMeshInsideChecker(mesh_m, bundle.mesh_faces)
    engage_config = load_engage_clearance_config(machine)
    head_xy = np.mean(mesh_m[:, :2], axis=0)

    def total_c(order: list[int], flips: list[bool]) -> float:
        ordered = apply_trace_order(gcode_list, order, flips)
        total = 0.0
        for i in range(len(ordered) - 1):
            total += c_delta_deg(ordered[i][-1], ordered[i + 1][0])
        return total

    legacy_order = list(range(len(gcode_list)))
    legacy_flips = [(i + 1) % 2 == 0 for i in legacy_order]
    c_order, c_flips = plan_trace_order_c_nearest_neighbor(
        gcode_list,
        machine,
        zsafe,
        head_center_xy=head_xy,
        checker=checker,
        engage_config=engage_config,
        start="first",
    )
    assert total_c(c_order, c_flips) < total_c(legacy_order, legacy_flips)


def test_skip_origin_flags_use_c_only_not_b():
    rows_a = np.array([[0, 0, 0, 10.0, 5.0, 500, 0], [1, 1, 1, 12.0, 8.0, 500, 0]])
    rows_b = np.array([[2, 2, 2, 90.0, 6.0, 500, 0], [3, 3, 3, 90.0, 40.0, 500, 0]])
    # dB=78, dC=2 — C-only policy should allow skip at 20 deg threshold.
    assert skip_origin_between_traces([rows_a, rows_b], max_delta_deg=20.0) == [True]
    assert skip_origin_between_traces([rows_a, rows_b], max_delta_deg=1.0) == [False]


def test_skip_origin_flags_follow_c_threshold():
    rows_a = np.array([[0, 0, 0, 10.0, 5.0, 500, 0], [1, 1, 1, 12.0, 8.0, 500, 0]])
    rows_b = np.array([[2, 2, 2, 11.0, 6.0, 500, 0], [3, 3, 3, 90.0, 40.0, 500, 0]])
    close = skip_origin_between_traces([rows_a, rows_b], max_delta_deg=20.0)
    far = skip_origin_between_traces([rows_a, rows_b], max_delta_deg=1.0)
    assert close == [True]
    assert far == [False]


def test_bc_plan_skips_origin_in_merge_when_close():
    bundle_dir = paths.REPO_ROOT / "data/output/bundles/subject_4"
    if not (bundle_dir / "manifest.json").is_file():
        pytest.skip("subject_4 bundle not present")
    pm_path = paths.postprocessor_subject_pm(4)
    if not pm_path.is_file():
        pytest.skip("subject_4 pm config not present")

    bundle = load_bundle(bundle_dir)
    pm = load_physical_landmarks(pm_path)
    machine = load_machine_config(paths.postprocessor_machine_config())
    channels, mesh = align_subject(bundle, JobConfig(physical_landmarks_mm=pm))
    mesh_m = registration_to_machine_frame(
        mesh,
        a_mm=machine.a_mm,
        d_mm=machine.d_mm,
        calgap_z_mm=machine.calgap_z_mm,
    )
    gcode_list = process_all_traces(
        channels, machine, 1, 0, mesh_points=mesh_m, mesh_faces=bundle.mesh_faces
    )
    zsafe = round(float(np.max(mesh[:, 2])) + machine.zsafe_margin_mm)

    plan = plan_trace_order(
        gcode_list,
        machine,
        zsafe,
        mesh_points=mesh_m,
        mesh_faces=bundle.mesh_faces,
        config=TraceOrderConfig(
            enabled=True,
            method="c_nearest_neighbor",
            skip_origin_when_bc_close=True,
            c_short_transfer_max_delta_deg=20.0,
        ),
    )
    ordered = apply_trace_order(gcode_list, plan.order, plan.flip)
    merged_skip = merge_traces(
        ordered,
        float(np.max(mesh[:, 2])),
        machine,
        0,
        mesh_points=mesh_m,
        mesh_faces=bundle.mesh_faces,
        alternate_flip=False,
        skip_origin_between=plan.skip_origin,
    )
    merged_full = merge_traces(
        ordered,
        float(np.max(mesh[:, 2])),
        machine,
        0,
        mesh_points=mesh_m,
        mesh_faces=bundle.mesh_faces,
        alternate_flip=False,
        skip_origin_between=[False] * (len(ordered) - 1),
    )
    counts = _safety_transition_counts(merged_skip, zsafe, machine.a_mm)
    assert counts["zsafe"] >= len(gcode_list)
    assert counts["m11"] >= len(gcode_list)
    assert sum(plan.skip_origin) > 0
    assert merged_skip.shape[0] == merged_full.shape[0] - sum(plan.skip_origin)
    assert _safety_transition_counts(merged_skip, zsafe, machine.a_mm)["origin_xy"] < _safety_transition_counts(
        merged_full, zsafe, machine.a_mm
    )["origin_xy"]


def test_bc_plan_reduces_air_travel_vs_full_origin():
    bundle_dir = paths.REPO_ROOT / "data/output/bundles/subject_4"
    if not (bundle_dir / "manifest.json").is_file():
        pytest.skip("subject_4 bundle not present")
    pm_path = paths.postprocessor_subject_pm(4)
    if not pm_path.is_file():
        pytest.skip("subject_4 pm config not present")

    bundle = load_bundle(bundle_dir)
    pm = load_physical_landmarks(pm_path)
    machine = load_machine_config(paths.postprocessor_machine_config())
    channels, mesh = align_subject(bundle, JobConfig(physical_landmarks_mm=pm))
    mesh_m = registration_to_machine_frame(
        mesh,
        a_mm=machine.a_mm,
        d_mm=machine.d_mm,
        calgap_z_mm=machine.calgap_z_mm,
    )
    gcode_list = process_all_traces(
        channels, machine, 1, 0, mesh_points=mesh_m, mesh_faces=bundle.mesh_faces
    )
    zsafe = round(float(np.max(mesh[:, 2])) + machine.zsafe_margin_mm)
    checker = HeadMeshInsideChecker(mesh_m, bundle.mesh_faces)
    engage_config = load_engage_clearance_config(machine)
    head_xy = np.mean(mesh_m[:, :2], axis=0)

    def hop_cost(ordered: list[np.ndarray], skip_flags: list[bool]) -> float:
        total = 0.0
        for i in range(len(ordered) - 1):
            total += transition_cost_mm(
                ordered[i][-1, :3],
                ordered[i + 1][0],
                machine,
                zsafe,
                head_center_xy=head_xy,
                checker=checker,
                engage_config=engage_config,
                skip_origin=skip_flags[i],
            )
        return total

    legacy = apply_trace_order(
        gcode_list, list(range(len(gcode_list))), [(i + 1) % 2 == 0 for i in range(len(gcode_list))]
    )
    full = hop_cost(legacy, [False] * (len(legacy) - 1))

    plan = plan_trace_order(
        gcode_list,
        machine,
        zsafe,
        mesh_points=mesh_m,
        mesh_faces=bundle.mesh_faces,
        config=TraceOrderConfig(
            enabled=True,
            method="c_nearest_neighbor",
            skip_origin_when_bc_close=True,
            c_short_transfer_max_delta_deg=20.0,
        ),
    )
    ordered = apply_trace_order(gcode_list, plan.order, plan.flip)
    short = hop_cost(ordered, plan.skip_origin)
    assert short < full
