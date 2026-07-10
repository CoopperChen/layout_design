"""Trace print-order planning to shorten inter-wire air travel."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.config_loader import load_defaults
from app.postprocess.gcode.kinematics.arm_clearance import HeadMeshInsideChecker
from app.postprocess.gcode.kinematics.engage_clearance import (
    EngageClearanceConfig,
    compute_engage_xy_offset,
    head_center_xy_from_mesh,
    load_engage_clearance_config,
)
from app.postprocess.gcode.models import MachineConfig


@dataclass(frozen=True)
class TraceOrderConfig:
    enabled: bool = False
    method: str = "nearest_neighbor"  # nearest_neighbor | c_nearest_neighbor
    start: str = "first"  # first | home
    skip_origin_when_bc_close: bool = True
    c_short_transfer_max_delta_deg: float = 20.0
    c_order_xy_weight: float = 0.05  # tie-break / blend for c_nearest_neighbor


@dataclass(frozen=True)
class TraceOrderPlan:
    order: list[int]
    flip: list[bool]
    skip_origin: list[bool]  # len n-1 between consecutive merged traces


def load_trace_order_config() -> TraceOrderConfig:
    pp = load_defaults().get("postprocess", {})
    method = str(pp.get("trace_order_method", "nearest_neighbor"))
    if method == "bc_nearest_neighbor":
        method = "c_nearest_neighbor"
    c_threshold = pp.get("c_short_transfer_max_delta_deg")
    if c_threshold is None:
        c_threshold = pp.get("bc_short_transfer_max_delta_deg", 20.0)
    c_xy_weight = pp.get("c_order_xy_weight")
    if c_xy_weight is None:
        c_xy_weight = pp.get("bc_order_xy_weight", 0.05)
    return TraceOrderConfig(
        enabled=bool(pp.get("optimize_trace_order", False)),
        method=method,
        start=str(pp.get("trace_order_start", "first")),
        skip_origin_when_bc_close=bool(pp.get("skip_origin_when_bc_close", True)),
        c_short_transfer_max_delta_deg=float(c_threshold),
        c_order_xy_weight=float(c_xy_weight),
    )


def _engage_row(rows: np.ndarray, flipped: bool) -> np.ndarray:
    return rows[-1] if flipped else rows[0]


def _exit_row(rows: np.ndarray, flipped: bool) -> np.ndarray:
    return rows[0] if flipped else rows[-1]


def _exit_xyz(rows: np.ndarray, flipped: bool) -> np.ndarray:
    return np.asarray(_exit_row(rows, flipped)[:3], dtype=float)


def _home_xyz(machine: MachineConfig, zsafe: float) -> np.ndarray:
    return np.array([0.0, -float(machine.a_mm), float(zsafe)], dtype=float)


def _c_axis_delta_deg(c1: float, c2: float) -> float:
    delta = abs(float(c1) - float(c2)) % 360.0
    return min(delta, 360.0 - delta)


def c_delta_deg(row_a: np.ndarray, row_b: np.ndarray) -> float:
    """Shortest |ΔC| between two gcode rows (B ignored for collision policy)."""
    return _c_axis_delta_deg(row_a[4], row_b[4])


def bc_delta_deg(row_a: np.ndarray, row_b: np.ndarray) -> float:
    """Sum of |ΔB| and shortest |ΔC| between two gcode rows."""
    return abs(float(row_a[3]) - float(row_b[3])) + c_delta_deg(row_a, row_b)


def skip_origin_between_traces(
    ordered_traces: list[np.ndarray],
    max_delta_deg: float,
) -> list[bool]:
    """After flips applied, engage is row 0 and exit is row -1 for each trace."""
    if len(ordered_traces) <= 1:
        return []
    flags: list[bool] = []
    for i in range(len(ordered_traces) - 1):
        delta = c_delta_deg(ordered_traces[i][-1], ordered_traces[i + 1][0])
        flags.append(delta <= float(max_delta_deg))
    return flags


def short_transition_cost_mm(
    from_xyz: np.ndarray,
    engage_row: np.ndarray,
    machine: MachineConfig,
    zsafe: float,
    *,
    head_center_xy: np.ndarray,
    checker: HeadMeshInsideChecker | None,
    engage_config: EngageClearanceConfig,
) -> float:
    """Z_safe retract then direct XY at Z_safe to next engage (no origin detour)."""
    x1, y1, z1 = np.asarray(from_xyz, dtype=float).reshape(3)
    engage = np.asarray(engage_row[:6], dtype=float)
    x2, y2, z2 = engage[0], engage[1], engage[2]
    offset_xy = compute_engage_xy_offset(
        engage,
        zsafe,
        head_center_xy,
        checker,
        machine,
        engage_config,
    )
    dx, dy = float(offset_xy[0]), float(offset_xy[1])
    ox, oy = x2 + dx, y2 + dy
    cost = abs(float(zsafe) - z1)
    cost += float(np.hypot(ox - x1, oy - y1))
    cost += abs(z2 - float(zsafe))
    cost += float(np.linalg.norm(offset_xy))
    return cost


def transition_cost_mm(
    from_xyz: np.ndarray,
    engage_row: np.ndarray,
    machine: MachineConfig,
    zsafe: float,
    *,
    head_center_xy: np.ndarray,
    checker: HeadMeshInsideChecker | None,
    engage_config: EngageClearanceConfig,
    skip_origin: bool = False,
) -> float:
    if skip_origin:
        return short_transition_cost_mm(
            from_xyz,
            engage_row,
            machine,
            zsafe,
            head_center_xy=head_center_xy,
            checker=checker,
            engage_config=engage_config,
        )
    x1, y1, z1 = np.asarray(from_xyz, dtype=float).reshape(3)
    engage = np.asarray(engage_row[:6], dtype=float)
    x2, y2, z2 = engage[0], engage[1], engage[2]
    a = float(machine.a_mm)

    offset_xy = compute_engage_xy_offset(
        engage,
        zsafe,
        head_center_xy,
        checker,
        machine,
        engage_config,
    )
    dx, dy = float(offset_xy[0]), float(offset_xy[1])
    ox, oy = x2 + dx, y2 + dy

    cost = abs(float(zsafe) - z1)
    cost += float(np.hypot(x1 - 0.0, y1 - (-a)))
    cost += float(np.hypot(ox - 0.0, oy - (-a)))
    if float(np.linalg.norm(offset_xy)) > 1e-9:
        cost += abs(z2 - float(zsafe))
        cost += float(np.hypot(dx, dy))
    else:
        cost += abs(z2 - float(zsafe))
    return cost


def first_trace_cost_mm(
    engage_row: np.ndarray,
    machine: MachineConfig,
    zsafe: float,
    *,
    head_center_xy: np.ndarray,
    checker: HeadMeshInsideChecker | None,
    engage_config: EngageClearanceConfig,
) -> float:
    return transition_cost_mm(
        _home_xyz(machine, zsafe),
        engage_row,
        machine,
        zsafe,
        head_center_xy=head_center_xy,
        checker=checker,
        engage_config=engage_config,
        skip_origin=False,
    )


def _best_flip_for_engage(
    rows: np.ndarray,
    machine: MachineConfig,
    zsafe: float,
    *,
    from_exit_row: np.ndarray | None,
    head_center_xy: np.ndarray,
    checker: HeadMeshInsideChecker | None,
    engage_config: EngageClearanceConfig,
    is_first: bool,
    cost_fn,
) -> tuple[bool, float]:
    best_flip = False
    best_cost = float("inf")
    for flipped in (False, True):
        engage = _engage_row(rows, flipped)
        if is_first:
            cost = first_trace_cost_mm(
                engage,
                machine,
                zsafe,
                head_center_xy=head_center_xy,
                checker=checker,
                engage_config=engage_config,
            )
        else:
            assert from_exit_row is not None
            cost = cost_fn(from_exit_row, engage)
        if cost < best_cost:
            best_cost = cost
            best_flip = flipped
    return best_flip, best_cost


def plan_trace_order_nearest_neighbor(
    gcode_list: list[np.ndarray],
    machine: MachineConfig,
    zsafe: float,
    *,
    head_center_xy: np.ndarray,
    checker: HeadMeshInsideChecker | None,
    engage_config: EngageClearanceConfig,
    start: str = "first",
) -> tuple[list[int], list[bool]]:
    n = len(gcode_list)
    if n <= 1:
        return list(range(n)), [False] * n

    def _travel_cost(exit_row: np.ndarray, engage_row: np.ndarray) -> float:
        return transition_cost_mm(
            exit_row[:3],
            engage_row,
            machine,
            zsafe,
            head_center_xy=head_center_xy,
            checker=checker,
            engage_config=engage_config,
            skip_origin=False,
        )

    order: list[int] = []
    flips: list[bool] = []
    remaining = set(range(n))

    if start == "first":
        first_idx = 0
        flip0, _ = _best_flip_for_engage(
            gcode_list[first_idx],
            machine,
            zsafe,
            from_exit_row=None,
            head_center_xy=head_center_xy,
            checker=checker,
            engage_config=engage_config,
            is_first=True,
            cost_fn=_travel_cost,
        )
        order.append(first_idx)
        flips.append(flip0)
        remaining.remove(first_idx)
        current_exit = _exit_row(gcode_list[first_idx], flip0)
    else:
        best_i = 0
        best_f = False
        best_cost = float("inf")
        for i in remaining:
            flip_i, cost_i = _best_flip_for_engage(
                gcode_list[i],
                machine,
                zsafe,
                from_exit_row=None,
                head_center_xy=head_center_xy,
                checker=checker,
                engage_config=engage_config,
                is_first=True,
                cost_fn=_travel_cost,
            )
            if cost_i < best_cost:
                best_cost = cost_i
                best_i = i
                best_f = flip_i
        order.append(best_i)
        flips.append(best_f)
        remaining.remove(best_i)
        current_exit = _exit_row(gcode_list[best_i], best_f)

    while remaining:
        best_j = -1
        best_f = False
        best_cost = float("inf")
        for j in remaining:
            for flipped in (False, True):
                engage = _engage_row(gcode_list[j], flipped)
                cost = _travel_cost(current_exit, engage)
                if cost < best_cost:
                    best_cost = cost
                    best_j = j
                    best_f = flipped
        if best_j < 0:
            break
        order.append(best_j)
        flips.append(best_f)
        remaining.remove(best_j)
        current_exit = _exit_row(gcode_list[best_j], best_f)

    return order, flips


def plan_trace_order_c_nearest_neighbor(
    gcode_list: list[np.ndarray],
    machine: MachineConfig,
    zsafe: float,
    *,
    head_center_xy: np.ndarray,
    checker: HeadMeshInsideChecker | None,
    engage_config: EngageClearanceConfig,
    start: str = "first",
    xy_weight: float = 0.05,
    skip_origin_max_delta_deg: float = 20.0,
) -> tuple[list[int], list[bool]]:
    """Order traces so consecutive C-axis retargeting is minimized."""
    n = len(gcode_list)
    if n <= 1:
        return list(range(n)), [False] * n

    threshold = float(skip_origin_max_delta_deg)

    def _c_cost(exit_row: np.ndarray, engage_row: np.ndarray) -> float:
        dc = c_delta_deg(exit_row, engage_row)
        cost = transition_cost_mm(
            exit_row[:3],
            engage_row,
            machine,
            zsafe,
            head_center_xy=head_center_xy,
            checker=checker,
            engage_config=engage_config,
            skip_origin=dc <= threshold,
        )
        return cost + 0.01 * dc

    order: list[int] = []
    flips: list[bool] = []
    remaining = set(range(n))

    if start == "first":
        first_idx = 0
        flip0, _ = _best_flip_for_engage(
            gcode_list[first_idx],
            machine,
            zsafe,
            from_exit_row=None,
            head_center_xy=head_center_xy,
            checker=checker,
            engage_config=engage_config,
            is_first=True,
            cost_fn=lambda _e, _g: 0.0,
        )
        order.append(first_idx)
        flips.append(flip0)
        remaining.remove(first_idx)
        current_exit = _exit_row(gcode_list[first_idx], flip0)
    else:
        best_i = 0
        best_f = False
        best_cost = float("inf")
        for i in remaining:
            for flipped in (False, True):
                engage = _engage_row(gcode_list[i], flipped)
                cost = c_delta_deg(
                    np.array([0.0, 0.0, zsafe, machine.b0_deg, machine.c0_deg, 0.0]),
                    engage,
                )
                if cost < best_cost:
                    best_cost = cost
                    best_i = i
                    best_f = flipped
        order.append(best_i)
        flips.append(best_f)
        remaining.remove(best_i)
        current_exit = _exit_row(gcode_list[best_i], best_f)

    while remaining:
        best_j = -1
        best_f = False
        best_cost = float("inf")
        for j in remaining:
            for flipped in (False, True):
                engage = _engage_row(gcode_list[j], flipped)
                cost = _c_cost(current_exit, engage)
                if cost < best_cost:
                    best_cost = cost
                    best_j = j
                    best_f = flipped
        if best_j < 0:
            break
        order.append(best_j)
        flips.append(best_f)
        remaining.remove(best_j)
        current_exit = _exit_row(gcode_list[best_j], best_f)

    return order, flips


def plan_trace_order_bc_nearest_neighbor(
    *args,
    **kwargs,
) -> tuple[list[int], list[bool]]:
    """Deprecated alias for :func:`plan_trace_order_c_nearest_neighbor`."""
    return plan_trace_order_c_nearest_neighbor(*args, **kwargs)


def plan_trace_order(
    gcode_list: list[np.ndarray],
    machine: MachineConfig,
    zsafe: float,
    *,
    mesh_points: np.ndarray | None,
    mesh_faces: np.ndarray | None,
    config: TraceOrderConfig,
) -> TraceOrderPlan:
    n = len(gcode_list)
    if not config.enabled or n <= 1:
        return TraceOrderPlan(
            order=list(range(n)),
            flip=[False] * n,
            skip_origin=[False] * max(0, n - 1),
        )

    head_center_xy = head_center_xy_from_mesh(mesh_points)
    engage_config = load_engage_clearance_config(machine)
    checker: HeadMeshInsideChecker | None = None
    if mesh_points is not None and mesh_faces is not None:
        checker = HeadMeshInsideChecker(mesh_points, mesh_faces)

    if config.method == "nearest_neighbor":
        order, flip = plan_trace_order_nearest_neighbor(
            gcode_list,
            machine,
            zsafe,
            head_center_xy=head_center_xy,
            checker=checker,
            engage_config=engage_config,
            start=config.start,
        )
    elif config.method in ("c_nearest_neighbor", "bc_nearest_neighbor"):
        order, flip = plan_trace_order_c_nearest_neighbor(
            gcode_list,
            machine,
            zsafe,
            head_center_xy=head_center_xy,
            checker=checker,
            engage_config=engage_config,
            start=config.start,
            xy_weight=config.c_order_xy_weight,
            skip_origin_max_delta_deg=config.c_short_transfer_max_delta_deg,
        )
    else:
        raise ValueError(f"Unsupported trace_order_method: {config.method}")

    ordered = apply_trace_order(gcode_list, order, flip)
    skip_origin: list[bool] = []
    if config.skip_origin_when_bc_close:
        skip_origin = skip_origin_between_traces(
            ordered, config.c_short_transfer_max_delta_deg
        )
    else:
        skip_origin = [False] * max(0, n - 1)

    return TraceOrderPlan(order=order, flip=flip, skip_origin=skip_origin)


def apply_trace_order(
    gcode_list: list[np.ndarray],
    order: list[int],
    flip: list[bool],
) -> list[np.ndarray]:
    if len(order) != len(gcode_list) or len(flip) != len(gcode_list):
        raise ValueError("order and flip must match gcode_list length")
    out: list[np.ndarray] = []
    for idx, flipped in zip(order, flip):
        rows = gcode_list[idx]
        if flipped:
            rows = np.flipud(rows)
        out.append(rows)
    return out


def legacy_alternate_flip_flags(n: int) -> list[bool]:
    return [(i + 1) % 2 == 0 for i in range(n)]
