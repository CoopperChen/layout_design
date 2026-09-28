"""B/C axis branch selection for continuous machine commands."""

from __future__ import annotations

import numpy as np


def c_step_deg(c1: float, c2: float) -> float:
    delta = abs(float(c1) - float(c2)) % 360.0
    return min(delta, 360.0 - delta)


def bc_step_deg(b1: float, c1: float, b2: float, c2: float) -> float:
    return abs(float(b1) - float(b2)) + c_step_deg(c1, c2)


def correct_flip(b_angles: np.ndarray, c_angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Reverse-sign prefix when C-axis jumps across zero with large magnitude.

    Matches MATLAB gcodeConverter_final14 lines 360-386.
    """
    b = b_angles.copy()
    c = c_angles.copy()
    npts = len(c)
    index_flip: list[int] = []

    for k in range(npts - 2, -1, -1):
        if (
            (c[k + 1] > 0 and c[k] < 0) or (c[k + 1] < 0 and c[k] > 0)
        ) and abs(c[k]) > 20 and abs(c[k + 1]) > 20:
            index_flip.append(k)

    nflips = len(index_flip)
    if nflips == 1:
        k = index_flip[0]
        c[: k + 1] = -c[: k + 1]
        b[: k + 1] = -b[: k + 1]
    elif nflips > 1:
        for k in index_flip:
            c[: k + 1] = -c[: k + 1]
            b[: k + 1] = -b[: k + 1]

    return b, c


def enforce_axis_continuity(
    b_angles: np.ndarray,
    c_angles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pick equivalent (B,C) or (-B,-C) at each point to minimize step from previous.

    Nozzle orientation is unchanged; only the commanded branch is selected.
    """
    b = b_angles.copy()
    c = c_angles.copy()
    if len(c) <= 1:
        return b, c

    if bc_step_deg(-b[0], -c[0], b[1], c[1]) < bc_step_deg(b[0], c[0], b[1], c[1]):
        b[0] = -b[0]
        c[0] = -c[0]

    for i in range(1, len(c)):
        direct = (b[i], c[i])
        flipped = (-b[i], -c[i])
        if bc_step_deg(b[i - 1], c[i - 1], *flipped) < bc_step_deg(
            b[i - 1], c[i - 1], *direct
        ):
            b[i], c[i] = flipped
    return b, c


def limit_c_slew(
    b_angles: np.ndarray,
    c_angles: np.ndarray,
    *,
    max_step_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cap consecutive |ΔC| so the printer cannot swing the arm tip per sample.

    Slight tip-orientation lag vs the raw normal is preferred over serpentine
    tracks from crown atan2 jitter. B is left unchanged.
    """
    b = np.asarray(b_angles, dtype=float).copy()
    c = np.asarray(c_angles, dtype=float).copy()
    if len(c) <= 1 or max_step_deg <= 0.0:
        return b, c

    limit = float(max_step_deg)
    for i in range(1, len(c)):
        # Signed shortest-path delta in (-180, 180]
        delta = (float(c[i]) - float(c[i - 1]) + 180.0) % 360.0 - 180.0
        if abs(delta) <= limit:
            continue
        c[i] = float(c[i - 1]) + float(np.sign(delta)) * limit
        # Keep C in a conventional range similar to axis-angle outputs.
        if c[i] > 180.0:
            c[i] -= 360.0
        elif c[i] <= -180.0:
            c[i] += 360.0
    return b, c


def _signed_c_delta(c_from: float, c_to: float) -> float:
    """Shortest signed step from ``c_from`` to ``c_to``, in (-180, 180]."""
    return (float(c_to) - float(c_from) + 180.0) % 360.0 - 180.0


def _wrap_c(angle: float) -> float:
    return (float(angle) + 180.0) % 360.0 - 180.0


def walk_pinned_crown_heading(
    b_angles: np.ndarray,
    c_angles: np.ndarray,
    *,
    max_step_deg: float,
    b_upright_deg: float = 20.0,
    unwind_deg: float = 4.0,
    step_deg: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Walk a pinned crown heading toward the normal in small C steps.

    ``limit_c_slew`` turns one crown heading change into a staircase of
    ``max_step_deg`` moves. A run pinned on that cap, with |B| below
    ``b_upright_deg``, is replaced by steps of at most ``step_deg`` toward
    the heading the trace settles to after the run. The caller rebuilds the
    normal from that C so the pose still lies on the normal.
    """
    b = np.asarray(b_angles, dtype=float).copy()
    c = np.asarray(c_angles, dtype=float).copy()
    original = c.copy()
    catchups: list[int] = []
    if len(c) <= 2 or max_step_deg <= 0.0:
        return b, c, catchups

    limit = float(max_step_deg) - 1e-2
    upright = float(b_upright_deg)
    unwind = float(unwind_deg)
    pinned = np.zeros(len(c), dtype=bool)
    for i in range(1, len(c)):
        if c_step_deg(c[i - 1], c[i]) < limit:
            continue
        if min(abs(float(b[i - 1])), abs(float(b[i]))) <= upright:
            pinned[i] = True

    visited = np.zeros(len(c), dtype=bool)
    for i in range(1, len(c)):
        if not pinned[i] or visited[i]:
            continue
        start = i
        end = i
        while end + 1 < len(c) and pinned[end + 1]:
            end += 1
        # One upright sample between two pinned steps is still the same walk.
        while (
            end + 2 < len(c)
            and not pinned[end + 1]
            and pinned[end + 2]
            and abs(float(b[end + 1])) <= upright
        ):
            end += 2
            while end + 1 < len(c) and pinned[end + 1]:
                end += 1
        while end + 1 < len(c):
            if abs(float(b[end + 1])) > upright:
                break
            if c_step_deg(c[end], c[end + 1]) < unwind:
                break
            end += 1
        while start > 1:
            if abs(float(b[start - 1])) > upright:
                break
            if c_step_deg(c[start - 2], c[start - 1]) < unwind:
                break
            start -= 1
        n_pinned = int(np.count_nonzero(pinned[start : end + 1]))
        if n_pinned < 2 or step_deg <= 0.0:
            visited[start : end + 1] = True
            continue
        target = float(original[end + 1]) if end + 1 < len(c) else float(original[end])
        step = float(step_deg)
        prev = float(original[start - 1])
        for k in range(start, len(c)):
            goal = target if k <= end else float(original[k])
            delta = _signed_c_delta(prev, goal)
            if abs(delta) <= step:
                c[k] = _wrap_c(prev + delta)
                prev = float(c[k])
                if k > end:
                    break
                continue
            c[k] = _wrap_c(prev + float(np.sign(delta)) * step)
            prev = float(c[k])
        visited[start : end + 1] = True
    return b, c, catchups


def _unwrap_c(c_angles: np.ndarray) -> np.ndarray:
    out = np.zeros(len(c_angles), dtype=float)
    if len(c_angles) == 0:
        return out
    out[0] = float(c_angles[0])
    for i in range(1, len(c_angles)):
        out[i] = out[i - 1] + _signed_c_delta(c_angles[i - 1], c_angles[i])
    return out


def _aim_half_width_deg(tilt_deg: float, aim_deg: float) -> float:
    """Largest |ΔC| whose tool axis stays within ``aim_deg`` of a normal at this tilt."""
    beta = np.deg2rad(min(abs(float(tilt_deg)), 89.0))
    aim = np.deg2rad(float(aim_deg))
    s2 = float(np.sin(beta) ** 2)
    if s2 < 1e-8:
        return 180.0
    cos_delta = (float(np.cos(aim)) - float(np.cos(beta) ** 2)) / s2
    if cos_delta >= 1.0:
        return 0.0
    if cos_delta <= -1.0:
        return 180.0
    return float(np.rad2deg(np.arccos(cos_delta)))


def _c_step_budget_deg(
    delta_b_deg: float,
    *,
    a_mm: float,
    d_mm: float,
    sample_mm: float,
    feed_ratio: float,
) -> float:
    """Largest |ΔC| that keeps pivot travel within ``feed_ratio`` times the tip step."""
    extra_mm = float(sample_mm) * max(float(feed_ratio) - 1.0, 0.0)
    b_mm = float(d_mm) * abs(float(delta_b_deg)) * np.pi / 180.0
    room_mm = extra_mm - b_mm
    if room_mm <= 0.0 or a_mm <= 0.0:
        return 0.0
    return float(np.rad2deg(room_mm / float(a_mm)))


def plan_trace_heading(
    b_angles: np.ndarray,
    c_angles: np.ndarray,
    *,
    a_mm: float,
    d_mm: float,
    sample_mm: float,
    speed_mm_min: float,
    max_speed_mm_min: float,
    aim_deg: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose C along one wire to minimize axis travel while matching the normal.

    Each sample may only move as far as the feed cap allows. Within that limit
    the path prefers the scalp heading, and where the nozzle is upright it
    holds C because many headings still fit the normal.
    """
    b = np.asarray(b_angles, dtype=float).copy()
    c = np.asarray(c_angles, dtype=float).copy()
    n = len(c)
    if n <= 1 or speed_mm_min <= 0.0:
        return b, c

    unwrapped = _unwrap_c(c)
    ratio = float(max_speed_mm_min) / float(speed_mm_min)
    lo = np.zeros(n)
    hi = np.zeros(n)
    step = np.zeros(n)
    for i in range(n):
        half = _aim_half_width_deg(b[i], aim_deg)
        lo[i] = unwrapped[i] - half
        hi[i] = unwrapped[i] + half
        if i > 0:
            step[i] = _c_step_budget_deg(
                float(b[i] - b[i - 1]),
                a_mm=a_mm,
                d_mm=d_mm,
                sample_mm=sample_mm,
                feed_ratio=ratio,
            )

    grid_lo = int(np.floor(min(float(np.min(lo)), float(unwrapped[0])) - 1.0))
    grid_hi = int(np.ceil(max(float(np.max(hi)), float(unwrapped[-1])) + 1.0))
    grid = np.arange(grid_lo, grid_hi + 1, 1.0)
    n_grid = len(grid)
    outside = np.zeros((n, n_grid))
    for i in range(n):
        for j in range(n_grid):
            gap = max(0.0, lo[i] - grid[j], grid[j] - hi[i])
            outside[i, j] = gap * gap + 0.01 * abs(grid[j] - unwrapped[i])

    inf = 1e18
    cost = outside[0].copy()
    back = np.zeros((n, n_grid), dtype=np.int32)
    for i in range(1, n):
        nxt = np.full(n_grid, inf)
        span = int(np.floor(step[i] + 1e-6))
        for j in range(n_grid):
            k0 = max(0, j - span)
            k1 = min(n_grid, j + span + 1)
            best = inf
            best_k = j
            for k in range(k0, k1):
                if abs(grid[k] - grid[j]) > step[i] + 1e-6:
                    continue
                val = cost[k] + abs(grid[k] - grid[j]) + outside[i, j]
                if val < best:
                    best = val
                    best_k = k
            nxt[j] = best
            back[i, j] = best_k
        cost = nxt

    chosen = np.zeros(n)
    j = int(np.argmin(cost))
    for i in range(n - 1, -1, -1):
        chosen[i] = grid[j]
        if i > 0:
            j = int(back[i, j])
    c_out = np.array([_wrap_c(v) for v in chosen])
    return b, c_out


def retarget_normals_to_c(
    normals: np.ndarray,
    c_angles: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Point each masked normal's XY heading at ``c_angles`` without changing tilt.

    ``find_caxis_angle`` is inverted while keeping ``sign(ny)``, so B stays
    the tilt of the same normal and the tool offset matches the commanded C.
    """
    out = np.asarray(normals, dtype=float).copy()
    c = np.asarray(c_angles, dtype=float)
    use = np.asarray(mask, dtype=bool)
    if out.ndim == 1:
        out = out.reshape(1, 3)
    for i in np.flatnonzero(use):
        nxy = float(np.hypot(out[i, 0], out[i, 1]))
        if nxy < 1e-8:
            continue
        if float(out[i, 1]) >= 0.0:
            alpha = np.deg2rad(90.0 - float(c[i]))
        else:
            alpha = np.deg2rad(-90.0 - float(c[i]))
        out[i, 0] = float(np.cos(alpha)) * nxy
        out[i, 1] = float(np.sin(alpha)) * nxy
        length = float(np.linalg.norm(out[i]))
        if length > 1e-12:
            out[i] /= length
    return out


def max_c_step_deg(c_angles: np.ndarray) -> float:
    if len(c_angles) <= 1:
        return 0.0
    return max(c_step_deg(c_angles[i - 1], c_angles[i]) for i in range(1, len(c_angles)))


def validate_axis_continuity(
    b_angles: np.ndarray,
    c_angles: np.ndarray,
    *,
    max_c_step_deg: float,
) -> None:
    """Raise when consecutive C commands exceed the allowed step."""
    threshold = float(max_c_step_deg)
    for i in range(1, len(c_angles)):
        step = c_step_deg(c_angles[i - 1], c_angles[i])
        if step > threshold:
            raise ValueError(
                f"C-axis step {step:.1f} deg at index {i} "
                f"(C {c_angles[i - 1]:.2f} -> {c_angles[i]:.2f}) "
                f"exceeds limit {threshold:.1f} deg"
            )
