"""Initial interconnect: balanced hubs, slot order, odd swaps, and even bends.

This is the pre-search wire picture used to place harness leads. A polar UV
lift draws those flat curves on the scalp. It does not run a genetic algorithm
or write interconnect G-code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TERMINALS = ("TERMINAL_LEFT", "TERMINAL_RIGHT")
SLOT_SPACING_MM = 6.0
PATH_POINTS = 32
ELECTRODE_ZONE_FACTOR = 0.05
TERMINAL_ZONE_FACTOR = 0.15
DETOUR_SCALES_MM = (8.0, 16.0, 28.0, 44.0)
ODD_SWAP_MAX_ROUNDS = 80
_CROSS_EPS = 1e-8
_POINT_DEDUP = 0.3


@dataclass
class WireLayout:
    assignments: dict[str, str]
    slots: dict[str, int]
    entries_2d: dict[str, np.ndarray]
    paths_2d: dict[str, np.ndarray]
    electrodes_2d: dict[str, np.ndarray]
    terminals_2d: dict[str, np.ndarray]
    odd_pairs_before: int
    crossings_before: int
    odd_pairs: int
    crossings: int
    notes: list[str] = field(default_factory=list)


def polar_projection(points: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Flatten points about ``origin``. Radius is the 3D distance; angle is atan2(y, x)."""
    rel = np.asarray(points, dtype=float) - np.asarray(origin, dtype=float)
    if rel.ndim == 1:
        rel = rel.reshape(1, 3)
    radius = np.linalg.norm(rel, axis=1)
    theta = np.arctan2(rel[:, 1], rel[:, 0])
    return np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))


def balanced_assignment(
    distances: dict[str, dict[str, float]],
    terminals: tuple[str, ...] = TERMINALS,
) -> dict[str, str]:
    """Near-equal hub counts with the smallest extra wire length.

    Starts from each electrode's nearer hub, then moves the cheapest leads off
    the overloaded hub until the counts match (10 and 9 for 19 electrodes).
    """
    if not distances:
        return {}
    terminals = tuple(terminals)
    capacity: dict[str, int] = {}
    base, remainder = divmod(len(distances), len(terminals))
    for index, terminal in enumerate(terminals):
        capacity[terminal] = base + (1 if index < remainder else 0)

    assignment = {
        name: min(options, key=options.get) for name, options in distances.items()
    }
    counts = {terminal: 0 for terminal in terminals}
    for terminal in assignment.values():
        counts[terminal] = counts.get(terminal, 0) + 1

    def overload(terminal: str) -> int:
        return counts.get(terminal, 0) - capacity.get(terminal, 0)

    def underload(terminal: str) -> int:
        return capacity.get(terminal, 0) - counts.get(terminal, 0)

    while True:
        heavy = max(terminals, key=overload)
        light = max(terminals, key=underload)
        if overload(heavy) <= 0 or underload(light) <= 0:
            break
        movable = [
            name
            for name, terminal in assignment.items()
            if terminal == heavy and light in distances[name]
        ]
        if not movable:
            break
        best = min(movable, key=lambda name: distances[name][light] - distances[name][heavy])
        assignment[best] = light
        counts[heavy] -= 1
        counts[light] += 1
    return assignment


def _mean_radius(points: dict[str, np.ndarray]) -> float:
    if not points:
        return 0.0
    return float(np.mean([np.linalg.norm(point) for point in points.values()]))


def electrode_zone_radius(electrodes_2d: dict[str, np.ndarray]) -> float:
    return max(_mean_radius(electrodes_2d) * ELECTRODE_ZONE_FACTOR, 1.0)


def terminal_zone_radius(electrodes_2d: dict[str, np.ndarray]) -> float:
    return max(_mean_radius(electrodes_2d) * TERMINAL_ZONE_FACTOR, SLOT_SPACING_MM)


def polar_layout(
    electrodes: dict[str, np.ndarray],
    terminals: dict[str, np.ndarray],
    cz: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    electrodes_2d = {
        name: polar_projection(np.asarray(point, dtype=float), cz)[0]
        for name, point in electrodes.items()
    }
    terminals_2d = {
        name: polar_projection(np.asarray(point, dtype=float), cz)[0]
        for name, point in terminals.items()
    }
    return electrodes_2d, terminals_2d


def straight_path(start: np.ndarray, end: np.ndarray, n: int = PATH_POINTS) -> np.ndarray:
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    samples = np.linspace(0.0, 1.0, max(2, n))
    return start + samples[:, None] * (end - start)


def _unit_perp(start: np.ndarray, end: np.ndarray) -> np.ndarray | None:
    direction = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    perp = np.array([-direction[1], direction[0]], dtype=float)
    length = float(np.linalg.norm(perp))
    if length < 1e-9:
        return None
    return perp / length


def _segment_point_distance(start: np.ndarray, end: np.ndarray, point: np.ndarray) -> float:
    delta = end - start
    length2 = float(np.dot(delta, delta))
    if length2 < 1e-18:
        return float(np.linalg.norm(point - start))
    weight = float(np.clip(np.dot(point - start, delta) / length2, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + weight * delta)))


def path_hits_electrodes(
    path: np.ndarray,
    electrode: str,
    centers: dict[str, np.ndarray],
    radius: float,
) -> int:
    hits = 0
    path = np.asarray(path, dtype=float)
    for name, center in centers.items():
        if name == electrode:
            continue
        center = np.asarray(center, dtype=float)
        close = False
        for start, end in zip(path[:-1], path[1:]):
            if _segment_point_distance(start, end, center) < radius:
                close = True
                break
        if close:
            hits += 1
    return hits


def path_with_electrode_detour(
    start: np.ndarray,
    end: np.ndarray,
    electrode: str,
    centers: dict[str, np.ndarray],
    radius: float,
    *,
    n: int = PATH_POINTS,
) -> np.ndarray:
    """Straight chord, or one sideways bend if another electrode disc blocks it."""
    straight = straight_path(start, end, n=n)
    if path_hits_electrodes(straight, electrode, centers, radius) == 0:
        return straight

    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    perp = _unit_perp(start, end)
    if perp is None:
        return straight
    best = straight
    best_hits = path_hits_electrodes(straight, electrode, centers, radius)
    mid = 0.5 * (start + end)

    for sign in (1.0, -1.0):
        for scale in DETOUR_SCALES_MM:
            waypoint = mid + sign * scale * perp
            trial = np.vstack(
                (
                    straight_path(start, waypoint, n=max(2, n // 2))[:-1],
                    straight_path(waypoint, end, n=max(2, n // 2)),
                )
            )
            hits = path_hits_electrodes(trial, electrode, centers, radius)
            if hits < best_hits:
                best = trial
                best_hits = hits
            if hits == 0:
                return trial
    return best


def _orient(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _collinear_overlap(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    if abs(_orient(a, b, c)) > _CROSS_EPS or abs(_orient(a, b, d)) > _CROSS_EPS:
        return False
    axis = 0 if abs(b[0] - a[0]) >= abs(b[1] - a[1]) else 1

    def span(p: np.ndarray, q: np.ndarray) -> tuple[float, float]:
        lo, hi = (p[axis], q[axis]) if p[axis] <= q[axis] else (q[axis], p[axis])
        return float(lo), float(hi)

    a0, a1 = span(a, b)
    c0, c1 = span(c, d)
    return min(a1, c1) - max(a0, c0) > _CROSS_EPS


def _proper_crossing(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray | None:
    ab = b - a
    cd = d - c
    denom = float(ab[0] * cd[1] - ab[1] * cd[0])
    if abs(denom) <= _CROSS_EPS:
        return None
    t = float(((c[0] - a[0]) * cd[1] - (c[1] - a[1]) * cd[0]) / denom)
    u = float(((c[0] - a[0]) * ab[1] - (c[1] - a[1]) * ab[0]) / denom)
    if _CROSS_EPS < t < 1.0 - _CROSS_EPS and _CROSS_EPS < u < 1.0 - _CROSS_EPS:
        return a + t * ab
    return None


def mutual_crossings(path_a: np.ndarray, path_b: np.ndarray) -> int:
    """Discrete interior crossings. A collinear overlap counts as one odd event."""
    path_a = np.asarray(path_a, dtype=float)
    path_b = np.asarray(path_b, dtype=float)
    if len(path_a) < 2 or len(path_b) < 2:
        return 0
    points: list[np.ndarray] = []
    for a, b in zip(path_a[:-1], path_a[1:]):
        for c, d in zip(path_b[:-1], path_b[1:]):
            if _collinear_overlap(a, b, c, d):
                return 1
            hit = _proper_crossing(a, b, c, d)
            if hit is None:
                continue
            if all(float(np.linalg.norm(hit - kept)) > _POINT_DEDUP for kept in points):
                points.append(hit)
    return len(points)


def crossing_count(paths: list[np.ndarray]) -> int:
    total = 0
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            total += mutual_crossings(paths[i], paths[j])
    return total


def odd_pair_count(paths: list[np.ndarray], path_terminals: list[str]) -> int:
    total = 0
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            if path_terminals[i] != path_terminals[j]:
                continue
            if mutual_crossings(paths[i], paths[j]) % 2 == 1:
                total += 1
    return total


def strip_tangent(center: np.ndarray) -> np.ndarray:
    radial = np.asarray(center, dtype=float)
    norm = float(np.linalg.norm(radial))
    if norm < 1e-9:
        return np.array([0.0, 1.0], dtype=float)
    radial = radial / norm
    return np.array([-radial[1], radial[0]], dtype=float)


def chord_hit(start: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    """Where the electrode→hub chord meets the hub circle."""
    start = np.asarray(start, dtype=float)
    center = np.asarray(center, dtype=float)
    direction = start - center
    dist = float(np.linalg.norm(direction))
    if dist < 1e-9:
        return center.copy()
    return center + float(radius) * direction / dist


def natural_slot_order(
    names: list[str],
    electrodes_2d: dict[str, np.ndarray],
    center: np.ndarray,
    radius: float,
) -> list[str]:
    tangent = strip_tangent(center)
    center = np.asarray(center, dtype=float)

    def key(name: str) -> float:
        hit = chord_hit(electrodes_2d[name], center, radius)
        return float(np.dot(hit - center, tangent))

    return sorted(names, key=key)


def place_slots(
    order: list[str],
    center: np.ndarray,
    radius: float,
    spacing: float,
) -> dict[str, np.ndarray]:
    """Even slots along the hub circle, centered on the side that faces Cz."""
    center = np.asarray(center, dtype=float)
    tangent = strip_tangent(center)
    inward = -center
    norm = float(np.linalg.norm(inward))
    inward = inward / norm if norm > 1e-9 else np.array([-1.0, 0.0], dtype=float)
    anchor = center + float(radius) * inward
    mid = (len(order) - 1) / 2.0
    entries: dict[str, np.ndarray] = {}
    for index, name in enumerate(order):
        raw = anchor + (index - mid) * float(spacing) * tangent
        direction = raw - center
        dist = float(np.linalg.norm(direction))
        if dist < 1e-9:
            entries[name] = anchor.copy()
        else:
            entries[name] = center + float(radius) * direction / dist
    return entries


def _pin(path: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    pinned = np.asarray(path, dtype=float).copy()
    if len(pinned) == 0:
        return pinned
    pinned[0] = np.asarray(start, dtype=float)
    pinned[-1] = np.asarray(end, dtype=float)
    return pinned


def odd_swap(
    paths: list[np.ndarray],
    names: list[str],
    path_terminals: list[str],
    electrodes_2d: dict[str, np.ndarray],
    entries: dict[str, np.ndarray],
    slots: dict[str, int],
    centers: dict[str, np.ndarray],
    radius: float,
    *,
    max_rounds: int = ODD_SWAP_MAX_ROUNDS,
) -> tuple[list[np.ndarray], dict[str, np.ndarray], dict[str, int], list[str]]:
    """Swap same-hub strip slots until odd mutual crossings are gone.

    While any same-hub pair crosses an odd number of times, a swap is kept
    only when that odd-pair count falls. Total crossings may stay or rise.
    After the odd count is zero, a swap is kept when it cuts the crossing
    count without creating a new odd pair.
    """
    paths = [np.asarray(path, dtype=float).copy() for path in paths]
    entries = {name: np.asarray(point, dtype=float).copy() for name, point in entries.items()}
    slots = {name: int(index) for name, index in slots.items()}
    tried: set[frozenset[str]] = set()
    notes: list[str] = []

    def replan(index: int) -> np.ndarray:
        name = names[index]
        return _pin(
            path_with_electrode_detour(
                electrodes_2d[name], entries[name], name, centers, radius
            ),
            electrodes_2d[name],
            entries[name],
        )

    def revert(name_a: str, name_b: str) -> None:
        entries[name_a], entries[name_b] = entries[name_b].copy(), entries[name_a].copy()
        if name_a in slots and name_b in slots:
            slots[name_a], slots[name_b] = slots[name_b], slots[name_a]

    for round_index in range(max_rounds):
        cross_total = crossing_count(paths)
        odd_total = odd_pair_count(paths, path_terminals)
        if cross_total == 0:
            break

        candidates: list[tuple[int, int, int]] = []
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                if path_terminals[i] != path_terminals[j]:
                    continue
                mutual = mutual_crossings(paths[i], paths[j])
                if mutual == 0:
                    continue
                is_odd = mutual % 2 == 1
                if odd_total > 0 and not is_odd:
                    continue
                candidates.append((0 if is_odd else 1, i, j))
        candidates.sort()

        improved = False
        for _, i, j in candidates:
            name_a = names[i]
            name_b = names[j]
            if name_a not in entries or name_b not in entries:
                continue
            pair = frozenset((name_a, name_b))
            if pair in tried:
                continue

            entries[name_a], entries[name_b] = entries[name_b].copy(), entries[name_a].copy()
            if name_a in slots and name_b in slots:
                slots[name_a], slots[name_b] = slots[name_b], slots[name_a]
            trial = list(paths)
            trial[i] = replan(i)
            trial[j] = replan(j)
            new_cross = crossing_count(trial)
            new_odd = odd_pair_count(trial, path_terminals)
            mutual = mutual_crossings(trial[i], trial[j])
            tried.add(pair)

            odd_improved = new_odd < odd_total
            count_improved = new_cross < cross_total
            even_mutual = mutual % 2 == 0
            if new_odd > odd_total:
                accept = False
            elif odd_total > 0:
                accept = odd_improved
            else:
                accept = new_odd == 0 and (count_improved or even_mutual)

            if accept:
                paths = trial
                improved = True
                if odd_improved:
                    tried.clear()
                    reason = f"odd {odd_total}->{new_odd}"
                elif count_improved:
                    reason = "fewer crossings"
                else:
                    reason = f"even mutual={mutual}"
                notes.append(
                    f"Odd swap {name_a}<->{name_b} on {path_terminals[i]}: "
                    f"crossings {cross_total}->{new_cross}, {reason}"
                )
                break

            revert(name_a, name_b)

        if not improved:
            if cross_total > 0:
                notes.append(
                    f"Odd swap stopped at round {round_index + 1} "
                    f"(crossings={cross_total}, odd_pairs={odd_total})"
                )
            break

    return paths, entries, slots, notes


_EVEN_SCALES = (8.0, 16.0, 28.0, 42.0, 60.0, 80.0)
_EVEN_BOTH_SCALES = (16.0, 28.0, 42.0, 60.0)


def bent_path(
    start: np.ndarray, end: np.ndarray, sign: float, scale: float, n: int = PATH_POINTS
) -> np.ndarray:
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    perp = _unit_perp(start, end)
    if perp is None:
        return straight_path(start, end, n=n)
    mid = 0.5 * (start + end) + sign * scale * perp
    half = max(2, n // 2)
    return np.vstack((straight_path(start, mid, n=half)[:-1], straight_path(mid, end, n=half)))


def _via_path(points: list[np.ndarray], n: int = PATH_POINTS) -> np.ndarray:
    legs = max(1, len(points) - 1)
    per = max(2, int(n) // legs)
    chunks: list[np.ndarray] = []
    for start, end in zip(points[:-1], points[1:]):
        leg = straight_path(start, end, n=per)
        chunks.append(leg if not chunks else leg[1:])
    return np.vstack(chunks)


def double_knee_path(
    start: np.ndarray, end: np.ndarray, sign: float, scale: float, n: int = PATH_POINTS
) -> np.ndarray:
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    perp = _unit_perp(start, end)
    if perp is None:
        return straight_path(start, end, n=n)
    direction = end - start
    offset = sign * scale * perp
    return _via_path(
        [start, start + direction / 3.0 + offset, start + 2.0 * direction / 3.0 + offset, end],
        n=n,
    )


def s_curve_path(
    start: np.ndarray, end: np.ndarray, sign: float, scale: float, n: int = PATH_POINTS
) -> np.ndarray:
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    perp = _unit_perp(start, end)
    if perp is None:
        return straight_path(start, end, n=n)
    direction = end - start
    return _via_path(
        [
            start,
            start + direction / 3.0 + sign * scale * perp,
            start + 2.0 * direction / 3.0 - sign * scale * perp,
            end,
        ],
        n=n,
    )


def _path_family(
    start: np.ndarray,
    end: np.ndarray,
    scales: tuple[float, ...] = _EVEN_SCALES,
) -> list[tuple[str, np.ndarray]]:
    family: list[tuple[str, np.ndarray]] = [("straight", straight_path(start, end))]
    for sign, side in ((1.0, "L"), (-1.0, "R")):
        for scale in scales:
            family.append((f"knee-{side}{scale:g}", bent_path(start, end, sign, scale)))
            family.append((f"dknee-{side}{scale:g}", double_knee_path(start, end, sign, scale)))
            family.append((f"scurve-{side}{scale:g}", s_curve_path(start, end, sign, scale)))
    return family


def even_uncross(
    paths: list[np.ndarray],
    names: list[str],
    path_terminals: list[str],
    electrodes_2d: dict[str, np.ndarray],
    entries: dict[str, np.ndarray],
    centers: dict[str, np.ndarray],
    radius: float,
    *,
    max_rounds: int = 40,
) -> tuple[list[np.ndarray], list[str]]:
    """Bend fixed-end wires until an even same-hub weave is gone.

    A bend is kept when that pair's crossing count falls, the layout crossing
    count does not rise, and the odd-pair count does not rise.
    """
    paths = [np.asarray(path, dtype=float).copy() for path in paths]
    notes: list[str] = []

    def pinned(name: str, raw: np.ndarray) -> np.ndarray | None:
        if name not in entries or name not in electrodes_2d:
            return None
        path = _pin(raw, electrodes_2d[name], entries[name])
        if path_hits_electrodes(path, name, centers, radius):
            return None
        return path

    for round_index in range(max_rounds):
        cross_total = crossing_count(paths)
        if cross_total == 0:
            break
        odd_total = odd_pair_count(paths, path_terminals)
        improved = False
        even_pairs = 0
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                if path_terminals[i] != path_terminals[j]:
                    continue
                mutual = mutual_crossings(paths[i], paths[j])
                if mutual == 0 or mutual % 2 == 1:
                    continue
                even_pairs += 1
                pair = ((i, names[i]), (j, names[j]))
                for index, name in pair:
                    for label, raw in _path_family(electrodes_2d[name], entries[name]):
                        trial_path = pinned(name, raw)
                        if trial_path is None:
                            continue
                        trial = list(paths)
                        trial[index] = trial_path
                        new_mutual = mutual_crossings(trial[i], trial[j])
                        new_cross = crossing_count(trial)
                        new_odd = odd_pair_count(trial, path_terminals)
                        if new_mutual < mutual and new_cross <= cross_total and new_odd <= odd_total:
                            paths = trial
                            improved = True
                            notes.append(
                                f"Even uncross {names[i]}<->{names[j]} via {name}:{label}: "
                                f"mutual {mutual}->{new_mutual}"
                            )
                            break
                    if improved:
                        break
                if improved:
                    break
                for label_i, raw_i in _path_family(electrodes_2d[names[i]], entries[names[i]], _EVEN_BOTH_SCALES):
                    left = pinned(names[i], raw_i)
                    if left is None:
                        continue
                    for label_j, raw_j in _path_family(electrodes_2d[names[j]], entries[names[j]], _EVEN_BOTH_SCALES):
                        right = pinned(names[j], raw_j)
                        if right is None:
                            continue
                        trial = list(paths)
                        trial[i] = left
                        trial[j] = right
                        new_mutual = mutual_crossings(trial[i], trial[j])
                        new_cross = crossing_count(trial)
                        new_odd = odd_pair_count(trial, path_terminals)
                        if new_mutual < mutual and new_cross <= cross_total and new_odd <= odd_total:
                            paths = trial
                            improved = True
                            notes.append(
                                f"Even uncross {names[i]}<->{names[j]} via both {label_i}+{label_j}: "
                                f"mutual {mutual}->{new_mutual}"
                            )
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            if cross_total > 0:
                notes.append(
                    f"Even uncross stopped at round {round_index + 1} "
                    f"(crossings={cross_total}, odd_pairs={odd_total}, even_pairs={even_pairs})"
                )
            break
    return paths, notes


def detour_uncross(
    paths: list[np.ndarray],
    names: list[str],
    electrodes_2d: dict[str, np.ndarray],
    entries: dict[str, np.ndarray],
    centers: dict[str, np.ndarray],
    radius: float,
    *,
    max_rounds: int = 40,
) -> list[np.ndarray]:
    """One more fixed-end bend pass when a pair still crosses."""
    paths = [np.asarray(path, dtype=float).copy() for path in paths]
    for _ in range(max_rounds):
        cross_total = crossing_count(paths)
        if cross_total == 0:
            break
        improved = False
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                if mutual_crossings(paths[i], paths[j]) == 0:
                    continue
                for index in (i, j):
                    name = names[index]
                    for sign in (1.0, -1.0):
                        for scale in (8.0, 16.0, 28.0, 42.0):
                            trial_path = _pin(
                                bent_path(electrodes_2d[name], entries[name], sign, scale),
                                electrodes_2d[name],
                                entries[name],
                            )
                            if path_hits_electrodes(trial_path, name, centers, radius):
                                continue
                            trial = list(paths)
                            trial[index] = trial_path
                            if crossing_count(trial) < cross_total:
                                paths = trial
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    return paths


def build_wire_layout(
    electrodes: dict[str, np.ndarray],
    terminals: dict[str, np.ndarray],
    cz: np.ndarray,
    distances: dict[str, dict[str, float]],
    *,
    spacing: float = SLOT_SPACING_MM,
) -> WireLayout:
    """Assign hubs, place strip slots, odd-swap, then bend any even weave that remains."""
    missing = [name for name in electrodes if name not in distances]
    if missing:
        raise ValueError(f"No hub distances for {missing}")
    for name, options in distances.items():
        absent = [terminal for terminal in TERMINALS if terminal not in options]
        if absent and name in electrodes:
            raise ValueError(f"{name} is missing distances to {absent}")

    electrodes_2d, terminals_2d = polar_layout(electrodes, terminals, cz)
    zone = electrode_zone_radius(electrodes_2d)
    hub_radius = terminal_zone_radius(electrodes_2d)
    assignments = balanced_assignment(
        {name: distances[name] for name in electrodes}
    )

    names: list[str] = []
    path_terminals: list[str] = []
    entries: dict[str, np.ndarray] = {}
    slots: dict[str, int] = {}
    for terminal in TERMINALS:
        if terminal not in terminals_2d:
            continue
        group = [name for name in electrodes if assignments.get(name) == terminal]
        order = natural_slot_order(group, electrodes_2d, terminals_2d[terminal], hub_radius)
        placed = place_slots(order, terminals_2d[terminal], hub_radius, spacing)
        for index, name in enumerate(order):
            entries[name] = placed[name]
            slots[name] = index
            names.append(name)
            path_terminals.append(terminal)

    paths = [
        _pin(
            path_with_electrode_detour(
                electrodes_2d[name], entries[name], name, electrodes_2d, zone
            ),
            electrodes_2d[name],
            entries[name],
        )
        for name in names
    ]
    odd_before = odd_pair_count(paths, path_terminals)
    cross_before = crossing_count(paths)
    paths, entries, slots, notes = odd_swap(
        paths,
        names,
        path_terminals,
        electrodes_2d,
        entries,
        slots,
        electrodes_2d,
        zone,
    )
    paths, even_notes = even_uncross(
        paths, names, path_terminals, electrodes_2d, entries, electrodes_2d, zone
    )
    notes.extend(even_notes)
    paths = detour_uncross(paths, names, electrodes_2d, entries, electrodes_2d, zone)
    paths = [_pin(path, electrodes_2d[name], entries[name]) for name, path in zip(names, paths)]
    path_map = {name: path for name, path in zip(names, paths)}
    return WireLayout(
        assignments=assignments,
        slots=slots,
        entries_2d=entries,
        paths_2d=path_map,
        electrodes_2d=electrodes_2d,
        terminals_2d=terminals_2d,
        odd_pairs_before=odd_before,
        crossings_before=cross_before,
        odd_pairs=odd_pair_count(paths, path_terminals),
        crossings=crossing_count(paths),
        notes=notes,
    )
