import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import unary_union
from PYTHON.tools.new2dAlterations import (
    MIN_PATH_SEPARATION,
    PHASE1_GREEDY_ELECTRODE_CLEARANCE_MULTIPLIER,
    PHASE2_GREEDY_SPACING_ELECTRODE_MULTIPLIER,
    PHASE2_GREEDY_TRACE_HALF_SEPARATION,
)

PATH_BUFFER_FACTOR = 1.3
CORRIDOR_BUFFER_FACTOR = 1.5


def _paths_equal(path_a, path_b):
    return np.array_equal(np.asarray(path_a, dtype=float), np.asarray(path_b, dtype=float))


def _build_forbidden_region(
    path,
    path_electrode_name,
    other_paths,
    electrode_zones,
    corridor_obstacle_paths=None,
    electrodes_only=False,
    electrode_clearance_multiplier=None,
    trace_buffer_half_width=None,
):
    base_clearance = electrode_zones['metadata']['electrode_zone_size']
    electrode_mult = (
        electrode_clearance_multiplier
        if electrode_clearance_multiplier is not None
        else PHASE1_GREEDY_ELECTRODE_CLEARANCE_MULTIPLIER
    )
    electrode_clearance = base_clearance * electrode_mult
    path_clearance = base_clearance * PATH_BUFFER_FACTOR
    if trace_buffer_half_width is not None:
        path_clearance = float(trace_buffer_half_width)
    corridor_clearance = max(path_clearance, MIN_PATH_SEPARATION * CORRIDOR_BUFFER_FACTOR)

    regions = []
    corridor_arrays = [
        np.asarray(p, dtype=float) for p in (corridor_obstacle_paths or [])
    ]

    if not electrodes_only:
        for corridor in corridor_arrays:
            line = LineString(corridor)
            if line.is_valid and not line.is_empty:
                regions.append(line.buffer(corridor_clearance))

        mesh_lines = [
            LineString(p)
            for p in other_paths
            if not _paths_equal(p, path)
            and not any(_paths_equal(p, corridor) for corridor in corridor_arrays)
        ]
        mesh_union = unary_union(mesh_lines) if mesh_lines else None

        if mesh_union is not None and not mesh_union.is_empty:
            regions.append(mesh_union.buffer(path_clearance))

    for name, md in electrode_zones['metadata'].items():
        if name in ('electrode_zone_size', 'terminal_zone_size'):
            continue
        if name == path_electrode_name:
            continue
        center = md['center']
        regions.append(Point(center).buffer(electrode_clearance))

    if not regions:
        return None

    forbidden = unary_union(regions)
    return forbidden.buffer(0)


def _segment_blocked_by_forbidden(start, end, forbidden):
    """True when the chord crosses the interior of the forbidden region."""
    if forbidden is None or forbidden.is_empty:
        return False

    chord = LineString([start, end])
    if not chord.intersects(forbidden):
        return False

    intersection = chord.intersection(forbidden)
    if intersection.is_empty:
        return False
    if intersection.geom_type in ('LineString', 'MultiLineString', 'Polygon', 'MultiPolygon'):
        return True
    if intersection.geom_type == 'GeometryCollection':
        return any(
            not geom.is_empty and geom.geom_type != 'Point'
            for geom in intersection.geoms
        )
    return intersection.geom_type != 'Point'


def _path_hits_foreign_electrode_zone(path, electrode_name, electrode_zones):
    path_line = LineString(path)
    for zone_name, zone in electrode_zones['zones'].items():
        if zone_name == electrode_name:
            continue
        if path_line.intersects(zone):
            return True
    return False


def _forbidden_boundary_for_route(start, end, forbidden):
    """Pick the forbidden component that actually blocks the route."""
    if forbidden is None or forbidden.is_empty:
        return None

    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    chord = LineString([start, end])
    chord_mid = (start + end) / 2.0

    if hasattr(forbidden, 'geoms'):
        polys = list(forbidden.geoms)
        blocking = [poly for poly in polys if poly.intersects(chord)]
        if blocking:
            poly = max(
                blocking,
                key=lambda p: float(p.intersection(chord).length)
                if p.intersection(chord).geom_type in ('LineString', 'MultiLineString')
                else p.area,
            )
        else:
            poly = min(polys, key=lambda p: p.distance(Point(chord_mid)))
        return [np.array(pt) for pt in poly.exterior.coords[:-1]]

    if hasattr(forbidden, 'exterior'):
        return [np.array(pt) for pt in forbidden.exterior.coords[:-1]]
    return None


def _polyline_length(route):
    route = np.asarray(route, dtype=float)
    if len(route) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(route, axis=0), axis=1)))


def _paths_same_polyline(path_a, path_b, atol=1e-5, rtol=1e-5):
    """True when two polylines match after resampling to a common point count."""
    path_a = np.asarray(path_a, dtype=float)
    path_b = np.asarray(path_b, dtype=float)
    if len(path_a) == len(path_b) and np.allclose(path_a, path_b, atol=atol, rtol=rtol):
        return True
    n = max(len(path_a), len(path_b))
    return np.allclose(
        _resample_polyline(path_a, n),
        _resample_polyline(path_b, n),
        atol=atol,
        rtol=rtol,
    )


def _shortcut_route(route, forbidden, start, end, n_points):
    """Remove unnecessary hull detours while staying outside the forbidden region."""
    if forbidden is None or forbidden.is_empty:
        return route

    route = np.asarray(route, dtype=float)
    step = max(1, len(route) // 40)
    pts = [np.asarray(route[i], dtype=float) for i in range(0, len(route), step)]
    if np.linalg.norm(pts[-1] - route[-1]) > 1e-6:
        pts.append(np.asarray(route[-1], dtype=float))

    pts[0] = np.asarray(start, dtype=float)
    pts[-1] = np.asarray(end, dtype=float)

    shortened = [pts[0]]
    idx = 0
    while idx < len(pts) - 1:
        best = min(idx + 1, len(pts) - 1)
        for j in range(len(pts) - 1, idx + 1, -1):
            if not _segment_blocked_by_forbidden(pts[idx], pts[j], forbidden):
                best = j
                break
        shortened.append(pts[best])
        if best == idx:
            break
        idx = best

    return _resample_polyline(shortened, n_points)


def _shortcut_route_if_electrode_clear(
    route,
    forbidden,
    start,
    end,
    n_points,
    path_electrode_name=None,
    electrode_zones=None,
):
    """
    Shortcut a hull route only when foreign electrode zones stay clear.

    Corner-cutting can re-enter real electrode discs even when chords avoid the
    inflated forbidden region; keep the raw hull when shortcut would violate.
    """
    shortened = _shortcut_route(route, forbidden, start, end, n_points)
    if path_electrode_name is None or electrode_zones is None:
        return shortened

    route = np.asarray(route, dtype=float)
    raw_clear = not _path_hits_foreign_electrode_zone(
        route, path_electrode_name, electrode_zones
    )
    short_clear = not _path_hits_foreign_electrode_zone(
        shortened, path_electrode_name, electrode_zones
    )
    if short_clear:
        return shortened
    if raw_clear:
        return route
    return route


def _refine_hull_route(
    start,
    end,
    forbidden,
    terminal_zones,
    n_points,
    forward,
    path_electrode_name=None,
    electrode_zones=None,
):
    candidate = _build_hull_route(start, end, forbidden, terminal_zones, n_points, forward)
    if candidate is None:
        return None
    return _shortcut_route_if_electrode_clear(
        candidate,
        forbidden,
        start,
        end,
        n_points,
        path_electrode_name=path_electrode_name,
        electrode_zones=electrode_zones,
    )


def _hull_sample_count(path, start, end, min_points=50):
    """Enough samples for long electrode→entry chords (right-hub routes)."""
    path = np.asarray(path, dtype=float)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    chord_len = float(np.linalg.norm(end - start))
    return max(min_points, len(path), int(chord_len / 2.5))


def _hull_route_points(start, end, boundary, forward=True):
    m = len(boundary)
    i0 = min(range(m), key=lambda i: np.linalg.norm(boundary[i] - start))
    i1 = min(range(m), key=lambda i: np.linalg.norm(boundary[i] - end))

    if forward:
        step, count = 1, (i1 - i0) % m
    else:
        step, count = -1, (i0 - i1) % m

    pts = []
    idx = i0
    for _ in range(count + 1):
        pts.append(boundary[idx])
        idx = (idx + step) % m
    return pts


def _trim_hull_to_terminal(hull_route, terminal_zones):
    for idx, pt in enumerate(hull_route):
        for term_zone in terminal_zones.values():
            if term_zone.contains(Point(pt)):
                term_center = np.array(term_zone.centroid.coords[0], dtype=float)
                return hull_route[:idx + 1] + [term_center], term_center
    return hull_route, None


def _resample_polyline(points, n_points):
    full = [np.asarray(points[0], dtype=float)]
    for pt in points[1:]:
        full.append(np.asarray(pt, dtype=float))

    cum = [0.0]
    for k in range(1, len(full)):
        cum.append(cum[-1] + np.linalg.norm(full[k] - full[k - 1]))
    total = cum[-1]
    if total <= 0:
        return np.tile(full[0], (n_points, 1))

    ds = np.linspace(0, total, n_points)
    out = []
    seg = 0
    for d in ds:
        while seg < len(cum) - 2 and cum[seg + 1] < d:
            seg += 1
        d0, d1 = cum[seg], cum[seg + 1]
        p0, p1 = full[seg], full[seg + 1]
        t = 0 if d1 == d0 else (d - d0) / (d1 - d0)
        out.append(p0 * (1 - t) + p1 * t)
    return np.vstack(out)


def _build_hull_route(start, end, forbidden, terminal_zones, n_points, forward):
    boundary = _forbidden_boundary_for_route(start, end, forbidden)
    if boundary is None:
        return None

    hull_route, term_center = _trim_hull_to_terminal(
        _hull_route_points(start, end, boundary, forward=forward),
        terminal_zones,
    )
    route_end = term_center if term_center is not None else end
    return _resample_polyline([start] + hull_route + [route_end], n_points)


def _path_has_foreign_electrode_violation(path, path_electrode_name, electrode_zones):
    return _path_hits_foreign_electrode_zone(path, path_electrode_name, electrode_zones)


def _chord_blocked_by_electrodes(start, end, path_electrode_name, electrode_zones):
    forbidden = _build_forbidden_region(
        path=np.array([start, end]),
        path_electrode_name=path_electrode_name,
        other_paths=[],
        electrode_zones=electrode_zones,
        electrodes_only=True,
    )
    return _segment_blocked_by_forbidden(start, end, forbidden)


def _classify_route_bypass_side(route, blocker_center, strip_tangent):
    """
    'left' / 'right': which side of the blocker the route passes (strip frame).

    Negative strip projection = low-slot / left side of the terminal strip.
    """
    if strip_tangent is None or blocker_center is None:
        return None
    route = np.asarray(route, dtype=float)
    blocker_center = np.asarray(blocker_center, dtype=float)
    strip_tangent = np.asarray(strip_tangent, dtype=float)
    norm = float(np.linalg.norm(strip_tangent))
    if norm < 1e-9:
        return None
    strip_tangent = strip_tangent / norm
    closest = route[int(np.argmin(np.linalg.norm(route - blocker_center, axis=1)))]
    along = float(np.dot(closest - blocker_center, strip_tangent))
    if abs(along) < 1e-6:
        return None
    return 'left' if along < 0 else 'right'


def _hull_directions_for_bypass_policy(bypass_side):
    """
    Map strip bypass side to hull sweep direction.

    Policy (terminal strip frame):
      - bypass left of blocker  → counter-clockwise (CCW) hull only
      - bypass right of blocker → clockwise (CW) hull only
      - no preference           → try both
    """
    if bypass_side == 'left':
        return [(False, 'ccw')]
    if bypass_side == 'right':
        return [(True, 'cw')]
    return [(True, 'cw'), (False, 'ccw')]


def greedy_electrode_avoidance(
    path,
    path_electrode_name,
    electrode_zones,
    terminal_zones,
    target_terminal_pos,
    n_points=None,
    quiet=False,
    bypass_side=None,
    blocker_center=None,
    strip_tangent=None,
    other_paths=None,
    spacing_mode=False,
):
    """
    Hull routing around foreign electrode zones (phase 1).

    spacing_mode=True adds buffered sibling traces to the forbidden region and uses
    PHASE2_GREEDY_SPACING_* keep-outs during phase 2 ordered resolution.
    """
    path = np.asarray(path, dtype=float)
    start = np.asarray(path[0], dtype=float)
    end = np.asarray(target_terminal_pos, dtype=float)
    n_points = _hull_sample_count(path, start, end, min_points=n_points or 50)

    if spacing_mode:
        sibling_paths = other_paths or []
        forbidden = _build_forbidden_region(
            path,
            path_electrode_name,
            other_paths=sibling_paths,
            electrode_zones=electrode_zones,
            electrodes_only=False,
            electrode_clearance_multiplier=PHASE2_GREEDY_SPACING_ELECTRODE_MULTIPLIER,
            trace_buffer_half_width=PHASE2_GREEDY_TRACE_HALF_SEPARATION,
        )
    else:
        forbidden = _build_forbidden_region(
            path,
            path_electrode_name,
            other_paths=[],
            electrode_zones=electrode_zones,
            electrodes_only=True,
        )

    if spacing_mode:
        chord_clear = not _segment_blocked_by_forbidden(start, end, forbidden)
    else:
        chord_clear = not _chord_blocked_by_electrodes(
            start, end, path_electrode_name, electrode_zones
        )

    if chord_clear:
        candidate = np.linspace(start, end, n_points)
        if not _path_has_foreign_electrode_violation(
            candidate, path_electrode_name, electrode_zones
        ):
            return candidate

    if forbidden is None or forbidden.is_empty:
        return path

    best_path = path
    policy_side = bypass_side

    for forward, label in _hull_directions_for_bypass_policy(policy_side):
        candidate = _refine_hull_route(
            start,
            end,
            forbidden,
            terminal_zones,
            n_points,
            forward,
            path_electrode_name=path_electrode_name,
            electrode_zones=electrode_zones,
        )
        if candidate is None:
            continue
        candidate_violation = _path_has_foreign_electrode_violation(
            candidate, path_electrode_name, electrode_zones
        )
        if not quiet:
            side = _classify_route_bypass_side(candidate, blocker_center, strip_tangent)
            print(
                f"[ELECTRODE] {path_electrode_name} {label} "
                f"(policy={bypass_side or 'both'}): "
                f"electrode_violation={candidate_violation}, route_side={side}"
            )
        if not candidate_violation:
            return candidate

    return best_path
