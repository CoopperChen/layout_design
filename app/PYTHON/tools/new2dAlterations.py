import numpy as np
import json
import os
from collections import defaultdict
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point
from shapely.ops import unary_union, nearest_points, substring
from scipy.spatial import KDTree
from scipy.interpolate import splprep, splev
import random
import datetime

from app.polish.phase2_profile import get_phase2_profile, profile_step

_pv_module = None
_init_conn_module = None


def _pyvista():
    """Lazy import — pyvista is only needed for UV grid / 3D visualization."""
    global _pv_module
    if _pv_module is None:
        import pyvista as pv
        _pv_module = pv
    return _pv_module


def _init_conn():
    """Lazy import — avoids loading pyvista when only 2D collision code is used."""
    global _init_conn_module
    if _init_conn_module is None:
        from PYTHON.tools import initiate3DConnections as init_conn
        _init_conn_module = init_conn
    return _init_conn_module


# --------------------------
# Configuration
# --------------------------

# Global padding configuration for bounds calculation
X_BOUNDS_PADDING_FACTOR = 0.15  # 15% of head width - adjust this single value to control all padding

# Minimum centerline clearance between traces in 2D projection units (wire spacing).
MIN_PATH_SEPARATION = 4.0
# Minimum spacing between trace entry points where they intersect a terminal zone.
TERMINAL_ZONE_ENTRY_MIN_SEPARATION = MIN_PATH_SEPARATION
# Closest-neighbor band at terminal entry: penalize when nearest co-terminal entry is
# closer than MIN or farther than MAX (ideal fan-in spacing lives inside the band).
TERMINAL_ENTRY_CLOSEST_MIN = 3.0
TERMINAL_ENTRY_CLOSEST_MAX = 5.0
# Fixed arc spacing between assigned entry slots on the terminal boundary (init).
TERMINAL_ENTRY_SLOT_SPACING = 4.5
# Phase 1 greedy: accept per-trace improvement when global count is flat (oscillation caps).
PHASE1_LOCAL_ACCEPT_MAX_PER_ROUND = 5
PHASE1_LOCAL_ACCEPT_MAX_PER_ELECTRODE = 1
# Phase 1 electrode-clearance pass: inflation on foreign electrode discs in greedy forbidden region.
PHASE1_GREEDY_ELECTRODE_CLEARANCE_MULTIPLIER = 1.05
# Phase 2: minimum clearance when checking against inner (fixed) traces.
PHASE2_INNER_TRACE_SEPARATION = 4.0
# Phase 2 spacing greedy: inflated electrode keep-outs and trace half-width buffer.
PHASE2_GREEDY_SPACING_ELECTRODE_MULTIPLIER = 1.35
PHASE2_GREEDY_TRACE_HALF_SEPARATION = MIN_PATH_SEPARATION / 2.0
PHASE2_SPACING_MAX_DETOUR_RATIO = 1.8
PHASE2_PAIR_RESOLUTION_MAX_ROUNDS = 4
PHASE2_RANDOM_ATTEMPTS_PER_TRACE = 4
PHASE2_SEPARATION_FOCUS_RANDOM_ATTEMPTS = 12
# Minimum centerline clearance between any two traces along their full route (fitness).
TRACE_SEPARATION_MIN = 4.0
TRACE_SEPARATION_SAMPLE_STEP = 0.25
# Electrode keep-out disc radius = avg electrode distance × this factor (2D projection units).
ELECTRODE_ZONE_FACTOR = 0.05
# Terminal zone radius uses a separate factor in create_zones().
TERMINAL_ZONE_FACTOR = 0.15
# Each unit of shared overlap length contributes this much to the collision score.
OVERLAP_LENGTH_WEIGHT = 1.0
# Weights for the composite collision score / fitness penalty (independent tuning).
CROSSING_SCORE_WEIGHT = 1.0
OVERLAP_SCORE_WEIGHT = OVERLAP_LENGTH_WEIGHT
TERMINAL_SPACING_SCORE_WEIGHT = 1.0
TRACE_SEPARATION_SCORE_WEIGHT = 1.0
ELECTRODE_SCORE_WEIGHT = 1.0
COLLISION_SCORE_EPSILON = 1e-6
# Unified layout penalty (lower is better). GA fitness = -layout_score.
CROSSING_LAYOUT_MULTIPLIER = 10.0
# Extra layout penalty per co-terminal ordered crossing (25 total with generic crossing weight).
SLOT_ORDERED_CROSSING_LAYOUT_MULTIPLIER = 25.0
SLOT_ORDERED_CROSSING_LAYOUT_EXTRA = (
    SLOT_ORDERED_CROSSING_LAYOUT_MULTIPLIER - CROSSING_LAYOUT_MULTIPLIER
)
# Layout penalty per trace that self-intersects or re-enters its electrode/terminal zone.
TRACE_REENTRY_LAYOUT_MULTIPLIER = 20.0
OVERLAP_LAYOUT_MULTIPLIER = 2.0
TRACE_SEPARATION_LAYOUT_MULTIPLIER = 4.0
LENGTH_LAYOUT_WEIGHT = 1.0
ELECTRODE_LAYOUT_WEIGHT = 1.0
PHASE2_SOLUTION_SCORE_THRESHOLD = 1E-6
# Only co-routed paths that share a terminal may overlap along their final approach.
# Large terminal buffers from create_zones() are for visualization/routing — NOT used here.
TERMINAL_MERGE_TAIL_LENGTH_MIN = 4.0
TERMINAL_MERGE_TAIL_LENGTH_FACTOR = 0.2  # × terminal_zone_size from create_zones metadata
TERMINAL_MERGE_PAD = 0.75  # small buffer around the shared tail so endpoints can meet
# Crossing checks use a much smaller co-terminal merge exclusion than overlap / trace-sep.
TERMINAL_CROSSING_MERGE_TAIL_LENGTH_MIN = 1.5
TERMINAL_CROSSING_MERGE_TAIL_FACTOR = 0.06
TERMINAL_CROSSING_MERGE_PAD = 0.2
# Densify terminal tails before intersection tests so sparse polylines cannot hide crossings.
NEAR_TERMINAL_CROSSING_DENSIFY_LENGTH_FACTOR = 3.0
NEAR_TERMINAL_CROSSING_SAMPLE_STEP = 0.5
NEAR_TERMINAL_CROSSING_POINT_DEDUP = 0.3
# Extra layout penalty per crossing inside the near-terminal region (on top of base crossing weight).
NEAR_TERMINAL_CROSSING_LAYOUT_MULTIPLIER = 15.0
# Segment-level supplement is O(segments²) per pair; densified Shapely intersection covers the same cases.
ENABLE_TERMINAL_CROSSING_SUPPLEMENT_SCAN = False
# Skip Phase 2 aggressive rerouting when Phase 1 already reduced the score below this.
AGGRESSIVE_PHASE_MIN_SCORE = 10.0
# Path length / chord length <= this → straight feeder; skip greedy reroute on it.
PATH_LOCKED_MAX_DETOUR_RATIO = 1.08

# --------------------------
# Core Functions
# --------------------------

def is_path_within_bounds(path, x_bounds):
    """Check if path stays within x-axis boundaries"""
    x_coords = path[:,0]
    return np.all(x_coords >= x_bounds[0]) and np.all(x_coords <= x_bounds[1])


def polar_projection(points_3d, cz_pos):
    """Flatten 3D points onto 2D plane with Cz at center"""
    rel_points = points_3d - cz_pos
    r = np.linalg.norm(rel_points, axis=1)
    theta = np.arctan2(rel_points[:,1], rel_points[:,0])
    return np.column_stack([r*np.cos(theta), r*np.sin(theta)])


def build_terminals_2d(electrodes_2d, fiducials, cz_pos, *, mode="inflated"):
    """
    Build 2D terminal anchors for routing and visualization.

    mode:
      inflated — legacy GA layout: hub angle from fiducial, radius pushed to ~1.2× max electrode r.
      fiducial — hub at polar projection of the clicked TERMINAL_* (target-native strip entries).
    """
    terminals_2d = {}
    for term in ["TERMINAL_LEFT", "TERMINAL_RIGHT"]:
        if term not in fiducials:
            continue
        pos = polar_projection(np.array([fiducials[term]]), cz_pos)[0]
        if mode == "fiducial":
            terminals_2d[term] = np.asarray(pos, dtype=float)
            continue
        max_dist = 1.2 * max(np.linalg.norm(list(electrodes_2d.values()), axis=1))
        angle = np.arctan2(pos[1], pos[0])
        terminals_2d[term] = max_dist * np.array([np.cos(angle), np.sin(angle)], dtype=float)
    return terminals_2d


def pin_path_endpoints_2d(path, start, end):
    """Lock the first and last points of a 2D path to canonical electrode/terminal anchors."""
    pinned = np.asarray(path, dtype=float).copy()
    if len(pinned) == 0:
        return pinned
    pinned[0] = np.asarray(start, dtype=float)
    pinned[-1] = np.asarray(end, dtype=float)
    return pinned


def path_end_target(path_entry, terminals_2d):
    """Return the 2D wire end (truncated end if set, else strip entry / terminal anchor)."""
    if path_entry.get('path_end_2d') is not None:
        return np.asarray(path_entry['path_end_2d'], dtype=float)
    if path_entry.get('entry_point_2d') is not None:
        return np.asarray(path_entry['entry_point_2d'], dtype=float)
    if isinstance(path_entry, dict):
        return np.asarray(terminals_2d[path_entry['terminal']], dtype=float)
    raise TypeError("path_entry must be a dict with terminal / entry_point_2d")


def pin_paths_to_layout(paths, path_specs, electrodes_2d, terminals_2d):
    """Pin each path to its electrode start and assigned entry / terminal end."""
    pinned_paths = []
    for path, spec in zip(paths, path_specs):
        if isinstance(spec, dict):
            electrode_name = spec['electrode']
            end = path_end_target(spec, terminals_2d)
        else:
            electrode_name, terminal_name = spec
            end = np.asarray(terminals_2d[terminal_name], dtype=float)
        pinned_paths.append(
            pin_path_endpoints_2d(
                path,
                electrodes_2d[electrode_name],
                end,
            )
        )
    return pinned_paths


def pin_child_paths_2d(child, electrodes_2d, terminals_2d):
    """Pin all modified paths inside a GA child genome."""
    for entry in child['paths']:
        entry['modified_path_2d'] = pin_path_endpoints_2d(
            entry['modified_path_2d'],
            electrodes_2d[entry['electrode']],
            path_end_target(entry, terminals_2d),
        ).tolist()
    return child


def create_zones(electrodes_2d, terminals_2d):
    """Create safety zones with separate size controls for electrodes and terminals"""
    # Calculate average inter-electrode distance
    avg_dist = np.mean([np.linalg.norm(pos) for pos in electrodes_2d.values()])

    electrode_zone_size = avg_dist * ELECTRODE_ZONE_FACTOR
    terminal_zone_size = avg_dist * TERMINAL_ZONE_FACTOR
    
    # Create electrode zones
    electrode_zones = {
        'zones': {},
        'metadata': {
            'electrode_zone_size': electrode_zone_size,
            'terminal_zone_size': terminal_zone_size
        }
    }
    
    for name, pos in electrodes_2d.items():
        zone = Point(pos).buffer(electrode_zone_size)
        electrode_zones['zones'][name] = zone
        electrode_zones['metadata'][name] = {
            'buffer_size': electrode_zone_size,
            'center': pos
        }
    
    # Create terminal zones with different size
    terminal_zones = {
        name: Point(pos).buffer(terminal_zone_size)
        for name, pos in terminals_2d.items()
    }
    
    return electrode_zones, terminal_zones


def _compute_x_bounds(electrodes_2d, terminals_2d, padding_factor=X_BOUNDS_PADDING_FACTOR):
    """Compute x-bounds from both electrodes and terminals so endpoint-preserving paths remain valid."""
    x_positions = [pos[0] for pos in electrodes_2d.values()]
    x_positions.extend(pos[0] for pos in terminals_2d.values())
    head_width = max(x_positions) - min(x_positions)
    x_padding = padding_factor * head_width
    return (min(x_positions) - x_padding, max(x_positions) + x_padding)


def load_zones_for_subject(SUBJECT_ID: int):
    """Build electrode and terminal safety zones for scoring (cached per subject)."""
    if not hasattr(load_zones_for_subject, "_cache"):
        load_zones_for_subject._cache = {}

    if SUBJECT_ID in load_zones_for_subject._cache:
        return load_zones_for_subject._cache[SUBJECT_ID]

    with open(f'data/json/electrode_positions_{SUBJECT_ID}.json') as f:
        electrodes = {k: np.array(v) for k, v in json.load(f).items()}
    with open(f'data/json/fiducials_{SUBJECT_ID}.json') as f:
        fiducials = {k: np.array(v) for k, v in json.load(f).items()}

    cz_pos = electrodes['Cz']
    electrodes_2d = {k: polar_projection(np.array([v]), cz_pos)[0] for k, v in electrodes.items()}
    max_dist = 1.2 * max(np.linalg.norm(list(electrodes_2d.values()), axis=1))
    terminals_2d = {}
    for term in ['TERMINAL_LEFT', 'TERMINAL_RIGHT']:
        if term in fiducials:
            pos = polar_projection(np.array([fiducials[term]]), cz_pos)[0]
            angle = np.arctan2(pos[1], pos[0])
            terminals_2d[term] = max_dist * np.array([np.cos(angle), np.sin(angle)])

    zones = create_zones(electrodes_2d, terminals_2d)
    load_zones_for_subject._cache[SUBJECT_ID] = zones
    return zones


def load_terminal_zones_for_subject(SUBJECT_ID: int):
    """Backward-compatible helper returning terminal zones only."""
    _, terminal_zones = load_zones_for_subject(SUBJECT_ID)
    return terminal_zones


_SUBJECT_LAYOUT_CACHE = {}
_UV_GRID_CACHE = {}
UV_GRID_RESOLUTION = 100


def warm_subject_caches(SUBJECT_ID, electrodes, fiducials, original_paths):
    """Pre-build static per-subject layout + UV grid once at GA start."""
    get_subject_layout(SUBJECT_ID, electrodes, fiducials, original_paths)
    ctx = _SUBJECT_LAYOUT_CACHE[SUBJECT_ID]
    get_cached_uv_grid(SUBJECT_ID, ctx['cz_pos'])


def get_subject_layout(SUBJECT_ID, electrodes=None, fiducials=None, original_paths=None):
    """Cached electrodes_2d, terminals, zones, optimized connections, and x_bounds."""
    if SUBJECT_ID in _SUBJECT_LAYOUT_CACHE:
        return _SUBJECT_LAYOUT_CACHE[SUBJECT_ID]

    if original_paths is None:
        with open(f'data/json/init_connection_paths_{SUBJECT_ID}.json') as f:
            original_paths = json.load(f)
    if electrodes is None:
        with open(f'data/json/electrode_positions_{SUBJECT_ID}.json') as f:
            electrodes = {k: np.array(v) for k, v in json.load(f).items()}
    if fiducials is None:
        with open(f'data/json/fiducials_{SUBJECT_ID}.json') as f:
            fiducials = {k: np.array(v) for k, v in json.load(f).items()}

    init_conn = _init_conn()
    initial_assignments = init_conn.load_or_create_terminal_assignments(
        SUBJECT_ID, original_paths
    )
    optimized = init_conn.select_connections_for_assignments(
        original_paths, initial_assignments, electrodes=electrodes
    )
    cz_pos = electrodes['Cz']
    electrodes_2d = {
        k: polar_projection(np.array([v]), cz_pos)[0] for k, v in electrodes.items()
    }
    terminals_2d = build_terminals_2d(electrodes_2d, fiducials, cz_pos)
    electrode_zones, terminal_zones = load_zones_for_subject(SUBJECT_ID)
    path_electrodes = [conn['electrode'] for conn in optimized]
    path_terminals = [conn['terminal'] for conn in optimized]
    original_paths_2d = []
    for conn in optimized:
        projected = polar_projection(np.array(conn['path_points']), cz_pos)
        original_paths_2d.append(
            pin_path_endpoints_2d(
                projected,
                electrodes_2d[conn['electrode']],
                terminals_2d[conn['terminal']],
            )
        )

    ctx = {
        'optimized': optimized,
        'electrodes_2d': electrodes_2d,
        'terminals_2d': terminals_2d,
        'electrode_zones': electrode_zones,
        'terminal_zones': terminal_zones,
        'cz_pos': cz_pos,
        'x_bounds': _compute_x_bounds(electrodes_2d, terminals_2d),
        'path_electrodes': path_electrodes,
        'path_terminals': path_terminals,
        'original_paths_2d': original_paths_2d,
    }
    _SUBJECT_LAYOUT_CACHE[SUBJECT_ID] = ctx
    return ctx


def get_cached_uv_grid(SUBJECT_ID, cz_pos, resolution=UV_GRID_RESOLUTION):
    """Build the head UV grid once per subject (expensive KDTree pass)."""
    key = (SUBJECT_ID, resolution)
    if key in _UV_GRID_CACHE:
        return _UV_GRID_CACHE[key]
    mesh = _pyvista().read(f"data/cleaned_scans/{SUBJECT_ID}.stl")
    uv_grid = create_uv_grid(mesh, cz_pos, resolution=resolution)
    _UV_GRID_CACHE[key] = uv_grid
    return uv_grid


def find_electrode_collisions(paths, electrode_zones, path_electrodes):
    """Find one marker point per foreign electrode zone intersected by each trace."""
    collisions = []

    for i, (path, electrode_name) in enumerate(zip(paths, path_electrodes)):
        try:
            path_line = _path_to_linestring(path)
            if path_line is None:
                continue

            for name, zone in electrode_zones['zones'].items():
                if name == electrode_name:
                    continue

                try:
                    if not zone.is_valid:
                        continue

                    intersection = path_line.intersection(zone)
                    if intersection.is_empty:
                        continue
                    if intersection.geom_type == 'Point':
                        collisions.append([intersection.x, intersection.y])
                    elif intersection.geom_type == 'MultiPoint':
                        pt = intersection.geoms[0]
                        collisions.append([pt.x, pt.y])
                    elif intersection.geom_type == 'LineString':
                        midpoint = intersection.interpolate(0.5, normalized=True)
                        collisions.append([midpoint.x, midpoint.y])
                    else:
                        centroid = intersection.centroid
                        collisions.append([centroid.x, centroid.y])
                except Exception:
                    continue

        except Exception:
            continue

    return np.array(collisions) if collisions else None


def count_electrode_violations(paths_2d, electrode_zones, path_electrodes):
    """Layout-wide electrode violation count (shared with logs, fitness, and plots)."""
    if electrode_zones is None or path_electrodes is None:
        return 0
    electrode_points = find_electrode_collisions(
        paths_2d, electrode_zones, path_electrodes
    )
    return 0 if electrode_points is None else len(electrode_points)


def count_single_trace_electrode_violations(path, path_electrode_name, electrode_zones):
    """Foreign electrode zone hits for one trace (same rules as layout count)."""
    if electrode_zones is None:
        return 0
    electrode_points = find_electrode_collisions(
        [path], electrode_zones, [path_electrode_name]
    )
    return 0 if electrode_points is None else len(electrode_points)


def _point_in_buffer_zone(coord, zone):
    pt = Point(coord)
    return zone.covers(pt) or zone.touches(pt)


def _path_segment_penetrates_zone(p0, p1, zone, endpoint_tolerance=1e-6):
    """True when a segment crosses zone interior (endpoint-only boundary touches excluded)."""
    if zone is None:
        return False

    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    if np.linalg.norm(p1 - p0) <= endpoint_tolerance:
        return _point_in_buffer_zone(p0, zone)

    seg = LineString([p0, p1])
    if not seg.is_valid or seg.is_empty:
        return False

    inter = seg.intersection(zone)
    if inter.is_empty:
        return False
    if inter.geom_type == 'Point':
        pt = np.array([inter.x, inter.y], dtype=float)
        if (
            np.linalg.norm(pt - p0) <= endpoint_tolerance
            or np.linalg.norm(pt - p1) <= endpoint_tolerance
        ):
            return False
        return True
    if inter.geom_type == 'MultiPoint':
        for geom in inter.geoms:
            pt = np.array([geom.x, geom.y], dtype=float)
            if (
                np.linalg.norm(pt - p0) > endpoint_tolerance
                and np.linalg.norm(pt - p1) > endpoint_tolerance
            ):
                return True
        return False
    return float(getattr(inter, 'length', 1.0)) > endpoint_tolerance


def _path_segment_uses_terminal_zone(p0, p1, zone):
    """True when a segment enters or runs inside the terminal zone."""
    if _point_in_buffer_zone(p1, zone):
        return True
    return _path_segment_penetrates_zone(p0, p1, zone)


def _terminal_zone_tail_start_segment(path, zone):
    """
    Index of the first segment in the final contiguous terminal approach.

    Segments from this index to the end may intersect the terminal zone; earlier
    segments must not. When the path ends outside the zone there is no terminal
    tail and all segments are treated as body.
    """
    coords = _clean_path_coords(path)
    if len(coords) < 2 or zone is None:
        return len(coords) - 1

    if not _point_in_buffer_zone(coords[-1], zone):
        return len(coords) - 1

    n_segments = len(coords) - 1
    tail_start = n_segments
    for seg_idx in range(n_segments - 1, -1, -1):
        if _path_segment_uses_terminal_zone(coords[seg_idx], coords[seg_idx + 1], zone):
            tail_start = seg_idx
        else:
            break
    return tail_start


def path_has_terminal_zone_reentry(path, zone):
    """
    True when a trace intersects its terminal zone before the final approach tail.

    Uses segment–zone tests so chord shortcuts through the buffer are caught even
    when no sampled vertex lies inside the zone.
    """
    if zone is None:
        return False

    coords = _clean_path_coords(path)
    if len(coords) < 2:
        return False

    tail_start = _terminal_zone_tail_start_segment(path, zone)
    if tail_start >= len(coords) - 1:
        return False

    for seg_idx in range(0, tail_start):
        if _path_segment_uses_terminal_zone(
            coords[seg_idx], coords[seg_idx + 1], zone
        ):
            return True
    return False


def path_has_zone_reentry(path, zone, allow_start_inside=False):
    """
    True when a path leaves a zone and enters it again.

    Own electrode zones allow starting inside; assigned terminal zones do not.
    """
    if zone is None:
        return False

    if not allow_start_inside:
        return path_has_terminal_zone_reentry(path, zone)

    coords = _clean_path_coords(path)
    if len(coords) < 2:
        return False

    outside_to_inside_count = 0
    left_zone_after_being_inside = False
    was_inside = _point_in_buffer_zone(coords[0], zone)

    for idx in range(1, len(coords)):
        now_inside = _point_in_buffer_zone(coords[idx], zone)
        if now_inside and not was_inside:
            outside_to_inside_count += 1
        if was_inside and not now_inside:
            left_zone_after_being_inside = True
        was_inside = now_inside

    if left_zone_after_being_inside and outside_to_inside_count >= 1:
        return True
    return False


def path_self_intersects(path):
    """True when non-adjacent segments of the same trace cross."""
    line = _path_to_linestring(path)
    if line is None:
        return False
    return not line.is_simple


def path_has_trace_reentry(
    path,
    path_electrode_name,
    terminal_name,
    electrode_zones,
    terminal_zones,
):
    """True when a trace self-intersects or re-enters its own electrode / terminal zone."""
    if path_self_intersects(path):
        return True

    if (
        electrode_zones
        and path_electrode_name
        and path_electrode_name in electrode_zones.get('zones', {})
    ):
        own_zone = electrode_zones['zones'][path_electrode_name]
        if path_has_zone_reentry(path, own_zone, allow_start_inside=True):
            return True

    if terminal_zones and terminal_name and terminal_name in terminal_zones:
        terminal_zone = terminal_zones[terminal_name]
        if path_has_terminal_zone_reentry(path, terminal_zone):
            return True

    return False


def count_trace_reentries(
    paths,
    path_electrodes,
    path_terminals,
    electrode_zones,
    terminal_zones,
):
    """Number of traces that violate the no-reentry policy."""
    count = 0
    for idx, path in enumerate(paths):
        electrode = path_electrodes[idx] if path_electrodes else None
        terminal = path_terminals[idx] if path_terminals else None
        if path_has_trace_reentry(
            path, electrode, terminal, electrode_zones, terminal_zones
        ):
            count += 1
    return count


def trace_indices_with_electrode_violations(paths, electrode_zones, path_electrodes):
    """Path indices whose route crosses a foreign electrode zone, worst first."""
    scored = []
    for idx, (path, name) in enumerate(zip(paths, path_electrodes)):
        n = count_single_trace_electrode_violations(path, name, electrode_zones)
        if n > 0:
            scored.append((n, idx))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [idx for _, idx in scored]


def _foreign_electrodes_hit_by_trace(path, path_electrode_name, electrode_zones):
    """Foreign electrode names whose keep-out zone this trace intersects."""
    path_line = _path_to_linestring(path)
    if path_line is None:
        return []

    hit_names = []
    for name, zone in electrode_zones['zones'].items():
        if name == path_electrode_name:
            continue
        try:
            if zone.is_valid and path_line.intersects(zone):
                hit_names.append(name)
        except Exception:
            continue
    return hit_names


def _primary_blocking_electrode_for_trace(
    path, path_electrode_name, electrode_zones, start, end
):
    """Blocker for bypass policy: chord blocker, else nearest foreign zone on the path."""
    blocker = _primary_blocking_electrode_on_chord(
        start, end, path_electrode_name, electrode_zones
    )
    if blocker is not None:
        return blocker

    start = np.asarray(start, dtype=float)
    best_name = None
    best_dist = float('inf')
    for name in _foreign_electrodes_hit_by_trace(path, path_electrode_name, electrode_zones):
        center = np.asarray(electrode_zones['metadata'][name]['center'], dtype=float)
        dist = float(np.linalg.norm(center - start))
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name


def _json_safe_number(value):
    """Convert numpy scalars to native int/float for JSON."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def analysis_to_collision_metrics(analysis):
    """Strip visualization points for JSON persistence."""
    return {
        key: _json_safe_number(value)
        for key, value in analysis.items()
        if key != 'points'
    }


def _empty_path_metrics(electrode_violations=0):
    """Minimal metrics dict when only electrode violations are computed."""
    return {
        'crossing_count': 0,
        'overlap_length': 0.0,
        'far_crossing_count': 0,
        'far_overlap_length': 0.0,
        'near_crossing_count': 0,
        'near_overlap_length': 0.0,
        'terminal_zone_spacing_deficit': 0.0,
        'terminal_zone_spacing_deficit_normalized': 0.0,
        'terminal_spacing_violation_pairs': 0,
        'trace_separation_deficit': 0.0,
        'trace_separation_deficit_normalized': 0.0,
        'trace_separation_violations': 0,
        'min_trace_separation': float('inf'),
        'trace_separation_min_required': TRACE_SEPARATION_MIN,
        'clearance_deficit': 0.0,
        'electrode_violations': electrode_violations,
        'collision_score': float(electrode_violations),
        'slot_ordered_crossing_count': 0,
        'trace_reentry_count': 0,
        'min_pairwise_distance': float('inf'),
        'min_terminal_entry_distance': float('inf'),
        'min_closest_neighbor_entry_distance': float('inf'),
        'terminal_merge_tail_length': TERMINAL_MERGE_TAIL_LENGTH_MIN,
        'points': None,
    }


def _clean_path_coords(path):
    """Remove consecutive duplicate vertices so LineString geometry is valid."""
    arr = np.asarray(path, dtype=float)
    if len(arr) < 2:
        return arr
    keep = [0]
    for i in range(1, len(arr)):
        if np.linalg.norm(arr[i] - arr[keep[-1]]) > 1e-10:
            keep.append(i)
    if len(keep) < 2:
        return arr[[0, -1]]
    return arr[keep]


def _path_to_linestring(path):
    coords = _clean_path_coords(path)
    if len(coords) < 2:
        return None
    line = LineString(coords)
    return line if line.is_valid and not line.is_empty else None


def _terminal_merge_tail_length(electrode_zones=None):
    """Length of each path's terminal approach that may overlap with a co-terminal sibling."""
    if electrode_zones and 'metadata' in electrode_zones:
        terminal_zone_size = electrode_zones['metadata'].get('terminal_zone_size')
        if terminal_zone_size:
            return max(
                TERMINAL_MERGE_TAIL_LENGTH_MIN,
                terminal_zone_size * TERMINAL_MERGE_TAIL_LENGTH_FACTOR,
            )
    return TERMINAL_MERGE_TAIL_LENGTH_MIN


def _crossing_merge_tail_length(electrode_zones=None):
    """Short co-terminal exclusion used only for crossing detection (stricter than overlap merge)."""
    if electrode_zones and 'metadata' in electrode_zones:
        terminal_zone_size = electrode_zones['metadata'].get('terminal_zone_size')
        if terminal_zone_size:
            return max(
                TERMINAL_CROSSING_MERGE_TAIL_LENGTH_MIN,
                terminal_zone_size * TERMINAL_CROSSING_MERGE_TAIL_FACTOR,
            )
    return TERMINAL_CROSSING_MERGE_TAIL_LENGTH_MIN


def _build_pair_crossing_merge_union(
    path_i, path_j, terminal_i, terminal_j, electrode_zones=None
):
    """Co-terminal merge exclusion for crossing detection (tighter than overlap merge)."""
    return _build_pair_merge_union(
        path_i,
        path_j,
        terminal_i,
        terminal_j,
        _crossing_merge_tail_length(electrode_zones),
        merge_pad=TERMINAL_CROSSING_MERGE_PAD,
    )


def _densify_path_terminal_tail(path, tail_length, sample_step=NEAR_TERMINAL_CROSSING_SAMPLE_STEP):
    """Add samples along the terminal tail so geometric crossings are not missed."""
    coords = _clean_path_coords(path)
    if len(coords) < 2:
        return coords

    line = _path_to_linestring(coords)
    if line is None:
        return coords
    if line.length <= tail_length + 1e-9:
        return _densify_polyline_coords(coords, sample_step)

    far_coords, near_coords = _split_path_far_near(path, tail_length)
    if near_coords is None or len(near_coords) < 2:
        return coords

    near_line = LineString(near_coords)
    n = max(len(near_coords), int(np.ceil(float(near_line.length) / sample_step)) + 1)
    dense_near = np.array(
        [near_line.interpolate(t, normalized=True).coords[0] for t in np.linspace(0.0, 1.0, n)],
        dtype=float,
    )
    if far_coords is None or len(far_coords) == 0:
        return dense_near
    if len(far_coords) == 1:
        return np.vstack([far_coords, dense_near[1:]]) if len(dense_near) > 1 else far_coords
    return np.vstack([far_coords[:-1], dense_near])


def _densify_polyline_coords(coords, sample_step):
    """Resample an entire polyline at fixed arc-length spacing."""
    coords = _clean_path_coords(coords)
    line = _path_to_linestring(coords)
    if line is None or line.length <= sample_step:
        return coords
    n = max(len(coords), int(np.ceil(float(line.length) / sample_step)) + 1)
    return np.array(
        [line.interpolate(t, normalized=True).coords[0] for t in np.linspace(0.0, 1.0, n)],
        dtype=float,
    )


def _path_for_crossing_detection(path, terminal_name, terminal_zones, electrode_zones):
    """Return a path polyline with a densified terminal tail for intersection tests."""
    densify_length = (
        _crossing_merge_tail_length(electrode_zones)
        * NEAR_TERMINAL_CROSSING_DENSIFY_LENGTH_FACTOR
    )
    return _densify_path_terminal_tail(path, densify_length)


def _build_crossing_detection_path_cache(paths, path_terminals, electrode_zones):
    """Pre-densify terminal tails once per layout (avoid O(N²) re-densify in pair loops)."""
    cache = {}
    for idx, path in enumerate(paths):
        terminal = (
            path_terminals[idx]
            if path_terminals is not None and idx < len(path_terminals)
            else None
        )
        cache[idx] = _path_for_crossing_detection(
            path, terminal, None, electrode_zones
        )
    return cache


def _dense_path_for_crossing(
    path,
    terminal_name,
    terminal_zones,
    electrode_zones,
    dense_path_cache=None,
    path_idx=None,
):
    if dense_path_cache is not None and path_idx is not None:
        return dense_path_cache[path_idx]
    return _path_for_crossing_detection(path, terminal_name, terminal_zones, electrode_zones)


def _pair_terminal_crossing_focus_region(
    path_a, path_b, terminal_a, terminal_b, terminal_zones, electrode_zones
):
    """Region around terminal hubs where segment-level crossing scans are enforced."""
    regions = []
    focus_tail = (
        _crossing_merge_tail_length(electrode_zones)
        * NEAR_TERMINAL_CROSSING_DENSIFY_LENGTH_FACTOR
    )
    for path, terminal in ((path_a, terminal_a), (path_b, terminal_b)):
        if terminal_zones and terminal in terminal_zones:
            regions.append(terminal_zones[terminal])
        tail = _get_path_tail_linestring(path, focus_tail)
        if tail is not None:
            buffered = tail.buffer(TERMINAL_CROSSING_MERGE_PAD * 2.0)
            if not buffered.is_empty:
                regions.append(buffered)
    if not regions:
        return None
    return unary_union(regions)


def _is_duplicate_xy_point(point, existing_points, tolerance=NEAR_TERMINAL_CROSSING_POINT_DEDUP):
    if not existing_points:
        return False
    pt = np.asarray(point, dtype=float)
    for other in existing_points:
        if np.linalg.norm(pt - np.asarray(other, dtype=float)) <= tolerance:
            return True
    return False


def _segment_pair_crossing_point(p0, p1, q0, q1, endpoint_tolerance=1e-6):
    """Return a single proper crossing point for two segments, or None."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    if np.linalg.norm(p1 - p0) <= endpoint_tolerance or np.linalg.norm(q1 - q0) <= endpoint_tolerance:
        return None

    inter = LineString([p0, p1]).intersection(LineString([q0, q1]))
    if inter.is_empty or inter.geom_type != 'Point':
        return None

    pt = np.array([inter.x, inter.y], dtype=float)
    if (
        np.linalg.norm(pt - p0) <= endpoint_tolerance
        or np.linalg.norm(pt - p1) <= endpoint_tolerance
        or np.linalg.norm(pt - q0) <= endpoint_tolerance
        or np.linalg.norm(pt - q1) <= endpoint_tolerance
    ):
        return None
    return pt


def _supplement_terminal_crossing_points(
    path_a,
    path_b,
    terminal_a,
    terminal_b,
    terminal_zones,
    electrode_zones,
    exclude_geom,
    existing_points,
):
    """
    Segment-level crossing scan near terminals.

    Catches X-junctions between sparse path samples that Shapely line intersections
    can miss when coarse polyline chords jump over a partner segment.
    """
    focus = _pair_terminal_crossing_focus_region(
        path_a, path_b, terminal_a, terminal_b, terminal_zones, electrode_zones
    )
    if focus is None or focus.is_empty:
        return []

    coords_a = _clean_path_coords(path_a)
    coords_b = _clean_path_coords(path_b)
    found = []
    for i in range(len(coords_a) - 1):
        seg_a = LineString([coords_a[i], coords_a[i + 1]])
        if not seg_a.intersects(focus):
            continue
        for j in range(len(coords_b) - 1):
            pt = _segment_pair_crossing_point(
                coords_a[i], coords_a[i + 1], coords_b[j], coords_b[j + 1]
            )
            if pt is None:
                continue
            if exclude_geom is not None and not _point_outside_terminal_merge(pt, exclude_geom):
                continue
            point = Point(pt)
            if not (focus.covers(point) or focus.touches(point) or focus.distance(point) < 1e-6):
                continue
            if _is_duplicate_xy_point(pt, existing_points) or _is_duplicate_xy_point(pt, found):
                continue
            found.append(pt.tolist())
    return found


def _collect_points_from_geometry(geom):
    points = []
    if geom is None or geom.is_empty:
        return points
    if geom.geom_type == 'Point':
        points.append([geom.x, geom.y])
    elif geom.geom_type == 'MultiPoint':
        for pt in geom.geoms:
            points.append([pt.x, pt.y])
    elif geom.geom_type in ('LineString', 'MultiLineString', 'GeometryCollection'):
        for sub in getattr(geom, 'geoms', [geom]):
            points.extend(_collect_points_from_geometry(sub))
    return points


def _compute_path_pair_crossing_geometry(
    path_a,
    path_b,
    terminal_a,
    terminal_b,
    terminal_zones,
    electrode_zones,
    supplemental_terminal_scan=None,
    dense_path_cache=None,
    path_idx_a=None,
    path_idx_b=None,
    use_dense_paths=True,
):
    """
    Intersection geometry for a path pair using crossing-tuned merge exclusion,
    densified terminal tails, and optional near-terminal segment scan.
    """
    if supplemental_terminal_scan is None:
        supplemental_terminal_scan = ENABLE_TERMINAL_CROSSING_SUPPLEMENT_SCAN
    if use_dense_paths:
        dense_a = _dense_path_for_crossing(
            path_a, terminal_a, terminal_zones, electrode_zones, dense_path_cache, path_idx_a
        )
        dense_b = _dense_path_for_crossing(
            path_b, terminal_b, terminal_zones, electrode_zones, dense_path_cache, path_idx_b
        )
    else:
        dense_a = path_a
        dense_b = path_b
    line_a = _path_to_linestring(dense_a)
    line_b = _path_to_linestring(dense_b)
    if line_a is None or line_b is None:
        return LineString(), None, []

    pair_merge_union = _build_pair_crossing_merge_union(
        path_a, path_b, terminal_a, terminal_b, electrode_zones
    )
    if not line_a.intersects(line_b):
        inter = LineString()
    else:
        inter = _clip_outside_terminal_merge(line_a.intersection(line_b), pair_merge_union)

    points = _collect_points_from_geometry(inter)
    if supplemental_terminal_scan:
        extra = _supplement_terminal_crossing_points(
            dense_a,
            dense_b,
            terminal_a,
            terminal_b,
            terminal_zones,
            electrode_zones,
            pair_merge_union,
            points,
        )
        if extra:
            points.extend(extra)
    return inter, pair_merge_union, points


def _infer_path_terminals(paths_2d, terminal_zones):
    """Assign each path to the terminal nearest its endpoint (fallback when metadata is missing)."""
    if not terminal_zones:
        return [None] * len(paths_2d)

    terminal_names = list(terminal_zones.keys())
    terminal_centers = {
        name: np.array([terminal_zones[name].centroid.x, terminal_zones[name].centroid.y])
        for name in terminal_names
    }
    assignments = []
    for path in paths_2d:
        end = np.asarray(path, dtype=float)[-1]
        nearest = min(
            terminal_names,
            key=lambda name: np.linalg.norm(end - terminal_centers[name]),
        )
        assignments.append(nearest)
    return assignments


def _get_path_tail_linestring(path_coords, tail_length):
    """Return the final `tail_length` units of a path as a LineString."""
    coords = _clean_path_coords(path_coords)
    if len(coords) < 2:
        return None

    line = LineString(coords)
    if not line.is_valid or line.is_empty:
        return None
    if line.length <= tail_length + 1e-9:
        return line

    start_dist = max(0.0, line.length - tail_length)
    tail = substring(line, start_dist, line.length)
    return tail if tail.is_valid and not tail.is_empty else None


def _build_pair_merge_union(path_i, path_j, terminal_i, terminal_j, tail_length, merge_pad=TERMINAL_MERGE_PAD):
    """
    Build an exclusion region only when two paths share the same terminal.

    Paths routed to different terminals are fully checked, even near a terminal anchor.
    """
    if terminal_i is None or terminal_j is None or terminal_i != terminal_j:
        return None

    tail_regions = []
    for path in (path_i, path_j):
        tail = _get_path_tail_linestring(path, tail_length)
        if tail is not None:
            tail_regions.append(tail.buffer(merge_pad))

    if not tail_regions:
        return None
    return unary_union(tail_regions)


def _clip_outside_near_terminal(geometry, near_terminal_union):
    """Remove geometry inside the near-terminal approach zone(s)."""
    if geometry is None or geometry.is_empty:
        return geometry
    if near_terminal_union is None or near_terminal_union.is_empty:
        return geometry
    try:
        clipped = geometry.difference(near_terminal_union)
        return clipped if not clipped.is_empty else geometry.__class__()
    except Exception:
        return geometry


def _build_near_terminal_region_union(paths_2d, tail_length, pad=TERMINAL_MERGE_PAD):
    """Union of buffered tail segments — the near-terminal zone (phase 2 edit region)."""
    regions = []
    for path in paths_2d:
        tail = _get_path_tail_linestring(path, tail_length)
        if tail is not None:
            buffered = tail.buffer(pad)
            if not buffered.is_empty:
                regions.append(buffered)
    if not regions:
        return None
    return unary_union(regions)


def _split_path_far_near(path, tail_length):
    """
    Split path into (far_coords, near_coords) at tail_length from terminal end.

    Junction point is shared: far_coords[-1] == near_coords[0].
    """
    coords = _clean_path_coords(path)
    if len(coords) < 2:
        return None, np.asarray(coords, dtype=float)

    line = LineString(coords)
    if not line.is_valid or line.is_empty:
        return None, np.asarray(coords, dtype=float)
    if line.length <= tail_length + 1e-9:
        return None, np.asarray(coords, dtype=float)

    split_dist = max(0.0, line.length - tail_length)
    far_line = substring(line, 0.0, split_dist)
    near_line = substring(line, split_dist, line.length)
    if far_line is None or near_line is None or far_line.is_empty or near_line.is_empty:
        return None, np.asarray(coords, dtype=float)

    far_coords = np.asarray(far_line.coords, dtype=float)
    near_coords = np.asarray(near_line.coords, dtype=float)
    if len(far_coords) < 1 or len(near_coords) < 2:
        return None, np.asarray(coords, dtype=float)
    return far_coords, near_coords


def _split_path_at_terminal_zone(path, terminal_name, terminal_zones):
    """
    Split a path into body vs final terminal approach using segment–zone tests.

    Returns (outside, inside). Only the final contiguous terminal tail is locked;
    earlier segment crossings through the zone are kept in the body so reentry is
    visible to validators rather than hidden inside the locked tail.
    """
    if terminal_name not in terminal_zones:
        coords = _clean_path_coords(path)
        return np.asarray(coords, dtype=float), None

    zone = terminal_zones[terminal_name]
    coords = _clean_path_coords(path)
    if len(coords) < 2:
        return np.asarray(coords, dtype=float), None

    tail_start_seg = _terminal_zone_tail_start_segment(path, zone)
    if tail_start_seg >= len(coords) - 1:
        return np.asarray(coords, dtype=float), None

    inside_start = tail_start_seg
    outside = np.asarray(coords[: inside_start + 1], dtype=float)
    inside = np.asarray(coords[inside_start:], dtype=float)
    if len(outside) < 2:
        return None, inside
    return outside, inside


def _combine_path_outside_locked_terminal_tail(outside, inside):
    """Splice a mutated outside segment back onto an unchanged terminal-zone tail."""
    outside = np.asarray(outside, dtype=float)
    inside = np.asarray(inside, dtype=float)
    if len(outside) == 0:
        return inside
    if len(inside) == 0:
        return outside
    if len(outside) == 1:
        return np.vstack([outside, inside[1:]]) if len(inside) > 1 else outside
    return np.vstack([outside[:-1], inside])


def _lock_terminal_zone_tail(
    original_path,
    trial_path,
    terminal_name,
    terminal_zones,
    start,
    end,
):
    """Keep the original terminal-zone tail; use only the trial path outside the zone."""
    _, locked_inside = _split_path_at_terminal_zone(
        original_path, terminal_name, terminal_zones
    )
    if locked_inside is None:
        return np.asarray(trial_path, dtype=float)

    trial_outside, _ = _split_path_at_terminal_zone(
        trial_path, terminal_name, terminal_zones
    )
    if trial_outside is None or len(trial_outside) < 2:
        return pin_path_endpoints_2d(np.asarray(original_path, dtype=float), start, end)

    zone = terminal_zones.get(terminal_name)
    if zone is not None and path_has_terminal_zone_reentry(
        pin_path_endpoints_2d(trial_outside, start, trial_outside[-1]), zone
    ):
        return pin_path_endpoints_2d(np.asarray(original_path, dtype=float), start, end)

    junction = np.asarray(trial_outside[-1], dtype=float)
    trial_outside[-1] = junction
    combined = _combine_path_outside_locked_terminal_tail(trial_outside, locked_inside)
    combined = pin_path_endpoints_2d(combined, start, end)
    if zone is not None and path_has_terminal_zone_reentry(combined, zone):
        return pin_path_endpoints_2d(np.asarray(original_path, dtype=float), start, end)
    return combined


def _greedy_spacing_outside_terminal_zone(
    path,
    terminal_name,
    terminal_zones,
    electrode_name,
    electrode_zones,
    start,
    end,
    partner_path,
):
    """Run spacing greedy on the path segment outside the terminal zone only."""
    from PYTHON.GA import greed

    outside, inside = _split_path_at_terminal_zone(path, terminal_name, terminal_zones)
    if inside is None or outside is None or len(outside) < 2:
        return None

    junction = np.asarray(outside[-1], dtype=float)
    rerouted = greed.greedy_electrode_avoidance(
        outside,
        electrode_name,
        electrode_zones,
        terminal_zones,
        target_terminal_pos=junction,
        n_points=max(len(outside), 50),
        quiet=True,
        spacing_mode=True,
        other_paths=[partner_path],
    )
    rerouted = pin_path_endpoints_2d(rerouted, start, junction)
    zone = terminal_zones[terminal_name]
    if path_has_terminal_zone_reentry(rerouted, zone):
        return None
    combined = _combine_path_outside_locked_terminal_tail(rerouted, inside)
    combined = pin_path_endpoints_2d(combined, start, end)
    if path_has_terminal_zone_reentry(combined, zone):
        return None
    return combined


def gently_modify_near_terminal_path(
    path,
    electrode_name,
    electrode_zones,
    terminal_zones,
    tail_length,
    x_bounds=None,
    target_electrode_pos=None,
    target_terminal_pos=None,
):
    """Apply gentle modification only on the near-terminal tail; far segment is unchanged."""
    far_coords, near_coords = _split_path_far_near(path, tail_length)
    if far_coords is None:
        return randomly_modify_path(
            path.copy(),
            electrode_name,
            electrode_zones,
            terminal_zones,
            x_bounds=x_bounds,
            target_electrode_pos=target_electrode_pos,
            target_terminal_pos=target_terminal_pos,
        )

    junction = far_coords[-1].copy()
    modified_near = randomly_modify_path(
        near_coords.copy(),
        electrode_name,
        electrode_zones,
        terminal_zones,
        x_bounds=x_bounds,
        target_electrode_pos=junction,
        target_terminal_pos=target_terminal_pos,
    )
    modified_near[0] = junction
    if len(far_coords) == 1:
        return np.vstack([far_coords, modified_near[1:]])
    return np.vstack([far_coords[:-1], modified_near])


def _clip_outside_terminal_merge(geometry, terminal_merge_union):
    if geometry is None or geometry.is_empty:
        return geometry
    if terminal_merge_union is None or terminal_merge_union.is_empty:
        return geometry
    try:
        clipped = geometry.difference(terminal_merge_union)
        return clipped if not clipped.is_empty else geometry.__class__()
    except Exception:
        return geometry


def _point_outside_terminal_merge(point, terminal_merge_union):
    if terminal_merge_union is None or terminal_merge_union.is_empty:
        return True
    pt = Point(point)
    return not (terminal_merge_union.covers(pt) or terminal_merge_union.touches(pt))


def _path_terminal_zone_entry_point(path_coords, terminal_zone):
    """Return the 2D point where a path first enters a terminal safety zone."""
    coords = _clean_path_coords(path_coords)
    if len(coords) < 2:
        return None

    for idx, coord in enumerate(coords):
        point = Point(coord)
        if not (terminal_zone.covers(point) or terminal_zone.touches(point)):
            continue

        if idx == 0:
            return np.asarray(coord, dtype=float)

        outside = np.asarray(coords[idx - 1], dtype=float)
        segment = LineString([coords[idx - 1], coords[idx]])
        crossing = segment.intersection(terminal_zone.boundary)
        if crossing.is_empty:
            return np.asarray(coord, dtype=float)

        if crossing.geom_type == 'Point':
            return np.array([crossing.x, crossing.y], dtype=float)
        if crossing.geom_type == 'MultiPoint':
            candidates = [np.array([pt.x, pt.y], dtype=float) for pt in crossing.geoms]
            return min(candidates, key=lambda candidate: np.linalg.norm(candidate - outside))

        try:
            rep = crossing.interpolate(0.0, normalized=True)
            return np.array([rep.x, rep.y], dtype=float)
        except Exception:
            return np.asarray(coord, dtype=float)

    return np.asarray(coords[-1], dtype=float)


def _boundary_point_at_arc_offset(zone, anchor_point, signed_arc_length):
    """Walk along the terminal zone boundary from anchor by signed arc length."""
    boundary = zone.boundary
    if boundary is None or boundary.is_empty:
        return np.asarray(anchor_point, dtype=float)
    anchor = Point(np.asarray(anchor_point, dtype=float))
    start_dist = float(boundary.project(anchor))
    target_dist = start_dist + signed_arc_length
    length = float(boundary.length)
    if length > 0:
        target_dist = target_dist % length
    pt = boundary.interpolate(target_dist)
    return np.array([pt.x, pt.y], dtype=float)


def _unwrap_arc_distances(arc_dists, boundary_length):
    if not arc_dists:
        return []
    unwrapped = [float(arc_dists[0])]
    for dist in arc_dists[1:]:
        d = float(dist)
        while d < unwrapped[-1]:
            d += boundary_length
        unwrapped.append(d)
    return unwrapped


def _count_arc_inversions(arc_dists, boundary_length):
    unwrapped = _unwrap_arc_distances(arc_dists, boundary_length)
    if len(unwrapped) < 2:
        return 0
    return sum(1 for i in range(len(unwrapped) - 1) if unwrapped[i + 1] < unwrapped[i])


def _base_strip_tangent(terminal_pos, head_center=(0.0, 0.0)):
    terminal_pos = np.asarray(terminal_pos, dtype=float)
    head_center = np.asarray(head_center, dtype=float)
    outward = terminal_pos - head_center
    norm = float(np.linalg.norm(outward))
    if norm < 1e-9:
        outward = np.array([1.0, 0.0], dtype=float)
    else:
        outward = outward / norm
    return np.array([-outward[1], outward[0]], dtype=float)


def _oriented_strip_tangent_for_terminal(terminal_pos, hit_points, zone):
    """Return strip tangent (increasing slot index direction) for a terminal hub."""
    terminal_pos = np.asarray(terminal_pos, dtype=float)
    base_tangent = _base_strip_tangent(terminal_pos)
    boundary = zone.boundary
    blen = float(boundary.length)

    best_tangent = base_tangent
    best_inversions = None
    for sign in (1.0, -1.0):
        tangent = sign * base_tangent

        def _key(hit, t=tangent):
            return float(np.dot(np.asarray(hit, dtype=float) - terminal_pos, t))

        order = sorted(range(len(hit_points)), key=lambda i: _key(hit_points[i]))
        ordered_arc = [
            float(boundary.project(Point(np.asarray(hit_points[i], dtype=float))))
            for i in order
        ]
        inversions = _count_arc_inversions(ordered_arc, blen)
        if best_inversions is None or inversions < best_inversions:
            best_inversions = inversions
            best_tangent = tangent
    return best_tangent


def _strip_sort_key_for_terminal(terminal_pos, hit_points, zone):
    """
    Pick +tangent or -tangent per terminal so strip order matches boundary arc order.

    LEFT and RIGHT hubs mirror each other; a fixed global tangent sign is wrong for one side.
    """
    tangent = _oriented_strip_tangent_for_terminal(terminal_pos, hit_points, zone)
    terminal_pos = np.asarray(terminal_pos, dtype=float)

    def sort_key(hit_point):
        hit = np.asarray(hit_point, dtype=float)
        return float(np.dot(hit - terminal_pos, tangent))

    return sort_key


def _primary_blocking_electrode_on_chord(start, end, path_electrode_name, electrode_zones):
    """Foreign electrode whose zone blocks the electrode→entry chord (closest to start)."""
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    chord = LineString([start, end])
    best_name = None
    best_dist = float('inf')

    for name, zone in electrode_zones['zones'].items():
        if name == path_electrode_name:
            continue
        try:
            if not chord.intersects(zone):
                continue
        except Exception:
            continue
        center = np.asarray(electrode_zones['metadata'][name]['center'], dtype=float)
        dist = float(np.linalg.norm(center - start))
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name


def _electrode_bypass_side_preference(
    blocked_electrode,
    blocking_electrode,
    entry_points_2d,
    slot_index_by_electrode,
    terminal_pos,
    zone,
):
    """
    Bypass on the side opposite the blocker's entry on the strip.

    Blocker entry right of blocked → bypass left of blocker (CCW hull only).
    Blocker entry left of blocked → bypass right of blocker (CW hull only).
    """
    if not entry_points_2d or blocking_electrode not in entry_points_2d:
        return None
    if blocked_electrode not in entry_points_2d:
        return None

    if slot_index_by_electrode:
        b_slot = slot_index_by_electrode.get(blocking_electrode)
        d_slot = slot_index_by_electrode.get(blocked_electrode)
        if b_slot is not None and d_slot is not None:
            if b_slot > d_slot:
                return 'left'
            if b_slot < d_slot:
                return 'right'
            return None

    terminal_pos = np.asarray(terminal_pos, dtype=float)
    hits = [entry_points_2d[blocking_electrode], entry_points_2d[blocked_electrode]]
    tangent = _oriented_strip_tangent_for_terminal(terminal_pos, hits, zone)
    b_coord = float(np.dot(entry_points_2d[blocking_electrode] - terminal_pos, tangent))
    d_coord = float(np.dot(entry_points_2d[blocked_electrode] - terminal_pos, tangent))
    if b_coord > d_coord:
        return 'left'
    if b_coord < d_coord:
        return 'right'
    return None


def assign_terminal_entry_slots(
    path_electrodes,
    path_terminals,
    straight_paths,
    terminal_zones,
    terminals_2d=None,
    spacing=TERMINAL_ENTRY_SLOT_SPACING,
):
    """
    Order electrodes per terminal along the strip tangent; place fixed-spacing slots
    on the zone boundary with the midpoint slot anchored at the center trace hit.
    """
    entry_points_by_electrode = {}
    slot_index_by_electrode = {}
    slot_order_by_terminal = {}

    terminals = sorted(set(path_terminals))
    for terminal in terminals:
        if terminal not in terminal_zones:
            continue
        zone = terminal_zones[terminal]
        terminal_pos = (
            np.asarray(terminals_2d[terminal], dtype=float)
            if terminals_2d and terminal in terminals_2d
            else np.array([zone.centroid.x, zone.centroid.y], dtype=float)
        )
        boundary = zone.boundary
        blen = float(boundary.length)

        raw_hits = []
        indices = [i for i, t in enumerate(path_terminals) if t == terminal]
        for idx in indices:
            electrode = path_electrodes[idx]
            hit = _path_terminal_zone_entry_point(straight_paths[idx], zone)
            if hit is None:
                hit = np.asarray(straight_paths[idx][-1], dtype=float)
            hit = np.asarray(hit, dtype=float)
            raw_hits.append((electrode, hit))

        if not raw_hits:
            continue

        hit_points = [hit for _, hit in raw_hits]
        strip_sort = _strip_sort_key_for_terminal(terminal_pos, hit_points, zone)

        hit_rows = []
        for electrode, hit in raw_hits:
            hit_rows.append(
                (
                    electrode,
                    strip_sort(hit),
                    hit,
                    float(boundary.project(Point(hit))),
                )
            )

        hit_rows.sort(key=lambda row: row[1])
        n = len(hit_rows)
        mid = n // 2
        anchor_hit = hit_rows[mid][2]
        anchor_s = float(boundary.project(Point(anchor_hit)))
        ordered_names = [row[0] for row in hit_rows]
        slot_order_by_terminal[terminal] = ordered_names

        natural_arcs = [row[3] for row in hit_rows]
        unwrapped = _unwrap_arc_distances(natural_arcs, blen)
        arc_sign = 1.0
        if len(unwrapped) > 1 and float(unwrapped[-1] - unwrapped[0]) < 0:
            arc_sign = -1.0

        for slot_idx, (name, _, _, _) in enumerate(hit_rows):
            entry_s = anchor_s + arc_sign * (slot_idx - mid) * spacing
            pt = boundary.interpolate(entry_s % blen)
            entry_points_by_electrode[name] = np.array([pt.x, pt.y], dtype=float)
            slot_index_by_electrode[name] = slot_idx

    return entry_points_by_electrode, slot_index_by_electrode, slot_order_by_terminal


def build_slot_order_maps(path_electrodes, path_terminals, slot_index_by_electrode):
    """Build per-terminal electrode name lists sorted by slot index."""
    by_terminal = {}
    for idx, (electrode, terminal) in enumerate(zip(path_electrodes, path_terminals)):
        by_terminal.setdefault(terminal, []).append(
            (slot_index_by_electrode.get(electrode, idx), electrode)
        )
    slot_order_by_terminal = {}
    for terminal, items in by_terminal.items():
        items.sort(key=lambda item: item[0])
        slot_order_by_terminal[terminal] = [name for _, name in items]
    return slot_order_by_terminal


def slot_metadata_from_child_paths(child_paths):
    """Extract entry points and slot order from saved GA path entries."""
    entry_points = {}
    slot_index = {}
    path_electrodes = [p['electrode'] for p in child_paths]
    path_terminals = [p['terminal'] for p in child_paths]
    for entry in child_paths:
        name = entry['electrode']
        if entry.get('path_end_2d') is not None:
            entry_points[name] = np.asarray(entry['path_end_2d'], dtype=float)
        elif entry.get('entry_point_2d') is not None:
            entry_points[name] = np.asarray(entry['entry_point_2d'], dtype=float)
        if entry.get('slot_index') is not None:
            slot_index[name] = int(entry['slot_index'])
    slot_order = build_slot_order_maps(path_electrodes, path_terminals, slot_index)
    return entry_points, slot_index, slot_order


def compute_terminal_zone_spacing_deficit(
    paths_2d,
    path_terminals,
    terminal_zones,
    closest_min_separation=TERMINAL_ENTRY_CLOSEST_MIN,
    closest_max_separation=TERMINAL_ENTRY_CLOSEST_MAX,
):
    """
    Penalize co-terminal entry points whose closest neighbor is outside [min, max].

    Each trace contributes at most one term based on the distance to its nearest
    co-terminal entry point (not all pairwise combinations).
    """
    if path_terminals is None:
        path_terminals = _infer_path_terminals(paths_2d, terminal_zones)

    entries_by_terminal = {name: [] for name in terminal_zones}
    for path, terminal in zip(paths_2d, path_terminals):
        if terminal not in terminal_zones:
            continue
        entry = _path_terminal_zone_entry_point(path, terminal_zones[terminal])
        if entry is not None:
            entries_by_terminal[terminal].append(np.asarray(entry, dtype=float))

    spacing_deficit_raw = 0.0
    spacing_deficit_normalized = 0.0
    n_spacing_violations = 0
    min_entry_distance = float('inf')
    min_closest_neighbor_distance = float('inf')
    tight_entry_midpoints = []

    for entries in entries_by_terminal.values():
        if len(entries) < 2:
            continue

        for i, entry in enumerate(entries):
            nearest_dist = float('inf')
            nearest_entry = None
            for j, other in enumerate(entries):
                if i == j:
                    continue
                dist = float(np.linalg.norm(entry - other))
                min_entry_distance = min(min_entry_distance, dist)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_entry = other

            if nearest_dist == float('inf'):
                continue

            min_closest_neighbor_distance = min(min_closest_neighbor_distance, nearest_dist)

            if nearest_dist < closest_min_separation:
                shortfall = closest_min_separation - nearest_dist
                spacing_deficit_raw += shortfall * shortfall
                spacing_deficit_normalized += (shortfall / closest_min_separation) ** 2
                n_spacing_violations += 1
                if nearest_entry is not None:
                    tight_entry_midpoints.append(((entry + nearest_entry) / 2.0).tolist())
            elif nearest_dist > closest_max_separation:
                excess = nearest_dist - closest_max_separation
                spacing_deficit_raw += excess * excess
                spacing_deficit_normalized += (excess / closest_max_separation) ** 2
                n_spacing_violations += 1
                if nearest_entry is not None:
                    tight_entry_midpoints.append(((entry + nearest_entry) / 2.0).tolist())

    return {
        'terminal_zone_spacing_deficit': spacing_deficit_raw,
        'terminal_zone_spacing_deficit_normalized': spacing_deficit_normalized,
        'terminal_spacing_violation_pairs': n_spacing_violations,
        'min_terminal_entry_distance': min_entry_distance,
        'min_closest_neighbor_entry_distance': min_closest_neighbor_distance,
        'terminal_entry_points': entries_by_terminal,
        'terminal_entry_midpoints': tight_entry_midpoints,
        'terminal_entry_closest_min': closest_min_separation,
        'terminal_entry_closest_max': closest_max_separation,
    }


def compute_collision_score(
    crossing_count,
    overlap_length,
    trace_separation_deficit_normalized,
    electrode_violations,
):
    """Weighted sum used by collision-resolution thresholds (unit weights)."""
    return (
        CROSSING_SCORE_WEIGHT * crossing_count
        + OVERLAP_SCORE_WEIGHT * overlap_length
        + TRACE_SEPARATION_SCORE_WEIGHT * trace_separation_deficit_normalized
        + ELECTRODE_SCORE_WEIGHT * electrode_violations
    )


def compute_layout_score(analysis, ga_phase=2, path_length_excess=0.0) -> float:
    """
    Single layout penalty (lower is better). GA fitness is -layout_score.

    Phase 1: electrode violations only.
    Phase 2: crossings, overlap, trace separation, path-length excess, electrodes.
    """
    electrode_violations = int(analysis.get('electrode_violations', 0))
    if ga_phase == 1:
        return ELECTRODE_LAYOUT_WEIGHT * electrode_violations
    trace_sep = analysis.get(
        'trace_separation_deficit_normalized',
        analysis.get('clearance_deficit', 0.0),
    )
    return (
        CROSSING_LAYOUT_MULTIPLIER * analysis.get('crossing_count', 0)
        + SLOT_ORDERED_CROSSING_LAYOUT_EXTRA
        * analysis.get('slot_ordered_crossing_count', 0)
        + TRACE_REENTRY_LAYOUT_MULTIPLIER * analysis.get('trace_reentry_count', 0)
        + NEAR_TERMINAL_CROSSING_LAYOUT_MULTIPLIER
        * analysis.get('near_crossing_count', 0)
        + OVERLAP_LAYOUT_MULTIPLIER * analysis.get('overlap_length', 0.0)
        + TRACE_SEPARATION_LAYOUT_MULTIPLIER * trace_sep
        + LENGTH_LAYOUT_WEIGHT * path_length_excess
        + ELECTRODE_LAYOUT_WEIGHT * electrode_violations
    )


def layout_score_from_metrics(metrics) -> float:
    """Read stored layout_score, falling back to legacy collision_score."""
    if metrics is None:
        return float('inf')
    if 'layout_score' in metrics:
        return float(metrics['layout_score'])
    if 'collision_score' in metrics:
        return float(metrics['collision_score'])
    return float('inf')


def _pair_crossing_point_count(
    path_a,
    path_b,
    terminal_a,
    terminal_b,
    terminal_zones,
    electrode_zones,
    dense_path_cache=None,
    path_idx_a=None,
    path_idx_b=None,
    use_dense_paths=True,
):
    """Point-crossing count for one path pair (terminal-merge exclusions applied)."""
    _, _, crossing_points = _compute_path_pair_crossing_geometry(
        path_a,
        path_b,
        terminal_a,
        terminal_b,
        terminal_zones,
        electrode_zones,
        dense_path_cache=dense_path_cache,
        path_idx_a=path_idx_a,
        path_idx_b=path_idx_b,
        use_dense_paths=use_dense_paths,
    )
    return len(crossing_points)


def _count_layout_crossings(
    paths_2d,
    path_terminals,
    terminal_zones,
    electrode_zones,
    use_dense_paths=True,
):
    """Count layout point crossings only (no overlap, electrodes, or policy metrics)."""
    if path_terminals is None:
        path_terminals = _infer_path_terminals(paths_2d, terminal_zones)
    dense_path_cache = (
        _build_crossing_detection_path_cache(paths_2d, path_terminals, electrode_zones)
        if use_dense_paths
        else None
    )
    total = 0
    n_paths = len(paths_2d)
    for i in range(n_paths):
        for j in range(i + 1, n_paths):
            total += _pair_crossing_point_count(
                paths_2d[i],
                paths_2d[j],
                path_terminals[i] if i < len(path_terminals) else None,
                path_terminals[j] if j < len(path_terminals) else None,
                terminal_zones,
                electrode_zones,
                dense_path_cache=dense_path_cache,
                path_idx_a=i if use_dense_paths else None,
                path_idx_b=j if use_dense_paths else None,
                use_dense_paths=use_dense_paths,
            )
    return total


def _count_crossings_involving_path(
    path_idx,
    paths_2d,
    path_terminals,
    terminal_zones,
    electrode_zones,
    use_dense_paths=True,
):
    """Crossings between paths_2d[path_idx] and every other path."""
    if path_terminals is None:
        path_terminals = _infer_path_terminals(paths_2d, terminal_zones)
    dense_path_cache = (
        _build_crossing_detection_path_cache(paths_2d, path_terminals, electrode_zones)
        if use_dense_paths
        else None
    )
    total = 0
    n_paths = len(paths_2d)
    for j in range(n_paths):
        if j == path_idx:
            continue
        i, jj = (path_idx, j) if path_idx < j else (j, path_idx)
        total += _pair_crossing_point_count(
            paths_2d[i],
            paths_2d[jj],
            path_terminals[i] if i < len(path_terminals) else None,
            path_terminals[jj] if jj < len(path_terminals) else None,
            terminal_zones,
            electrode_zones,
            dense_path_cache=dense_path_cache,
            path_idx_a=i if use_dense_paths else None,
            path_idx_b=jj if use_dense_paths else None,
            use_dense_paths=use_dense_paths,
        )
    return total


def _count_crossings_among_other_paths(
    path_idx,
    paths_2d,
    path_terminals,
    terminal_zones,
    electrode_zones,
    dense_path_cache=None,
    use_dense_paths=True,
):
    """Crossings between path pairs that do not include path_idx."""
    if path_terminals is None:
        path_terminals = _infer_path_terminals(paths_2d, terminal_zones)
    if use_dense_paths and dense_path_cache is None:
        dense_path_cache = _build_crossing_detection_path_cache(
            paths_2d, path_terminals, electrode_zones
        )
    total = 0
    n_paths = len(paths_2d)
    for i in range(n_paths):
        for j in range(i + 1, n_paths):
            if i == path_idx or j == path_idx:
                continue
            total += _pair_crossing_point_count(
                paths_2d[i],
                paths_2d[j],
                path_terminals[i] if i < len(path_terminals) else None,
                path_terminals[j] if j < len(path_terminals) else None,
                terminal_zones,
                electrode_zones,
                dense_path_cache=dense_path_cache,
                path_idx_a=i if use_dense_paths else None,
                path_idx_b=j if use_dense_paths else None,
                use_dense_paths=use_dense_paths,
            )
    return total


def _layout_crossing_count_if_replaced(
    path_idx,
    trial_path,
    paths_2d,
    path_terminals,
    terminal_zones,
    electrode_zones,
    crossings_among_others,
    dense_path_cache=None,
    use_dense_paths=True,
):
    """Global crossing count after replacing paths_2d[path_idx] with trial_path."""
    if path_terminals is None:
        path_terminals = _infer_path_terminals(paths_2d, terminal_zones)
    involving = 0
    n_paths = len(paths_2d)
    for j in range(n_paths):
        if j == path_idx:
            continue
        i, jj = (path_idx, j) if path_idx < j else (j, path_idx)
        path_i = trial_path if i == path_idx else paths_2d[i]
        path_j = trial_path if jj == path_idx else paths_2d[jj]
        involving += _pair_crossing_point_count(
            path_i,
            path_j,
            path_terminals[i] if i < len(path_terminals) else None,
            path_terminals[jj] if jj < len(path_terminals) else None,
            terminal_zones,
            electrode_zones,
            dense_path_cache=dense_path_cache,
            path_idx_a=i if use_dense_paths and i != path_idx else None,
            path_idx_b=jj if use_dense_paths and jj != path_idx else None,
            use_dense_paths=use_dense_paths,
        )
    return int(crossings_among_others) + int(involving)


def layout_crossing_count(
    paths_2d,
    terminal_zones,
    electrode_zones,
    path_electrodes,
    path_terminals,
):
    """Return layout crossing count (terminal-merge exclusions applied)."""
    del path_electrodes  # kept for call-site compatibility
    return _count_layout_crossings(
        paths_2d,
        path_terminals,
        terminal_zones,
        electrode_zones,
        use_dense_paths=True,
    )


def finalize_collision_metrics(analysis, ga_phase=2, path_length_excess=0.0):
    """Persist analysis fields plus unified layout_score for JSON logs."""
    metrics = analysis_to_collision_metrics(analysis)
    metrics['path_length_excess'] = float(path_length_excess)
    metrics['layout_score'] = compute_layout_score(
        metrics, ga_phase, path_length_excess
    )
    return metrics


def refresh_saved_individual_collision_metrics(
    SUBJECT_ID,
    INDIVIDUAL_ID,
    ga_phase=2,
    log_dir="data/output/logs",
):
    """Recompute collision_metrics from saved paths (e.g. after elite copy in phase 2)."""
    log_path = os.path.join(
        log_dir, f"GA_{SUBJECT_ID}_{INDIVIDUAL_ID}_mod_connection_paths.json"
    )
    with open(log_path, "r") as f:
        data = json.load(f)

    ctx = get_subject_layout(SUBJECT_ID)
    paths = [np.array(p["modified_path_2d"]) for p in data["paths"]]
    path_electrodes = [p["electrode"] for p in data["paths"]]
    path_terminals = [p["terminal"] for p in data["paths"]]

    metrics_mode = "electrodes_only" if ga_phase == 1 else "full"
    path_length_excess = 0.0
    if ga_phase == 2:
        path_length_excess = compute_layout_path_length_excess(
            paths,
            path_electrodes,
            path_terminals,
            ctx["electrodes_2d"],
            ctx["terminals_2d"],
        )

    analysis = analyze_path_collisions(
        paths,
        ctx["terminal_zones"],
        electrode_zones=ctx["electrode_zones"],
        path_electrodes=path_electrodes,
        path_terminals=path_terminals,
        metrics_mode=metrics_mode,
        ga_phase=ga_phase,
        path_length_excess=path_length_excess,
        slot_index_by_electrode=slot_metadata_from_child_paths(data["paths"])[1],
    )
    collision_metrics = finalize_collision_metrics(
        analysis, ga_phase=ga_phase, path_length_excess=path_length_excess
    )
    layout_score = collision_metrics["layout_score"]

    data["collision_metrics"] = collision_metrics
    for path_entry in data["paths"]:
        path_entry["n_collisions"] = layout_score

    with open(log_path, "w") as f:
        json.dump(data, f, indent=2)

    return collision_metrics


def _linestring_sample_points(line, exclude_region=None, sample_step=TRACE_SEPARATION_SAMPLE_STEP):
    """Sample points along a path, optionally skipping co-terminal merge tails."""
    if line is None or line.is_empty:
        return []
    n = max(2, int(np.ceil(float(line.length) / sample_step)))
    pts = []
    for t in np.linspace(0.0, 1.0, n):
        pt = line.interpolate(t, normalized=True)
        if exclude_region is not None and not exclude_region.is_empty:
            if exclude_region.covers(pt) or exclude_region.touches(pt):
                continue
        pts.append(pt)
    return pts


def compute_trace_separation_deficit(
    paths_2d,
    path_terminals=None,
    terminal_zones=None,
    min_separation=TRACE_SEPARATION_MIN,
    sample_step=TRACE_SEPARATION_SAMPLE_STEP,
    electrode_zones=None,
):
    """
    Penalize trace pairs whose centerlines come closer than min_separation anywhere
    along the route (excluding co-terminal merge tails).
    """
    lines = [_path_to_linestring(path) for path in paths_2d]
    merge_tail_length = _terminal_merge_tail_length(electrode_zones)
    if path_terminals is None:
        path_terminals = _infer_path_terminals(paths_2d, terminal_zones or {})

    deficit_raw = 0.0
    deficit_normalized = 0.0
    n_violations = 0
    violation_points = []
    min_trace_separation = float('inf')

    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            li, lj = lines[i], lines[j]
            if li is None or lj is None:
                continue

            pair_merge_union = _build_pair_merge_union(
                paths_2d[i],
                paths_2d[j],
                path_terminals[i] if i < len(path_terminals) else None,
                path_terminals[j] if j < len(path_terminals) else None,
                merge_tail_length,
            )

            for line, other in ((li, lj), (lj, li)):
                for pt in _linestring_sample_points(line, pair_merge_union, sample_step):
                    d = float(other.distance(pt))
                    min_trace_separation = min(min_trace_separation, d)
                    if d < min_separation:
                        shortfall = min_separation - d
                        deficit_raw += shortfall * shortfall
                        deficit_normalized += (shortfall / min_separation) ** 2
                        n_violations += 1
                        violation_points.append([pt.x, pt.y])

    return {
        'trace_separation_deficit': deficit_raw,
        'trace_separation_deficit_normalized': deficit_normalized,
        'trace_separation_violations': n_violations,
        'min_trace_separation': min_trace_separation,
        'trace_separation_violation_points': violation_points,
        'trace_separation_min_required': min_separation,
    }


def _accumulate_geometry_metrics(geom, points, stats):
    """Collect collision points, crossing count, and overlap length from a Shapely geometry."""
    if geom is None or geom.is_empty:
        return

    gtype = geom.geom_type
    if gtype == 'Point':
        points.append([geom.x, geom.y])
        stats['crossing_count'] += 1
    elif gtype == 'MultiPoint':
        for pt in geom.geoms:
            points.append([pt.x, pt.y])
            stats['crossing_count'] += 1
    elif gtype == 'LineString':
        stats['overlap_length'] += float(geom.length)
        n_samples = max(2, int(np.ceil(float(geom.length) / 0.5)))
        for t in np.linspace(0, 1, n_samples):
            pt = geom.interpolate(t, normalized=True)
            points.append([pt.x, pt.y])
    elif gtype in ('MultiLineString', 'GeometryCollection'):
        for sub in geom.geoms:
            _accumulate_geometry_metrics(sub, points, stats)
    else:
        try:
            rep = geom.representative_point()
            points.append([rep.x, rep.y])
            stats['crossing_count'] += 1
        except Exception:
            pass


def _accumulate_pair_crossing_metrics(
    inter,
    crossing_points,
    points,
    stats,
    stats_far,
    stats_near,
    near_terminal_union,
):
    """Accumulate crossing/overlap metrics for one path pair (includes supplemental points)."""
    _accumulate_geometry_metrics(inter, points, stats)
    geom_points = _collect_points_from_geometry(inter)
    for pt in crossing_points:
        pt_list = np.asarray(pt, dtype=float).tolist()
        if _is_duplicate_xy_point(pt_list, geom_points):
            continue
        points.append(pt_list)
        stats['crossing_count'] += 1
        geom_points.append(pt_list)
        if near_terminal_union is not None:
            point = Point(pt_list)
            if (
                near_terminal_union.covers(point)
                or near_terminal_union.touches(point)
                or near_terminal_union.distance(point) < 1e-6
            ):
                stats_near['crossing_count'] += 1
            else:
                stats_far['crossing_count'] += 1

    if near_terminal_union is not None:
        inter_far = _clip_outside_near_terminal(inter, near_terminal_union)
        _accumulate_geometry_metrics(inter_far, [], stats_far)
        try:
            inter_near = inter.intersection(near_terminal_union)
        except Exception:
            inter_near = inter.__class__()
        _accumulate_geometry_metrics(inter_near, [], stats_near)
    else:
        _accumulate_geometry_metrics(inter, [], stats_far)


def analyze_path_collisions(
    paths_2d,
    terminal_zones,
    electrode_zones=None,
    path_electrodes=None,
    path_terminals=None,
    min_separation=MIN_PATH_SEPARATION,
    metrics_mode='full',
    ga_phase=2,
    path_length_excess=0.0,
    slot_index_by_electrode=None,
):
    """
    Analyze path layout collisions using segment-based Shapely distance checks.

    metrics_mode:
      - 'full': all metrics (fitness phase 2, plots)
      - 'clearance': crossings/overlap/electrodes only (skip trace separation + terminal spacing)
      - 'electrodes_only': foreign electrode zone hits only (phase 1 fitness / greedy)
    """
    if metrics_mode == 'electrodes_only':
        electrode_violations = count_electrode_violations(
            paths_2d, electrode_zones, path_electrodes
        )
        metrics = _empty_path_metrics(electrode_violations)
        metrics['layout_score'] = compute_layout_score(metrics, ga_phase=1)
        return metrics

    merge_tail_length = _terminal_merge_tail_length(electrode_zones)
    crossing_near_tail = (
        _crossing_merge_tail_length(electrode_zones)
        * NEAR_TERMINAL_CROSSING_DENSIFY_LENGTH_FACTOR
    )
    if path_terminals is None:
        path_terminals = _infer_path_terminals(paths_2d, terminal_zones)

    near_terminal_union = _build_near_terminal_region_union(
        paths_2d,
        crossing_near_tail,
        pad=TERMINAL_CROSSING_MERGE_PAD * 2.0,
    )

    points = []
    stats = {'crossing_count': 0, 'overlap_length': 0.0}
    stats_far = {'crossing_count': 0, 'overlap_length': 0.0}
    stats_near = {'crossing_count': 0, 'overlap_length': 0.0}
    min_pairwise_distance = float('inf')
    slot_violating_pairs = set()
    track_slot_ordered = bool(slot_index_by_electrode and path_electrodes)

    dense_path_cache = _build_crossing_detection_path_cache(
        paths_2d, path_terminals, electrode_zones
    )

    for i in range(len(paths_2d)):
        for j in range(i + 1, len(paths_2d)):
            line_i = _path_to_linestring(paths_2d[i])
            line_j = _path_to_linestring(paths_2d[j])
            if line_i is not None and line_j is not None:
                min_pairwise_distance = min(
                    min_pairwise_distance, float(line_i.distance(line_j))
                )

            inter, _pair_merge, crossing_points = _compute_path_pair_crossing_geometry(
                paths_2d[i],
                paths_2d[j],
                path_terminals[i] if i < len(path_terminals) else None,
                path_terminals[j] if j < len(path_terminals) else None,
                terminal_zones,
                electrode_zones,
                dense_path_cache=dense_path_cache,
                path_idx_a=i,
                path_idx_b=j,
            )
            if track_slot_ordered:
                terminal_i = path_terminals[i]
                terminal_j = path_terminals[j]
                slot_i = slot_index_by_electrode.get(path_electrodes[i])
                slot_j = slot_index_by_electrode.get(path_electrodes[j])
                if (
                    terminal_i == terminal_j
                    and slot_i is not None
                    and slot_j is not None
                    and slot_i != slot_j
                    and _geometry_indicates_crossing(inter, crossing_points)
                ):
                    if slot_i < slot_j:
                        low_name, high_name = path_electrodes[i], path_electrodes[j]
                    else:
                        low_name, high_name = path_electrodes[j], path_electrodes[i]
                    slot_violating_pairs.add((terminal_i, low_name, high_name))

            if inter is None or (inter.is_empty and not crossing_points):
                continue

            _accumulate_pair_crossing_metrics(
                inter,
                crossing_points,
                points,
                stats,
                stats_far,
                stats_near,
                near_terminal_union,
            )

    if metrics_mode == 'clearance':
        electrode_violations = count_electrode_violations(
            paths_2d, electrode_zones, path_electrodes
        )
        unique_points = np.unique(np.asarray(points), axis=0) if points else None
        collision_score = compute_collision_score(
            stats['crossing_count'],
            stats['overlap_length'],
            0.0,
            electrode_violations,
        )
        result = {
            'points': unique_points,
            'crossing_count': stats['crossing_count'],
            'overlap_length': stats['overlap_length'],
            'far_crossing_count': stats_far['crossing_count'],
            'far_overlap_length': stats_far['overlap_length'],
            'near_crossing_count': stats_near['crossing_count'],
            'near_overlap_length': stats_near['overlap_length'],
            'terminal_zone_spacing_deficit': 0.0,
            'terminal_zone_spacing_deficit_normalized': 0.0,
            'terminal_spacing_violation_pairs': 0,
            'trace_separation_deficit': 0.0,
            'trace_separation_deficit_normalized': 0.0,
            'trace_separation_violations': 0,
            'min_trace_separation': min_pairwise_distance,
            'trace_separation_min_required': TRACE_SEPARATION_MIN,
            'clearance_deficit': 0.0,
            'electrode_violations': electrode_violations,
            'collision_score': collision_score,
            'min_pairwise_distance': min_pairwise_distance,
            'min_terminal_entry_distance': float('inf'),
            'min_closest_neighbor_entry_distance': float('inf'),
            'terminal_merge_tail_length': merge_tail_length,
            'slot_ordered_crossing_count': 0,
            'trace_reentry_count': 0,
        }
        result['layout_score'] = compute_layout_score(
            result, ga_phase=ga_phase, path_length_excess=path_length_excess
        )
        return result

    terminal_spacing = compute_terminal_zone_spacing_deficit(
        paths_2d,
        path_terminals,
        terminal_zones,
    )
    terminal_zone_spacing_deficit = terminal_spacing['terminal_zone_spacing_deficit']
    terminal_zone_spacing_deficit_normalized = terminal_spacing[
        'terminal_zone_spacing_deficit_normalized'
    ]
    trace_separation = compute_trace_separation_deficit(
        paths_2d,
        path_terminals=path_terminals,
        terminal_zones=terminal_zones,
        electrode_zones=electrode_zones,
    )
    trace_separation_deficit = trace_separation['trace_separation_deficit']
    trace_separation_deficit_normalized = trace_separation[
        'trace_separation_deficit_normalized'
    ]
    points.extend(trace_separation['trace_separation_violation_points'])
    points.extend(terminal_spacing['terminal_entry_midpoints'])

    electrode_violations = 0
    electrode_points = find_electrode_collisions(paths_2d, electrode_zones, path_electrodes) if (
        electrode_zones is not None and path_electrodes is not None
    ) else None
    if electrode_points is not None:
        electrode_violations = len(electrode_points)
        points.extend(np.asarray(electrode_points).tolist())

    unique_points = np.unique(np.asarray(points), axis=0) if points else None
    collision_score = compute_collision_score(
        stats['crossing_count'],
        stats['overlap_length'],
        trace_separation_deficit_normalized,
        electrode_violations,
    )

    result = {
        'points': unique_points,
        'crossing_count': stats['crossing_count'],
        'overlap_length': stats['overlap_length'],
        'far_crossing_count': stats_far['crossing_count'],
        'far_overlap_length': stats_far['overlap_length'],
        'near_crossing_count': stats_near['crossing_count'],
        'near_overlap_length': stats_near['overlap_length'],
        'terminal_zone_spacing_deficit': terminal_zone_spacing_deficit,
        'terminal_zone_spacing_deficit_normalized': terminal_zone_spacing_deficit_normalized,
        'terminal_spacing_violation_pairs': terminal_spacing['terminal_spacing_violation_pairs'],
        'trace_separation_deficit': trace_separation_deficit,
        'trace_separation_deficit_normalized': trace_separation_deficit_normalized,
        'trace_separation_violations': trace_separation['trace_separation_violations'],
        'min_trace_separation': trace_separation['min_trace_separation'],
        'trace_separation_min_required': trace_separation['trace_separation_min_required'],
        'clearance_deficit': trace_separation_deficit_normalized,
        'electrode_violations': electrode_violations,
        'collision_score': collision_score,
        'min_pairwise_distance': min_pairwise_distance,
        'min_terminal_entry_distance': terminal_spacing['min_terminal_entry_distance'],
        'min_closest_neighbor_entry_distance': terminal_spacing['min_closest_neighbor_entry_distance'],
        'terminal_merge_tail_length': merge_tail_length,
        'slot_ordered_crossing_count': (
            len(slot_violating_pairs)
            if track_slot_ordered
            else count_coterminal_ordered_crossings(
                paths_2d,
                path_electrodes,
                path_terminals,
                slot_index_by_electrode,
                electrode_zones,
                terminal_zones=terminal_zones,
                dense_path_cache=dense_path_cache,
            )
        ),
        'trace_reentry_count': count_trace_reentries(
            paths_2d,
            path_electrodes,
            path_terminals,
            electrode_zones,
            terminal_zones,
        ),
    }
    result['layout_score'] = compute_layout_score(
        result, ga_phase=ga_phase, path_length_excess=path_length_excess
    )
    return result


def is_far_region_clearance_free(collision_analysis) -> bool:
    """True when the body of the layout (outside near-terminal tails) is clear."""
    return (
        collision_analysis.get('far_crossing_count', collision_analysis['crossing_count']) == 0
        and collision_analysis.get('far_overlap_length', collision_analysis['overlap_length'])
        < COLLISION_SCORE_EPSILON
        and collision_analysis['electrode_violations'] == 0
    )


def is_layout_clearance_free(collision_analysis) -> bool:
    """True when crossings, overlap, and electrode violations are cleared."""
    return (
        collision_analysis['crossing_count'] == 0
        and collision_analysis['overlap_length'] < COLLISION_SCORE_EPSILON
        and collision_analysis['electrode_violations'] == 0
    )


def is_layout_collision_free(collision_analysis) -> bool:
    """True when there are no path, trace-separation, or electrode violations."""
    separation_deficit = collision_analysis.get(
        'trace_separation_deficit_normalized',
        collision_analysis.get('clearance_deficit', 0.0),
    )
    return (
        is_layout_clearance_free(collision_analysis)
        and separation_deficit < COLLISION_SCORE_EPSILON
    )


def is_layout_phase2_solution(collision_analysis) -> bool:
    """Phase 2 GA solution: electrode-free and layout_score below threshold."""
    path_length_excess = float(collision_analysis.get('path_length_excess', 0.0))
    layout_score = compute_layout_score(
        collision_analysis, ga_phase=2, path_length_excess=path_length_excess
    )
    return (
        int(collision_analysis.get('electrode_violations', 1)) == 0
        and layout_score < PHASE2_SOLUTION_SCORE_THRESHOLD
    )


def find_path_collisions(
    paths_2d,
    terminal_zones,
    min_separation=MIN_PATH_SEPARATION,
    electrode_zones=None,
    path_electrodes=None,
    path_terminals=None,
):
    """
    Return unique collision/near-miss points for visualization and legacy callers.

    Uses segment-based Shapely distance; see analyze_path_collisions for full metrics.
    """
    return analyze_path_collisions(
        paths_2d,
        terminal_zones,
        electrode_zones=electrode_zones,
        path_electrodes=path_electrodes,
        path_terminals=path_terminals,
        min_separation=min_separation,
    )['points']


def analyze_pair_path_penalty(
    path_a,
    path_b,
    terminal_zones,
    terminal_a=None,
    terminal_b=None,
    electrode_zones=None,
    dense_path_cache=None,
    path_idx_a=None,
    path_idx_b=None,
    pair_geometry=None,
):
    """Lightweight crossing/overlap penalty for one path pair (phase 2 inner traces)."""
    if pair_geometry is None:
        inter, _pair_merge, crossing_points = _compute_path_pair_crossing_geometry(
            path_a,
            path_b,
            terminal_a,
            terminal_b,
            terminal_zones,
            electrode_zones,
            dense_path_cache=dense_path_cache,
            path_idx_a=path_idx_a,
            path_idx_b=path_idx_b,
        )
    else:
        inter, crossing_points = pair_geometry
    return _crossing_overlap_penalty_from_geometry(inter, crossing_points)


def find_pair_path_collisions(
    path_a,
    path_b,
    terminal_zones,
    terminal_a=None,
    terminal_b=None,
    electrode_zones=None,
    min_separation=MIN_PATH_SEPARATION,
):
    """Collision check for a single path pair with correct terminal-aware exclusion."""
    return find_path_collisions(
        [path_a, path_b],
        terminal_zones,
        min_separation=min_separation,
        electrode_zones=electrode_zones,
        path_terminals=[terminal_a, terminal_b],
    )


def _point_in_terminal_zone(point, terminal_list):
    pt = Point(point)
    return any(zone.covers(pt) or zone.touches(pt) for zone in terminal_list)


def compute_path_separation_deficit(
    paths_2d,
    terminal_zones,
    closest_min_separation=TERMINAL_ENTRY_CLOSEST_MIN,
    closest_max_separation=TERMINAL_ENTRY_CLOSEST_MAX,
    electrode_zones=None,
    path_terminals=None,
):
    """Backward-compatible wrapper for terminal-zone entry spacing deficit."""
    if path_terminals is None:
        path_terminals = _infer_path_terminals(paths_2d, terminal_zones)

    spacing = compute_terminal_zone_spacing_deficit(
        paths_2d,
        path_terminals,
        terminal_zones,
        closest_min_separation=closest_min_separation,
        closest_max_separation=closest_max_separation,
    )
    return (
        spacing['terminal_zone_spacing_deficit_normalized'],
        spacing['min_closest_neighbor_entry_distance'],
    )


def visualize_uv_grid(mesh, uv_grid, title="UV Grid Visualization"):
    """Visualize the UV grid on the head surface"""
    pv = _pyvista()
    plotter = pv.Plotter()
    
    # Plot the head mesh
    plotter.add_mesh(mesh, color="white", opacity=0.7, 
                    ambient=0.2, diffuse=0.8, name="head")
    
    # Extract grid points
    grid_3d = np.array(uv_grid['grid_3d'])
    
    # Create point cloud for grid
    grid_points = pv.PolyData(grid_3d)
    
    # Plot grid points in neon green
    plotter.add_mesh(grid_points, color="#39FF14", point_size=10, 
                    render_points_as_spheres=True, name="grid_points")
    
    # Add connections between grid points (optional)
    resolution = int(np.sqrt(len(grid_3d)))
    for i in range(resolution):
        # Horizontal lines
        start = i * resolution
        end = (i+1) * resolution
        line = pv.lines_from_points(grid_3d[start:end])
        plotter.add_mesh(line, color="#39FF14", line_width=2, name=f"horiz_{i}")
        
        # Vertical lines
        points = grid_3d[i::resolution]
        line = pv.lines_from_points(points)
        plotter.add_mesh(line, color="#39FF14", line_width=2, name=f"vert_{i}")
    
    plotter.add_title(title, font_size=20)
    plotter.show()


# --------------------------
# Path Modification Functions
# --------------------------

def smooth_path(path, strength=0.3):
    """Smooth path using spline interpolation"""
    # Input validation
    if len(path) < 4:
        print(f"Warning: Path too short for smoothing ({len(path)} points), returning original")
        return path.copy()
    
    # Check for NaN or infinite values
    if np.any(~np.isfinite(path)):
        print("Warning: Path contains NaN or infinite values, cleaning...")
        path = path[np.all(np.isfinite(path), axis=1)]
        if len(path) < 4:
            print("Error: After cleaning, path has insufficient points")
            return path
    
    # Remove duplicate consecutive points
    if len(path) > 1:
        diffs = np.diff(path, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        valid_indices = np.concatenate([[0], np.where(distances > 1e-10)[0] + 1])
        path = path[valid_indices]
    
    if len(path) < 4:
        print(f"Warning: After removing duplicates, path too short ({len(path)} points)")
        return path
    
    try:
        tck, u = splprep(path.T, s=strength*len(path))
        new_points = splev(np.linspace(0, 1, len(path)), tck)
        smoothed = np.column_stack(new_points)
        smoothed[0] = path[0]
        smoothed[-1] = path[-1]
        return smoothed
    except Exception as e:
        print(f"Error in spline fitting: {str(e)}")
        print(f"Path shape: {path.shape}, contains NaN: {np.any(np.isnan(path))}")
        print(f"Path range: min={np.min(path, axis=0)}, max={np.max(path, axis=0)}")
        return path.copy()


def add_local_curve(path, curve_size=0.15, verbose=False):
    """
    Adds exactly ONE smooth curve affecting 10-35% of path length
    with smooth tapering at both ends.
    """
    if len(path) < 5:
        if verbose:
            print("Path too short for curve - returning original")
        return path.copy()
    
    # set curve_size to random value between 0.05 and 0.5
    curve_size = random.uniform(0.05, 0.5) # NEW
    
    path_length = len(path)
    curve_percentage = random.uniform(0.10, 0.35)
    curve_span = max(3, int(path_length * curve_percentage))
    
    # Choose curve center ensuring full curve fits
    center_idx = random.randint(curve_span//2, len(path)-1 - curve_span//2)
    start_idx = max(0, center_idx - curve_span//2)
    end_idx = min(len(path)-1, center_idx + curve_span//2)
    
    # Calculate direction perpendicular to path segment
    tangent = path[min(center_idx+1, len(path)-1)] - path[max(center_idx-1, 0)]
    normal = np.array([-tangent[1], tangent[0]])
    normal *= random.choice([-1, 1]) # Randomly flip direction so the curve can go either way
    normal = normal / (np.linalg.norm(normal) + 1e-8)  # Safe normalization
    
    modified = path.copy()
    curve_strength = curve_size * random.uniform(0.2, 2.0)
    
    # Apply smooth bell-shaped curve
    for i in range(start_idx, end_idx+1):
        # Normalized distance from center [0,1]
        t = abs(i - center_idx) / (curve_span/2)  
        # Gaussian-like weighting
        weight = np.exp(-4 * t**2)  
        modified[i] += normal * curve_strength * weight
    
    # Ensure endpoints stay fixed
    modified[0] = path[0]
    modified[-1] = path[-1]
    
    if verbose:
        print(f"Applied single curve to {len(path)}-point path:")
        print(f"- Center at point {center_idx}/{len(path)}")
        print(f"- Affecting points {start_idx} to {end_idx} ({end_idx-start_idx+1} points)")
        print(f"- Max displacement: {curve_strength:.3f} units")
    
    return modified


def add_global_bend(path, bend_strength=0.5):
    """
    Apply a STRONG single global bend (C or S shaped) 
    Handles both integer and float coordinate paths
    """
    if len(path) < 3:
        return path.copy()

    # Convert path to float if needed
    path = path.astype(float)
    
    # Choose bend type and direction
    bend_type = random.choice(['C', 'S'])
    bend_dir = np.array([random.choice([-1.0, 1.0]), random.choice([-1.0, 1.0])])  # Note: using float values
    bend_dir /= np.linalg.norm(bend_dir)  # Now safe to normalize
    
    # Calculate base magnitude (scales with path length)
    path_length = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=0))
    base_magnitude = bend_strength * path_length * 0.2
    
    # Select bend center (weighted toward middle 50%)
    center_bias = 0.5
    center_idx = int(len(path)//2 + (random.random()-0.5)*len(path)*(1-center_bias))
    center_idx = np.clip(center_idx, 1, len(path)-2)
    
    modified = path.copy()
    
    for i in range(1, len(path)-1):
        # Normalized distance from center [-1, 1]
        t = (i - center_idx) / (len(path)/2)  
        
        if bend_type == 'C':
            # Strong bell curve shape
            weight = np.exp(-3 * t**2)
            displacement = base_magnitude * weight * bend_dir
        else:  # S-curve
            # Sigmoid derivative shape
            weight = t * np.exp(-2 * t**2) * 2
            displacement = base_magnitude * weight * bend_dir
        
        modified[i] += displacement
    
    # Preserve exact endpoints
    modified[0] = path[0]
    modified[-1] = path[-1]
    
    # Light smoothing to remove any residual kinks
    return smooth_path(modified, strength=0.05)


def randomly_modify_path(
    path,
    path_electrode_name,
    electrode_zones,
    terminal_zones,
    max_attempts=20,
    x_bounds=(-150, 150),
    target_electrode_pos=None,
    target_terminal_pos=None,
    target_terminal_name=None,
):
    """Apply one random modification to path, ensuring it stays within bounds and avoids zones."""
    locked_terminal_tail = None
    mutate_path = path
    mutate_terminal_pos = target_terminal_pos

    if target_terminal_name and target_terminal_name in terminal_zones:
        outside, inside = _split_path_at_terminal_zone(
            path, target_terminal_name, terminal_zones
        )
        if inside is not None and (outside is None or len(outside) < 2):
            fallback = np.asarray(path, dtype=float).copy()
            if target_electrode_pos is not None or target_terminal_pos is not None:
                fallback = pin_path_endpoints_2d(
                    fallback,
                    target_electrode_pos if target_electrode_pos is not None else path[0],
                    target_terminal_pos if target_terminal_pos is not None else path[-1],
                )
            return fallback
        if inside is not None and outside is not None and len(outside) >= 2:
            locked_terminal_tail = inside
            mutate_path = outside
            mutate_terminal_pos = outside[-1]

    for attempt in range(max_attempts):

        USE_OLD_OPERATIONS_ONLY = True  # Set to True to use only old operations, False to include new ones
        if USE_OLD_OPERATIONS_ONLY:
            AVAILABLE_OPERATIONS = ['smooth', 'curve', 'bend', 'straight']
        else:
            AVAILABLE_OPERATIONS = ['curve', 'bend', 'straight']

        operation = random.choice(AVAILABLE_OPERATIONS)

        if operation == 'smooth':
            modified = smooth_path(mutate_path.copy(), strength=random.uniform(0.1, 0.3))

        elif operation == 'curve':
            modified = add_local_curve(mutate_path.copy(), curve_size=random.uniform(0.1, 0.4))

        elif operation == 'bend':
            modified = add_global_bend(mutate_path.copy(), bend_strength=random.uniform(0.1, 0.3))

        elif operation == 'straight':
            modified = straight_line_reroute(
                mutate_path.copy(),
                path_electrode_name,
                electrode_zones,
                terminal_zones,
                n_points=len(mutate_path)  # Keep same number of points as original
            )

        elif operation == 'rotate_segment':
            modified = rotate_segment(
                mutate_path.copy(),
                seg_frac=random.uniform(0.2, 0.4),
                max_angle=np.pi * random.uniform(0.1, 0.3)
            )

        elif operation == 'multi_curve_cascade':
            modified = multi_curve_cascade(
                mutate_path.copy(),
                n_curves=random.randint(2, 4),
                curve_size=random.uniform(0.05, 0.2)
            )

        # Apply zone avoidance
        modified = avoid_electrode_zones(
            modified,
            path_electrode_name,
            electrode_zones,
            terminal_zones
        )

        # Fix endpoint to correct terminal if provided
        if mutate_terminal_pos is not None or target_electrode_pos is not None:
            modified = np.array(modified, dtype=float)
            if target_electrode_pos is not None:
                modified[0] = np.asarray(target_electrode_pos, dtype=float)
            if mutate_terminal_pos is not None:
                modified[-1] = np.asarray(mutate_terminal_pos, dtype=float)

        if locked_terminal_tail is not None:
            modified = _combine_path_outside_locked_terminal_tail(
                modified, locked_terminal_tail
            )
            if target_electrode_pos is not None:
                modified[0] = np.asarray(target_electrode_pos, dtype=float)
            if target_terminal_pos is not None:
                modified[-1] = np.asarray(target_terminal_pos, dtype=float)

        if path_has_trace_reentry(
            modified,
            path_electrode_name,
            target_terminal_name,
            electrode_zones,
            terminal_zones,
        ):
            continue

        # Validate bounds
        if is_path_within_bounds(modified, x_bounds):
            return modified

    print(f"🔴 OUCH! All {max_attempts} modification attempts failed, returning original path")
    print(f"Path: {path_electrode_name}, Length: {len(path)}, Operation: {operation}")
    fallback = path.copy()
    if target_electrode_pos is not None or target_terminal_pos is not None:
        fallback = pin_path_endpoints_2d(
            fallback,
            target_electrode_pos if target_electrode_pos is not None else path[0],
            target_terminal_pos if target_terminal_pos is not None else path[-1],
        )
    return fallback


def avoid_electrode_zones(path, path_electrode_name, electrode_zones, terminal_zones, buffer_multiplier=1.1):
    """More aggressive zone avoidance with guaranteed clearance"""
    if len(path) < 2:
        return path.copy()
    
    PUSH_PATH_BUFFER_MULTIPLIER = 1.3
    
    # Convert and validate path
    try:
        modified = np.array(path, dtype=float)
        if np.any(np.isnan(modified)) or np.any(np.isinf(modified)):
            print("Invalid path coordinates (NaN or inf values detected)")
            return path.copy()
            
        path_line = LineString(modified)
        if not path_line.is_valid:
            print("Invalid path geometry created")
            return path.copy()
    except Exception as e:
        print(f"Error creating path geometry: {str(e)}")
        return path.copy()

    for name, zone in electrode_zones['zones'].items():
        if name == path_electrode_name:
            continue
            
        zone_data = electrode_zones['metadata'][name]
        buffer_size = zone_data['buffer_size']
        zone_center = zone_data['center']
        
        try:
            # Validate and expand zone
            if not hasattr(zone, 'buffer'):
                print(f"Skipping invalid zone for electrode {name}")
                continue
                
            expanded_zone = zone.buffer(buffer_size * buffer_multiplier)
            if not expanded_zone.is_valid or expanded_zone.is_empty:
                print(f"Skipping invalid expanded zone for electrode {name}")
                continue
                
            # Safe intersection check
            if not path_line.is_valid or not expanded_zone.is_valid:
                print(f"Skipping intersection check due to invalid geometries for {name}")
                continue
                
            if path_line.intersects(expanded_zone):
                # Find nearby points
                distances = np.linalg.norm(modified - zone_center, axis=1)
                nearby_indices = np.where(distances < buffer_size * 2)[0]
                
                if len(nearby_indices) > 0:
                    # Calculate push direction and strength
                    push_dir = modified[nearby_indices] - zone_center
                    push_dir = push_dir / (np.linalg.norm(push_dir, axis=1, keepdims=True) + 1e-8)
                    push_strength = (buffer_size * PUSH_PATH_BUFFER_MULTIPLIER - distances[nearby_indices]) 
                    push_strength = np.maximum(0, push_strength) * 1.2
                    
                    # Apply push to nearby points
                    modified[nearby_indices] += push_dir * push_strength.reshape(-1,1)
                    
                    # Smooth the transition area
                    if len(nearby_indices) > 3:
                        start_idx = max(0, nearby_indices[0] - 2)
                        end_idx = min(len(modified), nearby_indices[-1] + 2)
                        section_to_smooth = modified[start_idx:end_idx]
                        if len(section_to_smooth) >= 4:  # Ensure minimum points for smoothing
                            modified[start_idx:end_idx] = smooth_path(section_to_smooth, strength=0.3)
                        
        except Exception as e:
            print(f"Error processing zone {name}: {str(e)}")
            continue
    
    # Final smoothing pass
    start = modified[0].copy()
    end = modified[-1].copy()
    smoothed = smooth_path(modified, strength=0.1)
    smoothed[0] = start
    smoothed[-1] = end
    return smoothed


# --------------------------
# Visualization
# --------------------------

def plot_quad_comparison(versions, electrodes, terminals, electrode_zones, terminal_zones, all_path_collisions, all_electrode_collisions):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    titles = ["Original", "First Modification", "Second Modification", "Third Modification"]
    
    for i, ax in enumerate(axs.flat):
        if i < len(versions):
            plot_single_version(
                ax, 
                versions[i], 
                electrodes, 
                terminals,
                electrode_zones, 
                terminal_zones, 
                all_path_collisions[i] if i < len(all_path_collisions) else None,
                all_electrode_collisions[i] if i < len(all_electrode_collisions) else None,
                titles[i]
            )
        else:
            ax.axis('off')  # Hide unused subplots
    
    plt.tight_layout()
    plt.show()

    
def plot_single_version(ax, paths, electrodes, terminals, electrode_zones, terminal_zones, 
                       path_collisions, electrode_collisions, title, dpi=100,
                       show_plot=True, save_path=None,
                       entry_points_by_electrode=None, slot_index_by_electrode=None):
    """Plot electrode paths with safety zones and collision visualization.
    
    Generates a 2D visualization of electrode connection paths with:
    - Electrode and terminal positions
    - Safety zones around each electrode
    - Visual markers for path collisions and electrode violations
    - Customizable output options

    Parameters
    ----------
    ax : matplotlib.axes.Axes or None
        Axis to plot on. If None, creates new figure/axis.
    paths : list of ndarray
        List of 2D path arrays (Nx2) for each electrode connection
    electrodes : dict
        Dictionary of electrode positions {name: [x,y]}
    terminals : dict
        Dictionary of terminal positions {name: [x,y]}
    electrode_zones : dict
        Dictionary containing electrode safety zones and metadata
    terminal_zones : dict
        Dictionary of terminal safety zones
    path_collisions : ndarray or None
        Array of collision points between paths (Mx2)
    electrode_collisions : ndarray or None
        Array of collision points with electrodes (Kx2)
    title : str
        Plot title
    dpi : int, optional
        Figure resolution in dots per inch (default: 100)
    show_plot : bool, optional
        Whether to display the plot (default: True)
    save_path : str, optional
        If provided, saves plot to this file path (default: None)

    Returns
    -------
    None
        Shows and/or saves plot based on parameters

    Notes
    -----
    - Electrode zones are shown as red translucent circles
    - Terminal zones are shown as green translucent circles
    - Path collisions marked with magenta 'x' markers
    - Electrode violations marked with cyan 'x' markers
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10), dpi=dpi)
    else:
        fig = ax.figure
    
    # Plot electrode zones
    for name, zone in electrode_zones['zones'].items():
        try:
            x, y = zone.exterior.xy
            ax.fill(x, y, color='red', alpha=0.15, edgecolor='red', linewidth=1)
        except AttributeError:
            continue
    
    # Plot terminal zones
    for name, zone in terminal_zones.items():
        try:
            x, y = zone.exterior.xy
            ax.fill(x, y, color='green', alpha=0.15, edgecolor='green', linewidth=1)
        except AttributeError:
            continue
    
    # Plot electrodes and terminals
    for name, pos in electrodes.items():
        ax.plot(pos[0], pos[1], 'ro', markersize=10)
        ax.text(pos[0], pos[1]+0.03, name, ha='center', va='bottom', fontsize=10)
    
    for name, pos in terminals.items():
        ax.plot(pos[0], pos[1], 'ks', markersize=14)
        ax.text(pos[0], pos[1]+0.05, name.split('_')[-1],
               ha='center', va='bottom', fontsize=12, weight='bold')
    
    # Draw paths
    colors = plt.cm.tab20(np.linspace(0, 1, len(paths)))
    for i, path in enumerate(paths):
        ax.plot(path[:,0], path[:,1], '-', color=colors[i], linewidth=3, alpha=0.8)

    legend_handles = []
    legend_labels = []

    if entry_points_by_electrode:
        for name, pt in entry_points_by_electrode.items():
            pt = np.asarray(pt, dtype=float)
            slot = (
                slot_index_by_electrode.get(name)
                if slot_index_by_electrode else None
            )
            ax.plot(pt[0], pt[1], 'D', color='gold', markersize=7, markeredgecolor='black',
                    markeredgewidth=0.8, zorder=12)
            label = f"{name}" if slot is None else f"{name} s{slot}"
            ax.text(pt[0], pt[1] - 0.06, label, ha='center', va='top', fontsize=7, color='goldenrod')
        legend_handles.append(
            plt.Line2D([0], [0], marker='D', color='gold', markersize=8, linestyle='None',
                       markeredgecolor='black')
        )
        legend_labels.append('Entry slots')
    
    if path_collisions is not None:
        n_path_collisions = len(path_collisions)
        path_scatter = ax.scatter(path_collisions[:,0], path_collisions[:,1],
                                c='magenta', marker='x', s=50, linewidths=1,
                                zorder=10)
        legend_handles.append(plt.Line2D([0], [0], marker='x', color='magenta', 
                                       markersize=10, linestyle='None',
                                       markeredgewidth=2))
        legend_labels.append(f'Path collisions ({n_path_collisions})')
    
    if electrode_collisions is not None:
        n_electrode_collisions = len(electrode_collisions)
        electrode_scatter = ax.scatter(electrode_collisions[:,0], electrode_collisions[:,1],
                                     c='cyan', marker='x', s=50, linewidths=1,
                                     zorder=11)
        legend_handles.append(plt.Line2D([0], [0], marker='x', color='cyan', 
                                       markersize=12, linestyle='None',
                                       markeredgewidth=2))
        legend_labels.append(f'Electrode violations ({n_electrode_collisions})')
    
    if legend_handles:
        ax.legend(legend_handles, legend_labels, loc='upper right')
    
    ax.set_title(title, fontsize=14)
    ax.grid(True, linestyle=':', alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    # New show/save functionality (only addition)
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    if show_plot:
        plt.show()
    elif save_path:
        plt.close(fig)


def plot_individual_2d_layout(
    SUBJECT_ID,
    INDIVIDUAL_ID,
    electrodes=None,
    fiducials=None,
    show_plot=False,
    save_path=None,
    dpi=120,
    ga_phase=None,
    fitness_score=None,
):
    """Plot one GA individual's 2D paths, zones, entry slots, and collisions."""
    log_path = f"data/output/logs/GA_{SUBJECT_ID}_{INDIVIDUAL_ID}_mod_connection_paths.json"
    if not os.path.exists(log_path):
        print(f"Warning: plot skipped — log not found: {log_path}")
        return

    with open(log_path, "r") as f:
        data = json.load(f)

    if electrodes is None:
        with open(f"data/json/electrode_positions_{SUBJECT_ID}.json") as f:
            electrodes = {k: np.array(v) for k, v in json.load(f).items()}
    if fiducials is None:
        with open(f"data/json/fiducials_{SUBJECT_ID}.json") as f:
            fiducials = {k: np.array(v) for k, v in json.load(f).items()}

    cz_pos = electrodes['Cz']
    electrodes_2d = {
        k: polar_projection(np.array([v]), cz_pos)[0] for k, v in electrodes.items()
    }
    terminals_2d = build_terminals_2d(electrodes_2d, fiducials, cz_pos)
    electrode_zones, terminal_zones = load_zones_for_subject(SUBJECT_ID)

    paths = [np.array(p['modified_path_2d']) for p in data['paths']]
    path_electrodes = [p['electrode'] for p in data['paths']]
    path_terminals = [p['terminal'] for p in data['paths']]
    entry_points, slot_index, _ = slot_metadata_from_child_paths(data['paths'])

    electrode_collisions = find_electrode_collisions(
        paths, electrode_zones, path_electrodes
    )
    electrode_violations = count_electrode_violations(
        paths, electrode_zones, path_electrodes
    )

    analysis = analyze_path_collisions(
        paths,
        terminal_zones,
        electrode_zones=electrode_zones,
        path_electrodes=path_electrodes,
        path_terminals=path_terminals,
    )
    analysis['electrode_violations'] = electrode_violations
    path_collisions = analysis.get('points')

    metrics = data.get('collision_metrics') or {}
    score = metrics.get('collision_score', analysis['collision_score'])
    saved_electrodes = metrics.get('electrode_violations')
    saved_note = ""
    if saved_electrodes is not None and int(saved_electrodes) != electrode_violations:
        saved_note = f", saved={int(saved_electrodes)}"
    phase_str = f"phase {ga_phase}" if ga_phase is not None else "phase ?"
    fitness_str = f" | fitness={fitness_score:.3f}" if fitness_score is not None else ""
    title = (
        f"Subject {SUBJECT_ID} | {INDIVIDUAL_ID} | {phase_str}{fitness_str}\n"
        f"score={score:.2f}, crossings={analysis['crossing_count']}, "
        f"electrodes={electrode_violations}{saved_note}, "
        f"trace_sep={analysis['trace_separation_deficit_normalized']:.2f}"
    )

    if save_path is None:
        save_path = (
            f"data/output/plots/individuals/GA_{SUBJECT_ID}_{INDIVIDUAL_ID}.png"
        )

    plot_single_version(
        ax=None,
        paths=paths,
        electrodes=electrodes_2d,
        terminals=terminals_2d,
        electrode_zones=electrode_zones,
        terminal_zones=terminal_zones,
        path_collisions=path_collisions,
        electrode_collisions=electrode_collisions,
        title=title,
        dpi=dpi,
        show_plot=show_plot,
        save_path=save_path,
        entry_points_by_electrode=entry_points or None,
        slot_index_by_electrode=slot_index or None,
    )
    if save_path and not show_plot:
        print(f"  📊 Saved 2D plot: {save_path}")
    
    
def create_uv_grid(mesh, cz_pos, resolution=20):
    """Create a UV grid mapping head surface to 2D plane"""
    # Create polar projection for all mesh points
    points_3d = mesh.points
    points_2d = polar_projection(points_3d, cz_pos)
    
    # Find min/max for grid bounds
    min_x, min_y = np.min(points_2d, axis=0)
    max_x, max_y = np.max(points_2d, axis=0)
    
    # Create grid in 2D space
    x = np.linspace(min_x, max_x, resolution)
    y = np.linspace(min_y, max_y, resolution)
    grid_2d = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
    
    # Find corresponding 3D points for each grid cell
    kdtree_3d = KDTree(points_3d)
    kdtree_2d = KDTree(points_2d)
    
    grid_3d = []
    for point in grid_2d:
        _, idx = kdtree_2d.query(point)
        grid_3d.append(points_3d[idx])
    
    return {
        'grid_2d': grid_2d,
        'grid_3d': np.array(grid_3d),
        'bounds': [min_x, max_x, min_y, max_y]
    }
    
def straight_line_reroute(path, electrode_name, electrode_zones, terminal_zones, n_points=20):
    """
    Creates a straight-line path from electrode to terminal, then applies avoidance logic.
    
    Args:
        path (np.ndarray): Original path (Nx2)
        electrode_name (str): Name of the electrode (for zone avoidance)
        electrode_zones (dict): Electrode safety zones
        terminal_zones (dict): Terminal safety zones
        n_points (int): Number of points in the final path
        
    Returns:
        np.ndarray: Modified path (n_points x 2)
    """
    if len(path) < 2:
        return path.copy()
    
    # Extract start (electrode) and end (terminal) points
    start = path[0]
    end = path[-1]
    
    # Create a straight line (initially just start and end)
    straight_path = np.linspace(start, end, n_points)
    
    # Apply electrode avoidance to push the path away from other electrodes
    modified_path = avoid_electrode_zones(
        straight_path,
        electrode_name,
        electrode_zones,
        terminal_zones
    )
    
    # Light smoothing to remove any kinks from avoidance
    smoothed_path = smooth_path(modified_path, strength=0.1)
    
    # Ensure endpoints remain exact
    smoothed_path[0], smoothed_path[-1] = start, end
    
    return smoothed_path

        

def save_modified_paths_v2(original_connections, original_paths_2d, modified_paths, 
                        n_collisions, cz_pos, filename, visualize=False,
                        collision_metrics=None, electrodes_2d=None, terminals_2d=None,
                        entry_points_by_electrode=None, slot_index_by_electrode=None,
                        uv_grid=None, SUBJECT_ID=None):
    """Save modified 2D paths with UV grid for 3D reconstruction."""
    if uv_grid is None:
        if SUBJECT_ID is None:
            raise ValueError("SUBJECT_ID required when uv_grid is not provided")
        uv_grid = get_cached_uv_grid(SUBJECT_ID, cz_pos, resolution=UV_GRID_RESOLUTION)

    if collision_metrics is not None:
        collision_metrics = {
            k: _json_safe_number(v) for k, v in collision_metrics.items()
        }

    output_data = {
        'metadata': {
            'projection_center': np.asarray(cz_pos, dtype=float).tolist(),
            'timestamp': datetime.datetime.now().isoformat(),
            'grid_resolution': int(UV_GRID_RESOLUTION),
            'grid_bounds': [float(b) for b in uv_grid['bounds']],
            'terminal_entry_slot_spacing': float(TERMINAL_ENTRY_SLOT_SPACING),
        },
        'collision_metrics': collision_metrics or {
            'collision_score': _json_safe_number(n_collisions),
        },
        'uv_grid': {
            'points_2d': uv_grid['grid_2d'].tolist(),
            'points_3d': uv_grid['grid_3d'].tolist()
        },
        'paths': []
    }
    
    for orig_conn, orig_2d, mod_2d in zip(original_connections, original_paths_2d, modified_paths):
        electrode = orig_conn['electrode']
        path_meta = {
            'electrode': electrode,
            'terminal': orig_conn['terminal'],
        }
        if entry_points_by_electrode and electrode in entry_points_by_electrode:
            path_meta['entry_point_2d'] = np.asarray(
                entry_points_by_electrode[electrode], dtype=float
            ).tolist()
        if slot_index_by_electrode and electrode in slot_index_by_electrode:
            path_meta['slot_index'] = int(slot_index_by_electrode[electrode])
        if electrodes_2d is not None and terminals_2d is not None:
            orig_2d = pin_path_endpoints_2d(
                orig_2d,
                electrodes_2d[electrode],
                terminals_2d[orig_conn['terminal']],
            )
            mod_2d = pin_path_endpoints_2d(
                mod_2d,
                electrodes_2d[electrode],
                path_end_target(path_meta, terminals_2d),
            )
        output_data['paths'].append({
            **path_meta,
            'original_path_2d': np.asarray(orig_2d).tolist(),
            'modified_path_2d': np.asarray(mod_2d).tolist(),
            'n_collisions': _json_safe_number(n_collisions),
            'original_length': float(orig_conn['path_length']),
            'original_path_3d': np.asarray(orig_conn['path_points'], dtype=float).tolist(),
        })
    
    output_dir = os.path.dirname(filename)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(output_data, f, indent=2)
    

# --------------------------
# Main Execution
# --------------------------

def main():
    OUTPUTPATH = 'testing/mod_connection_paths.json'
    # Load data
    with open('data/json/electrode_positions_2.json') as f:
        electrodes = {k: np.array(v) for k,v in json.load(f).items()}
    with open('data/json/fiducials_2.json') as f:
        fiducials = {k: np.array(v) for k,v in json.load(f).items()}
    with open('data/json/init_connection_paths_2.json') as f:
        connections = json.load(f)
        
    mesh = _pyvista().read("data/cleaned_scans/2.stl")
    
    # Use consistent terminal assignments instead of shortest path selection
    SUBJECT_ID = 2  # Hard-coded for this test function
    init_conn = _init_conn()
    initial_assignments = init_conn.load_or_create_terminal_assignments(SUBJECT_ID, connections)

    optimized = init_conn.select_connections_for_assignments(
        connections, initial_assignments, electrodes=electrodes
    )
    
    # Project to 2D
    cz_pos = electrodes['Cz']
    electrodes_2d = {k: polar_projection(np.array([v]), cz_pos)[0] for k,v in electrodes.items()}
    original_paths = [polar_projection(np.array(conn['path_points']), cz_pos) for conn in optimized]
    
    # Position terminals and determine x-bounds
    max_dist = 1.2 * max(np.linalg.norm(list(electrodes_2d.values()), axis=1))
    terminals_2d = {}
    terminal_x_positions = []
    
    for term in ['TERMINAL_LEFT', 'TERMINAL_RIGHT']:
        if term in fiducials:
            pos = polar_projection(np.array([fiducials[term]]), cz_pos)[0]
            angle = np.arctan2(pos[1], pos[0])
            terminal_pos = max_dist * np.array([np.cos(angle), np.sin(angle)])
            terminals_2d[term] = terminal_pos
            terminal_x_positions.append(terminal_pos[0])
    
    # Set x-bounds with XX% padding beyond terminal positions (OLD- this was for if they are left and right )
    x_buffer = 0.3 * abs(terminal_x_positions[0] - terminal_x_positions[1])
    x_bounds = (min(terminal_x_positions) - x_buffer, max(terminal_x_positions) + x_buffer)
    
    # Create zones    
    electrode_zones, terminal_zones = create_zones(electrodes_2d, terminals_2d)
    
    # Create progressive modifications with boundary enforcement and zone avoidance
    versions = [original_paths]
    for _ in range(3):
        new_version = [
            randomly_modify_path(
                path.copy(), 
                optimized[i]['electrode'],  # Pass the electrode name
                electrode_zones,            # Pass electrode zones
                terminal_zones,             # Pass terminal zones
                x_bounds=x_bounds           # Pass bounds
            ) 
            for i, path in enumerate(versions[-1])
        ]
        versions.append(new_version)
    
    
    # Find collisions
    all_collisions = []
    for version in versions:
        print(version)
        exit()
        collisions = find_path_collisions(version, terminal_zones)
        all_collisions.append(collisions)

    # Find both types of collisions
    all_path_collisions = []
    all_electrode_collisions = []
    path_electrodes = [conn['electrode'] for conn in optimized]

    for version in versions:
        path_collisions = find_path_collisions(version, terminal_zones)
        electrode_collisions = find_electrode_collisions(version, electrode_zones, path_electrodes)
        all_path_collisions.append(path_collisions)
        all_electrode_collisions.append(electrode_collisions)

    # Visual comparison
    plot_quad_comparison(versions, electrodes_2d, terminals_2d,
                        electrode_zones, terminal_zones,
                        all_path_collisions, all_electrode_collisions)

    # SAVE THE FINAL STUFF
    final_version = versions[-1]
    final_collisions = all_collisions[-1]
    n_collisions = len(final_collisions) if final_collisions is not None else 0

    print("DISABLED SAVING DUE TO LEGACY STATUS")

if __name__ == "__main__":
    print("\nWe're also gonna show the grid net now lol...")
    main()


def create_and_save_new_2D_alteration(SUBJECT_ID: int, original_paths: list = None , electrodes: dict = None, fiducials: dict = None, INDIVIDUAL_ID: str=None):
    """Create a new 2D version of a new path version with random modifications based on the input paths."""
    
    if original_paths is None:
        raise ValueError("original_paths must be parsed as argument to create a random alteration set.")
    elif SUBJECT_ID == None:
        raise ValueError("SUBJECT_ID must be provided to create alterations.")
    elif INDIVIDUAL_ID is None:
        raise ValueError("INDIVIDUAL_ID must be provided to create alterations.")
    
    
    OUTPUTPATH = f"data/output/logs/GA_{SUBJECT_ID}_{INDIVIDUAL_ID}_mod_connection_paths.json"
    ctx = get_subject_layout(SUBJECT_ID, electrodes, fiducials, original_paths)
    optimized = ctx['optimized']
    electrodes_2d = ctx['electrodes_2d']
    terminals_2d = ctx['terminals_2d']
    electrode_zones = ctx['electrode_zones']
    terminal_zones = ctx['terminal_zones']
    cz_pos = ctx['cz_pos']
    x_bounds = ctx['x_bounds']
    path_electrodes = ctx['path_electrodes']
    path_terminals = ctx['path_terminals']
    original_paths_2d = ctx['original_paths_2d']

    # Initialization: straight chords for ordering, then fixed 4.5-spaced entry slots.
    chord_to_terminal = straighten_paths_to_chords(
        original_paths_2d,
        path_electrodes,
        path_terminals,
        electrodes_2d,
        terminals_2d,
    )
    entry_points, slot_index, slot_order = assign_terminal_entry_slots(
        path_electrodes,
        path_terminals,
        chord_to_terminal,
        terminal_zones,
        terminals_2d=terminals_2d,
        spacing=TERMINAL_ENTRY_SLOT_SPACING,
    )
    straight_paths = straighten_paths_to_chords(
        original_paths_2d,
        path_electrodes,
        path_terminals,
        electrodes_2d,
        terminals_2d,
        entry_points_2d=entry_points,
    )
    path_specs = []
    for conn in optimized:
        spec = dict(conn)
        electrode = conn['electrode']
        spec['entry_point_2d'] = entry_points[electrode].tolist()
        spec['slot_index'] = int(slot_index[electrode])
        path_specs.append(spec)
    final_version = pin_paths_to_layout(
        straight_paths, path_specs, electrodes_2d, terminals_2d
    )
    final_analysis = analyze_path_collisions(
        final_version,
        terminal_zones,
        electrode_zones=electrode_zones,
        path_electrodes=path_electrodes,
        path_terminals=path_terminals,
        metrics_mode='electrodes_only',
    )
    collision_metrics = analysis_to_collision_metrics(final_analysis)

    save_modified_paths_v2(
        original_connections=optimized,
        original_paths_2d=original_paths_2d,
        modified_paths=final_version,
        n_collisions=final_analysis['collision_score'],
        collision_metrics=collision_metrics,
        cz_pos=cz_pos,
        filename=OUTPUTPATH,
        visualize=False,
        electrodes_2d=electrodes_2d,
        terminals_2d=terminals_2d,
        entry_points_by_electrode=entry_points,
        slot_index_by_electrode=slot_index,
        SUBJECT_ID=SUBJECT_ID,
    )
    return
########### FUNCTIONS CALLED IN GA SCRIPT (BY GENETIC OPERATORS) #############

def getAltered2DpathFromChild(child: list, electrode_path_target: str)-> list:
    for path in child['paths']:
        if path['electrode'] == electrode_path_target:
            return path['modified_path_2d']

def mutateRandomElectrodePathsForSelectedChild(child: list, original_paths, electrodes, fiducials, MUTATE_N_ELECTRODES_PERCENTAGE: float=None, SUBJECT_ID: int=None, ga_phase=2) -> list:
    """Mutate a random percentage of the child's paths by applying random modifications.
    
    - Returns the same structure of the input.
    - `MUTATE_N_ELECTRODES_PERCENTAGE` determines how many paths to mutate (must be between 0 and 1).
    - Mutation is done by calling `randomly_modify_path` on each selected path.
    - Phase 2: revert a mutation if it increases layout crossing count vs immediately before.
    - All phases: revert a mutation that introduces trace reentry (self-cross or zone re-entry).
    """
    n_paths = len(child['paths'])
    n_mutations = max(1, int(n_paths * MUTATE_N_ELECTRODES_PERCENTAGE))
    ctx = get_subject_layout(SUBJECT_ID, electrodes, fiducials, original_paths)
    electrodes_2d = ctx['electrodes_2d']
    terminals_2d = ctx['terminals_2d']
    electrode_zones = ctx['electrode_zones']
    terminal_zones = ctx['terminal_zones']
    x_bounds = ctx['x_bounds']
    path_electrodes = [p['electrode'] for p in child['paths']]
    path_terminals = [p['terminal'] for p in child['paths']]
    reject_crossing_increase = ga_phase == 2
    for mutation_counter in range(n_mutations):
        # mutate one path only for testing
        rand_number = random.randint(0, n_paths-1) # may be happening to be the same path again but whatever
        random_path_for_mutation = child['paths'][rand_number]['electrode']
        assigned_terminal = child['paths'][rand_number]['terminal']
        target_terminal_pos = path_end_target(child['paths'][rand_number], terminals_2d)
        target_electrode_pos = electrodes_2d[random_path_for_mutation]
        
        # Mutate from the child's current path so improvements accumulate across generations
        before_paths = [np.array(p['modified_path_2d']) for p in child['paths']]
        crossings_before = (
            _count_crossings_involving_path(
                rand_number,
                before_paths,
                path_terminals,
                terminal_zones,
                electrode_zones,
            )
            if reject_crossing_increase
            else None
        )
        current_path = np.array(child['paths'][rand_number]['modified_path_2d'])
        mutated_path = randomly_modify_path(
            current_path.copy(),
            random_path_for_mutation,
            electrode_zones,
            terminal_zones,
            x_bounds=x_bounds,
            target_electrode_pos=target_electrode_pos,
            target_terminal_pos=target_terminal_pos,
            target_terminal_name=assigned_terminal,
        )

        
        # now, replace the updated (mutated) channel of the child...
        replaced = False
        for electrode_entry in child['paths']:
            if electrode_entry['electrode'] == random_path_for_mutation:
                electrode_entry['modified_path_2d'] = mutated_path
                replaced = True
        if replaced == False:
            raise ValueError("\n🔴 WHY WAS IT NOT REPLACED? 🔴\n")

        if reject_crossing_increase:
            after_paths = [np.array(p['modified_path_2d']) for p in child['paths']]
            crossings_after = _count_crossings_involving_path(
                rand_number,
                after_paths,
                path_terminals,
                terminal_zones,
                electrode_zones,
            )
            if crossings_after > crossings_before:
                for electrode_entry in child['paths']:
                    if electrode_entry['electrode'] == random_path_for_mutation:
                        electrode_entry['modified_path_2d'] = current_path.tolist()
                continue

        if path_has_trace_reentry(
            mutated_path,
            random_path_for_mutation,
            assigned_terminal,
            electrode_zones,
            terminal_zones,
        ):
            for electrode_entry in child['paths']:
                if electrode_entry['electrode'] == random_path_for_mutation:
                    electrode_entry['modified_path_2d'] = current_path.tolist()
            continue

    return pin_child_paths_2d(child, electrodes_2d, terminals_2d)
            
    

def only_save_new_2D_alteration(
    child: dict,
    SUBJECT_ID: int,
    electrodes: dict = None,
    fiducials: dict = None,
    INDIVIDUAL_ID: str = None,
    metrics_mode='full',
    collision_metrics=None,
    ga_phase=2,
):
    OUTPUTPATH = f"data/output/logs/GA_{SUBJECT_ID}_{INDIVIDUAL_ID}_mod_connection_paths.json"
    ctx = get_subject_layout(SUBJECT_ID, electrodes, fiducials)
    optimized = ctx['optimized']
    electrodes_2d = ctx['electrodes_2d']
    terminals_2d = ctx['terminals_2d']
    electrode_zones = ctx['electrode_zones']
    terminal_zones = ctx['terminal_zones']
    cz_pos = ctx['cz_pos']
    path_electrodes = ctx['path_electrodes']
    path_terminals = ctx['path_terminals']
    original_paths_2d = ctx['original_paths_2d']

    pin_child_paths_2d(child, electrodes_2d, terminals_2d)
    paths_2d = [np.array(i['modified_path_2d']) for i in child['paths']]
    entry_points, slot_index, _ = slot_metadata_from_child_paths(child['paths'])

    if collision_metrics is None:
        collision_metrics = child.pop('collision_metrics', None)

    if collision_metrics is None:
        analysis = analyze_path_collisions(
            paths_2d=paths_2d,
            terminal_zones=terminal_zones,
            electrode_zones=electrode_zones,
            path_electrodes=path_electrodes,
            path_terminals=path_terminals,
            metrics_mode=metrics_mode,
            ga_phase=ga_phase,
            slot_index_by_electrode=slot_index or None,
        )
        path_length_excess = 0.0
        if ga_phase == 2 and metrics_mode != 'electrodes_only':
            path_length_excess = compute_layout_path_length_excess(
                paths_2d,
                path_electrodes,
                path_terminals,
                electrodes_2d,
                terminals_2d,
            )
        collision_metrics = finalize_collision_metrics(
            analysis, ga_phase=ga_phase, path_length_excess=path_length_excess
        )
        n_collisions = collision_metrics['layout_score']
    else:
        n_collisions = layout_score_from_metrics(collision_metrics)

    save_modified_paths_v2(
        original_connections=optimized,
        original_paths_2d=original_paths_2d,
        modified_paths=paths_2d,
        n_collisions=n_collisions,
        collision_metrics=collision_metrics,
        cz_pos=cz_pos,
        filename=OUTPUTPATH,
        visualize=False,
        electrodes_2d=electrodes_2d,
        terminals_2d=terminals_2d,
        entry_points_by_electrode=entry_points or None,
        slot_index_by_electrode=slot_index or None,
        SUBJECT_ID=SUBJECT_ID,
    )

    return 200













####### NEW ALTERATION OPERATORS FOR GA ##########
def add_midpoint_detour(path, detour_frac=0.2):
    """
    Insert a single detour waypoint at the midpoint, offset perpendicular to the local tangent,
    then re-spline to the original number of points.
    """
    n = len(path)
    mid_idx = n // 2
    p_prev, p_next = path[mid_idx-1], path[mid_idx+1]
    tangent = p_next - p_prev
    normal = np.array([-tangent[1], tangent[0]])
    normal /= np.linalg.norm(normal) + 1e-8

    # size scales with overall path span
    span = np.linalg.norm(path[-1] - path[0])
    detour = path[mid_idx] + normal * (detour_frac * span)

    # build new sequence and re-sample
    pts = np.vstack([path[:mid_idx],
                     detour.reshape(1,2),
                     path[mid_idx:]])
    tck, _ = splprep(pts.T, s=0)
    new_pts = splev(np.linspace(0,1,n), tck)
    out = np.column_stack(new_pts)
    out[0], out[-1] = path[0], path[-1]
    return out


def add_perlin_noise_warp(path, warp_scale=0.1, octaves=3, seed=None):
    """
    Displace each point by a smooth, low-frequency “noise” warp.
    We approximate Perlin by generating random control values along the path
    and interpolating them.
    """
    if seed is not None:
        np.random.seed(seed)
    n = len(path)
    # generate coarse random vectors
    k = max(2, octaves)
    controls = np.random.randn(k, 2)
    u_coarse = np.linspace(0, 1, k)
    u_fine = np.linspace(0, 1, n)
    # interpolate noise onto path
    noise = np.vstack([
        np.interp(u_fine, u_coarse, controls[:,0]),
        np.interp(u_fine, u_coarse, controls[:,1])
    ]).T
    # scale by path span
    span = np.linalg.norm(path[-1] - path[0])
    out = path + noise * (warp_scale * span)
    out[0], out[-1] = path[0], path[-1]
    return out


def rotate_segment(path, seg_frac=0.3, max_angle=np.pi/6):
    """
    Select a contiguous middle segment of length seg_frac * len(path),
    rotate it around its midpoint by a random angle up to max_angle, then re-spline.
    """
    n = len(path)
    L = int(n * seg_frac)
    start = (n - L)//2
    end = start + L
    segment = path[start:end].copy()

    # rotate
    theta = np.random.uniform(-max_angle, max_angle)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    center = segment.mean(axis=0)
    seg_rot = ((segment - center) @ R.T) + center

    # rebuild and smooth
    pts = np.vstack([path[:start], seg_rot, path[end:]])
    tck, _ = splprep(pts.T, s=0)
    new_pts = splev(np.linspace(0,1,n), tck)
    out = np.column_stack(new_pts)
    out[0], out[-1] = path[0], path[-1]
    return out


def multi_curve_cascade(path, n_curves=3, curve_size=0.1):
    """
    Apply n_curves sequential local Gaussian bumps of size curve_size,
    each on a random section of the path.
    """
    out = path.copy()
    for _ in range(n_curves):
        out = add_local_curve(out, curve_size=curve_size)
    out[0], out[-1] = path[0], path[-1]
    return out


def force_based_repulsion_sweep(path, other_paths, thresh=5.0, strength=2):
    """
    Nudge each point away from any point on other_paths that is
    closer than thresh. Then re-smooth lightly.
    """
    out = path.astype(float).copy()
    for i, p in enumerate(path):
        disp = np.zeros(2)
        for op in other_paths:
            # skip if same object
            if np.array_equal(op, path): continue
            dists = p - op
            norms = np.linalg.norm(dists, axis=1)
            mask = norms < thresh
            if mask.any():
                # repel: sum of unit vectors
                vectors = dists[mask] / (norms[mask,None] + 1e-8)
                disp += vectors.sum()
        out[i] += strength * disp
    out[0], out[-1] = path[0], path[-1]
    # light smoothing
    tck, _ = splprep(out.T, s=0.1*len(path))
    pts = splev(np.linspace(0,1,len(path)), tck)
    res = np.column_stack(pts)
    res[0], res[-1] = path[0], path[-1]
    return res


def waypoint_rerouting(path, safe_waypoints, n_points=100):
    """
    Pick one random waypoint from safe_waypoints, then spline through
    [start -> waypoint -> end] into n_points samples.
    """
    wp = safe_waypoints[np.random.randint(len(safe_waypoints))]
    pts = np.vstack([path[0], wp, path[-1]])
    tck, _ = splprep(pts.T, s=0)
    new = splev(np.linspace(0,1,n_points), tck)
    out = np.column_stack(new)
    out[0], out[-1] = path[0], path[-1]
    return out


def apply_smart_collision_resolution(
    child,
    SUBJECT_ID,
    electrodes,
    fiducials,
    original_paths,
    use_greedy_aggressive=False,
    use_gentle_resolution=True,
    greedy_electrodes_only=False,
    phase2_max_pair_rounds: int | None = None,
    force_trace_resolution: bool = False,
    focus_separation: bool = False,
    fixed_endpoints: bool = False,
    max_crossing_count: int | None = None,
):
    """Apply smart collision resolution if there are many collisions"""
    
    THRESHOLD_COLLISIONS = 500 # UNDER THIS VALUE WE APPLY SMART COLLISION RESOLUTION
    
    ctx = get_subject_layout(SUBJECT_ID, electrodes, fiducials, original_paths)
    electrodes_2d = ctx['electrodes_2d']
    terminals_2d = ctx['terminals_2d']
    electrode_zones = ctx['electrode_zones']
    terminal_zones = ctx['terminal_zones']
    x_bounds = ctx['x_bounds']

    pin_child_paths_2d(child, electrodes_2d, terminals_2d)
    all_paths_2d = [np.array(path['modified_path_2d']) for path in child['paths']]
    path_electrodes = [path['electrode'] for path in child['paths']]
    path_terminals = [path['terminal'] for path in child['paths']]
    entry_points, slot_index, slot_order = slot_metadata_from_child_paths(child['paths'])
    metrics_mode = 'electrodes_only' if greedy_electrodes_only else 'full'
    analysis = analyze_path_collisions(
        all_paths_2d,
        terminal_zones,
        electrode_zones=electrode_zones,
        path_electrodes=path_electrodes,
        path_terminals=path_terminals,
        metrics_mode=metrics_mode,
        slot_index_by_electrode=slot_index or None,
    )

    if greedy_electrodes_only:
        if is_layout_electrode_free(analysis):
            child['collision_metrics'] = analysis_to_collision_metrics(analysis)
            return child
        print(
            f"Phase 1 resolution: electrode violations={analysis['electrode_violations']} "
            f"(crossings={analysis['crossing_count']} ignored for fitness)"
        )
    elif is_layout_collision_free(analysis):
        child['collision_metrics'] = analysis_to_collision_metrics(analysis)
        return child
    else:
        print(
            f"Detected collision score {analysis['collision_score']:.2f} "
            f"(crossings={analysis['crossing_count']}, overlap={analysis['overlap_length']:.2f}, "
            f"trace_sep={analysis['trace_separation_deficit_normalized']:.2f}, "
            f"min_trace_sep={analysis['min_trace_separation']:.2f}, "
            f"electrodes={analysis['electrode_violations']})"
        )

    if not entry_points:
        chord_to_terminal = straighten_paths_to_chords(
            all_paths_2d,
            path_electrodes,
            path_terminals,
            electrodes_2d,
            terminals_2d,
        )
        entry_points, slot_index, slot_order = assign_terminal_entry_slots(
            path_electrodes,
            path_terminals,
            chord_to_terminal,
            terminal_zones,
            terminals_2d=terminals_2d,
            spacing=TERMINAL_ENTRY_SLOT_SPACING,
        )
        for path_entry in child['paths']:
            electrode = path_entry['electrode']
            path_entry['entry_point_2d'] = entry_points[electrode].tolist()
            path_entry['slot_index'] = int(slot_index[electrode])

    resolved_paths = all_paths_2d

    run_resolution = (
        greedy_electrodes_only and analysis['electrode_violations'] > 0
    ) or (
        not greedy_electrodes_only
        and (
            analysis['collision_score'] <= THRESHOLD_COLLISIONS
            or force_trace_resolution
        )
    )
    if run_resolution:
        score_label = (
            f"electrode violations={analysis['electrode_violations']}"
            if greedy_electrodes_only
            else f"score {analysis['collision_score']:.2f}"
        )
        print(f'Applying smart collision resolution ({score_label})...')

        x_bounds = ctx['x_bounds']
        try:
            resolved_paths = smart_collision_resolution(
                paths=all_paths_2d,
                path_electrodes=path_electrodes,
                path_terminals=path_terminals,
                electrode_zones=electrode_zones,
                terminal_zones=terminal_zones,
                x_bounds=x_bounds,
                electrodes_2d=electrodes_2d,
                terminals_2d=terminals_2d,
                entry_points_2d=entry_points,
                slot_order_by_terminal=slot_order,
                slot_index_by_electrode=slot_index,
                max_attempts=10,
                use_greedy_aggressive=use_greedy_aggressive,
                use_gentle_resolution=use_gentle_resolution,
                greedy_electrodes_only=greedy_electrodes_only,
                phase2_max_pair_rounds=phase2_max_pair_rounds,
                focus_separation=focus_separation,
                fixed_endpoints=fixed_endpoints,
                max_crossing_count=max_crossing_count,
            )
            
            if greedy_electrodes_only:
                final_violations = find_electrode_collisions(
                    resolved_paths, electrode_zones, path_electrodes
                )
                if final_violations is not None:
                    print(
                        f"Warning: could not resolve all electrode violations "
                        f"(remaining: {len(final_violations)}); keeping partial repair"
                    )
                else:
                    print("All electrode zones respected")
                
        except Exception as e:
            print(f"Error in smart collision resolution: {str(e)}")
            resolved_paths = all_paths_2d
    
    # Update child paths only if resolution was successful
    for i, path in enumerate(child['paths']):
        path['modified_path_2d'] = pin_path_endpoints_2d(
            resolved_paths[i],
            electrodes_2d[path['electrode']],
            path_end_target(path, terminals_2d),
        ).tolist()

    if not run_resolution:
        child['collision_metrics'] = analysis_to_collision_metrics(analysis)
    else:
        final_analysis = analyze_path_collisions(
            resolved_paths,
            terminal_zones,
            electrode_zones=electrode_zones,
            path_electrodes=path_electrodes,
            path_terminals=path_terminals,
            metrics_mode=metrics_mode,
            slot_index_by_electrode=slot_index or None,
        )
        child['collision_metrics'] = analysis_to_collision_metrics(final_analysis)
    
    return child


def is_path_locked(path, start, end, max_detour_ratio=PATH_LOCKED_MAX_DETOUR_RATIO):
    """
    True for straight electrode→terminal feeders (path ≈ chord).

    Locked paths should not be greedy-rerouted; reroute the crossing partner instead.
    """
    path = np.asarray(path, dtype=float)
    if len(path) < 2:
        return True

    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    chord = float(np.linalg.norm(end - start))
    if chord < 1e-6:
        return True

    path_length = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    return (path_length / chord) <= max_detour_ratio


def _path_flexibility_ratio(path, start, end):
    """Higher means more bends / more freedom to reroute without leaving the corridor."""
    path = np.asarray(path, dtype=float)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    chord = float(np.linalg.norm(end - start))
    if chord < 1e-6:
        return 0.0
    path_length = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    return path_length / chord


def find_point_crossing_pairs(paths, path_terminals, terminal_zones, electrode_zones):
    """Return (i, j, crossing_xy) for path pairs with a point intersection (not overlap tail)."""
    pairs = []
    dense_path_cache = _build_crossing_detection_path_cache(
        paths, path_terminals, electrode_zones
    )

    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            inter, _pair_merge, crossing_points = _compute_path_pair_crossing_geometry(
                paths[i],
                paths[j],
                path_terminals[i] if i < len(path_terminals) else None,
                path_terminals[j] if j < len(path_terminals) else None,
                terminal_zones,
                electrode_zones,
                dense_path_cache=dense_path_cache,
                path_idx_a=i,
                path_idx_b=j,
            )
            if inter is None or (inter.is_empty and not crossing_points):
                continue

            emitted = []
            for pt in crossing_points:
                emitted.append(np.asarray(pt, dtype=float))
            if not emitted:
                if inter.geom_type == 'Point':
                    emitted.append(np.array([inter.x, inter.y], dtype=float))
                elif inter.geom_type == 'MultiPoint':
                    for pt in inter.geoms:
                        emitted.append(np.array([pt.x, pt.y], dtype=float))

            for pt in emitted:
                pairs.append((i, j, pt))

    return pairs


def _greedy_aggressive_candidate_indices(
    paths,
    path_electrodes,
    path_terminals,
    terminal_zones,
    electrode_zones,
    collision_scores,
    electrodes_2d=None,
    terminals_2d=None,
    max_paths=3,
):
    """
    Pick greedy reroute candidates: flexible path in each crossing pair, never locked feeders.

    Returns (candidate_indices, locked_indices, corridor_obstacles_by_idx).
    corridor_obstacles_by_idx maps reroute index → locked partner path(s) to hull around.
    """
    locked_indices = set()
    for idx, path in enumerate(paths):
        start = (
            electrodes_2d[path_electrodes[idx]]
            if electrodes_2d is not None
            else path[0]
        )
        end = (
            terminals_2d[path_terminals[idx]]
            if terminals_2d is not None
            else path[-1]
        )
        if is_path_locked(path, start, end):
            locked_indices.add(idx)

    crossing_pairs = find_point_crossing_pairs(
        paths, path_terminals, terminal_zones, electrode_zones
    )

    def _pick_flexible_partner(i, j):
        if i in locked_indices and j not in locked_indices:
            return j, i
        if j in locked_indices and i not in locked_indices:
            return i, j
        if i in locked_indices and j in locked_indices:
            return None, None
        fi = _path_flexibility_ratio(
            paths[i],
            electrodes_2d[path_electrodes[i]] if electrodes_2d is not None else paths[i][0],
            terminals_2d[path_terminals[i]] if terminals_2d is not None else paths[i][-1],
        )
        fj = _path_flexibility_ratio(
            paths[j],
            electrodes_2d[path_electrodes[j]] if electrodes_2d is not None else paths[j][0],
            terminals_2d[path_terminals[j]] if terminals_2d is not None else paths[j][-1],
        )
        if fi >= fj:
            return i, j
        return j, i

    ordered = []
    seen = set()
    corridor_obstacles_by_idx = {}

    for i, j, _crossing in crossing_pairs:
        reroute_idx, locked_idx = _pick_flexible_partner(i, j)
        if reroute_idx is None:
            continue
        if reroute_idx not in seen:
            ordered.append(reroute_idx)
            seen.add(reroute_idx)
        if locked_idx is not None:
            corridor_obstacles_by_idx.setdefault(reroute_idx, []).append(
                np.asarray(paths[locked_idx], dtype=float).copy()
            )

    for idx in sorted(collision_scores, key=lambda idy: -collision_scores[idy]):
        if len(ordered) >= max_paths:
            break
        if idx in locked_indices or idx in seen:
            continue
        if collision_scores[idx] > 0:
            ordered.append(idx)
            seen.add(idx)

    return ordered[:max_paths], locked_indices, corridor_obstacles_by_idx


def detect_trapped_path_indices(
    paths,
    path_electrodes,
    path_terminals,
    electrode_zones,
    terminal_zones,
    terminals_2d=None,
):
    """
    Return indices of paths that need hull routing rather than local deformation.

    Uses the combined forbidden region (foreign electrode zones + buffered foreign
    traces). A path is trapped when that region blocks the direct route, when it
    intersects a foreign electrode zone, or when it collides with other traces while
    running through the forbidden pocket.
    """
    from PYTHON.GA import greed

    trapped = []
    for idx in range(len(paths)):
        if greed.is_path_topologically_trapped(
            idx,
            paths,
            path_electrodes,
            path_terminals,
            electrode_zones,
            terminal_zones,
            terminals_2d=terminals_2d,
        ):
            trapped.append(idx)
    return trapped


def _phase2_candidate_indices(collision_scores, trapped_indices, max_paths=3):
    """Prioritize trapped paths, then the most collision-heavy remaining paths."""
    ordered = []
    seen = set()
    for idx in trapped_indices:
        if idx not in seen:
            ordered.append(idx)
            seen.add(idx)

    for idx in sorted(collision_scores, key=lambda idx: -collision_scores[idx]):
        if len(ordered) >= max_paths:
            break
        if collision_scores[idx] > 0 and idx not in seen:
            ordered.append(idx)
            seen.add(idx)

    return ordered[:max_paths]


def is_layout_electrode_free(collision_analysis) -> bool:
    """True when no path intersects a foreign electrode zone."""
    return collision_analysis['electrode_violations'] == 0


def is_layout_phase1_ready(collision_analysis) -> bool:
    """Phase 1 complete: no foreign electrode zone violations."""
    return is_layout_electrode_free(collision_analysis)


def straighten_paths_to_chords(
    paths,
    path_electrodes,
    path_terminals,
    electrodes_2d,
    terminals_2d,
    entry_points_2d=None,
    n_points=50,
):
    """Replace each path with a straight electrode→entry (or terminal) chord."""
    straight = []
    for idx, path in enumerate(paths):
        start = np.asarray(electrodes_2d[path_electrodes[idx]], dtype=float)
        electrode_name = path_electrodes[idx]
        if entry_points_2d and electrode_name in entry_points_2d:
            end = np.asarray(entry_points_2d[electrode_name], dtype=float)
        else:
            end = np.asarray(terminals_2d[path_terminals[idx]], dtype=float)
        n = max(len(path), n_points)
        straight.append(np.linspace(start, end, n))
    return straight


def compute_layout_path_length_excess(
    paths,
    path_electrodes,
    path_terminals,
    electrodes_2d,
    terminals_2d,
):
    """Sum of (path_length/chord - 1) per trace; 0 when all paths are straight chords."""
    excess = 0.0
    for path, electrode_name, terminal_name in zip(paths, path_electrodes, path_terminals):
        start = np.asarray(electrodes_2d[electrode_name], dtype=float)
        end = np.asarray(terminals_2d[terminal_name], dtype=float)
        chord = float(np.linalg.norm(end - start))
        if chord < 1e-6:
            continue
        path_length = float(np.sum(np.linalg.norm(np.diff(np.asarray(path, dtype=float), axis=0), axis=1)))
        excess += max(0.0, path_length / chord - 1.0)
    return excess


def rank_individual_layout_score(
    subject_id,
    individual_id,
    log_dir,
    layout_ctx=None,
) -> float:
    """
    Comparable phase-2 layout_score for ranking individuals across generations.
    Lower is better.
    """
    log_path = os.path.join(
        log_dir, f"GA_{subject_id}_{individual_id}_mod_connection_paths.json"
    )
    with open(log_path, "r") as f:
        data = json.load(f)

    paths = [np.array(p["modified_path_2d"]) for p in data["paths"]]
    path_electrodes = [p["electrode"] for p in data["paths"]]
    path_terminals = [p["terminal"] for p in data["paths"]]

    if layout_ctx is None:
        electrode_zones, terminal_zones = load_zones_for_subject(subject_id)
        subj = get_subject_layout(subject_id)
        electrodes_2d = subj["electrodes_2d"]
        terminals_2d = subj["terminals_2d"]
    else:
        electrode_zones = layout_ctx["electrode_zones"]
        terminal_zones = layout_ctx["terminal_zones"]
        electrodes_2d = layout_ctx["electrodes_2d"]
        terminals_2d = layout_ctx["terminals_2d"]

    path_length_excess = compute_layout_path_length_excess(
        paths, path_electrodes, path_terminals, electrodes_2d, terminals_2d
    )
    analysis = analyze_path_collisions(
        paths,
        terminal_zones,
        electrode_zones=electrode_zones,
        path_electrodes=path_electrodes,
        path_terminals=path_terminals,
        ga_phase=2,
        path_length_excess=path_length_excess,
    )
    return float(analysis["layout_score"])


def best_individual_by_layout_score(
    fitness_tracker,
    subject_id,
    log_dir,
    layout_ctx=None,
):
    """Return (individual_id, layout_score) for the best layout in a run."""
    best_id = None
    best_score = float("inf")
    for individual_id in fitness_tracker:
        score = rank_individual_layout_score(
            subject_id, individual_id, log_dir, layout_ctx=layout_ctx
        )
        if score < best_score:
            best_score = score
            best_id = individual_id
    if best_id is None:
        raise ValueError("No individuals found in fitness tracker")
    return best_id, best_score


def best_per_generation_by_layout_score(
    fitness_tracker,
    subject_id,
    log_dir,
    layout_ctx=None,
):
    """Return {generation: (individual_id, layout_score)} using lowest score per gen."""
    by_generation = defaultdict(list)
    for individual_id in fitness_tracker:
        generation = int(individual_id.split("-")[0])
        score = rank_individual_layout_score(
            subject_id, individual_id, log_dir, layout_ctx=layout_ctx
        )
        by_generation[generation].append((individual_id, score))
    return {
        generation: min(individuals, key=lambda item: item[1])
        for generation, individuals in by_generation.items()
    }


def parse_individual_id(individual_id):
    """Parse GA id '{generation}-{index}' (rsplit so multi-digit generations work)."""
    generation_str, index_str = str(individual_id).rsplit("-", 1)
    return int(generation_str), int(index_str)


def individual_id_sort_key(individual_id):
    generation, index = parse_individual_id(individual_id)
    return generation, index


def list_individual_ids_in_log_dir(log_dir, subject_id):
    """Individual ids that have a saved mod_connection_paths JSON in log_dir."""
    prefix = f"GA_{subject_id}_"
    suffix = "_mod_connection_paths.json"
    found = []
    try:
        names = os.listdir(log_dir)
    except OSError:
        return found
    for name in names:
        if not name.startswith(prefix) or not name.endswith(suffix):
            continue
        individual_id = name[len(prefix) : -len(suffix)]
        try:
            parse_individual_id(individual_id)
        except ValueError:
            continue
        found.append(individual_id)
    return found


def layout_score_from_saved_individual(log_dir, subject_id, individual_id):
    """Read layout_score from a saved individual JSON (no full collision recompute)."""
    log_path = os.path.join(
        log_dir, f"GA_{subject_id}_{individual_id}_mod_connection_paths.json"
    )
    if not os.path.isfile(log_path):
        return None
    with open(log_path, "r") as f:
        data = json.load(f)
    metrics = data.get("collision_metrics") or {}
    if "layout_score" in metrics:
        return float(metrics["layout_score"])
    if "collision_score" in metrics:
        return float(metrics["collision_score"])
    return None


def best_individual_from_fitness_tracker(fitness_tracker, prefer_latest_generation=True):
    """
    Fast best-individual lookup from GA fitness_tracker (fitness = -layout_score).

    When several individuals share the same fitness (common after phase-2 solutions),
    prefer_latest_generation picks the highest generation (e.g. 99-3 over 1-1).
    """
    if not fitness_tracker:
        raise ValueError("Fitness tracker is empty")
    best_fitness = max(fitness_tracker.values())
    candidates = [k for k, v in fitness_tracker.items() if v == best_fitness]
    if prefer_latest_generation:
        best_id = max(candidates, key=individual_id_sort_key)
    else:
        best_id = max(candidates, key=fitness_tracker.get)
    return best_id, -float(fitness_tracker[best_id])


def best_per_generation_from_fitness_tracker(fitness_tracker):
    """Fast {generation: (individual_id, layout_score)} from stored fitness values."""
    if not fitness_tracker:
        raise ValueError("Fitness tracker is empty")
    by_generation = defaultdict(list)
    for individual_id, fitness in fitness_tracker.items():
        generation, _ = parse_individual_id(individual_id)
        by_generation[generation].append((individual_id, -float(fitness)))
    return {
        generation: min(
            individuals,
            key=lambda item: (item[1], individual_id_sort_key(item[0])),
        )
        for generation, individuals in by_generation.items()
    }


def best_per_generation_from_log_dir(log_dir, subject_id, fitness_tracker=None):
    """
    Best individual per generation using JSON files present in log_dir.

    Uses fitness_tracker scores when available; otherwise reads collision_metrics
    from each individual file. This matches archived runs under records/{RUN_ID}/.
    """
    by_generation = defaultdict(list)
    for individual_id in list_individual_ids_in_log_dir(log_dir, subject_id):
        generation, _ = parse_individual_id(individual_id)
        if fitness_tracker is not None and individual_id in fitness_tracker:
            layout_score = -float(fitness_tracker[individual_id])
        else:
            layout_score = layout_score_from_saved_individual(
                log_dir, subject_id, individual_id
            )
            if layout_score is None:
                continue
        by_generation[generation].append((individual_id, layout_score))

    if not by_generation:
        if fitness_tracker:
            return best_per_generation_from_fitness_tracker(fitness_tracker)
        raise ValueError(f"No individuals found in log_dir: {log_dir}")

    return {
        generation: min(
            individuals,
            key=lambda item: (item[1], individual_id_sort_key(item[0])),
        )
        for generation, individuals in by_generation.items()
    }


def _path_needs_electrode_greedy(path, path_electrode_name, electrode_zones, start, end):
    from PYTHON.GA import greed

    if greed._path_has_foreign_electrode_violation(path, path_electrode_name, electrode_zones):
        return True
    return greed._chord_blocked_by_electrodes(start, end, path_electrode_name, electrode_zones)


def _center_outward_slot_order(n_slots, center_idx):
    """Visit slot indices center-first, then alternating outward neighbors."""
    order = [center_idx]
    for delta in range(1, max(n_slots - center_idx, center_idx + 1)):
        if center_idx + delta < n_slots:
            order.append(center_idx + delta)
        if center_idx - delta >= 0:
            order.append(center_idx - delta)
    return order


def _geometry_indicates_crossing(inter, crossing_points):
    """True when intersection geometry includes a point crossing (not overlap-only)."""
    if crossing_points:
        return True
    if inter is None or inter.is_empty:
        return False
    if inter.geom_type in ('Point', 'MultiPoint'):
        return True
    if inter.geom_type == 'LineString':
        return float(inter.length) > COLLISION_SCORE_EPSILON
    if inter.geom_type in ('MultiLineString', 'GeometryCollection'):
        for sub in inter.geoms:
            if sub.geom_type == 'LineString' and float(sub.length) > COLLISION_SCORE_EPSILON:
                return True
            if sub.geom_type == 'Point':
                return True
    return False


def _crossing_overlap_penalty_from_geometry(inter, crossing_points):
    """Crossing/overlap score for one pair from precomputed intersection geometry."""
    stats = {'crossing_count': 0, 'overlap_length': 0.0}
    _accumulate_geometry_metrics(inter, [], stats)
    geom_points = _collect_points_from_geometry(inter)
    for pt in crossing_points:
        if _is_duplicate_xy_point(pt, geom_points):
            continue
        stats['crossing_count'] += 1
        geom_points.append(pt)
    return (
        stats['crossing_count'] * CROSSING_SCORE_WEIGHT
        + stats['overlap_length'] * OVERLAP_SCORE_WEIGHT
    )


def _pair_crosses_outside_terminal_merge(
    path_a,
    path_b,
    terminal_a,
    terminal_b,
    electrode_zones,
    terminal_zones=None,
    dense_path_cache=None,
    path_idx_a=None,
    path_idx_b=None,
    inter=None,
    crossing_points=None,
):
    """True when path bodies intersect outside the co-terminal merge tail."""
    if terminal_a != terminal_b:
        return False
    if inter is None and crossing_points is None:
        inter, _pair_merge, crossing_points = _compute_path_pair_crossing_geometry(
            path_a,
            path_b,
            terminal_a,
            terminal_b,
            terminal_zones or {},
            electrode_zones,
            dense_path_cache=dense_path_cache,
            path_idx_a=path_idx_a,
            path_idx_b=path_idx_b,
        )
    return _geometry_indicates_crossing(inter, crossing_points)


def _pair_has_coterminal_ordered_crossing(
    path_a,
    path_b,
    terminal_a,
    terminal_b,
    slot_a,
    slot_b,
    electrode_zones,
    terminal_zones=None,
    dense_path_cache=None,
    path_idx_a=None,
    path_idx_b=None,
    inter=None,
    crossing_points=None,
):
    """Same terminal, different slots, crossing outside merge tail."""
    if terminal_a != terminal_b:
        return False
    if slot_a is None or slot_b is None or slot_a == slot_b:
        return False
    return _pair_crosses_outside_terminal_merge(
        path_a,
        path_b,
        terminal_a,
        terminal_b,
        electrode_zones,
        terminal_zones=terminal_zones,
        dense_path_cache=dense_path_cache,
        path_idx_a=path_idx_a,
        path_idx_b=path_idx_b,
        inter=inter,
        crossing_points=crossing_points,
    )


def _collect_coterminal_ordered_crossing_pairs(
    paths,
    path_electrodes,
    path_terminals,
    slot_index_by_electrode,
    electrode_zones,
    terminal_zones=None,
    dense_path_cache=None,
):
    """Pairs (terminal, lower-slot electrode, higher-slot electrode) that cross."""
    if not slot_index_by_electrode or path_electrodes is None or path_terminals is None:
        return frozenset()

    if dense_path_cache is None:
        dense_path_cache = _build_crossing_detection_path_cache(
            paths, path_terminals, electrode_zones
        )

    violating = set()
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            terminal_i = path_terminals[i]
            terminal_j = path_terminals[j]
            slot_i = slot_index_by_electrode.get(path_electrodes[i])
            slot_j = slot_index_by_electrode.get(path_electrodes[j])
            inter, _pair_merge, crossing_points = _compute_path_pair_crossing_geometry(
                paths[i],
                paths[j],
                terminal_i,
                terminal_j,
                terminal_zones or {},
                electrode_zones,
                dense_path_cache=dense_path_cache,
                path_idx_a=i,
                path_idx_b=j,
            )
            if not _pair_has_coterminal_ordered_crossing(
                paths[i],
                paths[j],
                terminal_i,
                terminal_j,
                slot_i,
                slot_j,
                electrode_zones,
                terminal_zones=terminal_zones,
                inter=inter,
                crossing_points=crossing_points,
            ):
                continue
            if slot_i < slot_j:
                low_name, high_name = path_electrodes[i], path_electrodes[j]
            else:
                low_name, high_name = path_electrodes[j], path_electrodes[i]
            violating.add((terminal_i, low_name, high_name))
    return frozenset(violating)


def count_coterminal_ordered_crossings(
    paths,
    path_electrodes,
    path_terminals,
    slot_index_by_electrode,
    electrode_zones,
    terminal_zones=None,
    dense_path_cache=None,
):
    return len(
        _collect_coterminal_ordered_crossing_pairs(
            paths,
            path_electrodes,
            path_terminals,
            slot_index_by_electrode,
            electrode_zones,
            terminal_zones=terminal_zones,
            dense_path_cache=dense_path_cache,
        )
    )


def _pair_is_coterminal_ordered_violation(
    i,
    j,
    paths,
    path_electrodes,
    path_terminals,
    slot_index_by_electrode,
    electrode_zones,
    terminal_zones=None,
    dense_path_cache=None,
):
    terminal_i = path_terminals[i]
    terminal_j = path_terminals[j]
    slot_i = slot_index_by_electrode.get(path_electrodes[i])
    slot_j = slot_index_by_electrode.get(path_electrodes[j])
    inter, _pair_merge, crossing_points = _compute_path_pair_crossing_geometry(
        paths[i],
        paths[j],
        terminal_i,
        terminal_j,
        terminal_zones or {},
        electrode_zones,
        dense_path_cache=dense_path_cache,
        path_idx_a=i,
        path_idx_b=j,
    )
    return _pair_has_coterminal_ordered_crossing(
        paths[i],
        paths[j],
        terminal_i,
        terminal_j,
        slot_i,
        slot_j,
        electrode_zones,
        terminal_zones=terminal_zones,
        inter=inter,
        crossing_points=crossing_points,
    )


def introduces_coterminal_ordered_crossings(
    before_paths,
    after_paths,
    path_electrodes,
    path_terminals,
    slot_index_by_electrode,
    electrode_zones,
    terminal_zones=None,
    changed_path_indices=None,
):
    """True when after_paths adds co-terminal ordered crossings vs before_paths."""
    if not slot_index_by_electrode:
        return False
    if changed_path_indices is not None:
        dense_cache = _build_crossing_detection_path_cache(
            after_paths, path_terminals, electrode_zones
        )
        for path_idx in changed_path_indices:
            for j in range(len(before_paths)):
                if j == path_idx:
                    continue
                i, jj = (path_idx, j) if path_idx < j else (j, path_idx)
                was_violating = _pair_is_coterminal_ordered_violation(
                    i,
                    jj,
                    before_paths,
                    path_electrodes,
                    path_terminals,
                    slot_index_by_electrode,
                    electrode_zones,
                    terminal_zones=terminal_zones,
                    dense_path_cache=dense_cache,
                )
                now_violating = _pair_is_coterminal_ordered_violation(
                    i,
                    jj,
                    after_paths,
                    path_electrodes,
                    path_terminals,
                    slot_index_by_electrode,
                    electrode_zones,
                    terminal_zones=terminal_zones,
                    dense_path_cache=dense_cache,
                )
                if now_violating and not was_violating:
                    return True
        return False

    before = _collect_coterminal_ordered_crossing_pairs(
        before_paths,
        path_electrodes,
        path_terminals,
        slot_index_by_electrode,
        electrode_zones,
        terminal_zones=terminal_zones,
    )
    after = _collect_coterminal_ordered_crossing_pairs(
        after_paths,
        path_electrodes,
        path_terminals,
        slot_index_by_electrode,
        electrode_zones,
        terminal_zones=terminal_zones,
    )
    return bool(after - before)


def _pair_centerline_min_distance(
    path_a,
    path_b,
    terminal_a,
    terminal_b,
    electrode_zones,
):
    """Minimum centerline distance between two paths outside terminal merge tails."""
    line_a = _path_to_linestring(path_a)
    line_b = _path_to_linestring(path_b)
    if line_a is None or line_b is None:
        return float("inf")

    merge_tail_length = _terminal_merge_tail_length(electrode_zones)
    pair_merge_union = _build_pair_merge_union(
        path_a, path_b, terminal_a, terminal_b, merge_tail_length
    )
    min_dist = float("inf")
    for pt in _linestring_sample_points(line_a, pair_merge_union):
        min_dist = min(min_dist, float(line_b.distance(pt)))
    return min_dist


def _pair_layout_penalty(
    path_a,
    path_b,
    terminal_a,
    terminal_b,
    terminal_zones,
    electrode_zones,
    min_separation=PHASE2_INNER_TRACE_SEPARATION,
    dense_path_cache=None,
    path_idx_a=None,
    path_idx_b=None,
    pair_geometry=None,
):
    """Crossing, overlap, and trace-separation penalty for one path pair."""
    penalty = analyze_pair_path_penalty(
        path_a,
        path_b,
        terminal_zones,
        terminal_a=terminal_a,
        terminal_b=terminal_b,
        electrode_zones=electrode_zones,
        dense_path_cache=dense_path_cache,
        path_idx_a=path_idx_a,
        path_idx_b=path_idx_b,
        pair_geometry=pair_geometry,
    )
    min_dist = _pair_centerline_min_distance(
        path_a, path_b, terminal_a, terminal_b, electrode_zones
    )
    if min_dist < min_separation:
        shortfall = min_separation - min_dist
        penalty += (shortfall / min_separation) ** 2
    return penalty


def _find_conflict_path_pairs(
    paths,
    path_terminals,
    terminal_zones,
    electrode_zones,
    min_separation=PHASE2_INNER_TRACE_SEPARATION,
    focus_separation: bool = False,
):
    """Return [(i, j, penalty)] for pairs with crossings, overlap, or tight separation."""
    pairs = {}
    pair_min_dist = {}
    dense_path_cache = _build_crossing_detection_path_cache(
        paths, path_terminals, electrode_zones
    )
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            inter, _pair_merge, crossing_points = _compute_path_pair_crossing_geometry(
                paths[i],
                paths[j],
                path_terminals[i],
                path_terminals[j],
                terminal_zones,
                electrode_zones,
                dense_path_cache=dense_path_cache,
                path_idx_a=i,
                path_idx_b=j,
            )
            penalty = _pair_layout_penalty(
                paths[i],
                paths[j],
                path_terminals[i],
                path_terminals[j],
                terminal_zones,
                electrode_zones,
                min_separation=min_separation,
                dense_path_cache=dense_path_cache,
                path_idx_a=i,
                path_idx_b=j,
                pair_geometry=(inter, crossing_points),
            )
            if penalty > COLLISION_SCORE_EPSILON:
                pairs[(i, j)] = penalty
                pair_min_dist[(i, j)] = _pair_centerline_min_distance(
                    paths[i],
                    paths[j],
                    path_terminals[i],
                    path_terminals[j],
                    electrode_zones,
                )

    if focus_separation:
        return sorted(
            ((i, j, penalty) for (i, j), penalty in pairs.items()),
            key=lambda item: (pair_min_dist[(item[0], item[1])], -item[2]),
        )
    return sorted(
        ((i, j, penalty) for (i, j), penalty in pairs.items()),
        key=lambda item: -item[2],
    )


def _path_chord_detour_ratio(path, start, end):
    path = np.asarray(path, dtype=float)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    chord = float(np.linalg.norm(end - start))
    if chord < 1e-6:
        return 1.0
    path_length = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    return path_length / chord


def _try_gentle_pair_trace_adjustment(
    paths,
    path_idx,
    partner_idx,
    path_electrodes,
    path_terminals,
    electrode_zones,
    terminal_zones,
    x_bounds,
    electrodes_2d,
    terminals_2d,
    entry_points_2d,
    slot_index_by_electrode=None,
    min_separation=PHASE2_INNER_TRACE_SEPARATION,
    max_random_attempts=PHASE2_RANDOM_ATTEMPTS_PER_TRACE,
    focus_separation: bool = False,
    fixed_endpoints: bool = False,
    max_crossing_count: int | None = None,
):
    """
    Gently adjust one trace in a conflicting pair (spacing greedy, then small random nudge).
    Terminal-zone tails are locked; only the outside segment may change.
    Rejects trials that cross foreign electrode discs, exceed detour limit, add
    co-terminal ordered crossings (same hub, different slot, outside merge tail),
    or re-enter the trace (self-intersection / electrode or terminal zone re-entry).
    """
    from PYTHON.GA import greed

    electrode_name = path_electrodes[path_idx]
    terminal_name = path_terminals[path_idx]
    start = np.asarray(electrodes_2d[electrode_name], dtype=float)
    end = np.asarray(
        entry_points_2d.get(electrode_name, terminals_2d[terminal_name]), dtype=float
    )
    path = paths[path_idx]
    partner_path = paths[partner_idx]
    baseline_penalty = _pair_layout_penalty(
        path,
        partner_path,
        terminal_name,
        path_terminals[partner_idx],
        terminal_zones,
        electrode_zones,
        min_separation=min_separation,
    )
    baseline_min_dist = _pair_centerline_min_distance(
        path,
        partner_path,
        terminal_name,
        path_terminals[partner_idx],
        electrode_zones,
    )
    if focus_separation:
        max_random_attempts = max(
            max_random_attempts, PHASE2_SEPARATION_FOCUS_RANDOM_ATTEMPTS
        )

    crossing_cap_context = None
    if max_crossing_count is not None:
        crossing_dense_cache = _build_crossing_detection_path_cache(
            paths, path_terminals, electrode_zones
        )
        crossings_among_others = _count_crossings_among_other_paths(
            path_idx,
            paths,
            path_terminals,
            terminal_zones,
            electrode_zones,
            dense_path_cache=crossing_dense_cache,
        )
        crossing_cap_context = (crossings_among_others, crossing_dense_cache)

    def _accept(trial_path):
        with profile_step("accept_total"):
            with profile_step("accept_pin"):
                if fixed_endpoints:
                    trial_path = pin_path_endpoints_2d(
                        np.asarray(trial_path, dtype=float), start, end
                    )
                else:
                    trial_path = _lock_terminal_zone_tail(
                        path, trial_path, terminal_name, terminal_zones, start, end
                    )
                    trial_path = pin_path_endpoints_2d(trial_path, start, end)
            if greed._paths_same_polyline(trial_path, path):
                return None
            with profile_step("accept_electrode_check"):
                if count_single_trace_electrode_violations(
                    trial_path, electrode_name, electrode_zones
                ) > 0:
                    return None
                if _path_chord_detour_ratio(trial_path, start, end) > PHASE2_SPACING_MAX_DETOUR_RATIO:
                    return None
            if max_crossing_count is not None:
                crossings_among_others, crossing_dense_cache = crossing_cap_context
                with profile_step("accept_global_crossing"):
                    trial_cross = _layout_crossing_count_if_replaced(
                        path_idx,
                        trial_path,
                        paths,
                        path_terminals,
                        terminal_zones,
                        electrode_zones,
                        crossings_among_others,
                        dense_path_cache=crossing_dense_cache,
                    )
                if int(trial_cross) > int(max_crossing_count):
                    return None
            trial_paths = [p.copy() for p in paths]
            trial_paths[path_idx] = trial_path
            with profile_step("accept_pair_metrics"):
                trial_penalty = _pair_layout_penalty(
                    trial_paths[path_idx],
                    trial_paths[partner_idx],
                    terminal_name,
                    path_terminals[partner_idx],
                    terminal_zones,
                    electrode_zones,
                    min_separation=min_separation,
                )
                trial_min_dist = _pair_centerline_min_distance(
                    trial_paths[path_idx],
                    trial_paths[partner_idx],
                    terminal_name,
                    path_terminals[partner_idx],
                    electrode_zones,
                )
            separation_improved = trial_min_dist > baseline_min_dist + 0.05
            penalty_improved = trial_penalty + 1e-9 < baseline_penalty
            if focus_separation:
                if not separation_improved and not penalty_improved:
                    return None
                if separation_improved and trial_penalty > baseline_penalty + 1e-6:
                    return None
            elif not penalty_improved:
                return None
            with profile_step("accept_coterminal_check"):
                if slot_index_by_electrode and introduces_coterminal_ordered_crossings(
                    paths,
                    trial_paths,
                    path_electrodes,
                    path_terminals,
                    slot_index_by_electrode,
                    electrode_zones,
                    terminal_zones=terminal_zones,
                    changed_path_indices=(path_idx,),
                ):
                    return None
            with profile_step("accept_reentry_check"):
                if path_has_trace_reentry(
                    trial_path,
                    electrode_name,
                    terminal_name,
                    electrode_zones,
                    terminal_zones,
                ):
                    return None
                zone = terminal_zones.get(terminal_name)
                if zone is not None and path_has_terminal_zone_reentry(trial_path, zone):
                    return None
            return trial_paths, trial_penalty

    with profile_step("pair_adjust"):
        with profile_step("pair_greedy"):
            spacing_trial = _greedy_spacing_outside_terminal_zone(
                np.asarray(path, dtype=float),
                terminal_name,
                terminal_zones,
                electrode_name,
                electrode_zones,
                start,
                end,
                partner_path,
            )
        if spacing_trial is not None:
            accepted = _accept(spacing_trial)
            if accepted is not None:
                return accepted

        with profile_step("pair_random_loop"):
            for _ in range(max_random_attempts):
                with profile_step("pair_random_modify"):
                    random_trial = randomly_modify_path(
                        np.asarray(path, dtype=float).copy(),
                        electrode_name,
                        electrode_zones,
                        terminal_zones,
                        x_bounds=x_bounds,
                        target_electrode_pos=start,
                        target_terminal_pos=end,
                        target_terminal_name=terminal_name,
                    )
                accepted = _accept(random_trial)
                if accepted is not None:
                    return accepted

    return None


def _pair_trace_penalty(
    path_a,
    path_b,
    electrode_a,
    electrode_b,
    terminal_a,
    terminal_b,
    terminal_zones,
    electrode_zones,
    min_separation=PHASE2_INNER_TRACE_SEPARATION,
):
    """Crossing + overlap penalty for one path pair."""
    del min_separation  # kept for API compatibility
    return analyze_pair_path_penalty(
        path_a,
        path_b,
        terminal_zones,
        terminal_a=terminal_a,
        terminal_b=terminal_b,
        electrode_zones=electrode_zones,
    )


def _penalty_against_inner_traces(
    path_idx,
    paths,
    inner_indices,
    path_electrodes,
    path_terminals,
    terminal_zones,
    electrode_zones,
    min_separation=PHASE2_INNER_TRACE_SEPARATION,
):
    total = 0.0
    for inner_idx in inner_indices:
        if inner_idx == path_idx:
            continue
        total += _pair_trace_penalty(
            paths[path_idx],
            paths[inner_idx],
            path_electrodes[path_idx],
            path_electrodes[inner_idx],
            path_terminals[path_idx],
            path_terminals[inner_idx],
            terminal_zones,
            electrode_zones,
            min_separation=min_separation,
        )
    return total


def _resolve_phase2_ordered_trace_resolution(
    paths,
    path_electrodes,
    path_terminals,
    electrode_zones,
    terminal_zones,
    x_bounds,
    electrodes_2d,
    terminals_2d,
    entry_points_2d,
    slot_order_by_terminal,
    slot_index_by_electrode=None,
    max_attempts_per_trace=8,
    min_separation=PHASE2_INNER_TRACE_SEPARATION,
    max_pair_rounds: int | None = None,
    focus_separation: bool = False,
    fixed_endpoints: bool = False,
    max_crossing_count: int | None = None,
):
    """
    Phase 2 gentle resolution: for each conflicting pair, try spacing-greedy and small
    random moves on BOTH traces (electrode-safe, detour-capped).
    """
    del max_attempts_per_trace  # pair-based pass uses fixed caps

    if slot_index_by_electrode is None:
        slot_index_by_electrode = {}
        for _terminal, ordered_names in (slot_order_by_terminal or {}).items():
            for slot_idx, name in enumerate(ordered_names):
                slot_index_by_electrode.setdefault(name, slot_idx)

    def _pin_all(paths_to_pin):
        pinned = []
        for idx, path in enumerate(paths_to_pin):
            electrode = path_electrodes[idx]
            end = entry_points_2d.get(electrode, terminals_2d[path_terminals[idx]])
            pinned.append(
                pin_path_endpoints_2d(
                    path,
                    electrodes_2d[electrode],
                    end,
                )
            )
        return pinned

    best_paths = _pin_all([path.copy() for path in paths])
    if max_crossing_count is None:
        with profile_step("phase2_baseline_crossings"):
            max_crossing_count = int(
                analyze_path_collisions(
                    best_paths,
                    terminal_zones,
                    electrode_zones=electrode_zones,
                    path_electrodes=path_electrodes,
                    path_terminals=path_terminals,
                )["crossing_count"]
            )
    pair_rounds = (
        int(max_pair_rounds)
        if max_pair_rounds is not None
        else PHASE2_PAIR_RESOLUTION_MAX_ROUNDS
    )

    for round_idx in range(pair_rounds):
        prof = get_phase2_profile()
        if prof is not None:
            prof.set_round(round_idx)
        with profile_step("find_conflict_pairs"):
            conflict_pairs = _find_conflict_path_pairs(
                best_paths,
                path_terminals,
                terminal_zones,
                electrode_zones,
                min_separation=min_separation,
                focus_separation=focus_separation,
            )
        if not conflict_pairs:
            label = "separation" if focus_separation else "conflicts"
            print(f"Phase 2 pair resolution: no {label} (round {round_idx + 1})")
            break

        print(
            f"Phase 2 pair resolution round {round_idx + 1}: "
            f"{len(conflict_pairs)} {'tight' if focus_separation else 'conflicting'} pair(s)"
        )
        round_improved = False
        for i, j, pair_penalty in conflict_pairs:
            for path_idx, partner_idx in ((i, j), (j, i)):
                result = _try_gentle_pair_trace_adjustment(
                    best_paths,
                    path_idx,
                    partner_idx,
                    path_electrodes,
                    path_terminals,
                    electrode_zones,
                    terminal_zones,
                    x_bounds,
                    electrodes_2d,
                    terminals_2d,
                    entry_points_2d,
                    slot_index_by_electrode=slot_index_by_electrode,
                    min_separation=min_separation,
                    focus_separation=focus_separation,
                    fixed_endpoints=fixed_endpoints,
                    max_crossing_count=max_crossing_count,
                )
                if result is None:
                    continue
                best_paths, new_penalty = result
                round_improved = True
                print(
                    f"  {path_electrodes[path_idx]} vs {path_electrodes[partner_idx]}: "
                    f"pair penalty {pair_penalty:.2f} -> {new_penalty:.2f}"
                )

        if not round_improved:
            print(f"Phase 2 pair resolution stopped (round {round_idx + 1}, no gain)")
            break
        if prof is not None:
            prof.print_round_summary(round_idx)

    with profile_step("phase2_final_analysis"):
        final = analyze_path_collisions(
            best_paths,
            terminal_zones,
            electrode_zones=electrode_zones,
            path_electrodes=path_electrodes,
            path_terminals=path_terminals,
        )
    print(
        f"Phase 2 pair final: crossings={final['crossing_count']}, "
        f"overlap={final['overlap_length']:.2f}, "
        f"trace_sep_norm={final['trace_separation_deficit_normalized']:.2f}, "
        f"layout_score={final.get('layout_score', final['collision_score']):.2f}"
    )
    return best_paths


def _resolve_phase1_straight_electrode_greedy(
    paths,
    path_electrodes,
    path_terminals,
    electrode_zones,
    terminal_zones,
    electrodes_2d,
    terminals_2d,
    entry_points_2d=None,
    slot_index_by_electrode=None,
    max_greedy_rounds=10,
):
    """Phase 1: greedy hull on every trace that crosses a foreign electrode zone."""
    from PYTHON.GA import greed

    if entry_points_2d is None:
        entry_points_2d = {}
    if slot_index_by_electrode is None:
        slot_index_by_electrode = {}

    def _path_end(electrode_name, terminal_name):
        return entry_points_2d.get(electrode_name, terminals_2d[terminal_name])

    def _pin_all(paths_to_pin):
        return [
            pin_path_endpoints_2d(
                path,
                electrodes_2d[path_electrodes[idx]],
                _path_end(path_electrodes[idx], path_terminals[idx]),
            )
            for idx, path in enumerate(paths_to_pin)
        ]

    best_paths = _pin_all([path.copy() for path in paths])

    round_violations = count_electrode_violations(
        best_paths, electrode_zones, path_electrodes
    )
    print(f"Phase 1 input: electrode violations={round_violations}")

    if round_violations > 0:
        print("Phase 1 electrode greedy (foreign zones only)...")
        for round_idx in range(max_greedy_rounds):
            if round_violations == 0:
                print(f"  Electrode clearance after {round_idx} round(s)")
                break

            round_improved = False
            local_only_accepts = 0
            local_accept_count_by_electrode = {}

            violating_indices = trace_indices_with_electrode_violations(
                best_paths, electrode_zones, path_electrodes
            )
            if not violating_indices:
                break

            violator_names = [path_electrodes[i] for i in violating_indices]
            print(
                f"  Round {round_idx + 1}: {len(violating_indices)} violating trace(s): "
                f"{', '.join(violator_names)}"
            )

            for path_idx in violating_indices:
                path = best_paths[path_idx]
                electrode_name = path_electrodes[path_idx]
                start = electrodes_2d[electrode_name]
                end = _path_end(electrode_name, path_terminals[path_idx])
                trace_violations_before = count_single_trace_electrode_violations(
                    path, electrode_name, electrode_zones
                )
                if trace_violations_before == 0:
                    continue

                n_points = max(len(path), 50)
                terminal_name = path_terminals[path_idx]
                terminal_pos = terminals_2d[terminal_name]
                terminal_zone = terminal_zones[terminal_name]
                terminal_entry_hits = [
                    entry_points_2d[e]
                    for e, t in zip(path_electrodes, path_terminals)
                    if t == terminal_name and e in entry_points_2d
                ]
                strip_tangent = (
                    _oriented_strip_tangent_for_terminal(
                        terminal_pos, terminal_entry_hits, terminal_zone
                    )
                    if terminal_entry_hits
                    else None
                )
                blocking_electrode = _primary_blocking_electrode_for_trace(
                    path, electrode_name, electrode_zones, start, end
                )
                bypass_side = None
                blocker_center = None
                if blocking_electrode is not None:
                    blocker_center = np.asarray(
                        electrode_zones['metadata'][blocking_electrode]['center'], dtype=float
                    )
                    bypass_side = _electrode_bypass_side_preference(
                        electrode_name,
                        blocking_electrode,
                        entry_points_2d,
                        slot_index_by_electrode,
                        terminal_pos,
                        terminal_zone,
                    )

                trial_path = greed.greedy_electrode_avoidance(
                    path,
                    electrode_name,
                    electrode_zones,
                    terminal_zones,
                    target_terminal_pos=end,
                    n_points=n_points,
                    quiet=True,
                    bypass_side=bypass_side,
                    blocker_center=blocker_center,
                    strip_tangent=strip_tangent,
                )
                trial_path = pin_path_endpoints_2d(trial_path, start, end)
                if greed._paths_same_polyline(trial_path, path):
                    continue

                trial_paths = _pin_all([p.copy() for p in best_paths])
                trial_paths[path_idx] = trial_path
                trial_violations = count_electrode_violations(
                    trial_paths, electrode_zones, path_electrodes
                )
                trial_trace_violations = count_single_trace_electrode_violations(
                    trial_path, electrode_name, electrode_zones
                )

                global_better = trial_violations < round_violations
                global_flat = trial_violations == round_violations
                trace_better = trial_trace_violations < trace_violations_before
                accept = False
                accept_kind = None

                if global_better:
                    accept = True
                    accept_kind = 'global'
                elif global_flat and trace_better:
                    if local_only_accepts < PHASE1_LOCAL_ACCEPT_MAX_PER_ROUND:
                        n_local = local_accept_count_by_electrode.get(electrode_name, 0)
                        if n_local < PHASE1_LOCAL_ACCEPT_MAX_PER_ELECTRODE:
                            accept = True
                            accept_kind = 'local'

                if accept and path_has_trace_reentry(
                    trial_path,
                    electrode_name,
                    terminal_name,
                    electrode_zones,
                    terminal_zones,
                ):
                    accept = False
                    accept_kind = None

                if accept:
                    best_paths = trial_paths
                    round_violations = trial_violations
                    round_improved = True
                    if accept_kind == 'local':
                        local_only_accepts += 1
                        local_accept_count_by_electrode[electrode_name] = (
                            local_accept_count_by_electrode.get(electrode_name, 0) + 1
                        )
                    print(
                        f"  Round {round_idx + 1}: electrode reroute {electrode_name} "
                        f"({accept_kind}: violations={trial_violations}, "
                        f"trace {trace_violations_before}->{trial_trace_violations})"
                    )

            if round_violations == 0:
                break
            if not round_improved:
                print(f"  Electrode greedy stopped after round {round_idx + 1} (no improvement)")
                break

    print(f"Phase 1 final: electrode violations={round_violations}")
    return best_paths


def _count_path_collision_points(path, collision_points, tolerance=0.1):
    if collision_points is None or len(collision_points) == 0:
        return 0
    path_line = LineString(path)
    count = 0
    for collision in collision_points:
        if path_line.distance(Point(collision)) < tolerance:
            count += 1
    return count


def smart_collision_resolution(
    paths,
    path_electrodes,
    electrode_zones,
    terminal_zones,
    x_bounds,
    path_terminals=None,
    electrodes_2d=None,
    terminals_2d=None,
    entry_points_2d=None,
    slot_order_by_terminal=None,
    slot_index_by_electrode=None,
    max_attempts=5,
    use_greedy_aggressive=False,
    use_gentle_resolution=True,
    greedy_electrodes_only=False,
    phase2_max_pair_rounds: int | None = None,
    focus_separation: bool = False,
    fixed_endpoints: bool = False,
    max_crossing_count: int | None = None,
):
    """Collision resolution; mode depends on GA phase (see greedy_electrodes_only / use_gentle_resolution)."""
    if entry_points_2d is None:
        entry_points_2d = {}

    clean_paths = []
    for path in paths:
        path_array = np.array(path, dtype=np.float64)
        mask = ~(np.isnan(path_array).any(axis=1) | np.isinf(path_array).any(axis=1))
        clean_path = path_array[mask]
        clean_paths.append(clean_path if len(clean_path) >= 2 else np.array([path[0], path[-1]]))
    
    paths = clean_paths
    if path_terminals is None:
        path_terminals = _infer_path_terminals(paths, terminal_zones)

    def _path_end(idx):
        electrode = path_electrodes[idx]
        return entry_points_2d.get(electrode, terminals_2d[path_terminals[idx]])

    def _pin_all(paths_to_pin):
        if electrodes_2d is None or terminals_2d is None:
            return paths_to_pin
        return [
            pin_path_endpoints_2d(
                path,
                electrodes_2d[path_electrodes[idx]],
                _path_end(idx),
            )
            for idx, path in enumerate(paths_to_pin)
        ]

    paths = _pin_all(paths)

    if greedy_electrodes_only:
        if electrodes_2d is None or terminals_2d is None:
            raise ValueError("greedy_electrodes_only requires electrodes_2d and terminals_2d")
        return _resolve_phase1_straight_electrode_greedy(
            paths,
            path_electrodes,
            path_terminals,
            electrode_zones,
            terminal_zones,
            electrodes_2d,
            terminals_2d,
            entry_points_2d=entry_points_2d,
            slot_index_by_electrode=slot_index_by_electrode,
        )

    initial = analyze_path_collisions(
        paths,
        terminal_zones,
        electrode_zones=electrode_zones,
        path_electrodes=path_electrodes,
        path_terminals=path_terminals,
    )
    if initial['collision_score'] < COLLISION_SCORE_EPSILON:
        return paths

    best_paths = [path.copy() for path in paths]

    if use_gentle_resolution:
        if slot_order_by_terminal is None:
            slot_index = {
                path_electrodes[idx]: idx for idx in range(len(path_electrodes))
            }
            slot_order_by_terminal = build_slot_order_maps(
                path_electrodes, path_terminals, slot_index
            )
        return _resolve_phase2_ordered_trace_resolution(
            best_paths,
            path_electrodes,
            path_terminals,
            electrode_zones,
            terminal_zones,
            x_bounds,
            electrodes_2d,
            terminals_2d,
            entry_points_2d,
            slot_order_by_terminal,
            slot_index_by_electrode=slot_index_by_electrode,
            max_attempts_per_trace=max(max_attempts, 8),
            max_pair_rounds=phase2_max_pair_rounds,
            min_separation=PHASE2_INNER_TRACE_SEPARATION,
            focus_separation=focus_separation,
            fixed_endpoints=fixed_endpoints,
            max_crossing_count=max_crossing_count,
        )

    best_collision_points = initial['points']
    best_score = initial['collision_score']

    # Legacy trace greedy (disabled in current two-phase GA; phase 1 uses electrode-only greedy)
    clearance_snapshot = analyze_path_collisions(
        best_paths,
        terminal_zones,
        electrode_zones=electrode_zones,
        path_electrodes=path_electrodes,
        path_terminals=path_terminals,
    )

    if use_greedy_aggressive and not greedy_electrodes_only and not is_layout_clearance_free(clearance_snapshot):
        from PYTHON.GA import greed

        print(
            f"Phase 2: Greedy aggressive resolution for score {best_score:.2f} "
            f"(target: 0 crossings)..."
        )
        max_greedy_rounds = 20
        for round_idx in range(max_greedy_rounds):
            round_analysis = analyze_path_collisions(
                best_paths,
                terminal_zones,
                electrode_zones=electrode_zones,
                path_electrodes=path_electrodes,
                path_terminals=path_terminals,
            )
            if is_layout_clearance_free(round_analysis):
                best_collision_points = round_analysis['points']
                best_score = round_analysis['collision_score']
                print(f"  Greedy clearance achieved after {round_idx + 1} round(s)")
                break

            collision_scores = {i: 0 for i in range(len(best_paths))}
            if best_collision_points is not None:
                for i, path in enumerate(best_paths):
                    collision_scores[i] = _count_path_collision_points(path, best_collision_points)

            candidate_indices, locked_indices, corridor_obstacles_by_idx = (
                _greedy_aggressive_candidate_indices(
                    best_paths,
                    path_electrodes,
                    path_terminals,
                    terminal_zones,
                    electrode_zones,
                    collision_scores,
                    electrodes_2d=electrodes_2d,
                    terminals_2d=terminals_2d,
                    max_paths=3,
                )
            )
            if locked_indices:
                locked_names = [path_electrodes[idx] for idx in sorted(locked_indices)]
                print(f"  Locked straight feeders (no reroute): {locked_names}")
            if not candidate_indices:
                break

            round_improved = False
            for path_idx in candidate_indices:
                electrode_name = path_electrodes[path_idx]
                terminal_name = path_terminals[path_idx]
                target_terminal_pos = (
                    terminals_2d[terminal_name]
                    if terminals_2d is not None
                    else best_paths[path_idx][-1]
                )
                n_points = max(len(best_paths[path_idx]), 50)
                orig_clearance = (
                    round_analysis['crossing_count']
                    + round_analysis['overlap_length']
                    + round_analysis['electrode_violations']
                )
                orig_pair_collisions = greed.count_path_pair_collisions(
                    best_paths[path_idx],
                    electrode_name,
                    best_paths,
                    path_electrodes,
                    path_terminals,
                    terminal_name,
                    terminal_zones,
                    electrode_zones,
                )

                corridor_obstacles = corridor_obstacles_by_idx.get(path_idx)
                if corridor_obstacles:
                    print(
                        f"  Rerouting flexible path {electrode_name} "
                        f"around locked corridor partner(s)"
                    )

                trial_path = greed.greedy_trap_escape(
                    best_paths[path_idx],
                    electrode_name,
                    best_paths,
                    path_electrodes,
                    path_terminals,
                    electrode_zones,
                    terminal_zones,
                    target_terminal_pos=target_terminal_pos,
                    target_terminal_name=terminal_name,
                    n_points=n_points,
                    allow_partial_improvement=True,
                    quiet=True,
                    corridor_obstacle_paths=corridor_obstacles,
                )

                trial_paths = [path.copy() for path in best_paths]
                trial_paths[path_idx] = pin_path_endpoints_2d(
                    trial_path,
                    electrodes_2d[electrode_name] if electrodes_2d else trial_path[0],
                    target_terminal_pos,
                )
                trial_paths = _pin_all(trial_paths)
                trial_analysis = analyze_path_collisions(
                    trial_paths,
                    terminal_zones,
                    electrode_zones=electrode_zones,
                    path_electrodes=path_electrodes,
                    path_terminals=path_terminals,
                )
                new_clearance = (
                    trial_analysis['crossing_count']
                    + trial_analysis['overlap_length']
                    + trial_analysis['electrode_violations']
                )
                new_pair_collisions = greed.count_path_pair_collisions(
                    trial_paths[path_idx],
                    electrode_name,
                    trial_paths,
                    path_electrodes,
                    path_terminals,
                    terminal_name,
                    terminal_zones,
                    electrode_zones,
                )

                accepted = (
                    is_layout_clearance_free(trial_analysis)
                    or new_clearance < orig_clearance
                    or trial_analysis['collision_score'] < round_analysis['collision_score']
                    or new_pair_collisions < orig_pair_collisions
                )
                if accepted:
                    best_paths = trial_paths
                    best_collision_points = trial_analysis['points']
                    best_score = trial_analysis['collision_score']
                    round_improved = True
                    print(
                        f"  Round {round_idx + 1}: greedy reroute {electrode_name} "
                        f"(crossings={trial_analysis['crossing_count']}, score={best_score:.2f})"
                    )
                    if is_layout_clearance_free(trial_analysis):
                        break

            if is_layout_clearance_free(analyze_path_collisions(
                best_paths, terminal_zones,
                electrode_zones=electrode_zones,
                path_electrodes=path_electrodes,
                path_terminals=path_terminals,
            )):
                break
            if not round_improved:
                print(f"  Greedy aggressive stopped after round {round_idx + 1} (no improvement)")
                break

        print(f"Final collision score after greedy Phase 2: {best_score:.2f}")

    return _pin_all(best_paths)

def complete_path_reroute(
    path,
    electrode_name,
    electrode_zones,
    terminal_zones,
    x_bounds,
    n_points=20,
    target_electrode_pos=None,
    target_terminal_pos=None,
):
    """Completely reroute a path while maintaining endpoints"""
    start = (
        np.asarray(target_electrode_pos, dtype=float)
        if target_electrode_pos is not None else np.asarray(path[0], dtype=float)
    )
    end = (
        np.asarray(target_terminal_pos, dtype=float)
        if target_terminal_pos is not None else np.asarray(path[-1], dtype=float)
    )
    
    # Generate a curved path avoiding zones
    for _ in range(5):  # Multiple attempts
        # Create control points
        mid1 = start + (end - start) * 0.3 + np.random.uniform(-0.5, 0.5, 2)
        mid2 = start + (end - start) * 0.7 + np.random.uniform(-0.5, 0.5, 2)
        
        # Create spline
        tck, _ = splprep(np.array([start, mid1, mid2, end]).T, s=0)
        new_points = splev(np.linspace(0, 1, n_points), tck)
        new_path = np.column_stack(new_points)
        
        # Ensure endpoints match exactly
        new_path[0], new_path[-1] = start, end
        
        # Apply zone avoidance
        new_path = avoid_electrode_zones(
            new_path,
            electrode_name,
            electrode_zones,
            terminal_zones
        )
        new_path[0], new_path[-1] = start, end
        
        # Check bounds
        if is_path_within_bounds(new_path, x_bounds):
            return new_path
    
    # Fallback to original if all attempts fail
    fallback = np.asarray(path, dtype=float).copy()
    fallback[0], fallback[-1] = start, end
    return fallback