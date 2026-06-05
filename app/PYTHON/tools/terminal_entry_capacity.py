"""
Check whether fixed-spacing entry slots fit on each terminal zone boundary.

Used to validate the proposed init scheme: anchor midpoint slot, neighbors at
±spacing along the terminal zone boundary arc.

Run from genetic_SHAPE/app:
    python -m PYTHON.tools.terminal_entry_capacity 1
    python -m PYTHON.tools.terminal_entry_capacity 1 --spacing 4.0 --full-circle
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import numpy as np

DEFAULT_SPACING = 4.5
TERMINAL_ZONE_FACTOR = 0.15


def polar_projection(points_3d, cz_pos):
    rel_points = points_3d - cz_pos
    r = np.linalg.norm(rel_points, axis=1)
    theta = np.arctan2(rel_points[:, 1], rel_points[:, 0])
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])


def build_terminals_2d(electrodes_2d, fiducials, cz_pos):
    max_dist = 1.2 * max(np.linalg.norm(list(electrodes_2d.values()), axis=1))
    terminals_2d = {}
    for term in ("TERMINAL_LEFT", "TERMINAL_RIGHT"):
        if term in fiducials:
            pos = polar_projection(np.array([fiducials[term]]), cz_pos)[0]
            angle = np.arctan2(pos[1], pos[0])
            terminals_2d[term] = max_dist * np.array([np.cos(angle), np.sin(angle)])
    return terminals_2d


def terminal_zone_size(electrodes_2d) -> float:
    avg_dist = float(np.mean([np.linalg.norm(pos) for pos in electrodes_2d.values()]))
    return avg_dist * TERMINAL_ZONE_FACTOR


def straight_chord(electrode_pos, terminal_pos, n_points=50):
    start = np.asarray(electrode_pos, dtype=float)
    end = np.asarray(terminal_pos, dtype=float)
    return np.linspace(start, end, n_points)


def chord_boundary_entry(path_coords, terminal_center, zone_radius):
    """First point where straight chord enters the terminal disc (approximate)."""
    coords = np.asarray(path_coords, dtype=float)
    center = np.asarray(terminal_center, dtype=float)
    for idx in range(len(coords)):
        if np.linalg.norm(coords[idx] - center) <= zone_radius:
            if idx == 0:
                return coords[idx]
            prev = coords[idx - 1]
            curr = coords[idx]
            # Linear step along segment until boundary radius
            for t in np.linspace(0.0, 1.0, 20):
                pt = prev + t * (curr - prev)
                dist = np.linalg.norm(pt - center)
                if abs(dist - zone_radius) < 0.05 or dist <= zone_radius:
                    return pt
            return curr
    return coords[-1]


def angular_span_on_boundary(entry_angles: list[float]) -> float:
    """Smallest arc on the circle containing all entry angles (radians)."""
    if len(entry_angles) < 2:
        return 0.0
    angles = np.sort(np.asarray(entry_angles, dtype=float))
    gaps = np.diff(angles)
    wrap_gap = (angles[0] + 2.0 * np.pi) - angles[-1]
    max_gap = float(max(np.max(gaps), wrap_gap))
    return float(2.0 * np.pi - max_gap)


def load_subject(SUBJECT_ID: int):
    with open(f"data/json/electrode_positions_{SUBJECT_ID}.json") as f:
        electrodes_3d = {k: np.array(v) for k, v in json.load(f).items()}
    with open(f"data/json/fiducials_{SUBJECT_ID}.json") as f:
        fiducials = {k: np.array(v) for k, v in json.load(f).items()}

    assignment_path = f"data/json/initial_terminal_assignments_{SUBJECT_ID}.json"
    if os.path.exists(assignment_path):
        with open(assignment_path) as f:
            assignments = json.load(f)
    else:
        with open(f"data/json/init_connection_paths_{SUBJECT_ID}.json") as f:
            connections = json.load(f)
        from PYTHON.tools import initiate3DConnections as init_conn

        assignments = init_conn.load_or_create_terminal_assignments(SUBJECT_ID, connections)

    cz_pos = electrodes_3d["Cz"]
    electrodes_2d = {
        k: polar_projection(np.array([v]), cz_pos)[0]
        for k, v in electrodes_3d.items()
    }
    terminals_2d = build_terminals_2d(electrodes_2d, fiducials, cz_pos)
    radius = terminal_zone_size(electrodes_2d)
    return assignments, electrodes_2d, terminals_2d, radius


def check_terminal_entry_capacity(
    SUBJECT_ID: int,
    spacing: float = DEFAULT_SPACING,
    use_half_circle: bool = True,
) -> dict:
    assignments, electrodes_2d, terminals_2d, radius = load_subject(SUBJECT_ID)
    full_arc = 2.0 * np.pi * radius
    half_arc = np.pi * radius
    available_arc = half_arc if use_half_circle else full_arc

    counts = Counter(assignments.values())
    terminals = sorted(set(assignments.values()) | set(terminals_2d.keys()))
    per_terminal = []
    all_ok = True

    for terminal in terminals:
        if terminal not in terminals_2d:
            continue
        terminal_center = terminals_2d[terminal]
        electrodes = sorted(
            name for name, term in assignments.items() if term == terminal
        )
        n = len(electrodes)
        required_arc = max(0, n - 1) * spacing

        entry_angles = []
        for name in electrodes:
            chord = straight_chord(electrodes_2d[name], terminal_center)
            hit = chord_boundary_entry(chord, terminal_center, radius)
            delta = hit - terminal_center
            entry_angles.append(float(np.arctan2(delta[1], delta[0])))

        init_span = angular_span_on_boundary(entry_angles) * radius
        geometric_ok = required_arc <= available_arc + 1e-9
        init_span_ok = required_arc <= init_span + 1e-9 if n > 1 else True
        ok = geometric_ok
        all_ok = all_ok and ok

        per_terminal.append(
            {
                "terminal": terminal,
                "n_electrodes": n,
                "spacing_requested": spacing,
                "required_arc": required_arc,
                "available_arc_half": half_arc,
                "available_arc_full": full_arc,
                "available_arc_used": available_arc,
                "max_fit_spacing_half": half_arc / max(n - 1, 1) if n > 1 else float("inf"),
                "max_fit_spacing_full": full_arc / max(n - 1, 1) if n > 1 else float("inf"),
                "geometric_ok_half": required_arc <= half_arc + 1e-9,
                "geometric_ok_full": required_arc <= full_arc + 1e-9,
                "init_chord_angular_span": init_span,
                "init_span_ok": init_span_ok,
                "ok": ok,
                "electrodes": electrodes,
            }
        )

    return {
        "subject_id": SUBJECT_ID,
        "spacing": spacing,
        "terminal_zone_radius": radius,
        "use_half_circle": use_half_circle,
        "all_ok": all_ok,
        "terminals": per_terminal,
    }


def format_capacity_report(report: dict) -> str:
    lines = [
        f"Terminal entry capacity — subject {report['subject_id']}",
        f"  spacing={report['spacing']:.2f}, zone radius={report['terminal_zone_radius']:.3f}",
        f"  mode={'half-circle' if report['use_half_circle'] else 'full-circle'}",
        "",
    ]
    for t in report["terminals"]:
        status = "OK" if t["ok"] else "FAIL"
        lines.extend(
            [
                f"[{status}] {t['terminal']}: {t['n_electrodes']} electrodes",
                f"       required arc (N-1)*spacing = {t['required_arc']:.2f}",
                f"       half-circle available      = {t['available_arc_half']:.2f}  "
                f"({'ok' if t['geometric_ok_half'] else 'NO'})",
                f"       full-circle available      = {t['available_arc_full']:.2f}  "
                f"({'ok' if t['geometric_ok_full'] else 'NO'})",
                f"       max spacing on half-circle = {t['max_fit_spacing_half']:.2f}",
                f"       max spacing on full-circle = {t['max_fit_spacing_full']:.2f}",
                f"       init chord span on boundary ~ {t['init_chord_angular_span']:.2f}  "
                f"(informational; slots are assigned, not inferred from chords)",
            ]
        )
        if not t["ok"]:
            lines.append(
                f"       -> need spacing <= {t['max_fit_spacing_half']:.2f} (half) "
                f"or <= {t['max_fit_spacing_full']:.2f} (full)"
            )
        lines.append("")

    lines.append(
        "Overall: "
        + (
            "all terminals fit at requested spacing."
            if report["all_ok"]
            else "NOT FEASIBLE at requested spacing."
        )
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Check terminal entry slot capacity.")
    parser.add_argument("subject_id", type=int, nargs="?", default=1)
    parser.add_argument(
        "--spacing",
        type=float,
        default=DEFAULT_SPACING,
        help="Fixed arc spacing between adjacent entry slots (default: 5.0).",
    )
    parser.add_argument(
        "--full-circle",
        action="store_true",
        help="Use full terminal boundary instead of half-circle.",
    )
    args = parser.parse_args()

    report = check_terminal_entry_capacity(
        args.subject_id,
        spacing=args.spacing,
        use_half_circle=not args.full_circle,
    )
    print(format_capacity_report(report))
    raise SystemExit(0 if report["all_ok"] else 1)


if __name__ == "__main__":
    main()
