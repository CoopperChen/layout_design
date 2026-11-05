"""Show the initial wire layout on the subject head. Called after the pipeline."""

from __future__ import annotations

import colorsys
import json
from pathlib import Path

import numpy as np
import pyvista as pv
import vtk
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from app import paths
from app.interact import KEYS_HELP, bind_accept_cancel
from app.interconnect.design import (
    SLOT_SPACING_MM,
    TERMINALS,
    WireLayout,
    build_wire_layout,
    terminal_zone_radius,
)
from app.interconnect.uv_lift import lift_layout
from app.layout.place import _snap_to_surface
from app.layout.session import _add_head, _sphere_radius
from app.mesh.io import load_head_mesh
from app.mesh.texture import visible_surface
from app.subject import load_layout

_LIFT_MM = 0.6


def surface_graph(points: np.ndarray, faces: np.ndarray) -> csr_matrix:
    """Undirected edge graph weighted by Euclidean length."""
    triangles = np.asarray(faces, dtype=np.int64).reshape(-1, 4)[:, 1:]
    edges = np.vstack((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    start = edges[:, 0]
    end = edges[:, 1]
    weight = np.linalg.norm(points[start] - points[end], axis=1)
    count = points.shape[0]
    data = np.concatenate((weight, weight))
    rows = np.concatenate((start, end))
    cols = np.concatenate((end, start))
    return csr_matrix((data, (rows, cols)), shape=(count, count))


def require_connected_surface(graph: csr_matrix, n_points: int) -> None:
    """Reject a broken shell. A chord through the head is not a wire."""
    count, labels = connected_components(graph, directed=False)
    largest = int(np.bincount(labels).max()) if n_points else 0
    if count > 1 and largest < 0.5 * n_points:
        raise RuntimeError(
            f"The head surface is {count} disconnected pieces, so a wire cannot stay on the scalp.\n"
            "Use the cleaned head.stl from: python -m app mesh --subject <id>"
        )


def _as_triangles(mesh: pv.DataSet) -> pv.PolyData:
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()
    if not bool(getattr(mesh, "is_all_triangles", False)):
        mesh = mesh.triangulate()
    if int(getattr(mesh, "n_cells", 0)) == 0:
        raise ValueError("Head mesh has no triangles.")
    return mesh


def load_display_mesh(subject_id: int | str) -> pv.PolyData:
    """Textured head for picking and viewing. Falls back to the cleaned STL."""
    obj = paths.head_obj(subject_id)
    if obj.is_file():
        mesh = visible_surface(load_head_mesh(obj))
        return _as_triangles(mesh)
    return load_route_mesh(subject_id)


def load_route_mesh(subject_id: int | str) -> pv.PolyData:
    """Connected scalp used to lay wires. The cleaned STL is one surface."""
    stl = paths.head_stl(subject_id)
    if stl.is_file():
        return _as_triangles(pv.read(stl))
    obj = paths.head_obj(subject_id)
    if not obj.is_file():
        raise FileNotFoundError(
            f"No head mesh for subject {subject_id}: {stl}\n"
            f"Run: python -m app mesh --subject {subject_id}"
        )
    return _as_triangles(visible_surface(load_head_mesh(obj)))


def _nearest_vertices(points: np.ndarray, queries: np.ndarray) -> np.ndarray:
    _, index = cKDTree(points).query(np.asarray(queries, dtype=float))
    return np.asarray(index, dtype=int)


def outward_normals(mesh: pv.PolyData) -> np.ndarray:
    """Point normals aimed out of the head, one per vertex."""
    if "Normals" in mesh.point_data:
        normals = np.asarray(mesh.point_normals, dtype=float)
    else:
        surfaced = mesh.compute_normals(cell_normals=False, point_normals=True, inplace=False)
        normals = np.asarray(surfaced.point_normals, dtype=float)
    radial = np.asarray(mesh.points, dtype=float) - np.asarray(mesh.center, dtype=float)
    flip = np.einsum("ij,ij->i", normals, radial) < 0.0
    normals = normals.copy()
    normals[flip] *= -1.0
    return normals


def _xyz_list(point: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(point, dtype=float).reshape(-1)]


def layout_document(
    layout: WireLayout,
    terminals: dict[str, np.ndarray],
    routes: dict[str, np.ndarray],
    *,
    spacing: float,
) -> dict:
    return {
        "terminals": {name: _xyz_list(point) for name, point in terminals.items()},
        "assignments": dict(layout.assignments),
        "slots": {name: int(index) for name, index in layout.slots.items()},
        "picked": True,
        "spacing_mm": float(spacing),
        "odd_pairs_before": int(layout.odd_pairs_before),
        "crossings_before": int(layout.crossings_before),
        "odd_pairs": int(layout.odd_pairs),
        "crossings": int(layout.crossings),
        "routes": {
            name: [_xyz_list(point) for point in np.asarray(route, dtype=float)]
            for name, route in routes.items()
        },
    }


def save_interconnect(subject_id: int | str, document: dict) -> Path:
    path = paths.interconnect_json(subject_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return path


def load_saved_terminals(subject_id: int | str) -> dict[str, np.ndarray] | None:
    path = paths.interconnect_json(subject_id)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data.get("picked"):
        return None
    raw = data.get("terminals") or {}
    if any(name not in raw for name in TERMINALS):
        return None
    return {name: np.asarray(raw[name], dtype=float) for name in TERMINALS}


def _color(index: int, count: int) -> str:
    red, green, blue = colorsys.hsv_to_rgb(index / max(count, 1), 0.72, 0.82)
    return f"#{int(red * 255):02x}{int(green * 255):02x}{int(blue * 255):02x}"


def _summary_lines(layout: WireLayout) -> list[str]:
    lines = [
        (
            f"Initial wires  ·  odd pairs {layout.odd_pairs_before}→{layout.odd_pairs}"
            f"  ·  crossings {layout.crossings_before}→{layout.crossings}"
        )
    ]
    for terminal, label in (("TERMINAL_LEFT", "LEFT"), ("TERMINAL_RIGHT", "RIGHT")):
        ordered = [name for name, hub in layout.assignments.items() if hub == terminal]
        ordered.sort(key=lambda name: layout.slots.get(name, 0))
        body = "  ".join(f"{layout.slots[name] + 1}:{name}" for name in ordered)
        lines.append(f"{label}  {body}" if body else f"{label}  (none)")
    return lines


def print_layout(layout: WireLayout) -> None:
    print(
        f"Odd pairs {layout.odd_pairs_before} -> {layout.odd_pairs}, "
        f"crossings {layout.crossings_before} -> {layout.crossings}"
    )
    for note in layout.notes:
        print(f"  {note}")
    for line in _summary_lines(layout)[1:]:
        print(line)


def pick_terminals(
    mesh: pv.PolyData,
    electrodes: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray] | None:
    """Click the left harness point on the scalp, then the right one."""
    sequence = (
        ("TERMINAL_LEFT", "1) Left terminal — right-click the scalp"),
        ("TERMINAL_RIGHT", "2) Right terminal — right-click the scalp"),
    )
    picked: dict[str, np.ndarray] = {}
    state = {"idx": 0, "last_pt": None, "discard": False}
    plotter = pv.Plotter(window_size=(1600, 1600))
    plotter.set_background("white")
    _add_head(plotter, mesh)
    radius = max(_sphere_radius(mesh) * 6.0, 1.5)
    if electrodes:
        for name, point in electrodes.items():
            plotter.add_mesh(
                pv.Sphere(center=np.asarray(point, dtype=float), radius=radius * 0.7),
                color="tomato",
                smooth_shading=True,
                name=f"electrode_{name}",
            )

    def show_instr(text: str) -> None:
        try:
            plotter.remove_actor("instr")
        except Exception:
            pass
        plotter.add_text(text, name="instr", font_size=14)

    show_instr(f"{sequence[0][1]}  ·  {KEYS_HELP}")
    picker = vtk.vtkCellPicker()
    picker.SetTolerance(0.0005)

    def on_right_click(obj, _event) -> None:
        x, y = obj.GetEventPosition()
        picker.Pick(x, y, 0, plotter.renderer)
        pos = picker.GetPickPosition()
        state["last_pt"] = (float(pos[0]), float(pos[1]), float(pos[2]))
        try:
            plotter.remove_actor("provisional")
        except Exception:
            pass
        plotter.add_mesh(
            pv.Sphere(center=state["last_pt"], radius=radius),
            color="white",
            opacity=0.65,
            name="provisional",
        )

    def on_confirm() -> None:
        pt = state["last_pt"]
        if pt is None:
            print("Right-click the head first.")
            return
        index = state["idx"]
        name, _label = sequence[index]
        picked[name] = np.asarray(pt, dtype=float)
        plotter.add_mesh(
            pv.Sphere(center=pt, radius=radius),
            color="black" if name.endswith("RIGHT") else "dimgray",
            name=name,
        )
        state["last_pt"] = None
        state["idx"] = index + 1
        try:
            plotter.remove_actor("provisional")
        except Exception:
            pass
        if state["idx"] >= len(sequence):
            plotter.close()
            return
        show_instr(f"{sequence[state['idx']][1]}  ·  {KEYS_HELP}")

    def on_discard() -> None:
        state["discard"] = True
        plotter.close()

    plotter.iren.add_observer("RightButtonPressEvent", on_right_click)
    bind_accept_cancel(plotter, on_confirm, on_discard)
    print(f"Terminal picks: right-click, then Space / Enter / S. {KEYS_HELP}.")
    plotter.show()
    if state["discard"] or len(picked) < len(sequence):
        return None
    return picked


def show_layout(
    mesh: pv.PolyData,
    electrodes: dict[str, np.ndarray],
    terminals: dict[str, np.ndarray],
    layout: WireLayout,
    routes: dict[str, np.ndarray],
    slot_xyz: dict[str, np.ndarray],
) -> None:
    plotter = pv.Plotter(window_size=(1600, 1600))
    plotter.set_background("white")
    _add_head(plotter, mesh)
    radius = max(_sphere_radius(mesh) * 6.0, 1.5)
    names = list(electrodes)
    colors = {name: _color(i, len(names)) for i, name in enumerate(names)}

    for name, point in electrodes.items():
        plotter.add_mesh(
            pv.Sphere(center=np.asarray(point, dtype=float), radius=radius),
            color=colors[name],
            smooth_shading=True,
            name=f"electrode_{name}",
        )
    plotter.add_point_labels(
        np.vstack([electrodes[name] for name in names]),
        names,
        font_size=12,
        text_color="black",
        point_size=1,
        shape_opacity=0.7,
        always_visible=True,
        name="electrode_labels",
    )

    for terminal, color, label in (
        ("TERMINAL_LEFT", "dimgray", "LEFT"),
        ("TERMINAL_RIGHT", "black", "RIGHT"),
    ):
        if terminal not in terminals:
            continue
        plotter.add_mesh(
            pv.Sphere(center=np.asarray(terminals[terminal], dtype=float), radius=radius * 1.4),
            color=color,
            smooth_shading=True,
            name=terminal,
        )
        plotter.add_point_labels(
            np.asarray(terminals[terminal], dtype=float).reshape(1, 3),
            [label],
            font_size=14,
            text_color="white",
            point_size=1,
            always_visible=True,
            name=f"label_{terminal}",
        )

    slot_names = [name for name in names if name in slot_xyz]
    if slot_names:
        plotter.add_point_labels(
            np.vstack([slot_xyz[name] for name in slot_names]),
            [str(layout.slots[name] + 1) for name in slot_names],
            font_size=12,
            text_color="black",
            point_size=8,
            always_visible=True,
            name="slot_labels",
        )

    for name, route in routes.items():
        line = pv.lines_from_points(np.asarray(route, dtype=float))
        try:
            line = line.tube(radius=max(radius * 0.18, 0.35))
        except Exception:
            pass
        plotter.add_mesh(line, color=colors.get(name, "orange"), smooth_shading=True, name=f"wire_{name}")

    corners = ("upper_left", "lower_left", "lower_right")
    for index, line in enumerate(_summary_lines(layout)):
        plotter.add_text(
            line,
            position=corners[min(index, len(corners) - 1)],
            font_size=11,
            color="black",
            name=f"summary_{index}",
        )
    print("Wire view: each number is that electrode's slot on its harness. Close the window when done.")
    plotter.show()


def run_interconnect(
    subject_id: int | str,
    *,
    pick: bool = False,
    show: bool = True,
    spacing: float = SLOT_SPACING_MM,
) -> Path | None:
    """Build and save the initial wire layout. Returns None if terminal picks are discarded."""
    layout_data = load_layout(subject_id)
    display = load_display_mesh(subject_id)
    route_mesh = load_route_mesh(subject_id)
    terminals = None if pick else load_saved_terminals(subject_id)
    if terminals is None:
        terminals = pick_terminals(display, layout_data.electrodes)
        if terminals is None:
            print("Discarded terminal picks. interconnect.json was not changed.")
            return None
    snapped = _snap_to_surface(route_mesh, np.vstack([terminals[name] for name in TERMINALS]))
    terminals = {name: np.asarray(snapped[i], dtype=float) for i, name in enumerate(TERMINALS)}

    points = np.asarray(route_mesh.points, dtype=float)
    graph = surface_graph(points, np.asarray(route_mesh.faces))
    require_connected_surface(graph, len(points))
    electrode_names = list(layout_data.electrodes)
    electrode_vertex = _nearest_vertices(
        points, np.vstack([layout_data.electrodes[name] for name in electrode_names])
    )
    terminal_vertex = _nearest_vertices(points, np.vstack([terminals[name] for name in TERMINALS]))
    distances: dict[str, dict[str, float]] = {name: {} for name in electrode_names}
    for index, terminal in enumerate(TERMINALS):
        reach = dijkstra(graph, directed=False, indices=int(terminal_vertex[index]))
        for electrode_index, name in enumerate(electrode_names):
            value = float(reach[int(electrode_vertex[electrode_index])])
            if not np.isfinite(value):
                raise RuntimeError(
                    f"{name} cannot reach {terminal} along the head surface. "
                    "Pick both terminals on the scalp."
                )
            distances[name][terminal] = value

    designed = build_wire_layout(
        layout_data.electrodes,
        terminals,
        layout_data.cz,
        distances,
        spacing=spacing,
    )
    surface_routes = lift_layout(
        designed,
        layout_data.electrodes,
        terminals,
        layout_data.cz,
        points,
        terminal_zone_radius(designed.electrodes_2d),
    )
    normals = outward_normals(route_mesh)
    routes = {
        name: np.asarray(route, dtype=float) + _LIFT_MM * normals[_nearest_vertices(points, route)]
        for name, route in surface_routes.items()
    }
    slots_3d = {name: np.asarray(route[-1], dtype=float) for name, route in routes.items()}
    document = layout_document(designed, terminals, routes, spacing=spacing)
    out = save_interconnect(subject_id, document)
    print_layout(designed)
    print(f"Wrote {out}")
    if show:
        show_layout(display, layout_data.electrodes, terminals, designed, routes, slots_3d)
    return out
