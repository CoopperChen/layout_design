"""Initial wire layout: hub balance, crossings, odd slot swaps, and the UV lift."""

import numpy as np
import pytest
from scipy.sparse.csgraph import dijkstra

from app.cli import build_parser
from app.interconnect.design import (
    WireLayout,
    balanced_assignment,
    build_wire_layout,
    mutual_crossings,
    odd_pair_count,
    odd_swap,
    path_hits_electrodes,
    path_with_electrode_detour,
    polar_layout,
    polar_projection,
    straight_path,
)
from app.interconnect.uv_lift import reconstruct_path
from app.interconnect.view import (
    layout_document,
    load_saved_terminals,
    polyline_length,
    require_connected_surface,
    surface_graph,
)
from app.pipeline.run import STAGES


def test_balanced_assignment_moves_the_cheapest_lead():
    distances = {
        "a": {"TERMINAL_LEFT": 1.0, "TERMINAL_RIGHT": 2.0},
        "b": {"TERMINAL_LEFT": 1.0, "TERMINAL_RIGHT": 3.0},
        "c": {"TERMINAL_LEFT": 1.0, "TERMINAL_RIGHT": 10.0},
    }
    assignment = balanced_assignment(distances)
    assert assignment["a"] == "TERMINAL_RIGHT"
    assert assignment["b"] == "TERMINAL_LEFT"
    assert assignment["c"] == "TERMINAL_LEFT"


def test_mutual_crossings_counts_an_x_and_an_overlap():
    assert mutual_crossings(straight_path([0, 0], [1, 1], n=2), straight_path([0, 1], [1, 0], n=2)) == 1
    assert mutual_crossings(straight_path([0, 0], [1, 0], n=2), straight_path([0, 1], [1, 1], n=2)) == 0
    assert mutual_crossings(straight_path([0, 0], [1, 0], n=2), straight_path([1, 0], [2, 1], n=2)) == 0
    overlap = mutual_crossings(straight_path([0, 0], [2, 0], n=2), straight_path([0.5, 0], [1.5, 0], n=2))
    assert overlap == 1


def test_detour_clears_a_foreign_electrode():
    centers = {"A": np.array([0.0, 0.0]), "B": np.array([10.0, 0.0])}
    path = path_with_electrode_detour(
        centers["A"], np.array([20.0, 0.0]), "A", centers, radius=3.0
    )
    assert path_hits_electrodes(straight_path(centers["A"], [20.0, 0.0]), "A", centers, 3.0) == 1
    assert path_hits_electrodes(path, "A", centers, 3.0) == 0


def test_odd_swap_exchanges_a_crossed_pair():
    electrodes = {
        "A": np.array([0.0, 0.0]),
        "B": np.array([0.0, 2.0]),
    }
    entries = {
        "A": np.array([10.0, 2.0]),
        "B": np.array([10.0, 0.0]),
    }
    paths = [straight_path(electrodes["A"], entries["A"]), straight_path(electrodes["B"], entries["B"])]
    assert mutual_crossings(paths[0], paths[1]) == 1
    swapped, new_entries, slots, notes = odd_swap(
        paths,
        ["A", "B"],
        ["TERMINAL_LEFT", "TERMINAL_LEFT"],
        electrodes,
        entries,
        {"A": 0, "B": 1},
        electrodes,
        radius=1.0,
    )
    assert mutual_crossings(swapped[0], swapped[1]) == 0
    assert slots["A"] == 1 and slots["B"] == 0
    np.testing.assert_allclose(new_entries["A"], [10.0, 0.0])
    assert any("A<->B" in note for note in notes)


def test_odd_swap_leaves_an_uncrossed_pair():
    electrodes = {"A": np.array([0.0, 0.0]), "B": np.array([0.0, 2.0])}
    entries = {"A": np.array([10.0, 0.0]), "B": np.array([10.0, 2.0])}
    names = ["A", "B"]
    terminals = ["TERMINAL_LEFT", "TERMINAL_LEFT"]
    paths = [straight_path(electrodes[name], entries[name]) for name in names]
    assert odd_pair_count(paths, terminals) == 0
    swapped, new_entries, slots, notes = odd_swap(
        paths,
        names,
        terminals,
        electrodes,
        entries,
        {"A": 0, "B": 1},
        electrodes,
        radius=0.5,
    )
    assert notes == []
    assert slots == {"A": 0, "B": 1}
    np.testing.assert_allclose(new_entries["A"], entries["A"])
    assert odd_pair_count(swapped, terminals) == 0


def test_build_wire_layout_splits_a_symmetric_set():
    cz = np.array([0.0, 0.0, 40.0])
    electrodes = {
        "Fp1": np.array([-15.0, 50.0, 30.0]),
        "Fp2": np.array([15.0, 50.0, 30.0]),
        "O1": np.array([-15.0, -50.0, 30.0]),
        "O2": np.array([15.0, -50.0, 30.0]),
    }
    terminals = {
        "TERMINAL_LEFT": np.array([-40.0, -10.0, 0.0]),
        "TERMINAL_RIGHT": np.array([40.0, -10.0, 0.0]),
    }
    distances = {}
    for name, point in electrodes.items():
        distances[name] = {
            hub: float(np.linalg.norm(point - terminals[hub])) for hub in terminals
        }
    layout = build_wire_layout(electrodes, terminals, cz, distances, spacing=6.0)
    assert layout.assignments["Fp1"] == "TERMINAL_LEFT"
    assert layout.assignments["Fp2"] == "TERMINAL_RIGHT"
    assert layout.assignments["O1"] == "TERMINAL_LEFT"
    assert layout.assignments["O2"] == "TERMINAL_RIGHT"
    assert layout.odd_pairs == 0
    assert set(layout.slots) == set(electrodes)


def test_saved_terminals_require_a_pick(tmp_path, monkeypatch):
    target = tmp_path / "interconnect.json"
    target.write_text(
        '{"terminals": {"TERMINAL_LEFT": [0, 0, 0], "TERMINAL_RIGHT": [1, 0, 0]}}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr("app.interconnect.view.paths.interconnect_json", lambda _sid: target)
    assert load_saved_terminals(1) is None
    target.write_text(
        '{"picked": true, "terminals": {"TERMINAL_LEFT": [0, 0, 0], "TERMINAL_RIGHT": [1, 0, 0]}}\n',
        encoding="utf-8",
    )
    loaded = load_saved_terminals(1)
    assert loaded is not None
    np.testing.assert_allclose(loaded["TERMINAL_RIGHT"], [1, 0, 0])


def test_clicked_hub_stays_at_its_polar_position():
    cz = np.zeros(3)
    electrodes = {"Cz": np.array([0.0, 0.0, 40.0])}
    terminals = {
        "TERMINAL_LEFT": np.array([-30.0, -20.0, 5.0]),
        "TERMINAL_RIGHT": np.array([30.0, -20.0, 5.0]),
    }
    _, flat = polar_layout(electrodes, terminals, cz)
    expected = polar_projection(terminals["TERMINAL_LEFT"].reshape(1, 3), cz)[0]
    np.testing.assert_allclose(flat["TERMINAL_LEFT"], expected)


def test_reconstruct_path_snaps_the_interior_onto_mesh_vertices():
    from scipy.spatial import cKDTree

    grid_2d = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]], dtype=float)
    grid_3d = np.array(
        [[0.0, 0.0, 5.0], [10.0, 0.0, 5.0], [0.0, 10.0, 5.0], [10.0, 10.0, 5.0]],
        dtype=float,
    )
    path_2d = np.array([[0.0, 0.0], [5.0, 5.0], [10.0, 0.0]], dtype=float)
    out = reconstruct_path(
        grid_3d[0],
        grid_3d[1],
        path_2d,
        cKDTree(grid_2d),
        grid_3d,
        cKDTree(grid_3d),
    )
    np.testing.assert_allclose(out[0], grid_3d[0])
    np.testing.assert_allclose(out[-1], grid_3d[1])
    assert any(np.allclose(out[1], row) for row in grid_3d)


def test_a_broken_shell_is_rejected():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [5.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [5.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    faces = np.hstack(([3, 0, 1, 2], [3, 3, 4, 5]))
    graph = surface_graph(points, faces)
    with pytest.raises(RuntimeError):
        require_connected_surface(graph, len(points) + 3)


def test_surface_distance_follows_mesh_edges():
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=float,
    )
    faces = np.hstack(([3, 0, 1, 2], [3, 0, 2, 3]))
    graph = surface_graph(points, faces)
    require_connected_surface(graph, len(points))
    distance = dijkstra(graph, directed=False, indices=1)
    assert float(distance[3]) == 2.0


def test_trace_length_follows_the_polyline():
    route = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
    assert polyline_length(route) == 7.0
    layout = WireLayout(
        assignments={"A": "TERMINAL_LEFT"},
        slots={"A": 0},
        entries_2d={},
        paths_2d={},
        electrodes_2d={},
        terminals_2d={},
        odd_pairs_before=0,
        crossings_before=0,
        odd_pairs=0,
        crossings=0,
    )
    document = layout_document(layout, {}, {"A": route}, spacing=6.0)
    assert document["lengths_mm"]["A"] == 7.0


def test_interconnect_command_is_outside_the_pipeline():
    args = build_parser().parse_args(["interconnect", "--subject", "2", "--no-show", "--spacing", "5"])
    assert args.func.__name__ == "cmd_interconnect"
    assert args.spacing == 5.0
    assert "interconnect" not in STAGES
