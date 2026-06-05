import pyvista as pv
import numpy as np
import json
import os
from pathlib import Path
from PYTHON.tools.helper import load_electrode_positions_and_fiducials

# How electrodes are split across terminals when assignments are first created.
#   "balanced"  - near-equal counts with minimal extra wire length (recommended)
#   "shortest"  - each electrode uses its shorter geodesic path (legacy)
TERMINAL_ASSIGNMENT_STRATEGY = "balanced"

def save_connection_data(connections, SUBJECT_ID: int):
    """Save connection paths with metadata in JSON format"""
    
    filename = f"data/json/init_connection_paths_{SUBJECT_ID}.json"
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    
    serializable_data = []
    for conn in connections:
        serializable_data.append({
            'electrode': conn['electrode'],
            'terminal': conn['terminal'],
            'path_points': conn['path'].points.tolist(),
            'path_length': conn['path'].length,
            'resolution': conn['resolution']
        })
    
    with open(filename, 'w') as f:
        json.dump(serializable_data, f, indent=2)
    print(f"Saved connection data to {filename} ✅")

def load_connection_data(SUBJECT_ID: int):
    """Load previously saved connection data"""
    
    filename = f"data/json/init_connection_paths_{SUBJECT_ID}.json"
    
    if not Path(filename).exists():
        return None
        
    with open(filename) as f:
        data = json.load(f)
    
    connections = []
    for item in data:
        connections.append({
            'electrode': item['electrode'],
            'terminal': item['terminal'],
            'path': pv.lines_from_points(np.array(item['path_points'])),
            'path_length': item['path_length'],
            'resolution': item['resolution']
        })
    print(f"Loaded {len(connections)} connections from {filename} ✅")
    return connections

# --------------------------
# Terminal Assignment
# --------------------------

def compute_terminal_path_options(connections):
    """Map each electrode to available terminal path lengths."""
    options = {}
    for conn in connections:
        electrode = conn['electrode']
        options.setdefault(electrode, {})[conn['terminal']] = conn['path_length']
    return options


def compute_shortest_terminal_assignments(connections):
    """Assign each electrode to its shorter geodesic terminal."""
    options = compute_terminal_path_options(connections)
    return {
        electrode: min(lengths, key=lengths.get)
        for electrode, lengths in options.items()
    }


def compute_balanced_terminal_assignments(
    connections,
    terminal_names=('TERMINAL_LEFT', 'TERMINAL_RIGHT'),
):
    """
    Assign electrodes to terminals with near-equal counts and minimal extra wire length.

    Starts from shortest-path assignment, then moves the cheapest candidates from the
    overloaded terminal until capacities are met (e.g. 10/9 for 19 electrodes).
    """
    options = compute_terminal_path_options(connections)
    terminal_names = tuple(terminal_names)
    n_electrodes = len(options)

    capacity = {}
    base = n_electrodes // len(terminal_names)
    remainder = n_electrodes % len(terminal_names)
    for idx, terminal in enumerate(terminal_names):
        capacity[terminal] = base + (1 if idx < remainder else 0)

    assignment = compute_shortest_terminal_assignments(connections)
    counts = {terminal: 0 for terminal in terminal_names}
    for terminal in assignment.values():
        if terminal in counts:
            counts[terminal] += 1

    def overload(terminal):
        return counts[terminal] - capacity[terminal]

    def underload(terminal):
        return capacity[terminal] - counts[terminal]

    while True:
        heavy = max(terminal_names, key=overload)
        light = max(terminal_names, key=underload)
        if overload(heavy) <= 0 or underload(light) <= 0:
            break

        movable = [
            electrode for electrode, terminal in assignment.items()
            if terminal == heavy and light in options[electrode]
        ]
        if not movable:
            break

        best_electrode = min(
            movable,
            key=lambda electrode: options[electrode][light] - options[electrode][heavy],
        )
        assignment[best_electrode] = light
        counts[heavy] -= 1
        counts[light] += 1

    return assignment, capacity


def summarize_terminal_assignments(assignments, capacity=None):
    """Return a readable count summary for logging."""
    counts = {}
    for terminal in assignments.values():
        counts[terminal] = counts.get(terminal, 0) + 1

    parts = []
    for terminal in sorted(counts):
        if capacity and terminal in capacity:
            parts.append(f"{terminal}: {counts[terminal]}/{capacity[terminal]}")
        else:
            parts.append(f"{terminal}: {counts[terminal]}")
    return ", ".join(parts)


def load_or_create_terminal_assignments(
    SUBJECT_ID,
    connections,
    strategy=None,
    force_recompute=False,
):
    """Load cached terminal assignments or create them from init connection paths."""
    if strategy is None:
        strategy = TERMINAL_ASSIGNMENT_STRATEGY

    assignment_path = f"data/json/initial_terminal_assignments_{SUBJECT_ID}.json"
    if not force_recompute and os.path.exists(assignment_path):
        with open(assignment_path, 'r') as f:
            assignments = json.load(f)
        print(f"Loaded terminal assignments ({summarize_terminal_assignments(assignments)})")
        return assignments

    if strategy == "balanced":
        assignments, capacity = compute_balanced_terminal_assignments(connections)
        summary = summarize_terminal_assignments(assignments, capacity)
        print(f"Created balanced terminal assignments: {summary}")
    elif strategy == "shortest":
        assignments = compute_shortest_terminal_assignments(connections)
        print(f"Created shortest-path terminal assignments: {summarize_terminal_assignments(assignments)}")
    else:
        raise ValueError(
            f"Unknown terminal assignment strategy '{strategy}'. Use 'balanced' or 'shortest'."
        )

    os.makedirs(os.path.dirname(assignment_path), exist_ok=True)
    with open(assignment_path, 'w') as f:
        json.dump(assignments, f, indent=2)
    print(f"Saved terminal assignments to {assignment_path} ✅")
    return assignments


def select_connections_for_assignments(connections, assignments, electrodes=None):
    """Pick one init path per electrode according to the terminal assignment map."""
    if electrodes is None:
        electrode_names = list(assignments.keys())
    else:
        electrode_names = list(electrodes.keys()) if isinstance(electrodes, dict) else list(electrodes)

    optimized = []
    for name in electrode_names:
        target_terminal = assignments.get(name)
        if target_terminal is None:
            continue
        relevant = [
            conn for conn in connections
            if conn['electrode'] == name and conn['terminal'] == target_terminal
        ]
        if relevant:
            optimized.append(relevant[0])
    return optimized

# --------------------------
# Path Creation Functions
# --------------------------

def project_to_surface(point, mesh):
    """Project a point to the nearest surface point"""
    return np.asarray(mesh.find_closest_point(point))

def create_surface_connection(electrode_name, electrode_pos, terminal_name, terminal_pos, mesh, resolution=50):
    """
    Create a geodesic path along the surface between two points
    Returns dictionary with connection info and path object
    """
    # Project points to surface
    start_surf = project_to_surface(electrode_pos, mesh)
    end_surf = project_to_surface(terminal_pos, mesh)
    
    # Create geodesic path
    geodesic = mesh.geodesic(start_surf, end_surf)
    
    # Sample the path at regular intervals
    if geodesic.n_points > 1:
        points = np.array([geodesic.points[int(i)] 
                         for i in np.linspace(0, geodesic.n_points-1, resolution)])
    else:
        points = np.linspace(start_surf, end_surf, resolution)
    
    return {
        'electrode': electrode_name,
        'terminal': terminal_name,
        'path': pv.lines_from_points(points),
        'path_length': geodesic.length,
        'resolution': resolution
    }

# --------------------------
# Visualization Functions
# --------------------------

def visualize_connections(electrodes, fiducials, connections=None, SUBJECT_ID: int=None):
    """Visualize the head model with electrodes, terminals and connections"""
    if SUBJECT_ID == None:
        raise ValueError("Need to parse `SUBJECT_ID: int`")
    
    pl = pv.Plotter(window_size=(1800, 1800))
    
    # Load and display head mesh
    mesh = pv.read(f"data/cleaned_scans/{SUBJECT_ID}.stl")
    pl.add_mesh(mesh, color="white", opacity=0.88)
    
    # Add electrodes
    for name, pos in electrodes.items():
        pl.add_mesh(pv.Sphere(center=pos, radius=mesh.length*0.008),
                   color='red', name=f"electrode_{name}")
        pl.add_point_labels([pos], [name], font_size=12)
    
    # Add terminals
    terminal_colors = {'TERMINAL_LEFT': 'gray', 'TERMINAL_RIGHT': 'black'}
    for term in ['TERMINAL_LEFT', 'TERMINAL_RIGHT']:
        if term in fiducials:
            pos = fiducials[term]
            pl.add_mesh(pv.Sphere(center=pos, radius=mesh.length*0.01), 
                       color=terminal_colors[term], name=f"terminal_{term}")
            pl.add_point_labels([pos], [term], font_size=12)
    
    # Add connections if provided
    if connections:
        if isinstance(connections[0], dict):  # If using full connection objects
            for conn in connections:
                pl.add_mesh(conn['path'], color='orange', line_width=3,
                           name=f"connection_{conn['electrode']}_{conn['terminal']}")
        else:  # If just path objects
            for i, path in enumerate(connections):
                pl.add_mesh(path, color='orange', line_width=3,
                           name=f"connection_{i}")
    
    pl.show()

# --------------------------
# Main Workflow Functions
# --------------------------

def generate_connections(electrodes, fiducials, mesh, save_to_file=True, SUBJECT_ID: int = None):
    """Generate all electrode-terminal connections"""
    if SUBJECT_ID == None:
        raise ValueError("SUBJECT_ID can not be `None`")
    connections = []
    for terminal_name in ['TERMINAL_LEFT', 'TERMINAL_RIGHT']:
        if terminal_name in fiducials:
            for electrode_name, pos in electrodes.items():
                connections.append(
                    create_surface_connection(
                        electrode_name, pos,
                        terminal_name, fiducials[terminal_name],
                        mesh
                    )
                )
    
    if save_to_file:
        save_connection_data(connections, SUBJECT_ID=SUBJECT_ID)
        print("Generated and saved new connections ✅")
    
    return connections


def createAndSaveInitConnections(SUBJECT_ID: int, electrodes, fiducials):
    mesh = pv.read(f"data/cleaned_scans/{SUBJECT_ID}.stl")
    connections = load_connection_data(SUBJECT_ID=SUBJECT_ID) or generate_connections(
        electrodes, fiducials, mesh, SUBJECT_ID=SUBJECT_ID
    )
    load_or_create_terminal_assignments(SUBJECT_ID, connections)
    print(f"Initial connections now exist in data/json/init_connection_paths_{SUBJECT_ID}.json ✅")
    print("Exiting createAndSaveInitConnections()...")
