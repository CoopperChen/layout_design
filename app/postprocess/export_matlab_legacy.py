#!/usr/bin/env python3
"""
MATLAB .mat File Exporter for Legacy GCode System
Exports smoothed paths in the exact format expected by gcodeConverter_final14.m

Creates the required .mat files:
- InterconnectElectrodePaths.mat
- HeadMesh.mat  
- Landmarks.mat
- LandmarkNames.mat
"""

import json
import numpy as np
import pyvista as pv
from scipy.spatial import KDTree
from scipy.io import savemat
import os
from pathlib import Path

def load_final_paths(json_filename):
    """Load the final smoothed paths from JSON file"""
    print(f" Loading final paths from: {json_filename}")
    
    with open(json_filename, 'r') as f:
        data = json.load(f)
    
    print(f" Loaded {len(data['final_paths'])} paths")
    return data

def resolve_mesh_file(json_filename, mesh_file_entry):
    """Resolve a mesh file entry from JSON into an existing mesh path."""
    json_path = Path(json_filename).resolve()
    mesh_entry = Path(mesh_file_entry)

    candidate_paths = []
    if mesh_entry.is_absolute():
        candidate_paths.append(mesh_entry)
    else:
        candidate_paths.append(json_path.parent / mesh_entry)

    if not mesh_entry.suffix:
        for extension in ('.stl', '.ply', '.vtk', '.obj'):
            candidate_paths.append(candidate_paths[0].with_suffix(extension))

    mesh_directory = candidate_paths[0].parent
    mesh_stem = mesh_entry.stem
    leading_digits = ''.join(character for character in mesh_stem if character.isdigit())
    if leading_digits:
        for mesh_path in mesh_directory.glob(f'{leading_digits}.*'):
            candidate_paths.append(mesh_path)

    for candidate_path in candidate_paths:
        if candidate_path.exists():
            return str(candidate_path)

    searched_paths = '\n'.join(f'   - {path}' for path in candidate_paths)
    raise FileNotFoundError(
        f"Could not resolve mesh file '{mesh_file_entry}' relative to '{json_path.parent}'.\n"
        f"Searched:\n{searched_paths}"
    )

def calculate_surface_normals(path_3d, mesh):
    """Calculate surface normal vectors for each point in the path"""
    print(f" Calculating surface normals for {len(path_3d)} points...")
    
    kdtree = KDTree(mesh.points)
    
    if not hasattr(mesh, 'point_normals') or mesh.point_normals is None:
        mesh = mesh.compute_normals(point_normals=True, cell_normals=False)
    
    normals = np.zeros_like(path_3d)
    
    for i, point in enumerate(path_3d):
        _, nearest_idx = kdtree.query(point)
        normal = mesh.point_normals[nearest_idx]
        normal = normal / np.linalg.norm(normal)
        normals[i] = normal
    
    return normals

def create_electrode_circles(final_paths_data, mesh, electrode_diameter_mm=13.8, resolution=20):
    """Create circular electrode toolpaths matching legacy format"""
    print(f" Creating electrode circles (diameter: {electrode_diameter_mm:.1f}mm)...")
    
    electrode_toolpaths = []
    radius = electrode_diameter_mm / 2.0
    kdtree = KDTree(mesh.points)
    
    if not hasattr(mesh, 'point_normals') or mesh.point_normals is None:
        mesh = mesh.compute_normals(point_normals=True, cell_normals=False)
    
    # Get unique electrode positions
    electrode_positions = {}
    for path_data in final_paths_data['final_paths']:
        electrode_name = path_data['electrode']
        if electrode_name not in electrode_positions:
            electrode_positions[electrode_name] = np.array(final_paths_data['electrode_positions'][electrode_name])
    
    for electrode_name, electrode_pos in electrode_positions.items():
        print(f"    Creating electrode circle: {electrode_name}")
        
        # Find surface normal at electrode position
        _, nearest_idx = kdtree.query(electrode_pos)
        surface_normal = mesh.point_normals[nearest_idx]
        surface_normal = surface_normal / np.linalg.norm(surface_normal)
        
        # Create orthogonal tangent vectors
        if abs(surface_normal[2]) < 0.9:
            tangent1 = np.cross(surface_normal, [0, 0, 1])
        else:
            tangent1 = np.cross(surface_normal, [1, 0, 0])
        tangent1 = tangent1 / np.linalg.norm(tangent1)
        
        tangent2 = np.cross(surface_normal, tangent1)
        tangent2 = tangent2 / np.linalg.norm(tangent2)
        
        # Create circle points
        circle_data = []
        angles = np.linspace(0, 2*np.pi, resolution, endpoint=False)
        
        for angle in angles:
            circle_point = (electrode_pos + 
                          radius * (np.cos(angle) * tangent1 + np.sin(angle) * tangent2))
            
            # Project to mesh surface
            _, proj_idx = kdtree.query(circle_point)
            projected_point = mesh.points[proj_idx]
            projected_normal = mesh.point_normals[proj_idx]
            projected_normal = projected_normal / np.linalg.norm(projected_normal)
            
            # Format: [x, y, z, nx, ny, nz]
            circle_data.append([
                projected_point[0], projected_point[1], projected_point[2],
                projected_normal[0], projected_normal[1], projected_normal[2]
            ])
        
        electrode_toolpaths.append(np.array(circle_data))
    
    return electrode_toolpaths, list(electrode_positions.keys())

def create_matlab_data_structure(final_paths_data, mesh):
    """
    Create the exact data structure expected by gcodeConverter_final14.m
    
    Expected format:
    InterconnectElectrodePaths = [allinterconnects, electrodesexport, PathNames]
    where:
    - allinterconnects: cell array of [x,y,z,nx,ny,nz] matrices
    - electrodesexport: cell array of [x,y,z,nx,ny,nz] matrices  
    - PathNames: cell array of electrode names
    """
    print(f"  Creating MATLAB data structure...")
    
    # Create interconnect toolpaths (smoothed paths with normals)
    interconnect_toolpaths = []
    path_names = []
    
    for path_data in final_paths_data['final_paths']:
        electrode = path_data['electrode']
        path_3d = np.array(path_data['path_3d'])
        
        print(f"    Processing interconnect: {electrode}")
        
        # Calculate surface normals
        normals = calculate_surface_normals(path_3d, mesh)
        
        # Format as [x, y, z, nx, ny, nz]
        interconnect_data = np.column_stack([path_3d, normals])
        interconnect_toolpaths.append(interconnect_data)
        path_names.append(electrode)
    
    # Create electrode circles
    electrode_toolpaths, electrode_names = create_electrode_circles(final_paths_data, mesh)
    
    # Verify same number of interconnects and electrodes
    if len(interconnect_toolpaths) != len(electrode_toolpaths):
        print(f"  Warning: {len(interconnect_toolpaths)} interconnects vs {len(electrode_toolpaths)} electrodes")
    
    print(f" Created {len(interconnect_toolpaths)} interconnects and {len(electrode_toolpaths)} electrodes")
    
    return interconnect_toolpaths, electrode_toolpaths, path_names

def create_mesh_data(mesh):
    """Create mesh data structure for MATLAB"""
    print(f" Creating mesh data structure...")
    
    # MATLAB triangulation format
    mesh_data = {
        'Points': mesh.points,
        'ConnectivityList': mesh.faces.reshape(-1, 4)[:, 1:4] + 1  # Convert to 1-based indexing
    }
    
    return mesh_data

def create_landmarks_data(final_paths_data):
    """Create landmarks data from terminal positions"""
    print(f" Creating landmarks data...")
    
    landmarks = []
    landmark_names = []
    
    # Use terminal positions as landmarks (required by legacy system)
    for term_name, term_pos in final_paths_data['terminal_positions'].items():
        landmarks.append(term_pos)
        landmark_names.append(term_name)
    
    # Legacy expects at least 3 landmarks, pad if necessary
    while len(landmarks) < 3:
        landmarks.append([0, 0, 0])
        landmark_names.append(f"DUMMY_{len(landmarks)}")
    
    return np.array(landmarks), landmark_names

def export_to_matlab_format(json_filename, output_folder="matlab_export"):
    """
    Export final paths to MATLAB .mat files compatible with gcodeConverter_final14.m
    
    Creates:
    - InterconnectElectrodePaths.mat
    - HeadMesh.mat
    - Landmarks.mat  
    - LandmarkNames.mat
    """
    print(f" Starting MATLAB export...")
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Load data
    final_paths_data = load_final_paths(json_filename)
    mesh_file = resolve_mesh_file(json_filename, final_paths_data['mesh_file'])
    print(f" Loading mesh from: {mesh_file}")
    mesh = pv.read(mesh_file)
    
    # Create MATLAB data structures
    interconnects, electrodes, path_names = create_matlab_data_structure(final_paths_data, mesh)
    mesh_data = create_mesh_data(mesh)
    landmarks, landmark_names = create_landmarks_data(final_paths_data)
    
    # Save InterconnectElectrodePaths.mat
    # Format: [allinterconnects, electrodesexport, PathNames]
    # Create cell arrays that MATLAB can properly read
    interconnect_cell = np.empty((len(interconnects), 1), dtype=object)
    for i, interconnect in enumerate(interconnects):
        interconnect_cell[i, 0] = interconnect
    
    electrode_cell = np.empty((len(electrodes), 1), dtype=object)
    for i, electrode in enumerate(electrodes):
        electrode_cell[i, 0] = electrode
    
    path_names_cell = np.empty((len(path_names), 1), dtype=object)
    for i, name in enumerate(path_names):
        path_names_cell[i, 0] = name
    
    # Create the final structure as expected by MATLAB
    interconnect_electrode_paths = np.empty((3, 1), dtype=object)
    interconnect_electrode_paths[0, 0] = interconnect_cell
    interconnect_electrode_paths[1, 0] = electrode_cell
    interconnect_electrode_paths[2, 0] = path_names_cell
    
    savemat(
        os.path.join(output_folder, 'InterconnectElectrodePaths.mat'),
        {'InterconnectElectrodePaths': interconnect_electrode_paths},
        format='5'
    )
    print("Saved InterconnectElectrodePaths.mat")

    # Save HeadMesh.mat
    savemat(
        os.path.join(output_folder, 'HeadMesh.mat'),
        {'dataref': mesh_data},
        format='5'
    )
    print(" HeadMesh.mat created and saved")
    # Save Landmarks.mat
    savemat(
        os.path.join(output_folder, 'Landmarks.mat'),
        {'Landmarks': landmarks},
        format='5'
    )
    print(f"Saved Landmarks.mat")
    # Save LandmarkNames.mat
    savemat(
        os.path.join(output_folder, 'LandmarkNames.mat'),
        {'LandmarkNames': np.array(landmark_names, dtype=object)},
        format='5'
    )
    print("Saved LandmarkNames.mat")
    # Summary
    print(f" MATLAB export complete!")
    print(f" Output folder: {output_folder}")
    print(f" Files created:")
    print(f"   - InterconnectElectrodePaths.mat ({len(interconnects)} paths)")
    print(f"   - HeadMesh.mat ({len(mesh.points)} vertices, {len(mesh.faces)} faces)")
    print(f"   - Landmarks.mat ({len(landmarks)} landmarks)")
    print(f"   - LandmarkNames.mat")
    print(f"")
    print(f" Ready for gcodeConverter_final14.m!")
    print(f"   1. Copy {output_folder}/ folder to your MATLAB workspace")
    print(f"   2. Set subject='{output_folder}' in gcodeConverter_final14.m")
    print(f"   3. Run the MATLAB script to generate GCode")
    
    return output_folder

if __name__ == "__main__":
    # Configuration
    INPUT_FILE = "NEW_SMOOTH_FINAL_PATHS_S1_0602_run1_99-2.json"
    print(f'CONVERTING COORDINATES FROM INPUT FILE: {INPUT_FILE}')
    OUTPUT_FOLDER = "subject_optimized"  # Folder name for MATLAB
    
    # Check if input file exists
    if not os.path.exists(INPUT_FILE):
        print(f" Input file not found: {INPUT_FILE}")
        print(" Available files:")
        for f in os.listdir('.'):
            if f.startswith('NEW_SMOOTH_FINAL_PATHS') and f.endswith('.json'):
                print(f"   - {f}")
        exit(1)
    
    # Export to MATLAB format
    output_folder = export_to_matlab_format(INPUT_FILE, OUTPUT_FOLDER)
    
    print(f"\n Next steps:")
    print(f"   1. Copy the '{output_folder}' folder to your MATLAB working directory")
    print(f"   2. Open gcodeConverter_final14.m")
    print(f"   3. Change line: subject='{output_folder}';")
    print(f"   4. Run the MATLAB script to generate final GCode")

