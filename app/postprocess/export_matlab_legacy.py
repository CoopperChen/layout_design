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
import os
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.io import savemat

from app import paths
from app.postprocess.bundle.schema import ELECTRODE_DIAMETER_MM, ELECTRODE_NLINES
from app.postprocess.electrode_finder import (
    build_electrode_disk_zigzag,
    export_electrode_xyzn,
)
from app.postprocess.mesh_export import (
    MeshExportContext,
    load_mesh_context,
    prepare_mesh_export_context,
    xyzn_from_path,
)


def load_final_paths(json_filename, *, verbose: bool = True):
    """Load the final smoothed paths from JSON file."""
    if verbose:
        print(f" Loading final paths from: {json_filename}")

    with open(json_filename, encoding="utf-8") as f:
        data = json.load(f)

    if verbose:
        print(f" Loaded {len(data['final_paths'])} paths")
    return data

def resolve_mesh_file(json_filename, mesh_file_entry):
    """Resolve a mesh file entry from JSON into an existing mesh path."""
    from app import paths as repo_paths

    json_path = Path(json_filename).resolve()
    mesh_entry = Path(mesh_file_entry)

    candidate_paths: list[Path] = []
    if mesh_entry.is_absolute():
        candidate_paths.append(mesh_entry)
    else:
        # smooth JSON stores mesh_file repo-relative (data/cleaned_scans/{id}.stl)
        candidate_paths.append(repo_paths.REPO_ROOT / mesh_entry)
        candidate_paths.append(json_path.parent / mesh_entry)

    if not mesh_entry.suffix:
        for extension in (".stl", ".ply", ".vtk", ".obj"):
            for base in list(candidate_paths):
                candidate_paths.append(base.with_suffix(extension))

    mesh_directories = {p.parent for p in candidate_paths}
    mesh_stem = mesh_entry.stem
    leading_digits = "".join(character for character in mesh_stem if character.isdigit())
    if leading_digits:
        for mesh_directory in mesh_directories:
            for mesh_path in mesh_directory.glob(f"{leading_digits}.*"):
                candidate_paths.append(mesh_path)

    seen: set[Path] = set()
    for candidate_path in candidate_paths:
        resolved = candidate_path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return str(resolved)

    searched_paths = "\n".join(f"   - {path}" for path in candidate_paths)
    raise FileNotFoundError(
        f"Could not resolve mesh file '{mesh_file_entry}' "
        f"(JSON: {json_path}).\n"
        f"Searched:\n{searched_paths}"
    )

def create_electrode_toolpaths(
    interconnect_toolpaths: list[np.ndarray],
    path_names: list[str],
    ctx: MeshExportContext,
    *,
    electrode_diameter_mm: float = ELECTRODE_DIAMETER_MM,
    nlines: int = ELECTRODE_NLINES,
    gap_size_mm: float | None = None,
    verbose: bool = True,
) -> tuple[list[np.ndarray], list[str], float]:
    """
    Planar electrode disk at ``surface + gap_size_mm`` along outward normal.

    Interconnect row 0 locates the pad site; perimeter zigzag stays coplanar.
    """
    if gap_size_mm is None:
        from app.postprocess.gcode.config_loader import load_machine_config

        gap_size_mm = load_machine_config(paths.postprocessor_machine_config()).gap_size_mm

    if verbose:
        print(
            f" Creating electrode disk zigzag "
            f"(diameter: {electrode_diameter_mm:.2f}mm, nlines={nlines}, "
            f"gap={gap_size_mm:.2f}mm)..."
        )

    electrode_toolpaths = []
    for name, interconnect in zip(path_names, interconnect_toolpaths):
        xyz, _origin, _surface, plane_normal = build_electrode_disk_zigzag(
            interconnect,
            ctx.mesh,
            ctx,
            electrode_diameter_mm,
            gap_size_mm,
            nlines=nlines,
        )
        if verbose:
            print(f"    Electrode: {name} ({len(xyz)} points)")
        electrode_toolpaths.append(export_electrode_xyzn(xyz, plane_normal))
    return electrode_toolpaths, path_names, float(gap_size_mm)


def create_matlab_data_structure(
    final_paths_data,
    mesh_or_ctx: pv.PolyData | MeshExportContext,
    *,
    verbose: bool = True,
):
    """
    Create the exact data structure expected by gcodeConverter_final14.m
    
    Expected format:
    InterconnectElectrodePaths = [allinterconnects, electrodesexport, PathNames]
    where:
    - allinterconnects: cell array of [x,y,z,nx,ny,nz] matrices
    - electrodesexport: cell array of [x,y,z,nx,ny,nz] matrices  
    - PathNames: cell array of electrode names
    """
    if verbose:
        print("  Creating MATLAB data structure...")

    ctx = (
        mesh_or_ctx
        if isinstance(mesh_or_ctx, MeshExportContext)
        else prepare_mesh_export_context(mesh_or_ctx)
    )

    interconnect_toolpaths = []
    path_names = []

    for path_data in final_paths_data["final_paths"]:
        electrode = path_data["electrode"]
        path_3d = np.asarray(path_data["path_3d"], dtype=np.float64)
        if verbose:
            print(f"    Processing interconnect: {electrode}")
        interconnect_toolpaths.append(xyzn_from_path(ctx, path_3d))
        path_names.append(electrode)

    electrode_toolpaths, electrode_names, _gap = create_electrode_toolpaths(
        interconnect_toolpaths,
        path_names,
        ctx,
        verbose=verbose,
    )
    
    # Verify same number of interconnects and electrodes
    if verbose and len(interconnect_toolpaths) != len(electrode_toolpaths):
        print(
            f"  Warning: {len(interconnect_toolpaths)} interconnects vs "
            f"{len(electrode_toolpaths)} electrodes"
        )

    if verbose:
        print(
            f" Created {len(interconnect_toolpaths)} interconnects and "
            f"{len(electrode_toolpaths)} electrodes"
        )

    return interconnect_toolpaths, electrode_toolpaths, path_names


def create_mesh_data(mesh, *, verbose: bool = True):
    """Create mesh data structure for MATLAB."""
    if verbose:
        print(" Creating mesh data structure...")
    
    # MATLAB triangulation format
    mesh_data = {
        'Points': mesh.points,
        'ConnectivityList': mesh.faces.reshape(-1, 4)[:, 1:4] + 1  # Convert to 1-based indexing
    }
    
    return mesh_data

def create_landmarks_data(
    final_paths_data,
    subject_id: int | None = None,
    *,
    strict_landmarks: bool = True,
    verbose: bool = True,
):
    """Create landmarks for legacy gcode (prefer preprocess calibration landmarks)."""
    if verbose:
        print(" Creating landmarks data...")

    if subject_id is not None:
        from app.postprocess.bundle.emit import (
            CalibrationLandmarksMissingError,
            require_calibration_landmarks,
        )

        if strict_landmarks:
            landmarks, landmark_names = require_calibration_landmarks(subject_id)
            if verbose:
                print(f" Using calibration landmarks from fiducials_{subject_id}.json")
            return landmarks, landmark_names

        try:
            from app.preprocess.fiducials_io import load_picks, matlab_landmarks_from_picks

            parsed = matlab_landmarks_from_picks(load_picks(subject_id))
            if parsed is not None:
                landmarks, landmark_names = parsed
                if verbose:
                    print(f" Using calibration landmarks from fiducials_{subject_id}.json")
                return landmarks, landmark_names
        except CalibrationLandmarksMissingError:
            pass
        except Exception:
            pass

    if strict_landmarks:
        raise CalibrationLandmarksMissingError(
            "Calibration landmarks required for export; use --allow-terminal-landmarks to override"
        )

    landmarks = []
    landmark_names = []
    for term_name, term_pos in final_paths_data["terminal_positions"].items():
        landmarks.append(term_pos)
        landmark_names.append(term_name)
    while len(landmarks) < 3:
        landmarks.append([0, 0, 0])
        landmark_names.append(f"DUMMY_{len(landmarks)}")
    return np.array(landmarks), landmark_names


def export_to_matlab_format(
    json_filename,
    output_folder="matlab_export",
    *,
    strict_landmarks: bool = True,
    skip_validation: bool = False,
    verbose: bool = True,
):
    """
    Export final paths to MATLAB .mat files compatible with gcodeConverter_final14.m
    
    Creates:
    - InterconnectElectrodePaths.mat
    - HeadMesh.mat
    - Landmarks.mat  
    - LandmarkNames.mat
    """
    if verbose:
        print(" Starting MATLAB export...")
    os.makedirs(output_folder, exist_ok=True)

    json_path = Path(json_filename)
    final_paths_data = load_final_paths(json_filename, verbose=verbose)
    if not skip_validation:
        from app.postprocess.validate_export import validate_smooth_for_export

        validate_smooth_for_export(
            final_paths_data,
            smooth_path=json_path.resolve(),
            require_collision_free=True,
        )

    mesh_file = resolve_mesh_file(json_filename, final_paths_data["mesh_file"])
    if verbose:
        print(f" Loading mesh from: {mesh_file}")
    ctx = load_mesh_context(mesh_file)

    interconnects, electrodes, path_names = create_matlab_data_structure(
        final_paths_data, ctx, verbose=verbose
    )
    mesh_data = create_mesh_data(ctx.mesh, verbose=verbose)
    mesh_stem = Path(mesh_file).stem
    subject_digits = "".join(c for c in mesh_stem if c.isdigit())
    subject_id = int(subject_digits) if subject_digits else None
    landmarks, landmark_names = create_landmarks_data(
        final_paths_data,
        subject_id=subject_id,
        strict_landmarks=strict_landmarks,
        verbose=verbose,
    )

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
    if verbose:
        print("Saved InterconnectElectrodePaths.mat")

    # Save HeadMesh.mat
    savemat(
        os.path.join(output_folder, 'HeadMesh.mat'),
        {'dataref': mesh_data},
        format='5'
    )
    if verbose:
        print(" HeadMesh.mat created and saved")
    # Save Landmarks.mat
    savemat(
        os.path.join(output_folder, 'Landmarks.mat'),
        {'Landmarks': landmarks},
        format='5'
    )
    if verbose:
        print("Saved Landmarks.mat")
    # Save LandmarkNames.mat
    savemat(
        os.path.join(output_folder, 'LandmarkNames.mat'),
        {'LandmarkNames': np.array(landmark_names, dtype=object)},
        format='5'
    )
    if verbose:
        print("Saved LandmarkNames.mat")
        print(" MATLAB export complete!")
        print(f" Output folder: {output_folder}")
        print(f"   - InterconnectElectrodePaths.mat ({len(interconnects)} paths)")
        print(f"   - HeadMesh.mat ({len(ctx.mesh.points)} vertices)")
        print(f"   - Landmarks.mat ({len(landmarks)} landmarks)")
        print("   - LandmarkNames.mat")

    return output_folder
