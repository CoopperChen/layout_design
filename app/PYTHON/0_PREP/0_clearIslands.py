import os
import pyvista as pv
import numpy as np
from pathlib import Path

SUBJECT_ID = int(os.environ.get("LAYOUT_SUBJECT_ID", "1"))

def remove_islands(mesh, min_size_ratio=0.01, min_absolute_size=100):
    """
    Remove disconnected components smaller than thresholds.
    
    Parameters:
        mesh: Input mesh
        min_size_ratio: Minimum relative size (ratio to largest component)
        min_absolute_size: Minimum absolute face count to keep
    """
    # Get all connected regions
    connected = mesh.connectivity(largest=False)
    regions = connected.cell_data['RegionId']
    unique_labels, counts = np.unique(regions, return_counts=True)
    
    # If only one region, return original mesh
    if len(unique_labels) <= 1:
        return mesh
    
    # Calculate size thresholds
    max_size = counts.max()
    size_threshold = max(max_size * min_size_ratio, min_absolute_size)
    
    # Find labels of regions to keep
    labels_to_keep = unique_labels[counts >= size_threshold]
    
    if len(labels_to_keep) == 0:
        return mesh
    
    # Extract all regions above threshold
    cleaned = connected.threshold(
        [labels_to_keep.min() - 0.5, labels_to_keep.max() + 0.5],
        scalars='RegionId',
        invert=False
    )
    
    return cleaned

def save_as_stl(mesh, filename, binary=True):
    """Properly save mesh as STL file"""
    # Convert to polydata if needed
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface(algorithm='dataset_surface').triangulate()

    # Use PyVista's built-in writer API to avoid relying on private VTK bindings.
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    mesh.save(filename, binary=binary)

def main(SUBJECT_ID: int):
    # Load the STL file
    file_path = f"data/raw/{SUBJECT_ID}.stl"
    print(f"Loading mesh from: {file_path}")
    original_mesh = pv.read(file_path)
    
    # Create before plot
    p_before = pv.Plotter(window_size=(800, 600), title="Before Cleaning")
    p_before.add_mesh(original_mesh, color="lightgray", opacity=1.0)
    p_before.add_text("Original Mesh (Before Cleaning)", position="upper_edge")
    p_before.show_bounds()
    
    # Process the mesh with more aggressive cleaning
    cleaned_mesh = remove_islands(
        original_mesh, 
        min_size_ratio=0.02,  # More aggressive relative threshold (2%)
        min_absolute_size=200  # Minimum 200 faces to keep
    )
    
    # Create after plot
    p_after = pv.Plotter(window_size=(800, 600), title="After Cleaning")
    p_after.add_mesh(cleaned_mesh, color="lightgray", opacity=1.0)
    p_after.add_text("Cleaned Mesh (Removed Small Fragments)", position="upper_edge")
    p_after.show_bounds()
    
    # Show both plots (non-blocking)
    p_before.show(auto_close=False)
    p_after.show()
    
    # Save cleaned mesh if desired
    save = input("Save cleaned mesh? (y/n): ").lower()
    if save == 'y':
        output_path = f"data/cleaned_scans/{SUBJECT_ID}.stl"
        save_as_stl(cleaned_mesh, output_path)
        print(f"Cleaned mesh saved to: {output_path}")

if __name__ == "__main__":
    main(SUBJECT_ID=SUBJECT_ID)