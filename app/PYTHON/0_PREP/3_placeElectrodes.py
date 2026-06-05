import os
import pyvista as pv
import numpy as np
import json
from pathlib import Path
import mne
import colorsys

SUBJECT_ID = int(os.environ.get("LAYOUT_SUBJECT_ID", "1"))

# Load data
# mesh = pv.read("../../[Pablo] Scan Samples/Mannequin Head/Mannequin_0110.stl")
mesh = pv.read(f"data/cleaned_scans/{SUBJECT_ID}.stl")
fid = json.load(open(f"data/json/fiducials_{SUBJECT_ID}.json"))  # nasion, lpa, rpa, inion, TERMINAL_RIGHT, TERMINAL_RIGHT
Cz = np.array(json.load(open(f"data/json/Cz_{SUBJECT_ID}.json"))['Cz'])

# Standard 10-20 channels with paired coloring
channel_pairs = [
    ["Fp1", "Fp2"], ["F7", "F8"], ["F3", "F4"],
    ["T7", "T8"], ["C3", "C4"], ["P7", "P8"],
    ["P3", "P4"], ["O1", "O2"]
]
single_channels = ["Fz", "Cz", "Pz"]
channels = [ch for pair in channel_pairs for ch in pair] + single_channels

# Get template positions
montage = mne.channels.make_standard_montage('standard_1020')
ch_pos = montage.get_positions()['ch_pos']

# Initialize plotter
pl = pv.Plotter(window_size=(1800, 1800))

# Store fiducial points
nasion = np.array(fid['nasion'])
lpa = np.array(fid['lpa'])
rpa = np.array(fid['rpa'])
inion = np.array(fid['inion'])

# Default size adjustment
size_adjustment = 0.8
electrode_positions = {}

def calculate_coordinate_system():
    # Calculate anatomical axes
    ap_axis = nasion - inion  # Anterior-posterior (always nasion→inion)
    ap_axis /= np.linalg.norm(ap_axis)
    
    # Calculate the left-right vector (from LPA to RPA)
    lr_vector = rpa - lpa
    
    # Create orthogonal coordinate system
    si_axis = np.cross(lr_vector, ap_axis)  # Superior-inferior
    si_axis /= np.linalg.norm(si_axis)
    
    # Recalculate proper left-right axis
    lr_axis = np.cross(ap_axis, si_axis)
    lr_axis /= np.linalg.norm(lr_axis)
    
    # Debug output
    print("\nCoordinate System:")
    print(f"AP (Nasion→Inion): {ap_axis}")
    print(f"LR (Left→Right): {lr_axis}") 
    print(f"SI (Superior→Inferior): {si_axis}")
    
    return ap_axis, lr_axis, si_axis

def calculate_electrode_positions(adjustment):
    ap_axis, lr_axis, si_axis = calculate_coordinate_system()
    
    def calculate_arc_length(points):
        return np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
    
    arc_points = np.vstack([
        np.linspace(nasion, Cz, 50),
        np.linspace(Cz, inion, 50)[1:]
    ])
    arc_length = calculate_arc_length(arc_points)
    
    template_arc = (np.linalg.norm(ch_pos['Fpz'] - ch_pos['Cz']) + 
                   np.linalg.norm(ch_pos['Cz'] - ch_pos['Oz']))
    scale_factor = (arc_length / template_arc) * adjustment
    
    # Project electrodes
    positions = {}
    for ch in channels:
        if ch not in ch_pos:
            continue
            
        templ_pos = np.array(ch_pos[ch]) - np.array(ch_pos['Cz'])
        scaled_pos = templ_pos * scale_factor
        subject_pos = Cz + (scaled_pos[0] * lr_axis + 
                          scaled_pos[1] * ap_axis + 
                          scaled_pos[2] * si_axis)
        idx = mesh.find_closest_point(subject_pos)
        positions[ch] = mesh.points[idx].tolist()  # Convert numpy array to list
        
    return positions

def update_plot(adjustment):
    global electrode_positions
    # Get list of actors to remove first
    to_remove = [name for name, actor in pl.actors.items() 
                if name and name in channels + ['legend', 'size_text']]
    
    # Now remove them
    for name in to_remove:
        pl.remove_actor(name)
    
    # Get new positions
    electrode_positions = calculate_electrode_positions(adjustment)
    
    # Create color mapping
    n_pairs = len(channel_pairs)
    pair_colors = []
    for i in range(n_pairs):
        rgb = colorsys.hsv_to_rgb(i/n_pairs, 0.8, 0.9)
        hex_color = '#%02x%02x%02x' % (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        pair_colors.append(hex_color)

    n_singles = len(single_channels)
    single_colors = []
    for i in range(n_singles):
        rgb = colorsys.hsv_to_rgb(i/n_singles, 0.6, 0.8)
        hex_color = '#%02x%02x%02x' % (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        single_colors.append(hex_color)

    color_map = {}
    for i, pair in enumerate(channel_pairs):
        for ch in pair:
            color_map[ch] = pair_colors[i]
    for i, ch in enumerate(single_channels):
        color_map[ch] = single_colors[i]
    
    # Add electrodes with colors but without labels
    for ch in channels:
        if ch not in electrode_positions:
            continue
        pos = electrode_positions[ch]
        pl.add_mesh(pv.Sphere(center=pos, radius=mesh.length*0.008),
                   color=color_map[ch], name=ch)
    
    # Add legend - MODIFIED TO INCLUDE FIDUCIALS AND TERMINAL POINTS
    legend_entries = []
    # Add EEG channels first
    for i, pair in enumerate(channel_pairs):
        legend_entries.append((f"{pair[0]}/{pair[1]}", pair_colors[i]))
    for i, ch in enumerate(single_channels):
        legend_entries.append((ch, single_colors[i]))
    
    # Add fiducials and other points
    legend_entries.extend([
        ('Nasion', 'red'),
        ('LPA', 'green'),
        ('RPA', 'blue'),
        ('Inion', 'purple'),
        ('Cz', 'yellow'),
        ('Terminal Left', 'gray'),
        ('Terminal Right', 'black')
    ])

    pl.add_legend(legend_entries, bcolor="w", name='legend')
    pl.add_text(f"EEG 10-20 Placement (Size: {adjustment:.1f}x)", 
               position=(0.1, 0.9), font_size=16, color='black', name='size_text')

def save_electrode_positions(SUBJECT_ID: int):
    global electrode_positions
    output_path = Path(f'data/json/electrode_positions_{SUBJECT_ID}.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w') as f:
        json.dump(electrode_positions, f, indent=4)
    print("Electrode coordinates saved ✅")

# Initial setup
pl.add_mesh(mesh, color="white", opacity=0.88)
for nm, col in zip(['nasion','lpa','rpa','inion'],
                  ['red','green','blue','purple']):
    pl.add_mesh(pv.Sphere(center=fid[nm], radius=mesh.length*0.01), color=col, name=nm)
pl.add_mesh(pv.Sphere(center=Cz, radius=mesh.length*0.01), color="yellow", name='Cz_marker')


# Add static terminal points (unaffected by scaling)
if 'TERMINAL_LEFT' in fid and 'TERMINAL_RIGHT' in fid:
    pl.add_mesh(pv.Sphere(center=fid['TERMINAL_LEFT'], radius=mesh.length*0.01), 
               color="gray", name='terminal_left')
    pl.add_mesh(pv.Sphere(center=fid['TERMINAL_RIGHT'], radius=mesh.length*0.01),
               color="black", name='terminal_right')
    print("Static terminal points displayed (gray=left, black=right)")
else:
    print("Warning: Terminal points not found in fiducials file")


# Add slider callback
def slider_callback(value):
    global size_adjustment
    size_adjustment = value
    update_plot(size_adjustment)

pl.add_slider_widget(
    slider_callback,
    [0.5, 1.5],
    value=0.8,
    title='Size Adjustment',
    pointa=(0.4, 0.9),
    pointb=(0.9, 0.9),
    style='modern'
)

# Add save functionality using observer pattern
def key_press_callback(SUBJECT_ID: int):
    def callback(iren, event):
        key = iren.GetKeySym()
        if key.lower() == 's':
            save_electrode_positions(SUBJECT_ID=SUBJECT_ID)
    return callback

pl.iren.add_observer('KeyPressEvent', key_press_callback(SUBJECT_ID=SUBJECT_ID))
pl.add_text("Press 'S' to save electrode positions", position='lower_edge', font_size=12)

# Initial plot
update_plot(size_adjustment)

# Show the plotter
pl.show()