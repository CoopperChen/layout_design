import os
import pyvista as pv
import numpy as np
import json
from pathlib import Path
import mne
import colorsys

SUBJECT_ID = int(os.environ.get("LAYOUT_SUBJECT_ID", "1"))

mesh = pv.read(f"data/cleaned_scans/{SUBJECT_ID}.stl")
fid = json.load(open(f"data/json/fiducials_{SUBJECT_ID}.json"))
Cz = np.array(json.load(open(f"data/json/Cz_{SUBJECT_ID}.json"))["Cz"])

channel_pairs = [
    ["Fp1", "Fp2"],
    ["F7", "F8"],
    ["F3", "F4"],
    ["T7", "T8"],
    ["C3", "C4"],
    ["P7", "P8"],
    ["P3", "P4"],
    ["O1", "O2"],
]
single_channels = ["Fz", "Cz", "Pz"]
channels = [ch for pair in channel_pairs for ch in pair] + single_channels

montage = mne.channels.make_standard_montage("standard_1020")
ch_pos = montage.get_positions()["ch_pos"]

pl = pv.Plotter(window_size=(1800, 1800))

nasion = np.asarray(fid["nasion"], dtype=float)
lpa = np.asarray(fid["lpa"], dtype=float)
rpa = np.asarray(fid["rpa"], dtype=float)
inion = np.asarray(fid["inion"], dtype=float)

size_adjustment = 0.8
electrode_positions = {}


def calculate_coordinate_system():
    ap_axis = nasion - inion
    ap_len = np.linalg.norm(ap_axis)
    if ap_len < 1e-6:
        raise ValueError("Nasion and inion are coincident — re-pick fiducials.")
    ap_axis /= ap_len

    lr_vector = rpa - lpa
    si_axis = np.cross(lr_vector, ap_axis)
    si_len = np.linalg.norm(si_axis)
    if si_len < 1e-6:
        # Nearly collinear fiducials: build SI from mesh / world Z
        fallback = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(fallback, ap_axis)) > 0.9:
            fallback = np.array([0.0, 1.0, 0.0])
        si_axis = np.cross(ap_axis, fallback)
        si_len = np.linalg.norm(si_axis)
        if si_len < 1e-6:
            raise ValueError(
                "Cannot build head coordinate system from fiducials — check nasion/LPA/RPA/inion."
            )
    si_axis /= si_len

    lr_axis = np.cross(ap_axis, si_axis)
    lr_axis /= np.linalg.norm(lr_axis)

    print("\nCoordinate System:")
    print(f"AP (Nasion→Inion): {ap_axis}")
    print(f"LR (Left→Right): {lr_axis}")
    print(f"SI (Superior→Inferior): {si_axis}")

    return ap_axis, lr_axis, si_axis


def calculate_electrode_positions(adjustment):
    ap_axis, lr_axis, si_axis = calculate_coordinate_system()

    def calculate_arc_length(points):
        return np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))

    arc_points = np.vstack([np.linspace(nasion, Cz, 50), np.linspace(Cz, inion, 50)[1:]])
    arc_length = calculate_arc_length(arc_points)

    template_arc = np.linalg.norm(ch_pos["Fpz"] - ch_pos["Cz"]) + np.linalg.norm(
        ch_pos["Cz"] - ch_pos["Oz"]
    )
    scale_factor = (arc_length / template_arc) * adjustment

    positions = {}
    for ch in channels:
        if ch not in ch_pos:
            continue

        templ_pos = np.array(ch_pos[ch]) - np.array(ch_pos["Cz"])
        scaled_pos = templ_pos * scale_factor
        subject_pos = Cz + (
            scaled_pos[0] * lr_axis + scaled_pos[1] * ap_axis + scaled_pos[2] * si_axis
        )
        idx = mesh.find_closest_point(subject_pos)
        positions[ch] = mesh.points[idx].tolist()

    return positions


def _build_color_map():
    n_pairs = len(channel_pairs)
    pair_colors = []
    for i in range(n_pairs):
        rgb = colorsys.hsv_to_rgb(i / n_pairs, 0.8, 0.9)
        pair_colors.append(
            "#%02x%02x%02x" % (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
        )

    n_singles = len(single_channels)
    single_colors = []
    for i in range(n_singles):
        rgb = colorsys.hsv_to_rgb(i / n_singles, 0.6, 0.8)
        single_colors.append(
            "#%02x%02x%02x" % (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
        )

    color_map = {}
    for i, pair in enumerate(channel_pairs):
        for ch in pair:
            color_map[ch] = pair_colors[i]
    for i, ch in enumerate(single_channels):
        color_map[ch] = single_colors[i]
    return color_map, pair_colors, single_colors


COLOR_MAP, PAIR_COLORS, SINGLE_COLORS = _build_color_map()


def _remove_electrode_actors():
    for name in list(pl.actors):
        if name and name in channels:
            pl.remove_actor(name)


def _draw_electrodes():
    radius = mesh.length * 0.008
    for ch in channels:
        if ch not in electrode_positions:
            continue
        pl.add_mesh(
            pv.Sphere(center=electrode_positions[ch], radius=radius),
            color=COLOR_MAP[ch],
            name=ch,
        )


def _update_size_text(adjustment):
    if "size_text" in pl.actors:
        pl.remove_actor("size_text")
    pl.add_text(
        f"EEG 10-20 Placement (Size: {adjustment:.1f}x)",
        position=(0.1, 0.9),
        font_size=16,
        color="black",
        name="size_text",
    )


def update_plot(adjustment):
    global electrode_positions
    _remove_electrode_actors()
    electrode_positions = calculate_electrode_positions(adjustment)
    _draw_electrodes()
    _update_size_text(adjustment)


def _add_static_legend():
    legend_entries = []
    for i, pair in enumerate(channel_pairs):
        legend_entries.append((f"{pair[0]}/{pair[1]}", PAIR_COLORS[i]))
    for i, ch in enumerate(single_channels):
        legend_entries.append((ch, SINGLE_COLORS[i]))
    legend_entries.extend(
        [
            ("Nasion", "red"),
            ("LPA", "green"),
            ("RPA", "blue"),
            ("Inion", "purple"),
            ("Cz", "yellow"),
            ("Terminal Left", "gray"),
            ("Terminal Right", "black"),
        ]
    )
    pl.add_legend(
        legend_entries,
        bcolor="w",
        face="none",
        loc="lower right",
        size=(0.18, 0.32),
        name="legend",
    )


def save_electrode_positions(SUBJECT_ID: int):
    global electrode_positions
    output_path = Path(f"data/json/electrode_positions_{SUBJECT_ID}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(electrode_positions, f, indent=4)
    print("Electrode coordinates saved ✅")


pl.add_mesh(mesh, color="white", opacity=0.88)
for nm, col in zip(["nasion", "lpa", "rpa", "inion"], ["red", "green", "blue", "purple"]):
    pl.add_mesh(
        pv.Sphere(center=np.asarray(fid[nm], dtype=float), radius=mesh.length * 0.01),
        color=col,
        name=nm,
    )
pl.add_mesh(pv.Sphere(center=Cz, radius=mesh.length * 0.01), color="yellow", name="Cz_marker")

if "TERMINAL_LEFT" in fid and "TERMINAL_RIGHT" in fid:
    pl.add_mesh(
        pv.Sphere(center=np.asarray(fid["TERMINAL_LEFT"], dtype=float), radius=mesh.length * 0.01),
        color="gray",
        name="terminal_left",
    )
    pl.add_mesh(
        pv.Sphere(center=np.asarray(fid["TERMINAL_RIGHT"], dtype=float), radius=mesh.length * 0.01),
        color="black",
        name="terminal_right",
    )
    print("Static terminal points displayed (gray=left, black=right)")
else:
    print("Warning: Terminal points not found in fiducials file")


def slider_callback(value):
    global size_adjustment
    size_adjustment = value
    update_plot(size_adjustment)


pl.add_slider_widget(
    slider_callback,
    [0.5, 1.5],
    value=0.8,
    title="Size Adjustment",
    pointa=(0.4, 0.9),
    pointb=(0.9, 0.9),
)


def key_press_callback(SUBJECT_ID: int):
    def callback(iren, event):
        key = iren.GetKeySym()
        if key.lower() == "s":
            save_electrode_positions(SUBJECT_ID=SUBJECT_ID)

    return callback


pl.iren.add_observer("KeyPressEvent", key_press_callback(SUBJECT_ID=SUBJECT_ID))
pl.add_text("Press 'S' to save electrode positions", position="lower_edge", font_size=12)

_add_static_legend()
update_plot(size_adjustment)
pl.show()
