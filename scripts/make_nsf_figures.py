"""Render publication-quality panels for the NSF scan-to-print pipeline figures.

Produces standalone PNG panels (transparent background) under
``data/output/figures/`` for manual compositing:

Figure 1 (hero scan-to-print strip, one subject):
  P1 fig1_p1_scan            true-color digitized point cloud
  P2 fig1_p2_landmarks       reconstructed head + fiducials/terminals/electrodes
  P3 fig1_p3_layout2d        collision-free 2D polar routing
  P4 fig1_p4_wires3d         synthesized wires on the scalp
  P5 fig1_p5_bundle          smoothed traces + electrode pads (canonical bundle)
  P6 fig1_p6_simulation      5-axis print simulation (printhead + jetting path)

Figure 2 (generalization): fig2_generalization_s{N}  one preset -> many heads

Usage:
  python scripts/make_nsf_figures.py all            # everything
  python scripts/make_nsf_figures.py fig1           # hero strip only
  python scripts/make_nsf_figures.py fig2           # generalization only
  python scripts/make_nsf_figures.py p4 --subject 5 # a single panel
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app import paths  # noqa: E402
from app.runtime import setup_runtime  # noqa: E402

# --- Shared style ---------------------------------------------------------
HERO_SUBJECT = 4
GENERALIZATION_SUBJECTS = (2, 3, 4, 5)

HEAD_CLAY = "#d9d2c7"
ELECTRODE_RED = "#c0392b"
FIDUCIAL_NAVY = "#1f3a5f"
TERMINAL_GOLD = "#e8a33d"
WINDOW = (2000, 1700)
# Posterior-superior 3/4 view (+Y = back, +Z = up, +X = left).
HEAD_VIEW = (0.75, 0.85, 0.7)


def _fig_dir() -> Path:
    d = paths.DATA_DIR / "output" / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _wire_colors(n: int) -> list[tuple[float, float, float]]:
    import matplotlib

    base = matplotlib.colormaps["tab20"]
    return [tuple(base(i % 20)[:3]) for i in range(n)]


def _add_wire(plotter, pts, color, radius) -> None:
    import pyvista as pv

    pts = np.asarray(pts, dtype=float)
    if len(pts) < 2:
        return
    spline = pv.Spline(pts, max(len(pts), 200))
    plotter.add_mesh(spline.tube(radius=radius), color=color, smooth_shading=True)


def _add_pad(plotter, pad_xyzn, color, *, snap_points=None) -> None:
    """Electrode contact pad as a flat disc, snapped flush onto the scalp."""
    import pyvista as pv

    pad = np.asarray(pad_xyzn, dtype=float)
    center = pad[:, :3].mean(axis=0)
    if pad.shape[1] >= 6:
        normal = pad[:, 3:6].mean(axis=0)
    else:
        normal = center.copy()
    normal = normal / (np.linalg.norm(normal) or 1.0)
    outer = float(np.max(np.linalg.norm(pad[:, :3] - center, axis=1))) or 5.0
    if snap_points is not None:
        pts = np.asarray(snap_points, dtype=float)
        nearest = pts[np.argmin(np.linalg.norm(pts - center, axis=1))]
        center = nearest + normal * (outer * 0.15)
    disc = pv.Disc(center=center, inner=0.0, outer=outer, normal=normal, c_res=48)
    plotter.add_mesh(disc, color=color, smooth_shading=True)


def _new_plotter():
    import pyvista as pv

    pl = pv.Plotter(off_screen=True, window_size=WINDOW)
    pl.set_background("white")
    pl.enable_anti_aliasing("ssaa")
    return pl


def _set_head_camera(plotter, mesh, direction=HEAD_VIEW, zoom=1.35) -> None:
    center = np.asarray(mesh.center, dtype=float)
    span = float(mesh.length) or 200.0
    d = np.asarray(direction, dtype=float)
    d = d / (np.linalg.norm(d) or 1.0)
    plotter.camera_position = [
        tuple(center + d * span * 1.9),
        tuple(center),
        (0.0, 0.0, 1.0),
    ]
    plotter.camera.zoom(zoom)


def _add_labels(plotter, points, labels, *, font_size=16, always_visible=False) -> None:
    import pyvista as pv

    if not labels:
        return
    cloud = pv.PolyData(np.asarray(points, dtype=float))
    plotter.add_point_labels(
        cloud,
        labels,
        font_size=font_size,
        text_color="black",
        show_points=False,
        shape=None,
        always_visible=always_visible,
    )


# --- P1: digitized point cloud -------------------------------------------
def panel_scan(subject: int = HERO_SUBJECT) -> Path:
    import pyvista as pv

    cloud = pv.read(str(paths.raw_point_cloud(subject)))
    pl = _new_plotter()
    if "RGB" in cloud.array_names:
        pl.add_points(
            cloud,
            scalars="RGB",
            rgb=True,
            point_size=6.0,
            render_points_as_spheres=True,
        )
    else:
        pl.add_points(cloud, color=HEAD_CLAY, point_size=6.0, render_points_as_spheres=True)
    pl.enable_eye_dome_lighting()
    _set_head_camera(pl, cloud, direction=(0.45, 1.0, 0.62), zoom=1.2)
    out = _fig_dir() / f"fig1_p1_scan_s{subject}.png"
    pl.screenshot(str(out), transparent_background=True)
    pl.close()
    print(f"P1 scan            -> {out}")
    return out


# --- P2: reconstructed head + landmarks ----------------------------------
def panel_landmarks(subject: int = HERO_SUBJECT) -> Path:
    import pyvista as pv

    setup_runtime()
    from PYTHON.tools.layoutPreset import load_subject_data

    electrodes, fiducials = load_subject_data(subject)
    mesh = pv.read(str(paths.cleaned_scan(subject)))
    pl = _new_plotter()
    pl.add_mesh(mesh, color=HEAD_CLAY, smooth_shading=True, specular=0.2)
    r = mesh.length * 0.006

    for name, pos in electrodes.items():
        if name == "Cz":
            continue
        pl.add_mesh(pv.Sphere(radius=r, center=pos), color=ELECTRODE_RED)
    label_pts, labels = [], []
    for key in ("nasion", "lpa", "rpa", "inion"):
        if key in fiducials:
            pl.add_mesh(pv.Sphere(radius=r * 1.5, center=fiducials[key]), color=FIDUCIAL_NAVY)
            label_pts.append(fiducials[key])
            labels.append(key)
    for term in ("TERMINAL_LEFT", "TERMINAL_RIGHT"):
        if term in fiducials:
            pos = fiducials[term]
            pl.add_mesh(
                pv.Cube(center=pos, x_length=r * 3, y_length=r * 3, z_length=r * 3),
                color=TERMINAL_GOLD,
            )
            label_pts.append(pos)
            labels.append(term.split("_")[-1] + " hub")
    _add_labels(pl, label_pts, labels, font_size=18, always_visible=True)
    _set_head_camera(pl, mesh)
    out = _fig_dir() / f"fig1_p2_landmarks_s{subject}.png"
    pl.screenshot(str(out), transparent_background=True)
    pl.close()
    print(f"P2 landmarks       -> {out}")
    return out


# --- P3: collision-free 2D polar layout ----------------------------------
def panel_layout_2d(subject: int = HERO_SUBJECT, layout_path: str | Path | None = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    setup_runtime()
    import PYTHON.tools.new2dAlterations as new2d

    layout_path = Path(layout_path) if layout_path else paths.layout_json(subject, "synth")
    data = json.loads(layout_path.read_text(encoding="utf-8"))
    electrodes = {
        k: np.asarray(v, dtype=float)
        for k, v in json.loads(paths.electrode_positions_json(subject).read_text()).items()
    }
    cz = electrodes["Cz"]

    def proj(p3d) -> np.ndarray:
        return new2d.polar_projection(np.asarray([p3d], dtype=float), cz)[0]

    term3d = data["metadata"].get("terminal_positions_3d", {})
    term2d = {k: proj(np.asarray(v, dtype=float)) for k, v in term3d.items()}
    wires = data["paths"]
    colors = _wire_colors(len(wires))

    entries = defaultdict(list)
    for p in wires:
        if p.get("entry_point_2d") is not None:
            entries[p["terminal"]].append(np.asarray(p["entry_point_2d"], dtype=float))

    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    for term, c2 in term2d.items():
        pts = entries.get(term, [])
        rad = float(np.median([np.linalg.norm(e - c2) for e in pts])) if pts else 16.0
        ax.add_patch(plt.Circle(c2, rad * 1.18, color="#2ecc71", alpha=0.12, zorder=0))
        ax.plot(c2[0], c2[1], "s", color="black", ms=11, zorder=6)
        ax.annotate(
            term.split("_")[-1],
            c2,
            textcoords="offset points",
            xytext=(0, 14),
            ha="center",
            fontsize=12,
            fontweight="bold",
        )

    for i, p in enumerate(wires):
        pts = np.asarray(p["modified_path_2d"], dtype=float)
        ax.plot(pts[:, 0], pts[:, 1], color=colors[i], lw=2.3, solid_capstyle="round", zorder=3)
        e2 = proj(electrodes[p["electrode"]])
        ax.plot(e2[0], e2[1], "o", color=ELECTRODE_RED, ms=7, zorder=5)
        ax.annotate(
            p["electrode"],
            e2,
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=8,
            zorder=7,
        )

    cm = data.get("collision_metrics", {})
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        f"Subject {subject} \u2014 collision-free routing\n"
        f"{cm.get('crossing_count', 0)} crossings, "
        f"{cm.get('electrode_violations', 0)} electrode violations",
        fontsize=13,
    )
    out = _fig_dir() / f"fig1_p3_layout2d_s{subject}.png"
    fig.savefig(out, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"P3 layout2d        -> {out}")
    return out


# --- P4: synthesized wires on the scalp ----------------------------------
def panel_wires_3d(
    subject: int = HERO_SUBJECT,
    *,
    source: str = "synth",
    show_labels: bool = True,
    out: Path | None = None,
) -> Path:
    import pyvista as pv

    setup_runtime()
    from PYTHON.tools.layoutPreset import load_subject_data

    electrodes, fiducials = load_subject_data(subject)
    mesh = pv.read(str(paths.cleaned_scan(subject)))

    if source == "smooth":
        doc = json.loads(paths.smooth_json(subject).read_text(encoding="utf-8"))
        wires = [np.asarray(fp["path_3d"], dtype=float) for fp in doc["final_paths"]]
    else:
        layout_path = (
            paths.layout_json(subject, "synth") if source == "synth" else Path(source)
        )
        doc = json.loads(layout_path.read_text(encoding="utf-8"))
        wires = [
            np.asarray(p["path_points"], dtype=float)
            for p in doc["paths"]
            if p.get("path_points")
        ]

    pl = _new_plotter()
    pl.add_mesh(mesh, color=HEAD_CLAY, smooth_shading=True, specular=0.2)
    colors = _wire_colors(len(wires))
    wire_r = mesh.length * 0.0018
    for w, c in zip(wires, colors):
        _add_wire(pl, w, c, wire_r)
    r = mesh.length * 0.006
    label_pts, labels = [], []
    for name, pos in electrodes.items():
        if name == "Cz":
            continue
        pl.add_mesh(pv.Sphere(radius=r, center=pos), color=ELECTRODE_RED)
        label_pts.append(pos)
        labels.append(name)
    for term in ("TERMINAL_LEFT", "TERMINAL_RIGHT"):
        if term in fiducials:
            pl.add_mesh(pv.Sphere(radius=r * 1.7, center=fiducials[term]), color="#34495e")
    if show_labels:
        _add_labels(pl, label_pts, labels, font_size=14, always_visible=True)
    _set_head_camera(pl, mesh)
    out = out or _fig_dir() / f"fig1_p4_wires3d_s{subject}.png"
    pl.screenshot(str(out), transparent_background=True)
    pl.close()
    print(f"P4 wires3d         -> {out}")
    return out


# --- P5: smoothed traces + electrode pads (canonical bundle) -------------
def panel_bundle(subject: int = HERO_SUBJECT) -> Path:
    import pyvista as pv

    setup_runtime()
    from app.postprocess.bundle.load import load_bundle

    bundle = load_bundle(paths.bundle_export_dir(subject))
    faces = np.hstack(
        [np.full((len(bundle.mesh_faces), 1), 3), bundle.mesh_faces]
    ).ravel()
    mesh = pv.PolyData(bundle.mesh_points, faces)

    pl = _new_plotter()
    pl.add_mesh(mesh, color=HEAD_CLAY, smooth_shading=True, specular=0.2)
    colors = _wire_colors(len(bundle.channels))
    wire_r = mesh.length * 0.0018
    for ch, c in zip(bundle.channels, colors):
        _add_wire(pl, np.asarray(ch.interconnect, dtype=float)[:, :3], c, wire_r)
        _add_pad(pl, np.asarray(ch.electrode, dtype=float), c, snap_points=bundle.mesh_points)
    _set_head_camera(mesh=mesh, plotter=pl)
    out = _fig_dir() / f"fig1_p5_bundle_s{subject}.png"
    pl.screenshot(str(out), transparent_background=True)
    pl.close()
    print(f"P5 bundle          -> {out}")
    return out


# --- P6: 5-axis print simulation -----------------------------------------
def panel_simulation(subject: int = HERO_SUBJECT) -> Path:
    setup_runtime()
    from app.simulator.cli import simulate_gcode

    gcode = paths.gcode_output_dir(subject) / "allinterconnects.txt"
    bundle = paths.bundle_export_dir(subject)
    out = _fig_dir() / f"fig1_p6_simulation_s{subject}.png"
    simulate_gcode(
        gcode,
        bundle,
        layers={"mesh", "tip", "arm", "landmarks"},
        screenshot=out,
        title="",
    )
    print(f"P6 simulation      -> {out}")
    return out


# --- Figure 2: generalization --------------------------------------------
def figure2_generalization(subjects=GENERALIZATION_SUBJECTS) -> list[Path]:
    outs = []
    for sid in subjects:
        out = _fig_dir() / f"fig2_generalization_s{sid}.png"
        panel_wires_3d(sid, source="synth", show_labels=False, out=out)
        outs.append(out)
    return outs


def build_fig1(subject: int = HERO_SUBJECT) -> None:
    panel_scan(subject)
    panel_landmarks(subject)
    panel_layout_2d(subject)
    panel_wires_3d(subject)
    panel_bundle(subject)
    panel_simulation(subject)


PANELS = {
    "p1": lambda a: panel_scan(a.subject),
    "p2": lambda a: panel_landmarks(a.subject),
    "p3": lambda a: panel_layout_2d(a.subject),
    "p4": lambda a: panel_wires_3d(a.subject),
    "p5": lambda a: panel_bundle(a.subject),
    "p6": lambda a: panel_simulation(a.subject),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target",
        choices=["all", "fig1", "fig2", *PANELS.keys()],
        help="What to render",
    )
    parser.add_argument("--subject", type=int, default=HERO_SUBJECT)
    args = parser.parse_args()

    if args.target == "all":
        build_fig1(args.subject)
        figure2_generalization()
    elif args.target == "fig1":
        build_fig1(args.subject)
    elif args.target == "fig2":
        figure2_generalization()
    else:
        PANELS[args.target](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
