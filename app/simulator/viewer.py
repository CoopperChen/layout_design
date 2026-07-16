"""PyVista 3D viewer — printhead arm and nozzle tip in machine space."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

LayerName = str

LEGEND_ENTRIES: list[tuple[str, str]] = [
    ("Head mesh (optional context)", "lightgray"),
    ("Landmark: left (registered)", "green"),
    ("Landmark: back (registered)", "blue"),
    ("Central landmark (machine frame)", "red"),
    ("Machine +X / +Y / +Z at C pivot (machine zero)", "crimson"),
    ("Machine-zero tip (0, −a, −d)", "dimgray"),
    ("Machine-zero C pivot X0 Y0 Z0", "black"),
    ("C-axis pivot (G-code X,Y,Z)", "dimgray"),
    ("Programmed G-code X,Y,Z", "silver"),
    ("Nozzle tip — rigid FK (M10 jetting)", "lime"),
    ("Nozzle tip — travel", "tomato"),
    ("Arm C → B (length a)", "orange"),
    ("Tool B → tip (length d, B rotates about arm)", "darkorange"),
    ("Nozzle tip at current step", "gold"),
]

LAYER_KEYS: dict[str, LayerName] = {
    "m": "mesh",
    "l": "landmarks",
    "o": "origin",
    "c": "cnc",
    "g": "programmed",
    "t": "tip",
    "a": "arm",
}


@dataclass
class SimulationScene:
    mesh_points: np.ndarray
    mesh_faces: np.ndarray
    landmarks: np.ndarray
    landmark_names: list[str]
    cnc_path: np.ndarray
    programmed_path: np.ndarray | None
    b_pivot_path: np.ndarray
    b_angles: np.ndarray
    c_angles: np.ndarray
    tip_path: np.ndarray
    markers: np.ndarray
    a_mm: float
    d_mm: float
    calgap_z_mm: float = 26.62
    b0_deg: float = 0.0
    c0_deg: float = 90.0
    layers: set[LayerName] = field(
        default_factory=lambda: {"mesh", "landmarks", "origin", "tip", "arm"}
    )


def _dedupe_consecutive_points(points: np.ndarray, *, min_dist: float = 1e-6) -> np.ndarray:
    """Drop consecutive duplicates so VTK line/tube filters stay valid."""
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        return pts.reshape(0, 3)
    out = [pts[0]]
    for p in pts[1:]:
        if np.linalg.norm(p - out[-1]) >= min_dist:
            out.append(p)
    return np.vstack(out)


def _dedupe_path_with_markers(
    points: np.ndarray,
    markers: np.ndarray,
    *,
    min_dist: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop duplicate path points and keep markers aligned."""
    pts = np.asarray(points, dtype=float)
    mk = np.asarray(markers, dtype=float)
    if len(pts) == 0:
        return pts.reshape(0, 3), mk.reshape(0)
    if len(mk) != len(pts):
        raise ValueError("markers must match path length")

    keep_pts = [pts[0]]
    keep_mk = [float(mk[0])]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - keep_pts[-1]) >= min_dist:
            keep_pts.append(pts[i])
            keep_mk.append(float(mk[i]))
        elif float(mk[i]) in (10.0, 11.0):
            keep_mk[-1] = float(mk[i])
    return np.vstack(keep_pts), np.asarray(keep_mk, dtype=float)


def _polyline(points: np.ndarray, *, min_dist: float = 1e-6):
    import pyvista as pv

    pts = _dedupe_consecutive_points(points, min_dist=min_dist)
    if len(pts) < 2:
        return None
    n = len(pts)
    lines = np.column_stack(
        [np.full(n - 1, 2), np.arange(n - 1), np.arange(1, n)]
    ).ravel()
    return pv.PolyData(pts, lines=lines)


def _add_line_mesh(plotter, polyline, *, color: str, line_width: float) -> object | None:
    """Add a polyline without tube extrusion (avoids vtkPlaneSource errors)."""
    if polyline is None:
        return None
    return plotter.add_mesh(
        polyline,
        color=color,
        line_width=line_width,
        render_lines_as_tubes=False,
    )


def _grouped_colored_segments(
    path: np.ndarray,
    segment_colors: list[str],
) -> list[tuple[np.ndarray, str]]:
    if len(path) < 2 or not segment_colors:
        return []

    segments: list[tuple[np.ndarray, str]] = []
    start = 0
    current = segment_colors[0]
    for i in range(1, len(segment_colors)):
        if segment_colors[i] != current:
            segments.append((path[start : i + 1], current))
            start = i
            current = segment_colors[i]
    segments.append((path[start:], current))
    return segments


def _segment_jet_on(markers: np.ndarray) -> list[bool]:
    """Jet active on motion from markers[i] to markers[i + 1]."""
    if len(markers) < 2:
        return []
    printing = False
    active: list[bool] = []
    for i in range(len(markers) - 1):
        m = float(markers[i])
        if m == 11:
            printing = False
        elif m == 10:
            printing = True
        active.append(printing)
    return active


def _tip_segment_colors(markers: np.ndarray) -> list[str]:
    return ["lime" if on else "tomato" for on in _segment_jet_on(markers)]


def _add_landmarks(
    plotter,
    scene: SimulationScene,
    *,
    scale: float,
    colors: tuple[str, ...],
) -> list:
    import pyvista as pv

    actors: list = []
    for i, (_, pos) in enumerate(zip(scene.landmark_names, scene.landmarks)):
        color = colors[i % len(colors)]
        actors.append(
            plotter.add_mesh(pv.Sphere(radius=scale, center=pos), color=color)
        )
    lm_cloud = pv.PolyData(np.asarray(scene.landmarks, dtype=float))
    actors.append(
        plotter.add_point_labels(
            lm_cloud,
            scene.landmark_names,
            font_size=12,
            point_color="black",
            show_points=False,
            shape=None,
        )
    )
    return actors


def _add_machine_origin(plotter, scene: SimulationScene, *, scale: float) -> list:
    """Machine-zero reference markers (same frame as mesh and G-code)."""
    import pyvista as pv

    from app.postprocess.gcode.kinematics.machine_fk import machine_zero_head_frame

    actors: list = []
    central, tip_mz, c_mz, _b_mz = machine_zero_head_frame(
        a_mm=scene.a_mm,
        d_mm=scene.d_mm,
        b0_deg=scene.b0_deg,
        c0_deg=scene.c0_deg,
        calgap_z_mm=scene.calgap_z_mm,
    )
    axis_len = scale * 2.5

    for direction, color in (
        ([1.0, 0.0, 0.0], "crimson"),
        ([0.0, 1.0, 0.0], "forestgreen"),
        ([0.0, 0.0, 1.0], "royalblue"),
    ):
        end = c_mz + axis_len * np.asarray(direction, dtype=float)
        line = _polyline(np.vstack([c_mz, end]))
        actor = _add_line_mesh(plotter, line, color=color, line_width=4)
        if actor is not None:
            actors.append(actor)

    actors.append(
        plotter.add_mesh(
            pv.Sphere(radius=scale * 0.35, center=tip_mz),
            color="dimgray",
        )
    )
    actors.append(
        plotter.add_mesh(
            pv.Sphere(radius=scale * 0.4, center=c_mz),
            color="black",
        )
    )

    label_pt = pv.PolyData(np.vstack([central, tip_mz, c_mz]))
    actors.append(
        plotter.add_point_labels(
            label_pt,
            [
                "central landmark",
                "tip @ machine zero",
                "C pivot X0 Y0 Z0",
            ],
            font_size=11,
            point_color="black",
            show_points=False,
            shape=None,
        )
    )
    return actors


def _add_arm_skeleton(
    plotter,
    scene: SimulationScene,
    idx: int,
    *,
    scale: float,
) -> list:
    import pyvista as pv

    actors: list = []
    c_center = np.asarray(scene.cnc_path[idx], dtype=float)
    b_pivot = np.asarray(scene.b_pivot_path[idx], dtype=float)
    tip = np.asarray(scene.tip_path[idx], dtype=float)

    arm_line = _polyline(np.vstack([c_center, b_pivot]))
    tool_line = _polyline(np.vstack([b_pivot, tip]))
    arm_actor = _add_line_mesh(plotter, arm_line, color="orange", line_width=8)
    if arm_actor is not None:
        actors.append(arm_actor)
    tool_actor = _add_line_mesh(plotter, tool_line, color="darkorange", line_width=6)
    if tool_actor is not None:
        actors.append(tool_actor)
    for pt, color, label in (
        (c_center, "dimgray", "C"),
        (b_pivot, "orange", "B"),
        (tip, "gold", "tip"),
    ):
        actors.append(
            plotter.add_mesh(pv.Sphere(radius=scale * 0.5, center=pt), color=color)
        )
    label_pts = np.vstack([c_center, b_pivot, tip])
    if len(_dedupe_consecutive_points(label_pts)) >= 2:
        actors.append(
            plotter.add_point_labels(
                label_pts,
                ["C", "B", "tip"],
                font_size=11,
                point_color="black",
                show_points=False,
                shape=None,
            )
        )
    return actors


def _marker_label(marker: float) -> str:
    m = int(marker)
    if m == 10:
        return "M10 jet ON"
    if m == 11:
        return "M11 jet OFF"
    return "—"


def render_simulation_screenshot(
    scene: SimulationScene,
    out_path: str,
    *,
    title: str = "G-code toolpath simulation",
    window_size: tuple[int, int] = (1800, 1500),
    transparent_background: bool = True,
    camera_direction: tuple[float, float, float] = (0.9, -0.5, 0.55),
    show_legend: bool = False,
    jetting_only: bool = True,
) -> str:
    """Off-screen still of the finished toolpath (full path + arm at last step).

    Reuses the module-level draw helpers so the framing matches the interactive
    viewer, but renders the whole path at once for a publication panel.
    """
    import pyvista as pv

    plotter = pv.Plotter(window_size=window_size, off_screen=True)
    plotter.set_background("white")

    faces = np.hstack(
        [np.full((len(scene.mesh_faces), 1), 3), scene.mesh_faces]
    ).ravel()
    mesh = pv.PolyData(scene.mesh_points, faces)
    scale = mesh.length * 0.012 if mesh.length > 0 else 5.0
    last = max(0, len(scene.tip_path) - 1)

    if "mesh" in scene.layers:
        plotter.add_mesh(mesh, color="#d9d2c7", opacity=0.45, smooth_shading=True)
    if "landmarks" in scene.layers:
        _add_landmarks(plotter, scene, scale=scale, colors=("red", "green", "blue"))
    if "origin" in scene.layers:
        _add_machine_origin(plotter, scene, scale=scale)

    if "cnc" in scene.layers and len(scene.cnc_path) >= 2:
        _add_line_mesh(plotter, _polyline(scene.cnc_path), color="dimgray", line_width=2)
    if (
        "programmed" in scene.layers
        and scene.programmed_path is not None
        and len(scene.programmed_path) >= 2
    ):
        _add_line_mesh(
            plotter, _polyline(scene.programmed_path), color="silver", line_width=2
        )

    if "tip" in scene.layers and len(scene.tip_path) >= 2:
        pts, mk = _dedupe_path_with_markers(scene.tip_path, scene.markers)
        colors = _tip_segment_colors(mk)
        for seg, color in _grouped_colored_segments(pts, colors):
            if jetting_only and color != "lime":
                continue
            _add_line_mesh(plotter, _polyline(seg), color=color, line_width=6)

    if "arm" in scene.layers:
        arm_step = last
        jet_on = _segment_jet_on(scene.markers)
        active = [i for i, on in enumerate(jet_on) if on]
        if active:
            arm_step = active[len(active) // 2]
        _add_arm_skeleton(plotter, scene, arm_step, scale=scale)

    if title:
        plotter.add_title(title, font_size=13)
    if show_legend:
        plotter.add_legend(
            labels=[[label, color] for label, color in LEGEND_ENTRIES],
            bcolor=(1.0, 1.0, 1.0),
            face=None,
            size=(0.28, 0.30),
            loc="upper right",
        )

    focal = np.asarray(scene.tip_path, dtype=float)
    focal = focal[np.isfinite(focal).all(axis=1)]
    center = focal.mean(axis=0) if len(focal) else mesh.center
    span = float(np.linalg.norm(mesh.bounds[1::2] - np.asarray(mesh.bounds[0::2])))
    span = span or 200.0
    d = np.asarray(camera_direction, dtype=float)
    d = d / (np.linalg.norm(d) or 1.0)
    plotter.camera_position = [
        tuple(center + d * span * 1.4),
        tuple(center),
        (0.0, 0.0, 1.0),
    ]

    plotter.screenshot(out_path, transparent_background=transparent_background)
    plotter.close()
    return out_path


def show_simulation(
    scene: SimulationScene,
    *,
    title: str = "G-code toolpath simulation",
    animate: bool = False,
) -> None:
    """Viewer: rigid arm (a) + tool (d, ⊥ arm) from G-code X,Y,Z,B,C."""
    import pyvista as pv

    plotter = pv.Plotter(window_size=(1400, 960))
    plotter.set_background("white")

    static_actors: dict[LayerName, list] = {"mesh": [], "landmarks": [], "origin": []}
    path_actors: list = []
    arm_actors: list = []

    faces = np.hstack(
        [np.full((len(scene.mesh_faces), 1), 3), scene.mesh_faces]
    ).ravel()
    mesh = pv.PolyData(scene.mesh_points, faces)
    landmark_colors = ("red", "green", "blue")
    scale = mesh.length * 0.012 if mesh.length > 0 else 5.0
    npts = len(scene.tip_path)
    step_state = {"idx": 0, "label_actor": None, "help_actor": None}

    # Normalized viewport Y (0=bottom): help → step HUD → slider
    _Y_HELP = 0.02
    _Y_STEP = 0.062
    _Y_SLIDER = 0.14

    def _help_line() -> str:
        layers = "m mesh  l landmarks  o origin  c C pivot  g prog  t tip  a arm"
        nav = (
            "←/→ step   p=advance (wrap)"
            if animate and npts > 1
            else "←/→ previous/next step"
            if npts > 1
            else ""
        )
        return f"{layers}    |    {nav}" if nav else layers

    def _visible(layer: LayerName) -> bool:
        return layer in scene.layers

    def _clear_actors(bucket: list) -> None:
        for actor in bucket:
            plotter.remove_actor(actor)
        bucket.clear()

    def _rebuild_static() -> None:
        _clear_actors(static_actors["mesh"])
        _clear_actors(static_actors["landmarks"])
        _clear_actors(static_actors["origin"])
        if _visible("mesh"):
            static_actors["mesh"].append(
                plotter.add_mesh(
                    mesh, color="lightgray", opacity=0.35, name="head_mesh"
                )
            )
        if _visible("landmarks"):
            static_actors["landmarks"] = _add_landmarks(
                plotter, scene, scale=scale, colors=landmark_colors
            )
        if _visible("origin"):
            static_actors["origin"] = _add_machine_origin(
                plotter, scene, scale=scale
            )

    def _path_up_to(path: np.ndarray, idx: int) -> np.ndarray:
        idx = max(0, min(int(idx), len(path) - 1))
        return path[: idx + 1]

    def _draw_up_to_step(idx: int) -> None:
        _clear_actors(path_actors)
        _clear_actors(arm_actors)

        if idx < 0 or npts == 0:
            return

        if _visible("cnc") and len(scene.cnc_path) >= 2:
            prefix = _path_up_to(scene.cnc_path, idx)
            prefix = _dedupe_consecutive_points(prefix)
            if len(prefix) >= 2:
                line = _polyline(prefix)
                actor = _add_line_mesh(plotter, line, color="dimgray", line_width=2)
                if actor is not None:
                    path_actors.append(actor)

        if (
            _visible("programmed")
            and scene.programmed_path is not None
            and len(scene.programmed_path) >= 2
        ):
            prefix = _path_up_to(scene.programmed_path, idx)
            prefix = _dedupe_consecutive_points(prefix)
            if len(prefix) >= 2:
                line = _polyline(prefix)
                actor = _add_line_mesh(plotter, line, color="silver", line_width=2)
                if actor is not None:
                    path_actors.append(actor)

        if _visible("tip") and len(scene.tip_path) >= 2:
            prefix = _path_up_to(scene.tip_path, idx)
            prefix_markers = scene.markers[: idx + 1]
            prefix, prefix_markers = _dedupe_path_with_markers(prefix, prefix_markers)
            if len(prefix) >= 2:
                colors = _tip_segment_colors(prefix_markers)
                for seg, color in _grouped_colored_segments(prefix, colors):
                    seg = _dedupe_consecutive_points(seg)
                    line = _polyline(seg)
                    actor = _add_line_mesh(plotter, line, color=color, line_width=5)
                    if actor is not None:
                        path_actors.append(actor)
                path_actors.append(
                    plotter.add_mesh(
                        pv.Sphere(radius=scale * 0.55, center=prefix[-1]),
                        color="gold",
                    )
                )

        if _visible("arm"):
            arm_actors.extend(_add_arm_skeleton(plotter, scene, idx, scale=scale))

    def _update_step(idx: int) -> None:
        step_state["idx"] = idx
        _draw_up_to_step(idx)

        mlabel = _marker_label(scene.markers[idx]) if idx < len(scene.markers) else ""
        b = float(scene.b_angles[idx]) if idx < len(scene.b_angles) else 0.0
        c = float(scene.c_angles[idx]) if idx < len(scene.c_angles) else 0.0
        xyz_path = scene.programmed_path if scene.programmed_path is not None else scene.cnc_path
        if idx < len(xyz_path):
            x, y, z = (float(v) for v in xyz_path[idx])
            coords = f"X={x:.2f}  Y={y:.2f}  Z={z:.2f}  B={b:.1f}  C={c:.1f}"
        else:
            coords = f"B={b:.1f}  C={c:.1f}"
        step_text = (
            f"Step {idx + 1}/{npts}  {coords}  {mlabel}"
            if npts
            else "Step — / —"
        )
        if step_state["label_actor"] is not None:
            plotter.remove_actor(step_state["label_actor"])
        step_state["label_actor"] = plotter.add_text(
            step_text, position=(0.02, _Y_STEP), font_size=9, color="black"
        )
        plotter.render()

    def _toggle(layer: LayerName) -> None:
        if layer in ("mesh", "landmarks", "origin"):
            if layer in scene.layers:
                scene.layers.discard(layer)
            else:
                scene.layers.add(layer)
            _rebuild_static()
        else:
            if layer in scene.layers:
                scene.layers.discard(layer)
            else:
                scene.layers.add(layer)
        _update_step(step_state["idx"])

    _rebuild_static()
    plotter.add_axes()
    plotter.add_title(title, font_size=14)

    plotter.add_legend(
        labels=[[label, color] for label, color in LEGEND_ENTRIES],
        bcolor=(1.0, 1.0, 1.0),
        face=None,
        size=(0.28, 0.30),
        loc="upper right",
    )

    step_state["help_actor"] = plotter.add_text(
        _help_line(),
        position=(0.02, _Y_HELP),
        font_size=7,
        color="dimgray",
    )

    for key, layer in LAYER_KEYS.items():
        plotter.add_key_event(key, lambda layer=layer: _toggle(layer))

    if npts > 0:

        def _go_prev() -> None:
            _update_step(max(0, step_state["idx"] - 1))

        def _go_next() -> None:
            _update_step(min(npts - 1, step_state["idx"] + 1))

        def _on_slider(value: float) -> None:
            _update_step(int(round(value)))

        plotter.add_slider_widget(
            _on_slider,
            rng=[0, max(0, npts - 1)],
            value=0,
            title="G-code step",
            pointa=(0.06, _Y_SLIDER),
            pointb=(0.94, _Y_SLIDER),
            style="classic",
            interaction_event="always",
        )
        plotter.add_key_event("Left", _go_prev)
        plotter.add_key_event("Right", _go_next)
        _update_step(0)

    if animate and npts > 1:

        def _auto_advance() -> None:
            idx = (step_state["idx"] + 1) % npts
            _update_step(idx)

        plotter.add_key_event("p", _auto_advance)

    plotter.show()
