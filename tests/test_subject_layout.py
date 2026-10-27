"""layout.json round trip and 10–20 projection."""

from pathlib import Path

import numpy as np
import pyvista as pv

from app import paths
from app.layout.place import compute_cz, project_electrodes
from app.layout.template import CHANNELS
from app.subject import Layout, layout_from_dict, layout_to_dict, save_layout


def test_layout_json_roundtrip(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path)
    anatomy = {
        "nasion": np.array([0.0, 80.0, 0.0]),
        "lpa": np.array([-70.0, 0.0, 0.0]),
        "rpa": np.array([70.0, 0.0, 0.0]),
        "inion": np.array([0.0, -80.0, 0.0]),
    }
    layout = Layout(
        anatomy=anatomy,
        cz=np.array([0.0, 0.0, 90.0]),
        calibration={
            "central": np.array([0.0, 10.0, 40.0]),
            "left": np.array([-30.0, 10.0, 40.0]),
            "back": np.array([0.0, -20.0, 30.0]),
        },
        electrodes={"Cz": np.array([0.0, 0.0, 100.0]), "Fp1": np.array([-20.0, 70.0, 10.0])},
    )
    save_layout(7, layout)
    loaded = layout_from_dict(layout_to_dict(layout))
    assert loaded.electrode_order() == ["Cz", "Fp1"]
    np.testing.assert_allclose(loaded.calibration_xyz()[1], layout.calibration["left"])


def test_project_electrodes_snaps_onto_sphere():
    mesh = pv.Sphere(radius=100.0, theta_resolution=40, phi_resolution=40)
    anatomy = {
        "nasion": np.array([0.0, 100.0, 0.0]),
        "inion": np.array([0.0, -100.0, 0.0]),
        "lpa": np.array([-100.0, 0.0, 0.0]),
        "rpa": np.array([100.0, 0.0, 0.0]),
    }
    cz = np.array([0.0, 0.0, 100.0])
    positions = project_electrodes(mesh, anatomy, cz, adjustment=0.8)
    assert list(positions) == list(CHANNELS)
    radii = [float(np.linalg.norm(pt)) for pt in positions.values()]
    assert max(abs(r - 100.0) for r in radii) < 8.0


def test_project_electrodes_ignores_unused_vertices():
    """Stray OBJ vertices must not pull sites off the scalp."""
    sphere = pv.Sphere(radius=100.0, theta_resolution=40, phi_resolution=40)
    stray = np.array(
        [[400.0, 50.0, 300.0], [-350.0, -40.0, 250.0], [0.0, 0.0, 800.0]],
        dtype=float,
    )
    mesh = pv.PolyData(np.vstack([np.asarray(sphere.points), stray]), sphere.faces)
    anatomy = {
        "nasion": np.array([0.0, 100.0, 0.0]),
        "inion": np.array([0.0, -100.0, 0.0]),
        "lpa": np.array([-100.0, 0.0, 0.0]),
        "rpa": np.array([100.0, 0.0, 0.0]),
    }
    positions = project_electrodes(mesh, anatomy, np.array([0.0, 0.0, 100.0]), adjustment=0.8)
    radii = [float(np.linalg.norm(pt)) for pt in positions.values()]
    assert max(abs(r - 100.0) for r in radii) < 8.0


def test_compute_cz_ignores_stray_high_vertex():
    sphere = pv.Sphere(radius=100.0, theta_resolution=48, phi_resolution=48)
    stray = np.array([[40.0, 30.0, 400.0]])
    mesh = pv.PolyData(np.vstack([np.asarray(sphere.points), stray]), sphere.faces)
    anatomy = {
        "nasion": np.array([0.0, 100.0, 0.0]),
        "inion": np.array([0.0, -100.0, 0.0]),
        "lpa": np.array([-100.0, 0.0, 0.0]),
        "rpa": np.array([100.0, 0.0, 0.0]),
    }
    cz = compute_cz(mesh, anatomy)
    assert cz[2] > 90.0
    assert abs(cz[0]) < 15.0
    assert abs(cz[1]) < 15.0
