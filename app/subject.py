"""Load and save one subject's layout and physical landmarks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from app import paths

ANATOMY_KEYS = ("nasion", "lpa", "rpa", "inion")
CALIBRATION_KEYS = ("central", "left", "back")
CALIBRATION_LABELS = ("landmark_central", "landmark_left", "landmark_back")

_MIN_EDGE_MM = 1.0
_MIN_CROSS_NORM = 1.0


def _xyz(value) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(3)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"non-finite coordinate: {value}")
    return arr


def _xyz_map(data: dict, keys: tuple[str, ...], label: str) -> dict[str, np.ndarray]:
    missing = [k for k in keys if k not in data]
    if missing:
        raise ValueError(f"{label} missing {missing}")
    return {k: _xyz(data[k]) for k in keys}


@dataclass
class Layout:
    anatomy: dict[str, np.ndarray]
    cz: np.ndarray
    calibration: dict[str, np.ndarray]
    electrodes: dict[str, np.ndarray]

    def calibration_xyz(self) -> np.ndarray:
        return np.vstack([self.calibration[k] for k in CALIBRATION_KEYS])

    def electrode_order(self) -> list[str]:
        return list(self.electrodes)


def layout_to_dict(layout: Layout) -> dict:
    def row(point: np.ndarray) -> list[float]:
        return [float(c) for c in np.asarray(point, dtype=float).reshape(3)]

    return {
        "anatomy": {k: row(layout.anatomy[k]) for k in ANATOMY_KEYS},
        "cz": row(layout.cz),
        "calibration": {k: row(layout.calibration[k]) for k in CALIBRATION_KEYS},
        "electrodes": {name: row(pt) for name, pt in layout.electrodes.items()},
    }


def layout_from_dict(data: dict) -> Layout:
    electrodes = data.get("electrodes") or {}
    if not electrodes:
        raise ValueError("layout.json has no electrodes")
    return Layout(
        anatomy=_xyz_map(data.get("anatomy") or {}, ANATOMY_KEYS, "anatomy"),
        cz=_xyz(data.get("cz")),
        calibration=_xyz_map(data.get("calibration") or {}, CALIBRATION_KEYS, "calibration"),
        electrodes={str(name): _xyz(pt) for name, pt in electrodes.items()},
    )


def load_layout(subject_id: int | str) -> Layout:
    path = paths.layout_json(subject_id)
    if not path.is_file():
        raise FileNotFoundError(
            f"No layout for subject {subject_id}: {path}\n"
            f"Run: python -m app layout --subject {subject_id}"
        )
    return layout_from_dict(json.loads(path.read_text(encoding="utf-8")))


def save_layout(subject_id: int | str, layout: Layout) -> Path:
    path = paths.layout_json(subject_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(layout_to_dict(layout), indent=2) + "\n", encoding="utf-8")
    return path


def validate_landmark_triangle(
    points: np.ndarray,
    *,
    label: str = "physical_landmarks_mm",
    path: Path | str | None = None,
) -> np.ndarray:
    """Reject a missing or degenerate landmark triangle before scan2phys."""
    arr = np.asarray(points, dtype=float)
    if arr.shape != (3, 3):
        raise ValueError(f"{label} must be 3x3, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{label} contains non-finite values")

    where = f" in {path}" if path is not None else ""
    if np.allclose(arr, 0.0, atol=1e-9):
        raise ValueError(
            f"{label}{where} is still all zeros.\n"
            f"  Capture landmarks: python -m app record-pm --subject <id>"
        )

    edges = [
        float(np.linalg.norm(arr[1] - arr[0])),
        float(np.linalg.norm(arr[2] - arr[0])),
        float(np.linalg.norm(arr[2] - arr[1])),
    ]
    if min(edges) < _MIN_EDGE_MM:
        raise ValueError(
            f"{label}{where} has coincident points "
            f"(edge lengths mm: {edges[0]:.3f}, {edges[1]:.3f}, {edges[2]:.3f}; "
            f"need >= {_MIN_EDGE_MM} mm)."
        )

    area2 = float(np.linalg.norm(np.cross(arr[1] - arr[0], arr[2] - arr[0])))
    if area2 < _MIN_CROSS_NORM:
        raise ValueError(
            f"{label}{where} is nearly colinear "
            f"(||(p1-p0)×(p2-p0)||={area2:.3g}; need >= {_MIN_CROSS_NORM})."
        )
    return arr


def save_physical_landmarks(
    path: Path | str,
    pm: np.ndarray,
    *,
    subject_id: int | str | None = None,
    capture_meta: dict | None = None,
) -> Path:
    out = Path(path)
    if not out.is_absolute():
        out = paths.REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(pm, dtype=float)
    if arr.shape != (3, 3):
        raise ValueError(f"physical_landmarks_mm must be 3x3, got {arr.shape}")

    sid = subject_id if subject_id is not None else out.parent.name
    lines = [
        f"# Physical landmarks (pm) for subject {sid}",
        "# Rows: central, left, back. Central is [0, 0, 0].",
        "physical_landmarks_mm:",
    ]
    for row, label in zip(arr, CALIBRATION_LABELS):
        lines.append(f"  - [{row[0]:.6g}, {row[1]:.6g}, {row[2]:.6g}]  # {label}")
    if capture_meta:
        lines.append("")
        lines.append("capture:")
        raw = capture_meta.get("raw_work_xyz_mm")
        if raw is not None:
            lines.append("  raw_work_xyz_mm:")
            for row, label in zip(raw, CALIBRATION_LABELS):
                lines.append(
                    f"    - [{row[0]:.6g}, {row[1]:.6g}, {row[2]:.6g}]  # {label}"
                )
        bc = capture_meta.get("work_bc_deg")
        if bc is not None:
            lines.append("  work_bc_deg:")
            for entry, label in zip(bc, CALIBRATION_LABELS):
                lines.append(
                    f"    - {{b: {float(entry['b']):.4g}, c: {float(entry['c']):.4g}}}  # {label}"
                )
        if "udp_port" in capture_meta:
            lines.append(f"  udp_port: {int(capture_meta['udp_port'])}")
    lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def load_physical_landmarks(
    path: Path | str,
    *,
    require_measured: bool = True,
) -> np.ndarray:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    pm = data.get("physical_landmarks_mm")
    if pm is None:
        raise ValueError(
            f"No physical_landmarks_mm in {path}. "
            f"Run: python -m app record-pm --subject <id>"
        )
    arr = np.asarray(pm, dtype=float)
    if arr.shape != (3, 3):
        raise ValueError(f"physical_landmarks_mm must be 3x3 in {path}, got {arr.shape}")
    if require_measured:
        validate_landmark_triangle(arr, path=path)
    return arr


def pm_is_measured(path: Path | str) -> bool:
    p = Path(path)
    if not p.is_file():
        return False
    try:
        load_physical_landmarks(p, require_measured=True)
    except (ValueError, OSError, yaml.YAMLError):
        return False
    return True


def load_pm(subject_id: int | str) -> np.ndarray:
    path = paths.pm_yaml(subject_id)
    if not path.is_file():
        raise FileNotFoundError(
            f"No physical landmarks for subject {subject_id}: {path}\n"
            f"Run: python -m app record-pm --subject {subject_id}"
        )
    return load_physical_landmarks(path)


def load_head_arrays(subject_id: int | str) -> tuple[np.ndarray, np.ndarray]:
    """STL vertices and triangle faces for registration and arm clearance."""
    import pyvista as pv

    path = paths.head_stl(subject_id)
    if not path.is_file():
        raise FileNotFoundError(
            f"No head mesh for subject {subject_id}: {path}\n"
            f"Run: python -m app mesh --subject {subject_id}"
        )
    mesh = pv.read(str(path))
    if isinstance(mesh, pv.MultiBlock):
        mesh = mesh.combine()
    points = np.asarray(mesh.points, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int).reshape(-1, 4)[:, 1:4]
    return points, faces
