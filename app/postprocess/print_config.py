"""Per-subject physical landmark (pm) configs for G-code conversion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from app import paths

PM_TEMPLATE = """# Physical landmarks (pm) for subject {subject_id}
# Measure with end-effector on printhead; touch landmark_central, left, back.
# Point 1 is origin [0, 0, 0]; points 2–3 are machine XYZ.
# Or automate: python -m app record-pm --subject {subject_id}
physical_landmarks_mm:
  - [0, 0, 0]
  - [0, 0, 0]  # TODO: landmark_left
  - [0, 0, 0]  # TODO: landmark_back
"""

_MIN_EDGE_MM = 1.0
_MIN_CROSS_NORM = 1.0  # mm^2 scale; rejects near-colinear / coincident points


def validate_landmark_triangle(
    points: np.ndarray,
    *,
    label: str = "physical_landmarks_mm",
    path: Path | str | None = None,
) -> np.ndarray:
    """
    Ensure three landmark points form a usable registration triangle.

    Rejects the empty ``init-print-config`` scaffold (all zeros) and
    near-degenerate geometry that would NaN in ``scan2phys``.
    """
    arr = np.asarray(points, dtype=float)
    if arr.shape != (3, 3):
        raise ValueError(f"{label} must be 3x3, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{label} contains non-finite values")

    where = f" in {path}" if path is not None else ""
    if np.allclose(arr, 0.0, atol=1e-9):
        raise ValueError(
            f"{label}{where} is still the empty scaffold (all zeros).\n"
            f"  Capture real landmarks first:\n"
            f"    python -m app record-pm --subject <id>\n"
            f"  Or edit left/back XYZ in the pm YAML (central stays [0,0,0])."
        )

    e01 = float(np.linalg.norm(arr[1] - arr[0]))
    e02 = float(np.linalg.norm(arr[2] - arr[0]))
    e12 = float(np.linalg.norm(arr[2] - arr[1]))
    if min(e01, e02, e12) < _MIN_EDGE_MM:
        raise ValueError(
            f"{label}{where} has coincident/near-coincident points "
            f"(edge lengths mm: {e01:.3f}, {e02:.3f}, {e12:.3f}; "
            f"need >= {_MIN_EDGE_MM} mm).\n"
            f"  Re-measure with record-pm or fix the YAML."
        )

    cross = np.cross(arr[1] - arr[0], arr[2] - arr[0])
    area2 = float(np.linalg.norm(cross))
    if area2 < _MIN_CROSS_NORM:
        raise ValueError(
            f"{label}{where} is nearly colinear "
            f"(||(p1-p0)×(p2-p0)||={area2:.3g}; need >= {_MIN_CROSS_NORM}).\n"
            f"  The three landmarks must span a plane (not a line)."
        )
    return arr


def init_print_config(
    subject_id: int | str,
    *,
    force: bool = False,
) -> Path:
    """Create pm-only YAML scaffold for a subject."""
    out = paths.postprocessor_subject_pm(subject_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not force:
        raise FileExistsError(
            f"Print config already exists: {out} (use --force to overwrite)"
        )
    out.write_text(PM_TEMPLATE.format(subject_id=subject_id), encoding="utf-8")
    return out


def save_physical_landmarks(
    path: Path | str,
    pm: np.ndarray,
    *,
    subject_id: int | str | None = None,
    capture_meta: dict | None = None,
) -> Path:
    """Write pm YAML (overwrites). Optional capture_meta stores raw DRO for audit."""
    out = Path(path)
    if not out.is_absolute():
        out = paths.REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(pm, dtype=float)
    if arr.shape != (3, 3):
        raise ValueError(f"physical_landmarks_mm must be 3x3, got {arr.shape}")

    sid = subject_id if subject_id is not None else out.stem.replace("subject_", "")
    lines = [
        f"# Physical landmarks (pm) for subject {sid}",
        "# Rows: landmark_central, landmark_left, landmark_back",
        "# Central is measurement origin [0, 0, 0]; left/back relative to central touch.",
        "physical_landmarks_mm:",
    ]
    labels = ("landmark_central", "landmark_left", "landmark_back")
    for row, label in zip(arr, labels):
        lines.append(
            f"  - [{row[0]:.6g}, {row[1]:.6g}, {row[2]:.6g}]  # {label}"
        )
    if capture_meta:
        lines.append("")
        lines.append("# Capture audit (not used by convert-gcode)")
        lines.append("capture:")
        raw = capture_meta.get("raw_work_xyz_mm")
        if raw is not None:
            lines.append("  raw_work_xyz_mm:")
            for row, label in zip(raw, labels):
                lines.append(
                    f"    - [{row[0]:.6g}, {row[1]:.6g}, {row[2]:.6g}]  # {label}"
                )
        bc = capture_meta.get("work_bc_deg")
        if bc is not None:
            lines.append("  work_bc_deg:")
            for entry, label in zip(bc, labels):
                lines.append(
                    f"    - {{b: {float(entry['b']):.4g}, c: {float(entry['c']):.4g}}}  # {label}"
                )
        if "udp_port" in capture_meta:
            lines.append(f"  udp_port: {int(capture_meta['udp_port'])}")
    lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def pm_is_measured(path: Path | str) -> bool:
    """True if YAML exists and ``physical_landmarks_mm`` passes triangle validation."""
    p = Path(path)
    if not p.is_file():
        return False
    try:
        load_physical_landmarks(p, require_measured=True)
    except (ValueError, OSError, yaml.YAMLError):
        return False
    return True


def load_physical_landmarks(
    path: Path | str,
    *,
    require_measured: bool = True,
) -> np.ndarray:
    """Load pm from a slim or legacy full YAML."""
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    pm = data.get("physical_landmarks_mm")
    if pm is None:
        pm = data.get("registration", {}).get("physical_landmarks_mm")
    if pm is None:
        raise ValueError(
            f"No physical_landmarks_mm in {path}; run: "
            f"python -m app init-print-config --subject <id>"
        )
    arr = np.asarray(pm, dtype=float)
    if arr.shape != (3, 3):
        raise ValueError(f"physical_landmarks_mm must be 3x3 in {path}, got {arr.shape}")
    if require_measured:
        validate_landmark_triangle(arr, path=path)
    return arr


def resolve_pm_config(
    bundle_dir: Path | str,
    *,
    config: Path | str | None = None,
    pm_file: Path | str | None = None,
) -> Path:
    """
    Resolve pm YAML path: explicit --pm-file / --config, else subject_{id}.yaml from bundle.
    """
    if pm_file is not None:
        p = Path(pm_file)
        if not p.is_absolute():
            p = paths.REPO_ROOT / p
        if not p.is_file():
            raise FileNotFoundError(f"pm file not found: {p}")
        return p

    if config is not None:
        p = Path(config)
        if not p.is_absolute():
            p = paths.REPO_ROOT / p
        if not p.is_file():
            raise FileNotFoundError(f"print config not found: {p}")
        return p

    from app.postprocess.gcode.io.load_bundle import load_bundle

    bundle_path = Path(bundle_dir)
    if not bundle_path.is_absolute():
        bundle_path = paths.REPO_ROOT / bundle_path
    bundle = load_bundle(bundle_path)
    subject_id = bundle.subject_id
    auto = paths.postprocessor_subject_pm(subject_id)
    if not auto.is_file():
        raise FileNotFoundError(
            f"No print config for subject {subject_id}: {auto}\n"
            f"Run: python -m app init-print-config --subject {subject_id}"
        )
    return auto
