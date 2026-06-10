"""Per-subject physical landmark (pm) configs for G-code conversion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from app import paths

PM_TEMPLATE = """# Physical landmarks (pm) for subject {subject_id}
# Measure with end-effector on printhead; touch landmark_central, left, back.
# Point 1 is origin [0, 0, 0]; points 2–3 are machine XYZ.
physical_landmarks_mm:
  - [0, 0, 0]
  - [0, 0, 0]  # TODO: landmark_left
  - [0, 0, 0]  # TODO: landmark_back
"""


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


def load_physical_landmarks(path: Path | str) -> np.ndarray:
    """Load pm from a slim or legacy full YAML."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
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
