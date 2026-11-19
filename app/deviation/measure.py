"""Electrode error after the printed head shares the planned frame."""

from __future__ import annotations

import json

import numpy as np

from app import paths


def apply_transform(points, transform) -> np.ndarray:
    """Apply a 4×4 matrix. ``x' = T @ [x, y, z, 1]``."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    homo = np.c_[pts, np.ones(len(pts))]
    return (np.asarray(transform, dtype=float) @ homo.T).T[:, :3]


def _xyz(point) -> list[float]:
    return [round(float(c), 4) for c in np.asarray(point, dtype=float).reshape(3)]


def electrode_record(planned, printed) -> dict:
    """``delta`` is printed minus planned, in the planned head frame."""
    planned_xyz = np.asarray(planned, dtype=float).reshape(3)
    printed_xyz = np.asarray(printed, dtype=float).reshape(3)
    if not np.isfinite(planned_xyz).all() or not np.isfinite(printed_xyz).all():
        raise ValueError("electrode coordinates must be finite")
    delta = printed_xyz - planned_xyz
    return {
        "planned_mm": _xyz(planned_xyz),
        "printed_mm": _xyz(printed_xyz),
        "delta_mm": _xyz(delta),
        "distance_mm": round(float(np.linalg.norm(delta)), 4),
    }


def summary_record(rows: dict[str, dict]) -> dict:
    if not rows:
        raise ValueError("no electrode deviations")
    names = list(rows)
    distance = np.array([float(rows[name]["distance_mm"]) for name in names], dtype=float)
    max_i = int(np.argmax(distance))
    return {
        "count": len(names),
        "mean_mm": round(float(distance.mean()), 4),
        "rms_mm": round(float(np.sqrt(np.mean(distance**2))), 4),
        "max_mm": round(float(distance[max_i]), 4),
        "max_name": names[max_i],
    }


def _scale_of(transform) -> float:
    linear = np.asarray(transform, dtype=float)[:3, :3]
    return float(np.mean(np.linalg.norm(linear, axis=0)))


def overlap_record(overlap) -> dict:
    transform = np.asarray(overlap.transform, dtype=float)
    return {
        "source": str(overlap.source),
        "scale": round(_scale_of(transform), 6),
        "fitness": round(float(overlap.fitness), 4),
        "mean_surface_mm": round(float(overlap.mean_dist_mm), 4),
        "init": str(overlap.init),
        "transform": [[round(float(value), 6) for value in row] for row in transform.tolist()],
    }


def build_document(subject_id: int | str, overlap, planned: dict, picked: dict) -> dict:
    order = list(planned)
    missing = [name for name in order if name not in picked]
    if missing:
        raise ValueError(f"Printed picks missing {missing}")
    unknown = [name for name in picked if name not in planned]
    if unknown:
        raise ValueError(f"Printed picks are not in the layout: {unknown}")
    rows = {name: electrode_record(planned[name], picked[name]) for name in order}
    return {
        "subject": int(subject_id),
        "delta_definition": "printed_mm - planned_mm in the planned head frame",
        "overlap": overlap_record(overlap),
        "electrodes": rows,
        "summary": summary_record(rows),
    }


def format_report(document: dict) -> str:
    lines = ["Printed electrode deviation (printed - planned)"]
    for name, row in document["electrodes"].items():
        dx, dy, dz = row["delta_mm"]
        lines.append(
            f"  {name:<6}  {row['distance_mm']:7.3f} mm"
            f"    dx={dx:+7.3f}  dy={dy:+7.3f}  dz={dz:+7.3f}"
        )
    summary = document["summary"]
    overlap = document["overlap"]
    lines.append(
        f"mean {summary['mean_mm']:.3f} mm   "
        f"rms {summary['rms_mm']:.3f} mm   "
        f"max {summary['max_mm']:.3f} mm ({summary['max_name']})"
    )
    lines.append(
        f"overlap {overlap['source']}  scale {overlap['scale']:.4g}  "
        f"mean surface {overlap['mean_surface_mm']:.3f} mm"
    )
    return "\n".join(lines)


def save_document(subject_id: int | str, document: dict):
    path = paths.deviation_json(subject_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return path
