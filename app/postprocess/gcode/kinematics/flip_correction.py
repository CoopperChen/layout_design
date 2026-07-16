"""B/C axis branch selection for continuous machine commands."""

from __future__ import annotations

import numpy as np


def c_step_deg(c1: float, c2: float) -> float:
    delta = abs(float(c1) - float(c2)) % 360.0
    return min(delta, 360.0 - delta)


def bc_step_deg(b1: float, c1: float, b2: float, c2: float) -> float:
    return abs(float(b1) - float(b2)) + c_step_deg(c1, c2)


def correct_flip(b_angles: np.ndarray, c_angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Reverse-sign prefix when C-axis jumps across zero with large magnitude.

    Matches MATLAB gcodeConverter_final14 lines 360-386.
    """
    b = b_angles.copy()
    c = c_angles.copy()
    npts = len(c)
    index_flip: list[int] = []

    for k in range(npts - 2, -1, -1):
        if (
            (c[k + 1] > 0 and c[k] < 0) or (c[k + 1] < 0 and c[k] > 0)
        ) and abs(c[k]) > 20 and abs(c[k + 1]) > 20:
            index_flip.append(k)

    nflips = len(index_flip)
    if nflips == 1:
        k = index_flip[0]
        c[: k + 1] = -c[: k + 1]
        b[: k + 1] = -b[: k + 1]
    elif nflips > 1:
        for k in index_flip:
            c[: k + 1] = -c[: k + 1]
            b[: k + 1] = -b[: k + 1]

    return b, c


def enforce_axis_continuity(
    b_angles: np.ndarray,
    c_angles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pick equivalent (B,C) or (-B,-C) at each point to minimize step from previous.

    Nozzle orientation is unchanged; only the commanded branch is selected.
    """
    b = b_angles.copy()
    c = c_angles.copy()
    if len(c) <= 1:
        return b, c

    if bc_step_deg(-b[0], -c[0], b[1], c[1]) < bc_step_deg(b[0], c[0], b[1], c[1]):
        b[0] = -b[0]
        c[0] = -c[0]

    for i in range(1, len(c)):
        direct = (b[i], c[i])
        flipped = (-b[i], -c[i])
        if bc_step_deg(b[i - 1], c[i - 1], *flipped) < bc_step_deg(
            b[i - 1], c[i - 1], *direct
        ):
            b[i], c[i] = flipped
    return b, c


def limit_c_slew(
    b_angles: np.ndarray,
    c_angles: np.ndarray,
    *,
    max_step_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cap consecutive |ΔC| so the printer cannot swing the arm tip per sample.

    Slight tip-orientation lag vs the raw normal is preferred over serpentine
    tracks from crown atan2 jitter. B is left unchanged.
    """
    b = np.asarray(b_angles, dtype=float).copy()
    c = np.asarray(c_angles, dtype=float).copy()
    if len(c) <= 1 or max_step_deg <= 0.0:
        return b, c

    limit = float(max_step_deg)
    for i in range(1, len(c)):
        # Signed shortest-path delta in (-180, 180]
        delta = (float(c[i]) - float(c[i - 1]) + 180.0) % 360.0 - 180.0
        if abs(delta) <= limit:
            continue
        c[i] = float(c[i - 1]) + float(np.sign(delta)) * limit
        # Keep C in a conventional range similar to axis-angle outputs.
        if c[i] > 180.0:
            c[i] -= 360.0
        elif c[i] <= -180.0:
            c[i] += 360.0
    return b, c


def max_c_step_deg(c_angles: np.ndarray) -> float:
    if len(c_angles) <= 1:
        return 0.0
    return max(c_step_deg(c_angles[i - 1], c_angles[i]) for i in range(1, len(c_angles)))


def validate_axis_continuity(
    b_angles: np.ndarray,
    c_angles: np.ndarray,
    *,
    max_c_step_deg: float,
) -> None:
    """Raise when consecutive C commands exceed the allowed step."""
    threshold = float(max_c_step_deg)
    for i in range(1, len(c_angles)):
        step = c_step_deg(c_angles[i - 1], c_angles[i])
        if step > threshold:
            raise ValueError(
                f"C-axis step {step:.1f} deg at index {i} "
                f"(C {c_angles[i - 1]:.2f} -> {c_angles[i]:.2f}) "
                f"exceeds limit {threshold:.1f} deg"
            )
