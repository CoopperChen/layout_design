"""Standard 10–20 positions in metres, relative to the MNE head frame.

Values are the colin27 10–20 montage. Only this table is required; MNE is not imported.
"""

from __future__ import annotations

import numpy as np

# x right, y anterior, z superior, metres.
STANDARD_1020_M: dict[str, tuple[float, float, float]] = {
    "Fp1": (-0.0294367, 0.0839171, -0.00699),
    "Fp2": (0.0298723, 0.0848959, -0.00708),
    "F7": (-0.0702629, 0.0424743, -0.01142),
    "F8": (0.0730431, 0.0444217, -0.012),
    "F3": (-0.0502438, 0.0531112, 0.042192),
    "F4": (0.0518362, 0.0543048, 0.040814),
    "T7": (-0.0841611, -0.0160187, -0.009346),
    "T8": (0.0850799, -0.0150203, -0.00949),
    "C3": (-0.0653581, -0.0116317, 0.064358),
    "C4": (0.0671179, -0.0109003, 0.06358),
    "P7": (-0.0724343, -0.0734527, -0.002487),
    "P8": (0.0730557, -0.0730683, -0.00254),
    "P3": (-0.0530073, -0.0787878, 0.05594),
    "P4": (0.0556667, -0.0785602, 0.056561),
    "O1": (-0.0294134, -0.112449, 0.008839),
    "O2": (0.0298426, -0.112156, 0.0088),
    "Fz": (0.0003122, 0.058512, 0.066462),
    "Cz": (0.0004009, -0.009167, 0.100244),
    "Pz": (0.0003247, -0.081115, 0.082615),
    "Fpz": (0.0001123, 0.088247, -0.001713),
    "Oz": (0.0001076, -0.114892, 0.014657),
}

CHANNEL_PAIRS = (
    ("Fp1", "Fp2"),
    ("F7", "F8"),
    ("F3", "F4"),
    ("T7", "T8"),
    ("C3", "C4"),
    ("P7", "P8"),
    ("P3", "P4"),
    ("O1", "O2"),
)
SINGLE_CHANNELS = ("Fz", "Cz", "Pz")
CHANNELS = tuple(ch for pair in CHANNEL_PAIRS for ch in pair) + SINGLE_CHANNELS


def template_xyz(name: str) -> np.ndarray:
    return np.asarray(STANDARD_1020_M[name], dtype=float)
