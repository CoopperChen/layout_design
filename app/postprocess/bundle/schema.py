"""Schema constants for eeg_subject_bundle handoff."""

from __future__ import annotations

SCHEMA_VERSION = "eeg_subject_bundle/1.0.0"

CALIBRATION_LANDMARK_KEYS = ("landmark_central", "landmark_left", "landmark_back")
CALIBRATION_LANDMARK_NAMES = (
    "Landmark(central)",
    "Landmark(left)",
    "Landmark(back)",
)
ANATOMICAL_KEYS = ("nasion", "lpa", "rpa", "inion")

ELECTRODE_AREA_CM2 = 1.5
ELECTRODE_NLINES = 10


def electrode_diameter_mm(area_cm2: float = ELECTRODE_AREA_CM2) -> float:
    """AdjPoints.m: diameter = 2*sqrt(areaelectrode/pi)*10 mm."""
    return float(2.0 * (area_cm2 / 3.141592653589793) ** 0.5 * 10.0)


ELECTRODE_DIAMETER_MM = electrode_diameter_mm()
