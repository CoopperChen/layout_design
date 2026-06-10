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

ELECTRODE_DIAMETER_MM = 13.8
ELECTRODE_CIRCLE_RESOLUTION = 20
