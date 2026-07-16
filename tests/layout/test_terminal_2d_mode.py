"""Terminal 2D mode aliases (fiducial_native ↔ fiducial)."""

from __future__ import annotations

import numpy as np

from app.runtime import setup_runtime

setup_runtime()
from PYTHON.tools import new2dAlterations as n2d  # noqa: E402


def test_normalize_terminal_2d_mode_aliases():
    assert n2d.normalize_terminal_2d_mode("fiducial_native") == "fiducial"
    assert n2d.normalize_terminal_2d_mode("fiducial") == "fiducial"
    assert n2d.normalize_terminal_2d_mode("inflated_legacy") == "inflated"
    assert n2d.normalize_terminal_2d_mode("inflated") == "inflated"


def test_build_terminals_2d_fiducial_native_matches_fiducial():
    cz = np.array([0.0, 0.0, 0.0])
    electrodes_2d = {
        "Cz": np.array([0.0, 0.0]),
        "Fp1": np.array([10.0, 0.0]),
        "O1": np.array([-10.0, 0.0]),
    }
    # Hub closer than inflated radius so modes differ.
    fiducials = {
        "TERMINAL_LEFT": np.array([5.0, 0.0, 0.0]),
        "TERMINAL_RIGHT": np.array([-5.0, 0.0, 0.0]),
    }
    native = n2d.build_terminals_2d(
        electrodes_2d, fiducials, cz, mode="fiducial_native"
    )
    fiducial = n2d.build_terminals_2d(
        electrodes_2d, fiducials, cz, mode="fiducial"
    )
    inflated = n2d.build_terminals_2d(
        electrodes_2d, fiducials, cz, mode="inflated_legacy"
    )
    np.testing.assert_allclose(native["TERMINAL_LEFT"], fiducial["TERMINAL_LEFT"])
    assert not np.allclose(native["TERMINAL_LEFT"], inflated["TERMINAL_LEFT"])
