"""Visualize should keep truncated wire ends for synthesize/polish layouts."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from app.layout.visualize import _surface_paths_3d


def test_surface_paths_keeps_truncated_uv_synthesize_paths():
    path_end = [1.0, 2.0, 3.0]
    entry = [10.0, 20.0, 30.0]
    data = {
        "metadata": {
            "target_subject_id": 5,
            "path_lift": "uv_surface_synthesize",
            "terminal_2d_mode": "fiducial_native",
        },
        "paths": [
            {
                "electrode": "Cz",
                "terminal": "TERMINAL_LEFT",
                "modified_path_2d": [[0.0, 0.0], [1.0, 1.0]],
                "path_points": [[0.0, 0.0, 0.0], path_end],
                "path_end_3d": path_end,
                "entry_position_3d": entry,
                "entry_point_2d": [5.0, 5.0],
                "path_end_2d": [1.0, 1.0],
            }
        ],
    }
    before = deepcopy(data["paths"][0]["path_points"])
    # Subject 5 mesh may be required if re-lift runs; this must not re-lift.
    out = _surface_paths_3d(data, subject_id=5)
    assert out["paths"][0]["path_points"] == before
    assert np.allclose(out["paths"][0]["path_points"][-1], path_end)
