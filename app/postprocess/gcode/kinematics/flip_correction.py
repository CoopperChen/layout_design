"""C-axis sign-flip prefix correction."""

from __future__ import annotations

import numpy as np


def correct_flip(b_angles: np.ndarray, c_angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Reverse-sign prefix when C-axis jumps across zero with large magnitude.

    Matches MATLAB gcodeConverter_final14 lines 360-386.
    """
    b = b_angles.copy()
    c = c_angles.copy()
    npts = len(c)
    index_flip: list[int] = []

    for i in range(npts - 1):
        k = npts - 1 - i
        if k + 1 >= npts:
            continue
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
