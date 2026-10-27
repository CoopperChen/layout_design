"""Plane-local coordinate rotation matrices (PlaneCoordinatesRotationMatrices2)."""

from __future__ import annotations

import numpy as np


def plane_coordinates_rotation_matrices(
    nhat: np.ndarray, vhat: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build T (global->local) and Ttilde (local->global) from plane normal and in-plane axis.

    Matches MATLAB PlaneCoordinatesRotationMatrices2.m.
    """
    nhat = np.asarray(nhat, dtype=float).ravel()
    vhat = np.asarray(vhat, dtype=float).ravel()
    if abs(np.dot(nhat, vhat)) > 1e-12:
        raise ValueError("The vectors need to be orthogonal.")

    zhat_prime = nhat / np.linalg.norm(nhat)
    xhat_prime = vhat / np.linalg.norm(vhat)
    yhat_prime = np.cross(zhat_prime, xhat_prime)
    yhat_prime = yhat_prime / np.linalg.norm(yhat_prime)

    ttilde = np.vstack([xhat_prime, yhat_prime, zhat_prime])
    t = np.linalg.inv(ttilde)
    return t, ttilde
