# Copyright © 2026 PHYDRA, Inc. All rights reserved.

from __future__ import annotations

from math import pi

import numpy as np
import numpy.typing as npt

from ....discretization.bem._rwg import RWGSurfaceCurrentSpace3D


def central_magnetic_matrix(
    space: RWGSurfaceCurrentSpace3D, wavenumber: float, /
) -> npt.NDArray[np.complex128]:
    """Principal-value free-space n cross curl S with centroid products.

    The self-panel principal value vanishes on a planar panel. Distinct-panel
    actions retain the actual outgoing Green gradient; the exterior half-jump
    is added with the exact Gram matrix by the boundary preparation owner.
    """
    surface = space.surface
    points = np.asarray(surface.face_centroids)
    normals = np.asarray(surface.face_normals)
    areas = np.asarray(surface.face_areas)
    edges = np.asarray(surface.face_edges)
    basis = np.asarray(space.centroid_basis)
    matrix = np.zeros((space.size, space.size), dtype=np.complex128)
    for target in range(surface.face_count):
        for source in range(surface.face_count):
            if target == source:
                continue
            displacement = points[target] - points[source]
            radius = np.linalg.norm(displacement)
            if radius <= 0.0:
                raise ValueError("Distinct Maxwell panels require distinct centroids.")
            gradient = (
                np.exp(1j * wavenumber * radius)
                * (1j * wavenumber * radius - 1.0)
                * displacement
                / (4.0 * pi * radius**3)
            )
            trial_curl = np.cross(gradient, basis[source])
            rotated = np.cross(normals[target], trial_curl)
            local = areas[target] * areas[source] * (basis[target] @ rotated.T)
            matrix[np.ix_(edges[target], edges[source])] += local
    return matrix
