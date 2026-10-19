# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Binary64-authored power sources for coordinate-law contract tests."""

from __future__ import annotations

import numpy as np
from jax import Array

from phydrax.discretization import ExactPowerCellGeometrySource, PeriodicCell
from phydrax.geometry._periodic_power_preparation import PeriodicPowerPreparation


def power_source() -> ExactPowerCellGeometrySource:
    """Return two weighted sites whose bisector meets three carrier edges."""
    return ExactPowerCellGeometrySource(
        np.asarray([[0.0, 0.0, 0.0], [0.7, 0.6, 0.8]], dtype=np.float64),
        np.asarray([0.03, -0.02], dtype=np.float64),
        np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        np.asarray([[0, 1, 2, 3]], dtype=np.int32),
        np.asarray([0, 1, 2, 3, 4, 6, 8, 10], dtype=np.int64),
        np.asarray([0, 0, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int32),
        np.asarray(
            [
                [0, -1, -1, -1],
                [1, -1, -1, -1],
                [2, -1, -1, -1],
                [3, -1, -1, -1],
                [0, 3, -1, -1],
                [1, 3, -1, -1],
                [2, 3, -1, -1],
            ],
            dtype=np.int32,
        ),
    )


def periodic_power_source() -> ExactPowerCellGeometrySource:
    """Return a retained lattice-image law with one authored site."""
    points = np.asarray([[0.2, 0.2, 0.2]], dtype=np.float64)
    weights = np.zeros(1, dtype=np.float64)
    carrier = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    preparation = PeriodicPowerPreparation(
        points, weights, carrier, PeriodicCell(4 * np.eye(3)), maximum_images=1000
    )
    image = int(np.flatnonzero(np.all(preparation.image_exponents == 0, axis=1))[0])
    return ExactPowerCellGeometrySource(
        points,
        weights,
        carrier,
        np.asarray([[0, 1, 2, 3]], dtype=np.int64),
        np.arange(5, dtype=np.int64),
        np.full(4, image, dtype=np.int64),
        np.column_stack(
            (np.arange(4, dtype=np.int64), np.full((4, 3), -1, dtype=np.int64))
        ),
        periodic_preparation=preparation,
    )


def source_arrays(
    source: ExactPowerCellGeometrySource, /
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    """Return the constructor inputs without changing scientific array bytes."""
    return (
        source.site_points,
        source.site_weights,
        source.carrier_points,
        source.carrier_tets,
        source.vertex_site_offsets,
        source.vertex_sites,
        source.vertex_carriers,
    )
