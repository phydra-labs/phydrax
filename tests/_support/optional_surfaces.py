"""Closed triangle surfaces for optional-provider consumer tests."""

from collections.abc import Sequence

import numpy as np

from phydrax import SpatialCoordinateContract
from phydrax.geometry.surface import SurfaceMetadata, SurfaceModel


def boolean_cube(offset: Sequence[float]) -> SurfaceModel:
    """Return a translated unit cube with outward triangles and stable source tags.

    Vertex and triangle ordering retain the Manifold fixture's source-face IDs.
    The triangulation is explicit so no optional fixture dependency is loaded.
    """
    vertices = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
            (0.0, 1.0, 1.0),
        ),
        dtype=np.float64,
    )
    faces = np.asarray(
        (
            (0, 2, 1),
            (0, 3, 2),
            (4, 5, 6),
            (4, 6, 7),
            (0, 1, 5),
            (0, 5, 4),
            (3, 7, 6),
            (3, 6, 2),
            (0, 4, 7),
            (0, 7, 3),
            (1, 2, 6),
            (1, 6, 5),
        ),
        dtype=np.int64,
    )
    return SurfaceModel.from_triangles(
        vertices + np.asarray(offset, dtype=np.float64),
        faces,
        SurfaceMetadata(
            source_id=str(offset),
            source_revision="0",
            coordinate_contract=SpatialCoordinateContract.si(),
            provenance=("qualification",),
        ),
    )
