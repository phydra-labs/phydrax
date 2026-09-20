from importlib.util import find_spec

import numpy as np
import pytest

from phydrax import SpatialCoordinateContract
from phydrax.geometry.surface import SurfaceMetadata, SurfaceModel
from phydrax.meshing import MeshingFailure
from phydrax.meshing.providers._manifold import ManifoldProvider, SurfaceBooleanOperation


pytestmark = [
    pytest.mark.meshing_manifold,
    pytest.mark.skipif(
        find_spec("manifold3d") is None,
        reason="optional manifold3d package is not installed",
    ),
]


def _cube(offset):
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
        )
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
        vertices + np.asarray(offset, dtype=float),
        faces,
        SurfaceMetadata(
            source_id=str(offset),
            source_revision="0",
            coordinate_contract=SpatialCoordinateContract.si(),
            provenance=("qualification",),
        ),
    )


@pytest.mark.parametrize(
    "operation, volume",
    [
        (SurfaceBooleanOperation.UNION, 1.5),
        (SurfaceBooleanOperation.DIFFERENCE, 0.5),
        (SurfaceBooleanOperation.INTERSECTION, 0.5),
    ],
)
def test_boolean_preserves_expected_solid_volume(operation, volume):
    result = ManifoldProvider().execute(_cube((0, 0, 0)), _cube((0.5, 0, 0)), operation)
    faces = np.asarray(result.mesh.blocks[0].vertices)
    points = np.asarray(result.mesh.coordinates)[faces]
    signed_volume = (
        np.sum(np.sum(points[:, 0] * np.cross(points[:, 1], points[:, 2]), axis=1)) / 6
    )
    assert signed_volume == pytest.approx(volume)
    assert result.audit.passed
    assert result.boundary is not None


def test_empty_intersection_is_not_a_successful_mesh():
    with pytest.raises(MeshingFailure, match="empty"):
        ManifoldProvider().execute(
            _cube((0, 0, 0)), _cube((2, 0, 0)), SurfaceBooleanOperation.INTERSECTION
        )
