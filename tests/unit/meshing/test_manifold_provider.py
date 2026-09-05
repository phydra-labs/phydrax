import manifold3d
import numpy as np
import pytest

from phydrax import SpatialCoordinateContract
from phydrax.geometry.surface import SurfaceMetadata, SurfaceModel
from phydrax.meshing import MeshingFailure
from phydrax.meshing.providers._manifold import ManifoldProvider, SurfaceBooleanOperation


def _cube(offset):
    arrays = manifold3d.Manifold.cube().translate(offset).to_mesh64()
    return SurfaceModel.from_triangles(
        arrays.vert_properties[:, :3],
        arrays.tri_verts,
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
