from importlib.util import find_spec
from typing import Any

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


def _cube(offset: Any) -> Any:
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
def test_boolean_preserves_expected_solid_volume(operation: Any, volume: Any) -> None:
    result = ManifoldProvider().execute((_cube((0, 0, 0)), _cube((0.5, 0, 0))), operation)
    faces = np.asarray(result.mesh.blocks[0].vertices)
    points = np.asarray(result.mesh.coordinates)[faces]
    signed_volume = (
        np.sum(np.sum(points[:, 0] * np.cross(points[:, 1], points[:, 2]), axis=1)) / 6
    )
    assert signed_volume == pytest.approx(volume)
    assert result.audit.passed
    assert result.boundary is not None


def test_empty_intersection_is_not_a_successful_mesh() -> None:
    with pytest.raises(MeshingFailure, match="empty"):
        ManifoldProvider().execute(
            (_cube((0, 0, 0)), _cube((2, 0, 0))), SurfaceBooleanOperation.INTERSECTION
        )


def _signed_volume(result: Any) -> Any:
    faces = np.asarray(result.mesh.blocks[0].vertices)
    points = np.asarray(result.mesh.coordinates)[faces]
    return np.sum(np.sum(points[:, 0] * np.cross(points[:, 1], points[:, 2]), axis=1)) / 6


@pytest.mark.parametrize(
    "operation, volume",
    [
        (SurfaceBooleanOperation.UNION, 1.75),
        (SurfaceBooleanOperation.DIFFERENCE, 0.5),
        (SurfaceBooleanOperation.INTERSECTION, 0.25),
    ],
)
def test_nary_boolean_combines_every_operand(operation: Any, volume: Any) -> None:
    operands = (_cube((0, 0, 0)), _cube((0.5, 0, 0)), _cube((0.75, 0, 0)))

    result = ManifoldProvider().execute(operands, operation)

    assert _signed_volume(result) == pytest.approx(volume)
    assert result.audit.passed


def test_vertex_properties_transfer_linearly_on_their_source_faces() -> None:
    left, right = _cube((0, 0, 0)), _cube((0.5, 0.25, 0.25))
    # Affine fields are reproduced exactly by barycentric transfer, and differ
    # between operands so cut-curve corners must keep their own face's value.
    left_values = 2.0 * np.asarray(left.mesh.coordinates) @ np.asarray((1.0, -1.0, 0.5))
    right_values = np.asarray(right.mesh.coordinates) @ np.asarray((0.0, 3.0, 1.0)) - 7.0

    result = ManifoldProvider().execute(
        (left, right),
        SurfaceBooleanOperation.UNION,
        vertex_properties={"temperature": (left_values, right_values)},
    )

    (attribute,) = result.attributes
    # ty: ignore[unresolved-attribute]
    faces = np.asarray(result.boundary.mesh.blocks[0].vertices)
    # ty: ignore[unresolved-attribute]
    face_ids = np.asarray(result.boundary.mesh.blocks[0].global_ids)
    # ty: ignore[unresolved-attribute]
    corners = np.asarray(result.boundary.mesh.coordinates)[faces][
        np.argsort(face_ids, kind="stable")
    ]
    sources = dict(
        zip(
            np.asarray(result.associations[0].target_global_ids).tolist(),
            [0] * len(result.associations[0].target_global_ids),
            strict=True,
        )
    )
    sources.update(
        zip(
            np.asarray(result.associations[1].target_global_ids).tolist(),
            [1] * len(result.associations[1].target_global_ids),
            strict=True,
        )
    )
    source = np.asarray([sources[identifier] for identifier in np.sort(face_ids)])
    expected = np.where(
        source[:, None] == 0,
        2.0 * corners @ np.asarray((1.0, -1.0, 0.5)),
        corners @ np.asarray((0.0, 3.0, 1.0)) - 7.0,
    )

    assert attribute.name == "face_corner_temperature"
    np.testing.assert_allclose(np.asarray(attribute.values), expected, atol=1e-12)
    assert set(source.tolist()) == {0, 1}
    achieved = dict(result.compliance.achieved)
    residual = "vertex_property:temperature:maximum_relative_interpolation_residual"
    assert achieved[residual] <= 1e-8


def test_vertex_properties_require_one_array_per_operand() -> None:
    left, right = _cube((0, 0, 0)), _cube((0.5, 0, 0))
    with pytest.raises(ValueError, match="one array per operand"):
        ManifoldProvider().execute(
            (left, right),
            SurfaceBooleanOperation.UNION,
            vertex_properties={"temperature": (np.zeros(8),)},
        )
