from importlib.util import find_spec
from typing import Any

import numpy as np
import pytest

from phydrax.geometry.surface import SurfaceBooleanOperation
from phydrax.meshing import MeshingFailure
from phydrax.meshing.providers import ManifoldProvider
from tests._support.optional_surfaces import boolean_cube


# Historical absolute affine-interpolation acceptance bound:
# test_manifold_provider.py at merge base d2dfcb941, unchanged here.
_AFFINE_INTERPOLATION_ATOL = 1.0e-12
# Manifold's published relative barycentric-transfer acceptance bound.
_PROPERTY_TRANSFER_RESIDUAL_LIMIT = 1.0e-8


pytestmark = [
    pytest.mark.meshing_manifold,
    pytest.mark.skipif(
        find_spec("manifold3d") is None,
        reason="optional manifold3d package is not installed",
    ),
]


@pytest.mark.parametrize(
    ("operation", "volume"),
    [
        pytest.param(SurfaceBooleanOperation.UNION, 1.5, id="union"),
        pytest.param(SurfaceBooleanOperation.DIFFERENCE, 0.5, id="difference"),
        pytest.param(SurfaceBooleanOperation.INTERSECTION, 0.5, id="intersection"),
    ],
)
def test_manifold_binary_boolean_volume(
    operation: SurfaceBooleanOperation,
    volume: float,
) -> None:
    result = ManifoldProvider().execute(
        (boolean_cube((0, 0, 0)), boolean_cube((0.5, 0, 0))), operation
    )
    assert _signed_volume(result) == pytest.approx(volume)
    assert result.audit.passed
    assert result.boundary is not None


def test_manifold_empty_intersection_is_refused() -> None:
    with pytest.raises(MeshingFailure, match="empty"):
        ManifoldProvider().execute(
            (boolean_cube((0, 0, 0)), boolean_cube((2, 0, 0))),
            SurfaceBooleanOperation.INTERSECTION,
        )


@pytest.mark.parametrize(
    ("operation", "volume"),
    [
        pytest.param(SurfaceBooleanOperation.UNION, 1.75, id="union"),
        pytest.param(SurfaceBooleanOperation.DIFFERENCE, 0.5, id="difference"),
        pytest.param(SurfaceBooleanOperation.INTERSECTION, 0.25, id="intersection"),
    ],
)
def test_manifold_nary_boolean_volume(
    operation: SurfaceBooleanOperation,
    volume: float,
) -> None:
    operands = (
        boolean_cube((0, 0, 0)),
        boolean_cube((0.5, 0, 0)),
        boolean_cube((0.75, 0, 0)),
    )
    result = ManifoldProvider().execute(operands, operation)
    assert _signed_volume(result) == pytest.approx(volume)
    assert result.audit.passed


def _signed_volume(result: Any) -> Any:
    faces = np.asarray(result.mesh.blocks[0].vertices)
    points = np.asarray(result.mesh.coordinates)[faces]
    return np.sum(np.sum(points[:, 0] * np.cross(points[:, 1], points[:, 2]), axis=1)) / 6


def test_vertex_properties_contracts() -> None:
    left, right = boolean_cube((0, 0, 0)), boolean_cube((0.5, 0.25, 0.25))
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
    np.testing.assert_allclose(
        np.asarray(attribute.values),
        expected,
        atol=_AFFINE_INTERPOLATION_ATOL,
    )
    assert set(source.tolist()) == {0, 1}
    achieved = dict(result.compliance.achieved)
    residual = "vertex_property:temperature:maximum_relative_interpolation_residual"
    assert achieved[residual] <= _PROPERTY_TRANSFER_RESIDUAL_LIMIT


def test_vertex_property_requires_one_array_per_operand() -> None:
    left, right = boolean_cube((0, 0, 0)), boolean_cube((0.5, 0, 0))
    with pytest.raises(ValueError, match="one array per operand"):
        ManifoldProvider().execute(
            (left, right),
            SurfaceBooleanOperation.UNION,
            vertex_properties={"temperature": (np.zeros(8),)},
        )
