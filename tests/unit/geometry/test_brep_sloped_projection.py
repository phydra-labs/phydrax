"""Public projection contracts for an imported face with sloped trim edges."""

from pathlib import Path

import numpy as np
import pytest

from phydrax._external_resource import ResourceLimits
from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry.brep import (
    BRepModel,
    BRepProjectionStatus,
    BRepSurfaceQueryResult,
    BRepTessellationPolicy,
    prepare_brep_projection,
    prepare_brep_query,
)
from phydrax.interchange import CadImportPolicy, read_brep_text


# Source fixture exported from the equilateral triangle in the mixed-planar
# provider regression. Its literal line-parameter endpoints differ at junctions;
# the native reader retains those coefficients rather than closing them by distance.
_SOURCE = Path(__file__).parents[2] / "data" / "cad" / "sloped_triangle.brep"
_POINTS = np.asarray(((0.3, np.sqrt(3.0) * 0.1, 0.0), (0.2, 0.1, 0.0)), dtype=np.float64)
# Unit-scale point evaluation uses fewer than 64 rounded elementary terms.
# This bound does not include uncertainty in the retained source-root enclosure.
_ROUNDING_BOUND = float(64 * np.finfo(np.float64).eps)


@pytest.fixture(scope="module")
def sloped_face() -> BRepModel:
    contract = SpatialCoordinateContract.si()
    policy = CadImportPolicy(
        contract,
        ResourceLimits(4194304, 64, 20000, 2000000, 16),
        tessellation=BRepTessellationPolicy(realize=False),
    )
    return read_brep_text(
        _SOURCE,
        policy,
        trusted_root=_SOURCE.parent,
        source_length_unit=contract.length_unit,
    ).model


@pytest.fixture(scope="module")
def sloped_closest(sloped_face: BRepModel) -> BRepSurfaceQueryResult:
    return prepare_brep_query(sloped_face).closest_point(_POINTS)


def test_imported_sloped_face_closest_points_resolve_interior_roots(
    sloped_closest: BRepSurfaceQueryResult,
) -> None:
    result = sloped_closest

    np.testing.assert_array_equal(
        result.status, np.full((2,), BRepProjectionStatus.UNIQUE, dtype=np.int8)
    )
    assert not np.any(np.asarray(result.unresolved))
    assert np.all(np.isfinite(np.asarray(result.point_error_bounds)))
    np.testing.assert_allclose(result.points, _POINTS, rtol=0.0, atol=_ROUNDING_BOUND)
    assert np.all(np.asarray(result.distance_lower_bounds) <= _ROUNDING_BOUND)
    evaluated_distance = np.linalg.norm(np.asarray(result.points) - _POINTS, axis=1)
    upper = np.asarray(result.distances)
    radius = np.asarray(result.point_error_bounds)
    assert np.all(evaluated_distance <= _ROUNDING_BOUND)
    assert np.all(upper >= evaluated_distance)
    # The source root is in the returned point's certified error ball. The
    # triangle inequality separates that radius from point-evaluation rounding.
    assert np.all(upper - evaluated_distance <= radius + _ROUNDING_BOUND)


def test_imported_sloped_face_owned_projection_resolves_cell_interiors(
    sloped_face: BRepModel,
    sloped_closest: BRepSurfaceQueryResult,
) -> None:
    result = prepare_brep_projection(sloped_face).project(
        _POINTS,
        np.full((2,), 2, dtype=np.int8),
        np.zeros((2,), dtype=np.int32),
    )

    np.testing.assert_array_equal(
        result.status, np.full((2,), BRepProjectionStatus.UNIQUE, dtype=np.int8)
    )
    assert np.all(np.asarray(result.resolved))
    np.testing.assert_allclose(result.points, _POINTS, rtol=0.0, atol=_ROUNDING_BOUND)
    evaluated_distance = np.linalg.norm(np.asarray(result.points) - _POINTS, axis=1)
    upper = np.asarray(result.residuals)
    radius = np.asarray(sloped_closest.point_error_bounds)
    assert np.all(evaluated_distance <= _ROUNDING_BOUND)
    assert np.all(upper >= evaluated_distance)
    # Native owned-face projection uses the same deterministic source-root
    # certificate; its residual retains the error-ball distance upper bound.
    assert np.all(upper - evaluated_distance <= radius + _ROUNDING_BOUND)
    np.testing.assert_array_equal(result.dimensions, np.full((2,), 2, dtype=np.int8))
    np.testing.assert_array_equal(result.indices, np.zeros((2,), dtype=np.int32))
