"""Cylinder projection preserves closest-point status, ownership and query charges."""

from __future__ import annotations

import numpy as np
import pytest

import phydrax as phx
from phydrax.geometry.brep import BRepQueryBudget
from phydrax.geometry.brep._patches import CylinderPatch, PlanePatch
from phydrax.geometry.brep._query_projection import NativeBRepProjection
from tests._support.cad_curving import CURVING_COORDINATE_CONTRACT


@pytest.fixture(scope="module")
def cylinder_projection_query() -> tuple[NativeBRepProjection, int, int]:

    model = phx.geometry.brep_cylinder(
        1.0,
        2.0,
        coordinate_contract=CURVING_COORDINATE_CONTRACT,
        tessellation=phx.geometry.BRepTessellationPolicy(realize=False),
    )
    projection = phx.geometry.prepare_brep_projection(model)
    wall = next(
        index
        for index, patch in enumerate(model.patches)
        if isinstance(patch, CylinderPatch)
    )
    bottom = next(
        index
        for index, patch in enumerate(model.patches)
        if isinstance(patch, PlanePatch) and np.asarray(patch.origin)[2] == 0.0
    )
    return projection, wall, bottom


def test_complete_cylinder_wall_closest_is_certified_without_refinement(
    cylinder_projection_query: tuple[NativeBRepProjection, int, int],
) -> None:
    projection, wall, _ = cylinder_projection_query
    result = projection.query.closest_point(
        np.asarray([[2.0, 0.0, 1.0]], dtype=np.float64), faces=(wall,)
    )
    np.testing.assert_array_equal(result.points, [[1.0, 0.0, 1.0]])
    assert result.distance_lower_bounds[0] <= 1.0 <= result.distances[0]
    assert not result.unresolved[0]
    assert result.status[0] == phx.geometry.BRepProjectionStatus.SEAM
    assert result.refinements[0] == 0
    assert result.point_error_bounds[0] <= projection.query.tolerances.distance
    assert result.parameter_bounds[0] <= projection.query.tolerances.parametric


def test_cylinder_axis_fiber_keeps_face_without_selecting_a_point(
    cylinder_projection_query: tuple[NativeBRepProjection, int, int],
) -> None:
    projection, wall, _ = cylinder_projection_query
    result = projection.project(
        np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64),
        np.asarray([2], dtype=np.int8),
        np.asarray([wall], dtype=np.int32),
    )
    assert result.status[0] == phx.geometry.BRepProjectionStatus.NON_ISOLATED
    assert result.dimensions[0] == 2 and result.indices[0] == wall
    assert np.isfinite(result.residuals[0]) and result.residuals[0] >= 1.0
    assert np.all(np.isnan(result.points[0]))
    assert np.isnan(result.parameters[0, 0])
    assert not result.resolved[0]


def test_circular_cap_closest_preserves_unique_center_and_rim_ownership(
    cylinder_projection_query: tuple[NativeBRepProjection, int, int],
) -> None:
    projection, _, bottom = cylinder_projection_query
    result = projection.query.closest_point(
        np.asarray([[0.0, 0.0, 1.0], [2.0, 0.0, 0.0]], dtype=np.float64),
        faces=(bottom,),
    )
    np.testing.assert_array_equal(result.points, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    assert np.all(np.asarray(result.distance_lower_bounds) <= 1.0)
    assert np.all(np.asarray(result.distances) >= 1.0)
    assert not np.any(result.unresolved)
    assert result.status[0] == phx.geometry.BRepProjectionStatus.UNIQUE
    assert result.status[1] == phx.geometry.BRepProjectionStatus.SEAM
    assert result.edge_indices[0] == -1 and result.edge_indices[1] >= 0
    assert np.isfinite(result.edge_parameters[1])
    assert np.all(np.asarray(result.refinements) == 0)


def test_analytic_closest_counts_input_points_once_across_faces(
    cylinder_projection_query: tuple[NativeBRepProjection, int, int],
) -> None:

    projection, _, _ = cylinder_projection_query
    budget = BRepQueryBudget(64, maximum_points=2, maximum_scratch_bytes=1 << 24)
    result = projection.query.closest_point(
        np.asarray([[2.0, 0.0, 1.0], [0.0, 0.0, 0.0]], dtype=np.float64),
        budget=budget,
    )
    assert not np.any(result.unresolved)
    assert budget.points == 2
    assert budget.operations == int(np.sum(np.asarray(result.query_operations))) == 6
    assert budget.subdivisions == 0
