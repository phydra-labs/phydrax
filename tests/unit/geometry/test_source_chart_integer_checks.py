"""Integer chart validation agrees with independent rational definitions."""

import random
from fractions import Fraction

import numpy as np
import pytest

from phydrax.discretization._cell_geometry import CellGeometrySpec
from phydrax.discretization._cell_geometry_validity import cell_geometry_id
from phydrax.discretization._cell_mesh import CellBlock, CellMesh
from phydrax.geometry._source_boundary_records import (
    _chart_tokens_in_parent_triangle,
    _chart_tokens_nondegenerate,
    _validated_rational_chart_tokens,
    SourceBoundaryChartWitness,
)
from phydrax.geometry._source_fidelity import _surface_target_chart_rows


def test_chart_integer_determinants_match_fraction_reference() -> None:
    rng = random.Random(719)
    for corners, dimension in ((2, 1), (2, 2), (2, 3), (3, 2), (3, 3)):
        for row in range(600):
            points = tuple(
                tuple(
                    Fraction(rng.randrange(-100, 101), rng.randrange(1, 101))
                    for _ in range(dimension)
                )
                for _ in range(corners)
            )
            if row % 7 == 0:
                points = (points[0],) * corners
            tokens = tuple(
                tuple((value.numerator, value.denominator) for value in point)
                for point in points
            )
            if corners == 2:
                expected = points[0] != points[1]
            else:
                first = tuple(
                    points[1][axis] - points[0][axis] for axis in range(dimension)
                )
                second = tuple(
                    points[2][axis] - points[0][axis] for axis in range(dimension)
                )
                expected = any(
                    first[left] * second[right] != first[right] * second[left]
                    for left in range(dimension)
                    for right in range(left + 1, dimension)
                )
            assert _chart_tokens_nondegenerate(tokens, dimension) == expected


def test_chart_integer_determinant_keeps_unbounded_near_ties() -> None:
    large = 2**1500
    points = (
        (Fraction(0), Fraction(0)),
        (Fraction(large, 3), Fraction(1, 5)),
        (Fraction(2 * large, 3), Fraction(2, 5) + Fraction(1, large)),
    )
    tokens = tuple(
        tuple((value.numerator, value.denominator) for value in point) for point in points
    )
    determinant = (points[1][0] - points[0][0]) * (points[2][1] - points[0][1]) - (
        points[1][1] - points[0][1]
    ) * (points[2][0] - points[0][0])
    assert determinant != 0
    assert _chart_tokens_nondegenerate(tokens, 2)


def test_parent_triangle_integer_test_matches_fraction_reference() -> None:
    rng = random.Random(831)
    for _ in range(600):
        points = tuple(
            tuple(
                Fraction(rng.randrange(-10, 21), rng.randrange(1, 11)) for _ in range(2)
            )
            for _ in range(3)
        )
        tokens = tuple(
            tuple((value.numerator, value.denominator) for value in point)
            for point in points
        )
        expected = all(x >= 0 and y >= 0 and x + y <= 1 for x, y in points)
        assert _chart_tokens_in_parent_triangle(tokens) == expected


@pytest.mark.parametrize("row", (0, 1, 2))
@pytest.mark.parametrize("target", (False, True))
@pytest.mark.parametrize(
    ("defect", "message"),
    [
        ("nonreduced", "exact coordinate differs from its binary64 realization"),
        ("denominator", "denominators must be positive"),
        ("mismatch", "exact coordinate differs from its binary64 realization"),
        ("degenerate", "contains a degenerate exact chart piece"),
    ],
)
def test_witness_refuses_one_bad_chart_in_row_order(
    row: int, target: bool, defect: str, message: str
) -> None:
    triangles = np.tile(
        np.asarray([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]], dtype=np.float64),
        (3, 1, 1),
    )
    target_charts = triangles.copy()
    source_charts = triangles.copy()
    token_row = (((0, 1), (0, 1)), ((1, 1), (0, 1)), ((0, 1), (1, 1)))
    rows = [[list(point) for point in token_row] for _ in range(3)]
    if defect == "nonreduced":
        rows[row][0][0] = (0, 2)
    elif defect == "denominator":
        rows[row][0][0] = (0, 0)
    elif defect == "mismatch":
        rows[row][0][0] = (1, 1)
    else:
        rows[row] = [list(token_row[0]) for _ in range(3)]
        (target_charts if target else source_charts)[row] = 0.0
    tokens = tuple(tuple(tuple(point) for point in chart) for chart in rows)
    ids = np.arange(3, dtype=np.int64)
    with pytest.raises(ValueError, match=message):
        SourceBoundaryChartWitness(
            mesh_id="mesh",
            geometry_id="geometry",
            topology_id="topology",
            source_id="source",
            source_revision="revision",
            source_query_id="query",
            chart_dimension=2,
            reference_dimension=2,
            cell_global_ids=ids,
            entity_global_ids=ids,
            local_entity_indices=np.full(3, -1, dtype=np.int32),
            target_reference_vertices=target_charts,
            target_reference_rational_tokens=tokens if target else None,
            source_chart_ids=("chart",) * 3,
            source_entity_ids=("face",) * 3,
            occurrence_paths=((),) * 3,
            source_parameters=source_charts,
            source_parameter_rational_tokens=None if target else tokens,
            orientations=np.ones(3, dtype=np.int8),
        )


@pytest.mark.parametrize("dimension", (2, 3))
def test_distinct_collinear_rational_charts_are_degenerate(dimension: int) -> None:
    first = (Fraction(0),) * dimension
    direction = (Fraction(1, 2), Fraction(1, 3), Fraction(1, 5))[:dimension]
    points = (first, direction, tuple(2 * value for value in direction))
    tokens = tuple(
        tuple((value.numerator, value.denominator) for value in point) for point in points
    )
    assert len(set(points)) == 3
    for left in range(dimension):
        for right in range(left + 1, dimension):
            assert direction[left] * points[2][right] != 0
            assert direction[right] * points[2][left] != 0
            assert (
                direction[left] * points[2][right] - direction[right] * points[2][left]
            ) == 0
    assert not _chart_tokens_nondegenerate(tokens, dimension)


@pytest.mark.parametrize("component", (False, np.bool_(False)))
def test_boolean_token_components_are_refused_by_validator(
    component: bool | np.bool_,
) -> None:
    chart = np.asarray([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]], dtype=np.float64)
    tokens = ((((component, 1), (0, 1)), ((1, 1), (0, 1)), ((0, 1), (1, 1))),)
    with pytest.raises(
        TypeError, match="coordinates must be integer numerator/denominator pairs"
    ):
        _validated_rational_chart_tokens(tokens, chart, 1, 3, 2, "chart")  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("component", (False, np.bool_(False)))
def test_boolean_token_components_are_refused_by_witness(
    component: bool | np.bool_,
) -> None:
    chart = np.asarray([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]], dtype=np.float64)
    tokens = ((((component, 1), (0, 1)), ((1, 1), (0, 1)), ((0, 1), (1, 1))),)
    with pytest.raises(
        TypeError, match="coordinates must be integer numerator/denominator pairs"
    ):
        SourceBoundaryChartWitness(
            mesh_id="mesh",
            geometry_id="geometry",
            topology_id="topology",
            source_id="source",
            source_revision="revision",
            source_query_id="query",
            chart_dimension=2,
            reference_dimension=2,
            cell_global_ids=np.asarray((0,), dtype=np.int64),
            entity_global_ids=np.asarray((0,), dtype=np.int64),
            local_entity_indices=np.asarray((-1,), dtype=np.int32),
            target_reference_vertices=chart,
            target_reference_rational_tokens=tokens,  # ty: ignore[invalid-argument-type]
            source_chart_ids=("chart",),
            source_entity_ids=("face",),
            occurrence_paths=((),),
            source_parameters=chart,
            orientations=np.asarray((1,), dtype=np.int8),
        )


def test_numpy_integer_tokens_are_stored_as_builtin_integers() -> None:
    chart = np.asarray([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]], dtype=np.float64)
    tokens = (
        (
            ((np.int64(0), np.int64(1)), (np.int64(0), np.int64(1))),
            ((np.int64(1), np.int64(1)), (np.int64(0), np.int64(1))),
            ((np.int64(0), np.int64(1)), (np.int64(1), np.int64(1))),
        ),
    )
    validated = _validated_rational_chart_tokens(tokens, chart, 1, 3, 2, "chart")  # ty: ignore[invalid-argument-type]
    assert all(
        type(value) is int
        for row in validated
        for point in row
        for pair in point
        for value in pair
    )
    witness = SourceBoundaryChartWitness(
        mesh_id="mesh",
        geometry_id="geometry",
        topology_id="topology",
        source_id="source",
        source_revision="revision",
        source_query_id="query",
        chart_dimension=2,
        reference_dimension=2,
        cell_global_ids=np.asarray((0,), dtype=np.int64),
        entity_global_ids=np.asarray((0,), dtype=np.int64),
        local_entity_indices=np.asarray((-1,), dtype=np.int32),
        target_reference_vertices=chart,
        target_reference_rational_tokens=tokens,  # ty: ignore[invalid-argument-type]
        source_chart_ids=("chart",),
        source_entity_ids=("face",),
        occurrence_paths=((),),
        source_parameters=chart,
        orientations=np.asarray((1,), dtype=np.int8),
    )
    assert all(
        type(value) is int
        for row in witness.target_reference_rational_tokens
        for point in row
        for pair in point
        for value in pair
    )


def test_target_chart_orientations_distinguish_opposite_shared_edge() -> None:
    points = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=np.float64,
    )
    mesh = CellMesh(
        points,
        (
            CellBlock(
                "surface",
                "triangle",
                np.asarray(((0, 1, 2),), dtype=np.int64),
                global_ids=np.asarray((31,), dtype=np.int64),
            ),
        ),
    )
    geometry = CellGeometrySpec.affine(mesh)
    charts = np.asarray(
        (
            ((0.5, 0.0), (0.0, 0.5), (0.0, 0.0)),
            ((0.5, 0.0), (0.0, 0.5), (0.5, 0.5)),
            ((0.5, 0.0), (1.0, 0.0), (0.5, 0.5)),
            ((0.0, 0.5), (0.5, 0.5), (0.0, 1.0)),
        ),
        dtype=np.float64,
    )
    witness = SourceBoundaryChartWitness(
        mesh_id=mesh.mesh_id,
        geometry_id=cell_geometry_id(geometry),
        topology_id=mesh.topology_id,
        source_id="source",
        source_revision="revision",
        source_query_id="query",
        chart_dimension=2,
        reference_dimension=2,
        cell_global_ids=np.full(4, 31, dtype=np.int64),
        entity_global_ids=np.full(4, 31, dtype=np.int64),
        local_entity_indices=np.full(4, -1, dtype=np.int32),
        target_reference_vertices=charts,
        source_chart_ids=("chart",) * 4,
        source_entity_ids=("face",) * 4,
        occurrence_paths=((),) * 4,
        source_parameters=charts,
        orientations=np.ones(4, dtype=np.int8),
    )
    _, reference, orientations = _surface_target_chart_rows(
        mesh, geometry, None, witness, full_source=True
    )
    by_triangle = dict(zip(reference, orientations, strict=True))
    edge = ((Fraction(1, 2), Fraction(0)), (Fraction(0), Fraction(1, 2)))
    assert by_triangle[(*edge, (Fraction(0), Fraction(0)))] == 1
    assert by_triangle[(*edge, (Fraction(1, 2), Fraction(1, 2)))] == -1
