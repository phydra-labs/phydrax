from __future__ import annotations

import random
from fractions import Fraction

import numpy as np
import pytest

from phydrax.discretization import (
    CellBlock,
    CellGeometrySpec,
    CellMesh,
)
from phydrax.discretization._cell_geometry import coordinate_effective_basis
from phydrax.discretization._coordinate_enclosure import (
    add,
    constant,
    evaluate_expression,
    scale,
)
from phydrax.discretization._exact_algebraic_geometry import ExactScalar, RadicalElement
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.discretization.fem._reference import (
    lagrange_element,
    lagrange_reference_node_data,
)
from phydrax.geometry._embedding_cells import prepare_embedding_cells
from phydrax.meshing import audit_cell_mesh
from phydrax.meshing._curving import _exact_coordinate_bank_source_map


def _geometry(
    degree: int, rows: tuple[tuple[tuple[ExactScalar, ...], ...], ...]
) -> tuple[CellMesh, CellGeometrySpec]:
    element = lagrange_element("triangle", degree)
    basis = coordinate_effective_basis(element)
    references = tuple(
        tuple(Fraction(float(value)) for value in point)
        for point in reference_cell_topology("triangle").vertices
    )
    vertices = []
    for local in rows:
        for point in references:
            vertices.append(
                tuple(
                    sum(
                        (
                            evaluate_expression(function, point) * coefficient[axis]
                            for function, coefficient in zip(basis, local, strict=True)
                        ),
                        Fraction(0),
                    )
                    for axis in range(2)
                )
            )
    execution_vertices = np.asarray(
        [[float(value) for value in row] for row in vertices], dtype=np.float64
    )
    mesh = CellMesh(
        execution_vertices,
        (CellBlock("triangles", "triangle", np.arange(3 * len(rows)).reshape(-1, 3)),),
    )
    bank = tuple(value for local in rows for value in local)
    execution = np.asarray([[float(value) for value in row] for row in bank])
    source = _exact_coordinate_bank_source_map(
        mesh, "integer-map-test", "integer-map-test-r1", ("triangles",), bank, execution
    )
    routes = np.arange(len(bank)).reshape(len(rows), len(rows[0]))
    geometry = CellGeometrySpec(
        {"triangles": element},
        {"triangles": routes},
        execution,
        source_map=source,
        source_mesh=mesh,
    )
    return mesh, geometry


def _audit_geometry(
    degree: int, quadratic: Fraction, cubic: Fraction
) -> tuple[CellMesh, CellGeometrySpec]:
    # A degree-two representation also exercises an exactly affine map through
    # the mapped-measure audit route rather than the vertex-only quality path.
    nodes = lagrange_reference_node_data(lagrange_element("triangle", degree))
    rows = []
    for shift in (0, 3):
        local = []
        for node in nodes:
            assert node.exact_point is not None
            x, y = node.exact_point
            local.append((x + quadratic * x * x + cubic * x * x * x + shift, y))
        rows.append(tuple(local))
    return _geometry(degree, tuple(rows))


@pytest.mark.parametrize(
    ("degree", "quadratic", "cubic"),
    [
        (2, Fraction(0), Fraction(0)),
        (2, Fraction(1, 5), Fraction(0)),
        (3, Fraction(1, 5), Fraction(1, 7)),
    ],
    ids=("affine", "quadratic", "cubic"),
)
def test_audit_measures_match_exact_integration(
    degree: int, quadratic: Fraction, cubic: Fraction
) -> None:
    mesh, geometry = _audit_geometry(degree, quadratic, cubic)
    actual = audit_cell_mesh(mesh, geometry)
    reference = audit_cell_mesh(mesh, geometry)
    assert actual.validity.all_certified
    assert actual.quality.evaluation.measure_exact is True
    expected = float(Fraction(1, 2) + quadratic / 3 + cubic / 4)
    np.testing.assert_allclose(
        actual.quality.evaluation.measures, [expected, expected], rtol=1e-6
    )
    np.testing.assert_array_equal(
        actual.quality.evaluation.measures, reference.quality.evaluation.measures
    )
    np.testing.assert_array_equal(
        actual.quality.evaluation.measure_absolute_errors,
        reference.quality.evaluation.measure_absolute_errors,
    )
    assert (
        actual.quality.evaluation.measure_geometry_id
        == reference.quality.evaluation.measure_geometry_id
    )
    assert (
        actual.quality.evaluation.evaluation_id
        == reference.quality.evaluation.evaluation_id
    )
    assert actual.quality.report_id == reference.quality.report_id
    assert actual.report_id == reference.report_id
    assert actual.issues == reference.issues


def test_audit_measures_with_one_out_of_range_rational_cell() -> None:
    nodes = lagrange_reference_node_data(lagrange_element("triangle", 2))
    rows = []
    quadratic_over_width = Fraction((1 << 5000) + 1, 1 << 5000)
    for shift, quadratic in ((0, quadratic_over_width), (3, Fraction(1, 5))):
        local = []
        for node in nodes:
            assert node.exact_point is not None
            x, y = node.exact_point
            local.append((x + quadratic * x * x + shift, y))
        rows.append(tuple(local))
    mesh, geometry = _geometry(2, tuple(rows))
    actual = audit_cell_mesh(mesh, geometry)
    assert actual.validity.all_certified
    assert actual.quality.evaluation.measure_exact is True
    expected = np.asarray(
        [
            float(Fraction(1, 2) + quadratic_over_width / 3),
            float(Fraction(17, 30)),
        ],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(
        np.asarray(actual.quality.evaluation.measures).view(np.uint64),
        expected.view(np.uint64),
    )


def test_audit_measures_change_with_coordinate_input() -> None:
    baseline_mesh, baseline_geometry = _audit_geometry(2, Fraction(1, 5), Fraction(0))
    altered_mesh, altered_geometry = _audit_geometry(2, Fraction(2, 5), Fraction(0))
    baseline = audit_cell_mesh(baseline_mesh, baseline_geometry)
    altered = audit_cell_mesh(altered_mesh, altered_geometry)
    np.testing.assert_allclose(
        baseline.quality.evaluation.measures,
        [float(Fraction(17, 30))] * 2,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        altered.quality.evaluation.measures,
        [float(Fraction(19, 30))] * 2,
        rtol=1e-6,
    )
    assert (
        altered.quality.evaluation.evaluation_id
        != baseline.quality.evaluation.evaluation_id
    )


@pytest.mark.parametrize("degree", (1, 2, 3))
@pytest.mark.parametrize("radical", (False, True))
def test_embedding_expressions_equal_generic_exact_basis(
    degree: int, radical: bool
) -> None:
    rng = random.Random(4100 + degree)
    element = lagrange_element("triangle", degree)
    basis = coordinate_effective_basis(element)
    root = RadicalElement.sqrt(Fraction(5))
    rows = []
    for cell in range(3):
        local = []
        for _ in basis:
            point = []
            for axis in range(2):
                value: ExactScalar = Fraction(
                    rng.randint(-9, 9), rng.choice((3, 5, 7, 11))
                )
                if radical:
                    value += Fraction(rng.randint(1, 4), 13) * root
                if axis == 0:
                    value += 4 * cell
                point.append(value)
            local.append(tuple(point))
        rows.append(tuple(local))
    mesh, geometry = _geometry(degree, tuple(rows))
    cells = prepare_embedding_cells(mesh, geometry)
    for cell, local in zip(cells, rows, strict=True):
        for axis, actual in enumerate(cell.coordinates):
            expected = constant(0, 2)
            for function, point in zip(basis, local, strict=True):
                expected = add(expected, scale(function, point[axis]))
            assert actual == expected
