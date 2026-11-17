from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from phydrax._meshcore import meshcore_available
from phydrax.discretization import CellBlock, CellGeometrySpec, CellMesh
from phydrax.discretization._cell_geometry import PolynomialComposedCellGeometryElement
from phydrax.discretization._coordinate_enclosure import BernsteinDomain, Polynomial
from phydrax.discretization._exact_coordinate_bank import ExactCoordinateBank
from phydrax.discretization._mapped_geometry_measure import exact_block_cell_measures
from phydrax.discretization.fem import lagrange_element
from phydrax.geometry._domain_certification import _target_region_measures
from phydrax.geometry._domain_preparation import prepare_domain_coverage
from phydrax.meshing._curving import _exact_coordinate_bank_source_map
from tests.unit.discretization.test_mapped_geometry_measure import (
    _polynomial_measure_batch,
)
from tests.unit.geometry.test_domain_coverage import _split_square


def _batch(kind: str, degree: int) -> tuple[CellMesh, CellGeometrySpec, Fraction]:
    source = lagrange_element(kind, 1)
    nodes = np.asarray(source.reference_nodes, dtype=np.float64)
    dimension = nodes.shape[1]
    axis = tuple(int(index == 0) for index in range(dimension))
    power = (degree,) + (0,) * (dimension - 1)
    first = {axis: Fraction(1)}
    first[power] = first.get(power, Fraction(0)) + Fraction(1, 2)
    chart = (first,) + tuple(
        {tuple(int(index == component) for index in range(dimension)): Fraction(1)}
        for component in range(1, dimension)
    )
    element = PolynomialComposedCellGeometryElement(source, kind, chart)
    width = len(nodes)
    vertex_routes = np.arange(4 * width, dtype=np.int64).reshape(4, width)
    mesh = CellMesh(
        np.tile(nodes, (4, 1)),
        (
            CellBlock("first", kind, vertex_routes[:2], global_ids=np.asarray((11, 12))),
            CellBlock("second", kind, vertex_routes[2:], global_ids=np.asarray((13, 14))),
        ),
    )
    coordinates = np.concatenate(
        [nodes * np.asarray((scale,) + (1,) * (dimension - 1)) for scale in (1, 2, 3, 4)]
    )
    geometry = CellGeometrySpec(
        {"first": element, "second": element},
        {"first": vertex_routes[:2], "second": vertex_routes[2:]},
        coordinates,
    )
    if kind in ("triangle", "tetrahedron"):
        unit = Fraction(1, math.factorial(dimension)) + Fraction(
            math.factorial(degree), 2 * math.factorial(degree - 1 + dimension)
        )
    else:
        unit = Fraction(3, 2)
    return mesh, geometry, unit


def _coverage_measures(
    mesh: CellMesh, geometry: CellGeometrySpec, measures: dict[int, Fraction] | None
) -> tuple[object, ...]:
    domain, _, _, _ = _split_square()
    preparation = prepare_domain_coverage(mesh, geometry, np.asarray((0, 1, 0, 1)), 2)
    findings = []
    result = _target_region_measures(
        mesh, geometry, domain, preparation, findings, cell_measures=measures
    )
    assert not findings
    return result


@pytest.mark.meshcore
@pytest.mark.skipif(not meshcore_available(), reason="meshcore unavailable")
@pytest.mark.parametrize(
    "kind", ("triangle", "quadrilateral", "tetrahedron", "hexahedron")
)
@pytest.mark.parametrize("degree", (1, 2, 3))
def test_block_basis_measures_match_coverage_symbolic_integrals(
    kind: str, degree: int
) -> None:
    mesh, geometry, unit = _batch(kind, degree)
    measures = exact_block_cell_measures(mesh, geometry)
    assert measures == {row: (row + 1) * unit for row in range(4)}
    assert _coverage_measures(mesh, geometry, measures) == (4 * unit, 6 * unit)
    assert _coverage_measures(mesh, geometry, None) == (4 * unit, 6 * unit)


@pytest.mark.meshcore
@pytest.mark.skipif(not meshcore_available(), reason="meshcore unavailable")
def test_distinct_per_cell_charts_match_coverage_integrals() -> None:
    mesh, geometry, expected = _polynomial_measure_batch("triangle", 3, 823)
    measures = exact_block_cell_measures(mesh, geometry)
    assert measures == dict(enumerate(expected))
    assert _coverage_measures(mesh, geometry, measures) == (
        expected[0] + expected[2],
        expected[1] + expected[3],
    )
    assert _coverage_measures(mesh, geometry, None) == (
        expected[0] + expected[2],
        expected[1] + expected[3],
    )


def test_embedded_maps_are_not_reported_as_signed_volume() -> None:
    source = lagrange_element("triangle", 1)
    nodes = np.asarray(source.reference_nodes, dtype=np.float64)
    points = np.column_stack((nodes, np.zeros(len(nodes), dtype=np.float64)))
    route = np.arange(len(nodes), dtype=np.int64)[None, :]
    mesh = CellMesh(points, (CellBlock("surface", "triangle", route),))
    geometry = CellGeometrySpec({"surface": source}, {"surface": route}, points)
    assert exact_block_cell_measures(mesh, geometry) == {}


@pytest.mark.meshcore
@pytest.mark.skipif(not meshcore_available(), reason="meshcore unavailable")
def test_sparse_rational_bank_and_item_local_range_refusal_reach_coverage() -> None:
    mesh, geometry, unit = _batch("triangle", 2)
    elements, routes, _ = geometry.resolve(mesh)
    nodes = np.asarray(lagrange_element("triangle", 1).reference_nodes, dtype=np.float64)
    scales = (
        Fraction(1, 3),
        Fraction(1) + Fraction(1, 1 << 6000),
        Fraction(3),
        Fraction(4),
    )
    exact = tuple(
        (scale * Fraction(float(point[0])), Fraction(float(point[1])))
        for scale in scales
        for point in nodes
    )
    execution = np.asarray([[float(value) for value in row] for row in exact])
    source = _exact_coordinate_bank_source_map(
        mesh, "sparse-basis-test", "revision", (), exact, execution
    )
    geometry = CellGeometrySpec(
        {"first": elements[0], "second": elements[1]},
        {"first": routes[0], "second": routes[1]},
        execution,
        source_map=source,
        source_mesh=mesh,
    )
    measures = exact_block_cell_measures(mesh, geometry)
    assert measures == {0: unit / 3, 2: 3 * unit, 3: 4 * unit}
    expected = ((Fraction(1, 3) + 3) * unit, (scales[1] + 4) * unit)
    assert _coverage_measures(mesh, geometry, measures) == expected
    assert _coverage_measures(mesh, geometry, None) == expected


@pytest.mark.meshcore
@pytest.mark.skipif(not meshcore_available(), reason="meshcore unavailable")
def test_coverage_retains_zero_and_negative_signed_measures() -> None:
    mesh, original, unit = _batch("triangle", 2)
    elements, routes, _ = original.resolve(mesh)
    nodes = np.asarray(lagrange_element("triangle", 1).reference_nodes, dtype=np.float64)
    coordinates = np.concatenate(
        [nodes * np.asarray((scale, 1)) for scale in (1, -2, 0, 4)]
    )
    geometry = CellGeometrySpec(
        {"first": elements[0], "second": elements[1]},
        {"first": routes[0], "second": routes[1]},
        coordinates,
    )
    measures = exact_block_cell_measures(mesh, geometry)
    assert measures == {0: unit, 1: -2 * unit, 2: Fraction(0), 3: 4 * unit}
    domain, _, _, _ = _split_square()
    preparation = prepare_domain_coverage(mesh, geometry, np.asarray((0, 1, 0, 1)), 2)
    findings = []
    assert _target_region_measures(
        mesh, geometry, domain, preparation, findings, cell_measures=measures
    ) == (unit, 2 * unit)
    assert [(finding.check, finding.entity_ids) for finding in findings] == [
        ("target_cell_orientation", (12,)),
        ("target_cell_orientation", (13,)),
    ]


@pytest.mark.meshcore
@pytest.mark.skipif(not meshcore_available(), reason="meshcore unavailable")
def test_coverage_falls_back_for_one_native_refusal_and_detects_one_numerator_fault(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import phydrax.discretization._mapped_geometry_measure as supplier

    mesh, geometry, unit = _batch("triangle", 3)
    native = supplier.exact_basis_jacobian_integrals

    def refuse_one(
        basis: tuple[Polynomial, ...],
        bank: ExactCoordinateBank,
        routes: np.ndarray,
        domain: BernsteinDomain,
        dimension: int,
        /,
        *,
        parallelism: int | None = None,
    ) -> list[Fraction | None]:
        values = native(basis, bank, routes, domain, dimension, parallelism=parallelism)
        values[1] = None
        return values

    monkeypatch.setattr(supplier, "exact_basis_jacobian_integrals", refuse_one)
    measures = exact_block_cell_measures(mesh, geometry)
    assert measures == {0: unit, 2: 3 * unit}
    assert _coverage_measures(mesh, geometry, measures) == (4 * unit, 6 * unit)
    corrupted_once = False

    def corrupt_one(
        basis: tuple[Polynomial, ...],
        bank: ExactCoordinateBank,
        routes: np.ndarray,
        domain: BernsteinDomain,
        dimension: int,
        /,
        *,
        parallelism: int | None = None,
    ) -> list[Fraction | None]:
        nonlocal corrupted_once
        values = native(basis, bank, routes, domain, dimension, parallelism=parallelism)
        value = values[0]
        if value is not None and not corrupted_once:
            values[0] = Fraction(value.numerator + 1, value.denominator)
            corrupted_once = True
        return values

    monkeypatch.setattr(supplier, "exact_basis_jacobian_integrals", corrupt_one)
    corrupted = exact_block_cell_measures(mesh, geometry)
    assert _coverage_measures(mesh, geometry, corrupted) != (4 * unit, 6 * unit)
