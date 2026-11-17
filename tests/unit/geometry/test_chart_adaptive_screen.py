"""Adaptive sphere-chart decisions against the original exact target route."""

from fractions import Fraction

import numpy as np
import pytest

from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization import lagrange_element
from phydrax.discretization._cell_geometry import (
    CellGeometrySpec,
    ScalarCoordinateElement,
)
from phydrax.discretization._cell_geometry_validity import cell_geometry_id
from phydrax.discretization._cell_mesh import CellBlock, CellMesh
from phydrax.discretization._coordinate_enclosure import (
    compose_expression,
    coordinate_source_basis,
    evaluate_expression,
    Polynomial,
    polynomial_bounds,
    scale,
    subtract,
    sum_polynomials,
)
from phydrax.geometry._domain_boundary import MeshingDomainBoundarySource
from phydrax.geometry._meshing_domain import MeshingDomain, SourcePreparation
from phydrax.geometry._source_boundary_records import SourceBoundaryChartWitness
from phydrax.geometry._source_fidelity import _affine_triangle_arguments, _box_gap_upper
from phydrax.geometry._source_interpolation import (
    _exact_component_norm,
    _target_endpoint_upper,
    _TargetChartBasis,
    certify_domain_surface_charts,
    SourceChartFidelity,
)
from phydrax.geometry.brep import brep_sphere
from phydrax.linalg._hermitian_spectral import fraction_sqrt_interval


_REFERENCE = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))


def _domain() -> MeshingDomain:
    return MeshingDomain.from_brep(
        brep_sphere(
            1.0,
            coordinate_contract=SpatialCoordinateContract.si(),
            source_id="adaptive-screen-sphere",
        )
    )


def _case(
    domain: MeshingDomain,
    charts: np.ndarray,
    order: np.ndarray,
    *,
    degree: int = 2,
) -> tuple[
    CellMesh, CellGeometrySpec, MeshingDomainBoundarySource, SourceBoundaryChartWitness
]:
    count = len(charts)
    source = MeshingDomainBoundarySource(domain, (0,))
    preparation = SourcePreparation(domain)
    points = charts.reshape(-1, 2)
    lower, upper = preparation.derivative_program(domain.patches[0].surface, 0).evaluate(
        points, points
    )
    vertices = (np.asarray(lower) + np.asarray(upper)) / 2
    ids = np.arange(101, 101 + count, dtype=np.int64)
    mesh = CellMesh(
        vertices,
        (
            CellBlock(
                "surface",
                "triangle",
                np.arange(3 * count).reshape(count, 3),
                global_ids=ids,
            ),
        ),
    )
    element = lagrange_element("triangle", degree)
    nodes = np.asarray(element.reference_nodes, dtype=np.float64)
    coordinates = []
    for row in range(count):
        p = vertices[3 * row : 3 * row + 3]
        values = p[0] + nodes[:, :1] * (p[1] - p[0]) + nodes[:, 1:] * (p[2] - p[0])
        # A non-affine target exercises the exact composition and residual budget.
        values[:, 2] += 0.015625 * nodes[:, 0] * nodes[:, 1]
        coordinates.extend(values)
    geometry = CellGeometrySpec(
        {"surface": element},
        {"surface": np.arange(count * len(nodes)).reshape(count, len(nodes))},
        np.asarray(coordinates),
    )
    entity = domain.entity_id(2, 0)
    witness = SourceBoundaryChartWitness(
        mesh_id=mesh.mesh_id,
        geometry_id=cell_geometry_id(geometry),
        topology_id=mesh.topology_id,
        source_id=source.source_id,
        source_revision=source.source_revision,
        source_query_id=source.source_scope_id,
        chart_dimension=2,
        reference_dimension=2,
        cell_global_ids=ids[order],
        entity_global_ids=ids[order],
        local_entity_indices=np.full(len(order), -1, dtype=np.int32),
        target_reference_vertices=np.tile(_REFERENCE[None], (len(order), 1, 1)),
        source_chart_ids=(entity,) * len(order),
        source_entity_ids=(entity,) * len(order),
        occurrence_paths=((),) * len(order),
        source_parameters=charts[order],
        orientations=np.ones(len(order), dtype=np.int8),
    )
    return mesh, geometry, source, witness


def _oracle(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    source: MeshingDomainBoundarySource,
    witness: SourceBoundaryChartWitness,
    rows: np.ndarray,
    tolerance: float,
    evidence: SourceChartFidelity,
) -> tuple[bool, ...]:
    """Reconstruct target polynomials before using the exact adaptive budget."""
    domain = source.domain
    preparation = SourcePreparation(domain)
    elements, routes, _ = geometry.resolve(mesh)
    coefficients = geometry.source_coordinates()
    program = preparation.derivative_program(domain.patches[0].surface, 0)
    limit = Fraction.from_float(tolerance)
    decisions = []
    for witness_row, cell_row in enumerate(rows):
        reference = (
            (Fraction(0), Fraction(0)),
            (Fraction(1), Fraction(0)),
            (Fraction(0), Fraction(1)),
        )
        arguments = _affine_triangle_arguments(reference)
        local = tuple(coefficients[int(i)] for i in np.asarray(routes[0])[cell_row])
        assert isinstance(elements[0], ScalarCoordinateElement)
        basis = coordinate_source_basis(elements[0], cell_row=int(cell_row))
        physical = tuple(
            sum_polynomials(
                tuple(
                    scale(function, values[axis])
                    for function, values in zip(basis, local, strict=True)
                ),
                2,
            )
            for axis in range(3)
        )
        mapped = tuple(compose_expression(component, arguments) for component in physical)
        corners = tuple(
            tuple(evaluate_expression(component, p) for component in mapped)
            for p in reference
        )
        assert evidence.target_corners[witness_row] == corners
        assert evidence.target_affine[witness_row] == all(
            isinstance(expression, dict) and all(sum(power) <= 1 for power in expression)
            for expression in mapped
        )
        components = []
        for axis, expression in enumerate(mapped):
            affine: Polynomial = {
                (0, 0): corners[0][axis],
                (1, 0): corners[1][axis] - corners[0][axis],
                (0, 1): corners[2][axis] - corners[0][axis],
            }
            residual = subtract(expression, affine)
            if not residual:
                continue
            lo, hi = polynomial_bounds(residual, "simplex", 2)
            components.append(Fraction.from_float(max(abs(lo), abs(hi))))
        target = _exact_component_norm(tuple(components))
        chart = tuple(
            (Fraction(float(p[0])), Fraction(float(p[1])))
            for p in witness.source_parameter_chart(witness_row)
        )
        points = np.asarray(chart, dtype=np.float64)
        lower, upper = program.evaluate(points, points)
        endpoint = max(
            _exact_component_norm(
                tuple(
                    _box_gap_upper(
                        float(lower[i, axis]), float(upper[i, axis]), corners[i][axis]
                    )
                    for axis in range(3)
                )
            )
            for i in range(3)
        )
        (bound,) = domain.adaptive_interpolation_bounds(
            0, (chart,), (limit - endpoint - target,), preparation=preparation
        )
        assert bound is not None
        decisions.append(bound + endpoint + target <= limit)
    return tuple(decisions)


def _decisions(
    domain: MeshingDomain,
    charts: np.ndarray,
    order: np.ndarray,
    tolerance: float,
    *,
    degree: int = 2,
) -> tuple[tuple[bool, ...], tuple[bool, ...]]:
    mesh, geometry, source, witness = _case(domain, charts, order, degree=degree)
    rows = np.asarray(witness.cell_global_ids, dtype=np.int64) - 101
    result = certify_domain_surface_charts(
        mesh,
        geometry,
        source,
        witness,
        np.zeros(len(rows), dtype=np.int64),
        tuple((0, int(row)) for row in rows),
        tolerance=tolerance,
    )
    unresolved = {
        int(f.entity_ids[0])
        for f in result.findings
        if f.check == "source_fidelity_tolerance"
    }
    assert all(f.check == "source_fidelity_tolerance" for f in result.findings)
    actual = tuple(int(cell_id) not in unresolved for cell_id in witness.cell_global_ids)
    return actual, _oracle(mesh, geometry, source, witness, rows, tolerance, result)


def test_random_sphere_charts_match_exact_budget_oracle() -> None:
    domain = _domain()
    rng = np.random.default_rng(48291)
    origins = rng.uniform(0.1, 0.8, (5, 2))
    sizes = rng.uniform(0.08, 0.3, (5, 2))
    charts = np.stack(
        (origins, origins + sizes * (1, 0), origins + sizes * (0, 1)), axis=1
    )
    order = np.arange(len(charts))
    for tolerance in (0.001, 0.01, 0.1):
        actual, expected = _decisions(domain, charts, order, tolerance)
        assert actual == expected


def test_chart_certification_matches_generic_affine_quadratic_and_cubic_maps() -> None:
    domain = _domain()
    charts = np.asarray(
        (
            ((0.2, 0.1), (0.45, 0.1), (0.2, 0.35)),
            ((0.5, 0.3), (0.625, 0.3), (0.5, 0.425)),
        ),
        dtype=np.float64,
    )
    for degree in (1, 2, 3):
        for tolerance in (0.001, 0.02, 0.2):
            actual, expected = _decisions(
                domain, charts, np.arange(len(charts)), tolerance, degree=degree
            )
            assert actual == expected


def test_adaptive_chart_decisions_are_batch_independent() -> None:
    domain = _domain()
    charts = np.asarray(
        (
            ((0.0, 0.0), (0.25, 0.0), (0.0, 0.25)),
            ((0.5, 0.25), (0.625, 0.25), (0.5, 0.375)),
            ((0.75, 0.5), (1.0, 0.5), (0.75, 0.75)),
        )
    )
    tolerance = 0.02
    order = np.asarray((2, 0, 1))
    batch, oracle = _decisions(domain, charts, order, tolerance)
    assert batch == oracle

    def evidence(selected: np.ndarray) -> SourceChartFidelity:
        mesh, geometry, source, witness = _case(domain, charts, selected)
        rows = np.asarray(witness.cell_global_ids, dtype=np.int64) - 101
        return certify_domain_surface_charts(
            mesh,
            geometry,
            source,
            witness,
            np.zeros(len(rows), dtype=np.int64),
            tuple((0, int(row)) for row in rows),
            tolerance=tolerance,
        )

    batch_evidence = evidence(order)
    _, _, _, batch_witness = _case(domain, charts, order)
    batch_ids = tuple(int(value) for value in batch_witness.cell_global_ids)
    assert len(set(batch_evidence.deviation_bounds[:, 1].tolist())) > 1
    for row in order:
        position = batch_ids.index(101 + int(row))
        single, expected = _decisions(domain, charts, np.asarray((row,)), tolerance)
        assert single == expected == (batch[position],)
        one = evidence(np.asarray((row,)))
        np.testing.assert_array_equal(
            batch_evidence.deviation_bounds[position].view(np.uint64),
            one.deviation_bounds[0].view(np.uint64),
        )


def test_target_basis_batch_preserves_rows_and_certification() -> None:
    domain = _domain()
    charts = np.asarray(
        (
            ((0.0, 0.0), (0.25, 0.0), (0.0, 0.25)),
            ((0.5, 0.25), (0.625, 0.25), (0.5, 0.375)),
            ((0.75, 0.5), (1.0, 0.5), (0.75, 0.75)),
        )
    )
    order = np.asarray((2, 0, 1))
    mesh, geometry, source, witness = _case(domain, charts, order)
    elements, routes, _ = geometry.resolve(mesh)
    assert isinstance(elements[0], ScalarCoordinateElement)
    basis = _TargetChartBasis(
        elements[0],
        0,
        _affine_triangle_arguments(
            (
                (Fraction(0), Fraction(0)),
                (Fraction(1), Fraction(0)),
                (Fraction(0), Fraction(1)),
            )
        ),
    )
    if basis.prepared.lower is None:
        pytest.skip("Native interval basis bounds are unavailable")
    coordinates = geometry.source_coordinates()
    locals_ = tuple(
        tuple(coordinates[int(index)] for index in np.asarray(routes[0])[row])
        for row in order
    )
    affine = tuple((Fraction(0), Fraction(0), Fraction(0)) for _ in locals_[0])
    mixed = (locals_[0], (), affine, locals_[1], locals_[2])
    singles = tuple(basis.screen_batch((local,))[0] for local in mixed)
    assert singles[1] is None
    assert singles[2] is not None and singles[2][1]
    assert basis.screen_batch(mixed) == singles


def test_natural_near_threshold_sphere_chart_matches_exact_route() -> None:
    domain = _domain()
    charts = np.asarray((((0.0, 0.0), (0.25, 0.0), (0.0, 0.25)),))
    order = np.asarray((0,))
    low, high = 0.0, 1.0
    for _ in range(10):
        tolerance = (low + high) / 2
        actual, expected = _decisions(domain, charts, order, tolerance)
        assert actual == expected
        if expected[0]:
            high = tolerance
        else:
            low = tolerance
    lower, lower_oracle = _decisions(domain, charts, order, low)
    upper, upper_oracle = _decisions(domain, charts, order, high)
    assert lower == lower_oracle == (False,)
    assert upper == upper_oracle == (True,)


def test_endpoint_norm_screen_encloses_exact_and_catches_inward_fault() -> None:
    rng = np.random.default_rng(1937)
    for _ in range(64):
        target = rng.uniform(-10.0, 10.0, (3, 3))
        first = rng.uniform(-10.0, 10.0, (3, 3))
        second = rng.uniform(-10.0, 10.0, (3, 3))
        lower, upper = np.minimum(first, second), np.maximum(first, second)
        corners = tuple(
            tuple(Fraction(float(value)) for value in point) for point in target
        )
        bound = _target_endpoint_upper(corners, lower, upper)
        assert bound is not None
        exact = max(
            fraction_sqrt_interval(
                sum(
                    (
                        max(
                            abs(corners[row][axis] - Fraction(float(lower[row, axis]))),
                            abs(corners[row][axis] - Fraction(float(upper[row, axis]))),
                        )
                        ** 2
                        for axis in range(3)
                    ),
                    Fraction(0),
                )
            )[1]
            for row in range(3)
        )
        assert bound >= exact
        assert bound / 2 < exact
