# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""White-box mapped-density integration and whole-map correspondence contracts."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager, ExitStack
from fractions import Fraction
from itertools import permutations
from typing import NamedTuple

import numpy as np
import pytest

from phydrax._meshcore import meshcore_available, NativeExecutionScope
from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization import (
    _coordinate_enclosure as enclosure,
    CellBlock,
    CellGeometrySpec,
    CellMesh,
)
from phydrax.discretization._cell_geometry import (
    coordinate_lagrange_element,
    PolynomialComposedCellGeometryElement,
    RestrictedCellGeometryElement,
    ScalarCoordinateElement,
)
from phydrax.discretization._cell_geometry_transfer import plc_nested_approximation
from phydrax.discretization._cell_geometry_transition_types import (
    NestedReferenceWitnesses,
)
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from phydrax.discretization._coordinate_enclosure import (
    CoordinateCoefficients,
    evaluate,
    Expression,
    multiply,
    Polynomial,
)
from phydrax.discretization._coordinate_operands import (
    _signed_determinant_operand_batch,
    _signed_determinant_operands_for,
    CoordinateOperandCollector,
    PreparedCoordinateOperands,
)
from phydrax.discretization._mapped_geometry_measure import (
    affine_nested_geometry_pair_bound,
    certified_sqrt_polynomial_integral_state,
    certify_nested_geometry_pairs,
    mapped_density_expression,
    polynomial_nested_geometry_pair_bound,
    reuse_embedded_polynomial_area,
)
from phydrax.discretization._nested_reference_types import NestedReferencePair
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.discretization._reference_cell_types import CellGeometryElement
from phydrax.discretization.fem._reference_types import FiniteElementSpec
from phydrax.geometry._embedding_state import MeshCertificateLimits
from phydrax.meshing import MeshingLimits
from phydrax.meshing._measurements import NativeExecutionRecord
from phydrax.meshing._native_preparation import native_preparation_ledger
from phydrax.meshing._result import CellMeshingResult
from tests._support.cell_meshes import kuhn_cube
from tests._support.coordinate_references import coordinate_reference_budget


# Preserve the unit-scale solver accuracy allocation (2**20 binary64 eps) with
# its stricter decimal threshold; the integer/Fraction formulas are the oracle.
_SOLVE_ACCURACY = 1e-10
# This is the requested relative moment enclosure, not an empirical tolerance.
_MOMENT_RELATIVE_ERROR = 1e-11


@pytest.mark.parametrize(
    ("kind", "expected"), (("triangle", (409, 400)), ("quadrilateral", (1059, 400)))
)
def test_high_degree_sqrt_density_preserves_weighted_reference_moments(
    kind: str,
    expected: tuple[int, int],
    coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget],
) -> None:

    root: Polynomial = {
        (0, 0): Fraction(1),
        (3, 0): Fraction(1, 10),
        (0, 3): Fraction(1, 10),
    }
    weight: Polynomial = {(0, 0): Fraction(1), (1, 0): Fraction(1), (0, 1): Fraction(2)}
    ledger = coordinate_budget(1_000_000, 16 * 1024**2)
    with ledger.activate():
        value, error = certified_sqrt_polynomial_integral_state(
            multiply(root, root, budget=ledger),
            weight,
            kind,
            Fraction(1, 10**11),
            _MOMENT_RELATIVE_ERROR,
            10000,
            32,
            [0, 1_000_000],
            budget=ledger,
        )
    assert abs(Fraction(value) - Fraction(*expected)) <= Fraction(error)
    assert error <= _MOMENT_RELATIVE_ERROR * float(Fraction(*expected))


@pytest.mark.parametrize("increment", ((1, 10**11), (1, 2)))
def test_area_reuse_encloses_changed_gram_or_integrates_it_fresh(
    increment: tuple[int, int],
    coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget],
) -> None:

    delta = Fraction(*increment)
    root: Polynomial = {(0, 0): Fraction(2), (1, 0): delta}
    reference: tuple[Polynomial, Fraction, float, float] = (
        {(0, 0): Fraction(4)},
        Fraction(4),
        1.0,
        0.0,
    )
    ledger = coordinate_budget(1_000_000, 16 * 1024**2)
    with ledger.activate():
        (value, error), prepared = reuse_embedded_polynomial_area(
            multiply(root, root, budget=ledger),
            "triangle",
            [reference],
            Fraction(1, 10**10),
            _SOLVE_ACCURACY,
            10000,
            32,
            [0, 1_000_000],
            budget=ledger,
        )
    assert abs(Fraction(value) - (Fraction(1) + delta / 6)) <= Fraction(error)
    assert error <= max(_SOLVE_ACCURACY, _SOLVE_ACCURACY * (value - error))
    if delta < Fraction(1, 1000):
        assert value == 1.0
        assert error > 0.0
        assert prepared is None
    else:
        assert value > 1.08
        assert prepared is not None


def test_polynomial_square_preserves_cancelled_consumer_values_under_work_cap(
    budget: CoordinateEnclosureBudget,
    coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget],
) -> None:

    polynomial: Polynomial = {
        (a, b): Fraction((-1) ** (a + b) * (a + 1), 2 ** (a + b + 3))
        for a in range(11)
        for b in range(11 - a)
    }
    ledger = coordinate_budget(2800, 16 * 1024**2)
    with ledger.activate():
        squared = multiply(polynomial, polynomial, budget=ledger)
    for point in (
        (Fraction(0), Fraction(0)),
        (Fraction(1, 3), Fraction(-2, 5)),
        (Fraction(7, 6), Fraction(2, 3)),
    ):
        assert (
            evaluate(squared, point, budget=budget)
            == evaluate(polynomial, point, budget=budget) ** 2
        )
    cancellation: Polynomial = {
        (0, 0): Fraction(1),
        (1, 0): Fraction(-2),
        (2, 0): Fraction(1),
    }
    squared = multiply(cancellation, cancellation, budget=budget)
    assert evaluate(squared, (Fraction(1), Fraction(0)), budget=budget) == 0
    assert evaluate(squared, (Fraction(3), Fraction(0)), budget=budget) == 16


@pytest.mark.parametrize("coarsening", (False, True))
def test_affine_correspondence_bounds_corner_difference(
    coarsening: bool, coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget]
) -> None:

    source = kuhn_cube(1, orientation_swap=(0, 1))
    coordinates = np.asarray(source.coordinates).copy()
    coordinates[0, 0] += 0.125
    target = source.with_coordinates(coordinates, numeric_version="perturbed-map")
    ids = np.concatenate([np.asarray(block.global_ids) for block in source.blocks])
    reference = np.concatenate((np.zeros((1, 3)), np.eye(3)), axis=0)
    witness = NestedReferenceWitnesses(
        ids,
        ids,
        np.broadcast_to(reference, (ids.size, 4, 3)).copy(),
    )
    empty = NestedReferenceWitnesses(
        np.zeros(0, dtype=np.int64),
        np.zeros(0, dtype=np.int64),
        np.zeros((0, 4, 3), dtype=np.float64),
    )
    ledger = coordinate_budget(100_000, 16 * 1024**2)
    with ledger.activate():
        bounds = plc_nested_approximation(
            source,
            CellGeometrySpec.affine(source),
            target,
            CellGeometrySpec.affine(target),
            empty if coarsening else witness,
            witness if coarsening else empty,
            budget=ledger,
        )
    assert bounds == ((0.0, 0.125) if coarsening else (0.125, 0.0))
    assert ledger.work_units > 0


def test_constant_embedded_area_work_is_independent_of_prior_candidates(
    coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget],
) -> None:

    def measure(candidate_count: int) -> tuple[tuple[float, float], int]:
        ledger = coordinate_budget(1_000_000, 64 * 1024**2)
        work = [0, 1_000_000]
        with ledger.activate():
            _, reference = reuse_embedded_polynomial_area(
                {(0, 0): Fraction(1), (2, 0): Fraction(1, 4)},
                "triangle",
                [],
                Fraction(1, 1000),
                0.0,
                1000,
                32,
                work,
                budget=ledger,
            )
            if reference is None:
                raise RuntimeError(
                    "measure_fixture.reference must retain positive varying-Gram evidence; got None."
                )
            starting_work = ledger.work_units
            result, _ = reuse_embedded_polynomial_area(
                {(0, 0): Fraction(2)},
                "triangle",
                [reference] * candidate_count,
                Fraction(1, 10**10),
                0.0,
                1000,
                32,
                work,
                budget=ledger,
            )
        return result, ledger.work_units - starting_work

    empty, empty_work = measure(0)
    crowded, crowded_work = measure(100)
    assert crowded == empty
    assert crowded_work == empty_work > 0
    value, error = crowded
    assert abs(value - np.sqrt(2.0) / 2.0) <= error


@pytest.mark.parametrize("kind", ("interval", "triangle", "tetrahedron"))
@pytest.mark.parametrize(
    ("defect", "accepted"),
    ((0.0, True), (2.0**-49, True), (2.0**-40, False)),
    ids=("exact", "within-rounding-bound", "different-physical-map"),
)
def test_affine_nested_corner_proof_matches_polynomial_bound_and_status(
    kind: str, defect: float, accepted: bool
) -> None:

    element = coordinate_lagrange_element(kind, 1)
    nodes = np.asarray(element.reference_nodes, dtype=np.float64)
    corners = np.asarray(reference_cell_topology(kind).vertices, dtype=np.float64)
    dimension = corners.shape[1]
    matrix = 0.5 * np.eye(dimension, dtype=np.float64)
    if dimension > 1:
        matrix[0, 1] = 0.125
    offset = np.full((dimension,), 0.125, dtype=np.float64)
    action = np.diag(np.arange(1, dimension + 1, dtype=np.float64))
    origin = np.ones((dimension,), dtype=np.float64)

    def physical(reference: np.ndarray, *, fine: bool) -> np.ndarray:
        mapped = reference @ matrix.T + offset if fine else reference
        values = mapped @ action.T + origin
        if fine:
            values[:, 0] += defect * (1.0 + 2.0 * reference[:, 0])
        return values

    row = np.arange(corners.shape[0], dtype=np.int32)[None]
    coarse_mesh = CellMesh(
        physical(corners, fine=False),
        (CellBlock("cells", kind, row),),
    )
    fine_mesh = CellMesh(
        physical(corners, fine=True),
        (CellBlock("cells", kind, row),),
    )
    route = np.arange(nodes.shape[0], dtype=np.int32)[None]
    coarse_values = physical(nodes, fine=False)
    fine_values = physical(nodes, fine=True)
    coarse_geometry = CellGeometrySpec(
        {"cells": element}, {"cells": route}, coarse_values
    )
    fine_geometry = CellGeometrySpec({"cells": element}, {"cells": route}, fine_values)
    polynomial_geometry = CellGeometrySpec(
        {
            "cells": RestrictedCellGeometryElement(
                element,
                kind,
                np.eye(dimension, dtype=np.float64),
                np.zeros((dimension,), dtype=np.float64),
            )
        },
        {"cells": route},
        fine_values,
    )
    pair = NestedReferencePair(0, 0, False, matrix, offset)
    with coordinate_reference_budget() as ledger:
        affine_bound = affine_nested_geometry_pair_bound(
            element, fine_values, element, coarse_values, pair, budget=ledger
        )
        polynomial_bound = polynomial_nested_geometry_pair_bound(
            element, fine_values, element, coarse_values, pair, budget=ledger
        )
        assert affine_bound is not None and affine_bound == polynomial_bound
        if accepted:
            assert (
                certify_nested_geometry_pairs(
                    coarse_mesh,
                    coarse_geometry,
                    fine_mesh,
                    fine_geometry,
                    (pair,),
                    budget=ledger,
                )
                == polynomial_bound
            )
            assert (
                certify_nested_geometry_pairs(
                    coarse_mesh,
                    coarse_geometry,
                    fine_mesh,
                    polynomial_geometry,
                    (pair,),
                    budget=ledger,
                )
                == polynomial_bound
            )
        else:
            with pytest.raises(ValueError, match="nested_geometry.physical_map"):
                certify_nested_geometry_pairs(
                    coarse_mesh,
                    coarse_geometry,
                    fine_mesh,
                    fine_geometry,
                    (pair,),
                    budget=ledger,
                )
            with pytest.raises(ValueError, match="nested_geometry.physical_map"):
                certify_nested_geometry_pairs(
                    coarse_mesh,
                    coarse_geometry,
                    fine_mesh,
                    polynomial_geometry,
                    (pair,),
                    budget=ledger,
                )


def _expanded_nested_pair_bound(
    fine: ScalarCoordinateElement,
    fine_bank: CoordinateCoefficients,
    coarse: ScalarCoordinateElement,
    coarse_bank: CoordinateCoefficients,
    pair: NestedReferencePair,
    *,
    budget: CoordinateEnclosureBudget,
) -> float:
    # Expand every individual composed basis function before contracting the
    # physical bank. This oracle does not use the coordinate-map pullback cache
    # or the definitional pair-law shortcut.
    def expanded(
        element: ScalarCoordinateElement, bank: CoordinateCoefficients
    ) -> tuple[Polynomial, ...]:
        basis = enclosure.source_basis(element, budget=budget)
        assert basis is not None
        rows = enclosure.coordinate_source_rows(bank)
        return enclosure.linear_combinations(
            basis, tuple(zip(*rows, strict=True)), budget=budget
        )

    actual = expanded(fine, fine_bank)
    expected = enclosure.restrict_chart_expressions(
        expanded(coarse, coarse_bank),
        coarse.cell_kind,
        fine.cell_kind,
        np.asarray(pair.exact_offset, dtype=object)
        if pair.exact_offset is not None else pair.offset,
        np.asarray(pair.exact_matrix, dtype=object)
        if pair.exact_matrix is not None else pair.matrix,
        budget=budget,
    )
    bound = 0.0
    domain = (
        "simplex" if fine.cell_kind in ("interval", "triangle", "tetrahedron")
        else "prism" if fine.cell_kind == "prism" else "box"
    )
    for left, right in zip(actual, expected, strict=True):
        difference = enclosure.expression_add(
            left, enclosure.expression_scale(right, -1, budget=budget), budget=budget
        )
        if difference:
            lower, upper = enclosure.expression_bounds(
                difference, domain, fine.topological_dimension, budget=budget
            )
            bound = max(bound, abs(lower), abs(upper))
    return bound


@pytest.mark.parametrize("source_law", ("nonlinear", "noncanonical", "nonunit"))
def test_composed_nested_pair_definition_matches_full_basis_expansion(
    source_law: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from phydrax.discretization import _mapped_geometry_measure as measure
    from phydrax.discretization._cell_geometry import BarycentricCellGeometryElement
    from phydrax.discretization._nested_reference_types import exact_reference_pair

    kind = "triangle" if source_law == "nonunit" else "quadrilateral"
    chart = coordinate_lagrange_element(kind, 2)
    root = coordinate_lagrange_element(kind, 1 if source_law == "nonunit" else 2)
    nodes = tuple(tuple(Fraction(float(v)) for v in row) for row in chart.reference_nodes)
    source_nodes = tuple(
        tuple(Fraction(float(v)) for v in row) for row in root.reference_nodes
    )
    # The nonlinear physical source folds on its full reference cell. Equality
    # of restrictions must not require a global parent positivity certificate.
    bank = tuple((x * x - x, y + y * y / 8) for x, y in source_nodes)
    pair = exact_reference_pair(
        0, 0, False,
        ((Fraction(1, 3), Fraction(1, 12)), (Fraction(0), Fraction(1, 4))),
        (Fraction(1, 8), Fraction(1, 8)),
    )
    points = tuple(
        (Fraction(1, 8) + x / 3 + y / 12, Fraction(1, 8) + y / 4)
        for x, y in nodes
    )
    with coordinate_reference_budget() as preparation:
        source: ScalarCoordinateElement = root
        if source_law == "noncanonical":
            source = _polynomial_chart(
                root, chart, tuple((x / 2 + x * x / 16, y / 2) for x, y in nodes),
                budget=preparation,
            )
        elif source_law == "nonunit":
            weights = np.eye(root.local_dof_count)
            weights[0, 0] = 2
            weights[1, 0] = Fraction(1, 4)
            source = BarycentricCellGeometryElement(root, weights)
            bank = tuple((1 + x, 2 + y) for x, y in source_nodes)
        fine = _polynomial_chart(source, chart, points, budget=preparation)
    with coordinate_reference_budget() as independent:
        assert _expanded_nested_pair_bound(
            fine, bank, source, bank, pair, budget=independent
        ) == 0

    def no_expansion(*args: object, **kwargs: object) -> None:
        pytest.fail("definitional pair equality must not expand physical maps")

    monkeypatch.setattr(measure, "mapped_coordinate_expressions", no_expansion)
    with coordinate_reference_budget() as ledger:
        assert polynomial_nested_geometry_pair_bound(
            fine, bank, source, bank, pair, budget=ledger
        ) == 0
        assert not ledger.coordinate_cache
        assert not ledger.density_cache


@pytest.mark.parametrize("mutation", ("bank", "chart", "pair", "nonlinear_chart"))
def test_composed_nested_pair_reauthenticates_full_current_operands(
    mutation: str,
) -> None:
    source = coordinate_lagrange_element("quadrilateral", 2)
    nodes = tuple(tuple(Fraction(float(v)) for v in row) for row in source.reference_nodes)
    bank = np.asarray([(float(x + x * x / 8), float(y)) for x, y in nodes])
    pair = NestedReferencePair(0, 0, False, np.eye(2) / 2, np.zeros(2))
    with coordinate_reference_budget() as preparation:
        fine = _polynomial_chart(
            source, source, tuple((x / 2, y / 2) for x, y in nodes),
            budget=preparation,
        )
        changed = _polynomial_chart(
            source, source,
            tuple((x / 2 + x * (1 - x) / 8, y / 2) for x, y in nodes),
            budget=preparation,
        )
    fine_bank = bank.copy()
    with coordinate_reference_budget() as ledger:
        assert polynomial_nested_geometry_pair_bound(
            fine, fine_bank, source, bank, pair, budget=ledger
        ) == 0
        if mutation == "bank":
            interior = next(i for i, point in enumerate(nodes) if point == (Fraction(1, 2),) * 2)
            fine_bank[interior, 0] += Fraction(1, 8)
        elif mutation == "chart":
            # Keep the declared ID and every corner fixed; change the exact
            # interior polynomial, not a fitted affine/corner representation.
            object.__setattr__(fine, "chart_coefficients", changed.chart_coefficients)
            object.__setattr__(fine, "chart_coordinates", changed.chart_coordinates)
        elif mutation == "pair":
            pair.matrix[0, 0] = Fraction(1, 4)
        else:
            fine = changed
        actual = polynomial_nested_geometry_pair_bound(
            fine, fine_bank, source, bank, pair, budget=ledger
        )
    with coordinate_reference_budget() as independent:
        assert actual == _expanded_nested_pair_bound(
            fine, fine_bank, source, bank, pair, budget=independent
        )
    assert actual > 0


def test_composed_nested_pair_authenticates_actual_source_not_declared_id() -> None:
    from phydrax.discretization._cell_geometry import BarycentricCellGeometryElement

    root = coordinate_lagrange_element("triangle", 1)
    weights = np.eye(3)
    weights[0, 0] = 2
    source = BarycentricCellGeometryElement(root, weights)
    coarse = BarycentricCellGeometryElement(root, weights)
    chart = coordinate_lagrange_element("triangle", 2)
    nodes = tuple(tuple(Fraction(float(v)) for v in row) for row in chart.reference_nodes)
    bank = tuple((Fraction(1 + i), Fraction(4 - i)) for i in range(3))
    pair = NestedReferencePair(0, 0, False, np.eye(2) / 2, np.zeros(2))
    with coordinate_reference_budget() as preparation:
        fine = _polynomial_chart(
            source, chart, tuple((x / 2, y / 2) for x, y in nodes), budget=preparation
        )
    with coordinate_reference_budget() as ledger:
        assert polynomial_nested_geometry_pair_bound(
            fine, bank, coarse, bank, pair, budget=ledger
        ) == 0
        weights[1, 1] = 3
        object.__setattr__(coarse, "barycentric_weights", weights)
        actual = polynomial_nested_geometry_pair_bound(
            fine, bank, coarse, bank, pair, budget=ledger
        )
    with coordinate_reference_budget() as independent:
        assert actual == _expanded_nested_pair_bound(
            fine, bank, coarse, bank, pair, budget=independent
        )
    assert actual > 0


def test_composed_nested_pair_scalar_representations_keep_generic_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.discretization import _mapped_geometry_measure as measure
    from phydrax.discretization._exact_algebraic_geometry import RadicalElement

    source = coordinate_lagrange_element("quadrilateral", 2)
    nodes = tuple(tuple(Fraction(float(v)) for v in row) for row in source.reference_nodes)
    rational = tuple((x + x * x / 8, y) for x, y in nodes)
    radical = tuple(tuple(RadicalElement(value) for value in row) for row in rational)
    assert rational == radical
    pair = NestedReferencePair(0, 0, False, np.eye(2) / 2, np.zeros(2))
    with coordinate_reference_budget() as preparation:
        fine = _polynomial_chart(
            source, source, tuple((x / 2, y / 2) for x, y in nodes), budget=preparation
        )
    calls = []
    original = measure.mapped_coordinate_expressions

    def expanded(
        element: CellGeometryElement,
        local: CoordinateCoefficients,
        *,
        budget: CoordinateEnclosureBudget,
    ) -> tuple[Expression, ...]:
        calls.append(element)
        return original(element, local, budget=budget)

    monkeypatch.setattr(measure, "mapped_coordinate_expressions", expanded)
    with coordinate_reference_budget() as ledger:
        assert polynomial_nested_geometry_pair_bound(
            fine, radical, source, rational, pair, budget=ledger
        ) == 0
    assert len(calls) == 2


def test_composed_nested_pair_authenticates_exact_rounded_pair_binding() -> None:
    from phydrax.discretization._nested_reference_types import exact_reference_pair

    source = coordinate_lagrange_element("quadrilateral", 2)
    nodes = tuple(tuple(Fraction(float(v)) for v in row) for row in source.reference_nodes)
    bank = tuple((x + x * x / 8, y) for x, y in nodes)
    pair = exact_reference_pair(
        0, 0, False,
        ((Fraction(1, 3), Fraction(0)), (Fraction(0), Fraction(1, 3))),
        (Fraction(0), Fraction(0)),
    )
    with coordinate_reference_budget() as preparation:
        fine = _polynomial_chart(
            source, source, tuple((x / 3, y / 3) for x, y in nodes), budget=preparation
        )
    with coordinate_reference_budget() as ledger:
        assert polynomial_nested_geometry_pair_bound(
            fine, bank, source, bank, pair, budget=ledger
        ) == 0
        pair.matrix[0, 0] = np.nextafter(pair.matrix[0, 0], np.inf)
        with pytest.raises(ValueError, match="exact_pair must round"):
            polynomial_nested_geometry_pair_bound(
                fine, bank, source, bank, pair, budget=ledger
            )


@pytest.mark.parametrize("resource", ("work", "storage"))
def test_composed_nested_pair_definition_obeys_original_budget(
    resource: str,
    coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget],
) -> None:
    source = coordinate_lagrange_element("quadrilateral", 2)
    nodes = tuple(tuple(Fraction(float(v)) for v in row) for row in source.reference_nodes)
    bank = tuple((x + x * x / 8, y) for x, y in nodes)
    with coordinate_reference_budget() as preparation:
        fine = _polynomial_chart(
            source, source, tuple((x / 2, y / 2) for x, y in nodes), budget=preparation
        )
    pair = NestedReferencePair(0, 0, False, np.eye(2) / 2, np.zeros(2))
    ledger = coordinate_budget(
        1 if resource == "work" else 1_000_000,
        64 * 1024**2 if resource == "work" else 1,
    )
    with ledger.activate(), pytest.raises(CoordinateEnclosureResourceError):
        polynomial_nested_geometry_pair_bound(
            fine, bank, source, bank, pair, budget=ledger
        )


def _polynomial_chart(
    source: ScalarCoordinateElement,
    chart: FiniteElementSpec,
    points: tuple[tuple[Fraction, ...], ...],
    *,
    budget: CoordinateEnclosureBudget,
) -> PolynomialComposedCellGeometryElement:
    return PolynomialComposedCellGeometryElement(
        source,
        chart,
        np.asarray([[value.numerator for value in row] for row in points], dtype=object),
        np.asarray([[value.denominator for value in row] for row in points], dtype=object),
        budget=budget,
    )


def _expanded_mapped_density(
    element: ScalarCoordinateElement,
    coefficients: CoordinateCoefficients,
    *,
    budget: CoordinateEnclosureBudget,
) -> Expression:
    # Independent oracle: expand every physical coordinate before differentiating,
    # rather than composing either source or chart determinant.
    coordinates = enclosure.coordinate_expressions(element, coefficients, budget=budget)
    assert coordinates is not None
    dimension = element.topological_dimension
    jacobian = tuple(
        tuple(enclosure.expression_derivative(value, axis, budget=budget)
              for axis in range(dimension))
        for value in coordinates
    )
    total: Expression = {}
    for ordering in permutations(range(dimension)):
        inversions = sum(
            ordering[left] > ordering[right]
            for left in range(dimension) for right in range(left + 1, dimension)
        )
        term: Expression = {(0,) * dimension: Fraction((-1) ** inversions)}
        for row, column in enumerate(ordering):
            term = enclosure.expression_multiply(term, jacobian[row][column], budget=budget)
        total = enclosure.expression_add(total, term, budget=budget)
    return total


@pytest.mark.parametrize("kind", ("interval", "quadrilateral", "hexahedron"))
def test_composed_density_chain_rule_matches_independent_nonlinear_expansion(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from phydrax.discretization import _mapped_geometry_measure as measure

    source = coordinate_lagrange_element(kind, 2)
    chart = coordinate_lagrange_element(kind, 2)
    nodes = tuple(tuple(Fraction(float(v)) for v in row) for row in source.reference_nodes)
    dimension = source.topological_dimension
    coefficients = tuple(
        tuple(
            x + x * x / 8 + x * point[(axis + 1) % dimension] / 16
            for axis, x in enumerate(point)
        )
        for point in nodes
    )
    elements = []
    with coordinate_reference_budget() as preparation:
        for offset in (Fraction(0), Fraction(1, 4)):
            points = tuple(
                tuple(
                    offset + x / 4 + x * x / 32
                    + x * point[(axis + 1) % dimension] / 64
                    for axis, x in enumerate(point)
                )
                for point in nodes
            )
            elements.append(_polynomial_chart(source, chart, points, budget=preparation))
    calls = []
    original = measure._numeric_mapped_density

    def record_density(
        coordinates: tuple[Expression, ...],
        dimension: int,
        *,
        budget: CoordinateEnclosureBudget,
    ) -> Expression:
        calls.append(len(budget.composed_source_density_cache))
        return original(coordinates, dimension, budget=budget)

    monkeypatch.setattr(measure, "_numeric_mapped_density", record_density)
    with coordinate_reference_budget() as shared:
        first_source_density = None
        for element in elements:
            actual = mapped_density_expression(element, coefficients, budget=shared)
            with coordinate_reference_budget() as independent:
                assert actual == _expanded_mapped_density(
                    element, coefficients, budget=independent
                )
            key = enclosure.coordinate_bank_cache_key(source, coefficients)
            source_density = shared.composed_source_density_cache[key]
            if first_source_density is not None:
                assert source_density is first_source_density
            first_source_density = source_density
            assert enclosure.coordinate_bank_cache_key(
                element, coefficients
            ) not in shared.coordinate_cache
            assert mapped_density_expression(element, coefficients, budget=shared) is actual
        assert len(shared.composed_source_density_cache) == 1
        assert len(shared.composed_chart_density_cache) == 2
        assert len(shared.density_cache) == 2
    assert calls == [0, 1, 1]


def test_composed_density_keeps_both_reversed_orientation_signs() -> None:
    source = coordinate_lagrange_element("quadrilateral", 2)
    nodes = tuple(tuple(Fraction(float(v)) for v in row) for row in source.reference_nodes)
    coefficients = tuple((-x - x * x / 8, y + y * y / 8) for x, y in nodes)
    points = tuple((1 - x / 2 - x * x / 16, y / 2 + y * y / 16) for x, y in nodes)
    with coordinate_reference_budget() as budget:
        element = _polynomial_chart(source, source, points, budget=budget)
        with pytest.raises(ValueError, match="strictly positive certified bound"):
            mapped_density_expression(source, coefficients, budget=budget)
        actual = mapped_density_expression(element, coefficients, budget=budget)
        source_density = budget.composed_source_density_cache[
            enclosure.coordinate_bank_cache_key(source, coefficients)
        ]
        chart_density = budget.composed_chart_density_cache[enclosure.element_identity(element)]
        assert enclosure.expression_bounds(source_density, "box", 2, budget=budget)[1] < 0
        assert enclosure.expression_bounds(chart_density, "box", 2, budget=budget)[1] < 0
        assert enclosure.expression_bounds(actual, "box", 2, budget=budget)[0] > 0
    with coordinate_reference_budget() as independent:
        assert actual == _expanded_mapped_density(element, coefficients, budget=independent)


def test_composed_density_certifies_only_positive_restricted_portion_of_folded_source() -> None:
    source = coordinate_lagrange_element("quadrilateral", 2)
    nodes = tuple(tuple(Fraction(float(v)) for v in row) for row in source.reference_nodes)
    coefficients = tuple((x * x - x, y + y * y / 8) for x, y in nodes)
    with coordinate_reference_budget() as budget:
        whole = _polynomial_chart(source, source, nodes, budget=budget)
        positive = _polynomial_chart(
            source,
            source,
            tuple((Fraction(3, 4) + x / 8 + x * x / 32, y / 2) for x, y in nodes),
            budget=budget,
        )
        with pytest.raises(ValueError, match="strictly positive certified bound"):
            mapped_density_expression(whole, coefficients, budget=budget)
        assert not budget.density_cache
        retained = budget.composed_source_density_cache[
            enclosure.coordinate_bank_cache_key(source, coefficients)
        ]
        lower, upper = enclosure.expression_bounds(retained, "box", 2, budget=budget)
        assert lower < 0 < upper
        actual = mapped_density_expression(positive, coefficients, budget=budget)
        assert budget.composed_source_density_cache[
            enclosure.coordinate_bank_cache_key(source, coefficients)
        ] is retained
        assert len(budget.density_cache) == 1
        # A successful restriction does not turn the numerical source cache into
        # a positivity verdict for the original folded chart.
        with pytest.raises(ValueError, match="strictly positive certified bound"):
            mapped_density_expression(whole, coefficients, budget=budget)
        assert len(budget.density_cache) == 1
    with coordinate_reference_budget() as independent:
        assert actual == _expanded_mapped_density(positive, coefficients, budget=independent)


def test_composed_density_authenticates_current_bank_and_actual_chart_binding() -> None:
    source = coordinate_lagrange_element("quadrilateral", 2)
    nodes = tuple(tuple(Fraction(float(v)) for v in row) for row in source.reference_nodes)
    bank = np.asarray([(float(x + x * x / 8), float(y)) for x, y in nodes])
    with coordinate_reference_budget() as budget:
        first = _polynomial_chart(
            source, source, tuple((x / 2, y / 2) for x, y in nodes), budget=budget
        )
        second = _polynomial_chart(
            source, source, tuple((x / 4, y / 2) for x, y in nodes), budget=budget
        )
        # Declared IDs alone must not authenticate a chart's actual operands.
        object.__setattr__(second, "element_id", first.element_id)
        first_density = mapped_density_expression(first, bank, budget=budget)
        second_density = mapped_density_expression(second, bank, budget=budget)
        assert second_density != first_density
        assert len(budget.composed_chart_density_cache) == 2
        original_bank = bank.copy()
        bank[:, 0] *= 2
        changed_bank = bank.copy()
        changed_density = mapped_density_expression(first, bank, budget=budget)
        assert changed_density != first_density
        assert len(budget.composed_source_density_cache) == 2
        assert len(budget.composed_chart_density_cache) == 2
        bank[:] = original_bank
        assert mapped_density_expression(first, bank, budget=budget) is first_density
    with coordinate_reference_budget() as independent:
        assert changed_density == _expanded_mapped_density(
            first, changed_bank, budget=independent
        )


@pytest.mark.parametrize("embedded", (False, True))
def test_composed_density_preserves_embedded_and_nonsquare_original_route(
    embedded: bool,
) -> None:
    source = coordinate_lagrange_element(
        "quadrilateral" if embedded else "hexahedron", 2
    )
    chart = coordinate_lagrange_element("quadrilateral", 2)
    nodes = tuple(tuple(Fraction(float(v)) for v in row) for row in source.reference_nodes)
    coefficients = tuple(
        (point[0] + point[0] ** 2 / 8, point[1] + point[1] ** 2 / 8)
        + ((Fraction(3),) if embedded else ())
        for point in nodes
    )
    points = tuple(
        (Fraction(float(u)) / 2, Fraction(float(v)) / 2)
        + (() if embedded else (Fraction(1, 4),))
        for u, v in chart.reference_nodes
    )
    with coordinate_reference_budget() as budget:
        element = _polynomial_chart(source, chart, points, budget=budget)
        actual = mapped_density_expression(element, coefficients, budget=budget)
        assert not budget.composed_source_density_cache
        assert not budget.composed_chart_density_cache
    with coordinate_reference_budget() as independent:
        coordinates = enclosure.coordinate_expressions(
            element, coefficients, budget=independent
        )
        assert coordinates is not None
        expected = enclosure.expression_determinant(
            tuple(
                tuple(
                    enclosure.expression_derivative(value, axis, budget=independent)
                    for axis in range(2)
                )
                for value in coordinates[:2]
            ),
            budget=independent,
        )
        assert actual == expected


@pytest.mark.parametrize("resource", ("work", "storage"))
def test_composed_density_numeric_preparation_obeys_original_budget(
    resource: str,
    coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget],
) -> None:
    source = coordinate_lagrange_element("quadrilateral", 2)
    nodes = tuple(tuple(Fraction(float(v)) for v in row) for row in source.reference_nodes)
    coefficients = tuple((x + x * x / 8, y + y * y / 8) for x, y in nodes)
    with coordinate_reference_budget() as preparation:
        element = _polynomial_chart(
            source, source, tuple((x / 2, y / 2) for x, y in nodes), budget=preparation
        )
    ledger = coordinate_budget(
        1 if resource == "work" else 1_000_000,
        64 * 1024**2 if resource == "work" else 1,
    )
    with ledger.activate():
        with pytest.raises(CoordinateEnclosureResourceError):
            mapped_density_expression(element, coefficients, budget=ledger)
    assert not ledger.density_cache
    assert not ledger.composed_source_density_cache
    assert not ledger.composed_chart_density_cache


@pytest.fixture
def budget() -> Iterator[CoordinateEnclosureBudget]:
    with coordinate_reference_budget() as ledger:
        yield ledger


@pytest.fixture
def coordinate_budget() -> Iterator[Callable[[int, int], CoordinateEnclosureBudget]]:
    with ExitStack() as resources:

        def create(work: int, memory: int) -> CoordinateEnclosureBudget:
            return resources.enter_context(
                coordinate_reference_budget(
                    MeshCertificateLimits(
                        maximum_work_units=work,
                        maximum_scratch_bytes=memory,
                    )
                )
            )

        yield create


@pytest.mark.parametrize("kind", ("triangle", "tetrahedron", "quadrilateral", "prism", "hexahedron"))
def test_restricted_nested_pair_definition_matches_full_nonlinear_expansion(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from phydrax.discretization import _mapped_geometry_measure as measure

    source = coordinate_lagrange_element(kind, 2)
    dimension = source.topological_dimension
    matrix = np.eye(dimension) / 2
    matrix[0, -1] = 0.25
    offset = np.zeros(dimension)
    fine = RestrictedCellGeometryElement(source, kind, matrix, offset)
    nodes = tuple(tuple(Fraction(float(value)) for value in row) for row in source.reference_nodes)
    bank = tuple(
        tuple(value * value - value if axis == 0 else value + value * value / 8
              for axis, value in enumerate(point))
        for point in nodes
    )
    pair = NestedReferencePair(0, 0, False, matrix, offset)
    with coordinate_reference_budget() as independent:
        assert _expanded_nested_pair_bound(
            fine, bank, source, bank, pair, budget=independent
        ) == 0

    def no_expansion(*args: object, **kwargs: object) -> None:
        pytest.fail("equal complete restrictions must not expand physical maps")

    monkeypatch.setattr(measure, "mapped_coordinate_expressions", no_expansion)
    with coordinate_reference_budget() as budget:
        assert polynomial_nested_geometry_pair_bound(
            fine, bank, source, bank, pair, budget=budget
        ) == 0
        assert not budget.density_cache


@pytest.mark.parametrize("change", ("bank", "matrix"))
def test_restricted_nested_pair_definition_reauthenticates_complete_inputs(change: str) -> None:
    source = coordinate_lagrange_element("tetrahedron", 2)
    matrix = np.asarray(((1, 0, 0.25), (0, 1, 0.25), (0, 0, 0.25)), dtype=np.float64)
    offset = np.zeros(3)
    fine = RestrictedCellGeometryElement(source, "tetrahedron", matrix, offset)
    nodes = np.asarray(source.reference_nodes)
    bank = tuple(tuple(Fraction(float(value)) for value in row) for row in nodes)
    pair = NestedReferencePair(0, 0, False, matrix, offset)
    with coordinate_reference_budget() as budget:
        assert polynomial_nested_geometry_pair_bound(
            fine, bank, source, bank, pair, budget=budget
        ) == 0
        actual_bank = bank
        if change == "bank":
            index = next(index for index, row in enumerate(nodes) if np.any(row == 0.5))
            rows = list(bank)
            rows[index] = (rows[index][0] + Fraction(1, 13), *rows[index][1:])
            actual_bank = tuple(rows)
        else:
            changed = matrix.copy()
            changed[0, 2] = 0.125
            changed.setflags(write=False)
            object.__setattr__(fine, "matrix", changed)
        actual = polynomial_nested_geometry_pair_bound(
            fine, actual_bank, source, bank, pair, budget=budget
        )
        assert actual > 0
    with coordinate_reference_budget() as independent:
        assert actual == _expanded_nested_pair_bound(
            fine, actual_bank, source, bank, pair, budget=independent
        )


def test_restricted_nested_pair_requires_exact_not_only_rounded_chart_equality() -> None:
    from phydrax.discretization._nested_reference_types import exact_reference_pair

    source = coordinate_lagrange_element("quadrilateral", 2)
    fine = RestrictedCellGeometryElement(source, "quadrilateral", np.eye(2) / 2, np.zeros(2))
    nodes = tuple(tuple(Fraction(float(value)) for value in row) for row in source.reference_nodes)
    bank = tuple((x + x * x / 8, y + y * y / 8) for x, y in nodes)
    pair = exact_reference_pair(
        0, 0, False,
        ((Fraction(1, 2) + Fraction(1, 2**80), Fraction(0)),
         (Fraction(0), Fraction(1, 2))),
        (Fraction(0), Fraction(0)),
    )
    assert np.array_equal(pair.matrix, np.eye(2) / 2)
    with coordinate_reference_budget() as budget:
        actual = polynomial_nested_geometry_pair_bound(
            fine, bank, source, bank, pair, budget=budget
        )
        assert actual > 0
    with coordinate_reference_budget() as independent:
        assert actual == _expanded_nested_pair_bound(
            fine, bank, source, bank, pair, budget=independent
        )


class _EarnedSignedDensity(NamedTuple):
    mesh: CellMesh
    geometry: CellGeometrySpec
    element: FiniteElementSpec
    prepared: PreparedCoordinateOperands
    receipt: NativeExecutionRecord
    contract: SpatialCoordinateContract
    local: enclosure.CoordinateSourceBank
    polynomial: Polynomial
    sign: int
    inputs: dict[str, object]


@pytest.fixture(params=(1, -1), ids=("positive", "negative"))
def earned_signed_density(request: pytest.FixtureRequest) -> _EarnedSignedDensity:
    from phydrax.discretization._cell_geometry_validity import (
        _prepare_polynomial_differential,
    )

    if not meshcore_available():
        pytest.skip("native meshcore required for actual paid signed operands")
    sign = request.param
    element = coordinate_lagrange_element("quadrilateral", 2)
    nodes = np.asarray(element.reference_nodes)

    def physical(points: np.ndarray) -> np.ndarray:
        x, y = points.T
        return np.stack((sign * (x + x * x * y / 8), y + y * y / 16), axis=1)

    corners = np.asarray(reference_cell_topology("quadrilateral").vertices)
    mesh = CellMesh(
        physical(corners),
        (CellBlock("cells", "quadrilateral", np.arange(4, dtype=np.int32)[None]),),
    )
    geometry = CellGeometrySpec(
        {"cells": element},
        {"cells": np.arange(len(nodes), dtype=np.int32)[None]},
        physical(nodes),
    )
    contract = SpatialCoordinateContract.si()
    limits = MeshingLimits(
        maximum_work_units=1_000_000,
        maximum_scratch_bytes=32 * 1024**2,
        maximum_wall_seconds=120.0,
    )
    collector = CoordinateOperandCollector(capture_signed_determinants=True)
    public_bank = np.asarray(((1 / 3, 2 / 7),), dtype=np.float64)
    public_bank.flags.writeable = False
    inputs: dict[str, object] = {
        "bank": public_bank,
        "nonbinary": Fraction(1, 3),
        "policy": ["signed", 2],
    }
    with NativeExecutionScope(limits.native_limits()) as earning:
        collector.begin(
            earning, "signed-density-earning",
            native_preparation_ledger(earning).overlap(earning, None),
        )
        with earning.ticket_view() as ticket:
            budget = CoordinateEnclosureBudget(
                ticket, maximum_work_units=limits.maximum_work_units,
                maximum_memory_bytes=limits.maximum_scratch_bytes,
            )
            budget.capture_coordinate_operands = True
            budget.capture_signed_determinant_operands = True
            bank = enclosure.prepared_coordinate_source_bank(geometry, budget=budget)
            elements, routes, _ = geometry.resolve(mesh, budget=budget)
            assert elements[0] is element
            local = tuple(bank[int(index)] for index in np.asarray(routes[0])[0])
            differential = _prepare_polynomial_differential(
                element, local, "quadrilateral", 2, False, budget,
            )
            polynomial = differential.polynomial
            assert isinstance(polynomial, dict)
            expected = dict(polynomial)
            collector.capture(
                mesh, geometry, budget, earning, coordinate_contract=contract,
                input_owners=(inputs,),
            )
            assert 0 < budget.work_units <= limits.maximum_work_units
            assert 0 < budget.peak_bytes_upper <= limits.maximum_scratch_bytes
    receipt = NativeExecutionRecord(
        earning.completed_evidence(), owner_id="signed-density-earning",
    )
    collector.complete(receipt)
    prepared = collector.for_result(mesh, geometry)
    assert prepared is not None
    # A caller's retained working output cannot change the private issued atoms.
    polynomial[(0, 0)] = Fraction(987)
    with coordinate_reference_budget() as independent:
        assert expected == _expanded_mapped_density(element, local, budget=independent)
    return _EarnedSignedDensity(
        mesh, geometry, element, prepared, receipt, contract, local, expected, sign,
        inputs,
    )


@contextmanager
def _signed_density_consumer_budget(
    earned: _EarnedSignedDensity, *, paid: bool = True,
    work: int = 1_000_000, storage: int = 32 * 1024**2,
) -> Iterator[CoordinateEnclosureBudget]:
    limits = MeshingLimits(
        maximum_work_units=1_000_000,
        maximum_scratch_bytes=32 * 1024**2,
        maximum_wall_seconds=120.0,
    )
    with NativeExecutionScope(limits.native_limits()) as scope:
        if paid:
            native_preparation_ledger(scope).import_receipt(scope, earned.receipt)
        with scope.ticket_view() as ticket:
            budget = CoordinateEnclosureBudget(
                ticket, maximum_work_units=work, maximum_memory_bytes=storage,
            )
            with budget.activate():
                yield budget


def test_paid_signed_density_reuses_raw_numeric_data_but_certifies_fresh(
    earned_signed_density: _EarnedSignedDensity, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.discretization import _mapped_geometry_measure as measure

    earned = earned_signed_density
    certified: list[Expression] = []
    original = measure._certify_mapped_density

    def certify(
        element: ScalarCoordinateElement, density: Expression, axis_plane: bool,
        *, budget: CoordinateEnclosureBudget,
    ) -> Expression:
        certified.append(density)
        assert not axis_plane
        return original(element, density, axis_plane, budget=budget)

    def repeated_numeric_work(*args: object, **kwargs: object) -> Expression:
        raise AssertionError("paid signed determinant must not be differentiated again")

    monkeypatch.setattr(measure, "_certify_mapped_density", certify)
    monkeypatch.setattr(measure, "expression_derivative", repeated_numeric_work)
    monkeypatch.setattr(measure, "expression_determinant", repeated_numeric_work)
    with _signed_density_consumer_budget(earned) as budget:
        earned.prepared.require(
            earned.mesh, earned.geometry, earned.receipt, budget,
            coordinate_contract=earned.contract,
        )
        assert not budget.density_cache
        before = budget.work_units
        if earned.sign > 0:
            actual = mapped_density_expression(
                earned.geometry.elements[0], earned.local, budget=budget,
            )
            assert actual == earned.polynomial
            assert len(budget.density_cache) == 1
        else:
            with pytest.raises(ValueError, match="strictly positive certified bound"):
                mapped_density_expression(
                    earned.geometry.elements[0], earned.local, budget=budget,
                )
            assert not budget.density_cache
        assert certified == [earned.polynomial]
        assert 0 < budget.work_units - before <= budget.maximum_work_units
        assert 0 < budget.peak_bytes_upper <= budget.maximum_memory_bytes


@pytest.mark.parametrize("change", ("source", "frame"))
def test_paid_signed_density_reauthenticates_actual_source_and_frame_at_use(
    earned_signed_density: _EarnedSignedDensity, change: str,
) -> None:
    earned = earned_signed_density
    with _signed_density_consumer_budget(earned) as budget:
        earned.prepared.require(
            earned.mesh, earned.geometry, earned.receipt, budget,
            coordinate_contract=earned.contract,
        )
        if change == "source":
            object.__setattr__(
                earned.geometry, "coordinates",
                earned.geometry.coordinates.at[0, 0].add(0.25),
            )
        else:
            object.__setattr__(earned.contract, "reference_frame", "changed-frame")
        with pytest.raises(ValueError, match="changed|current source|frame"):
            mapped_density_expression(
                earned.geometry.elements[0], earned.local, budget=budget,
            )
        assert not budget.density_cache


@pytest.mark.parametrize("change", ("control", "axis_order", "sign"))
def test_paid_signed_density_does_not_alias_changed_current_numeric_bank(
    earned_signed_density: _EarnedSignedDensity, change: str,
) -> None:
    earned = earned_signed_density
    rows = list(earned.local)
    if change == "control":
        rows[len(rows) // 2] = (rows[len(rows) // 2][0] + Fraction(1, 128),
                               rows[len(rows) // 2][1])
    elif change == "axis_order":
        rows = [(y, x) for x, y in rows]
    else:
        rows = [(-x, y) for x, y in rows]
    local = tuple(rows)
    element = earned.element
    with coordinate_reference_budget() as independent:
        expected = _expanded_mapped_density(element, local, budget=independent)
        lower, _ = enclosure.expression_bounds(expected, "box", 2, budget=independent)
    assert expected != earned.polynomial
    with _signed_density_consumer_budget(earned) as budget:
        earned.prepared.require(
            earned.mesh, earned.geometry, earned.receipt, budget,
            coordinate_contract=earned.contract,
        )
        if lower > 0:
            assert mapped_density_expression(element, local, budget=budget) == expected
        else:
            with pytest.raises(ValueError, match="strictly positive certified bound"):
                mapped_density_expression(element, local, budget=budget)
            assert not budget.density_cache


def test_signed_density_requires_paid_receipt_and_cannot_cross_cold_boundary(
    earned_signed_density: _EarnedSignedDensity,
) -> None:
    import copy
    import pickle

    earned = earned_signed_density
    with _signed_density_consumer_budget(earned, paid=False) as budget:
        with pytest.raises(ValueError, match="already be paid"):
            earned.prepared.require(
                earned.mesh, earned.geometry, earned.receipt, budget,
                coordinate_contract=earned.contract,
            )
        assert not budget.composed_source_density_cache
        assert not budget.density_cache
    for detach in (copy.copy, pickle.dumps):
        with pytest.raises(TypeError, match="archive/cold"):
            detach(earned.prepared)


@pytest.mark.parametrize("resource", ("work", "storage"))
def test_paid_signed_density_admission_keeps_actual_resource_bounds(
    earned_signed_density: _EarnedSignedDensity, resource: str,
) -> None:
    earned = earned_signed_density
    with _signed_density_consumer_budget(
        earned, work=1 if resource == "work" else 1_000_000,
        storage=1 if resource == "storage" else 32 * 1024**2,
    ) as budget:
        with pytest.raises(CoordinateEnclosureResourceError):
            earned.prepared.require(
                earned.mesh, earned.geometry, earned.receipt, budget,
                coordinate_contract=earned.contract,
            )
        assert not budget.density_cache


def test_paid_signed_density_keeps_current_measure_publication_tolerance(
    earned_signed_density: _EarnedSignedDensity,
) -> None:
    from phydrax.discretization._cell_geometry_transition_types import (
        CellGeometryTransitionError,
    )
    from phydrax.discretization._mapped_geometry_measure import certified_cell_measures

    earned = earned_signed_density
    with _signed_density_consumer_budget(earned) as budget:
        earned.prepared.require(
            earned.mesh, earned.geometry, earned.receipt, budget,
            coordinate_contract=earned.contract,
        )
        assert not budget.exact_cell_measure_cache
        if earned.sign < 0:
            with pytest.raises(ValueError, match="strictly positive certified bound"):
                certified_cell_measures(earned.mesh, earned.geometry, budget=budget)
            assert not budget.exact_cell_measure_cache
            return
        values, errors, exact = certified_cell_measures(
            earned.mesh, earned.geometry, budget=budget,
        )
        assert exact
        assert abs(Fraction(float(values[0])) - Fraction(217, 192)) <= Fraction(float(errors[0]))
        assert errors[0] > 0
        with pytest.raises(CellGeometryTransitionError, match="publication_error"):
            certified_cell_measures(
                earned.mesh, earned.geometry, absolute_tolerance=1e-30,
                relative_tolerance=0, budget=budget,
            )


def test_reconstructed_signed_artifact_has_no_decoded_authority(
    earned_signed_density: _EarnedSignedDensity,
) -> None:
    from dataclasses import replace

    earned = earned_signed_density
    decoded = replace(earned.prepared)
    with _signed_density_consumer_budget(earned) as budget:
        with pytest.raises(ValueError, match="current source.*exact paid earning receipt"):
            decoded.require(
                earned.mesh, earned.geometry, earned.receipt, budget,
                coordinate_contract=earned.contract,
            )
        assert not budget.composed_source_density_cache
        assert not budget.density_cache


def test_paid_signed_source_density_composes_both_signs_before_target_certification(
    earned_signed_density: _EarnedSignedDensity, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.discretization import _mapped_geometry_measure as measure

    earned = earned_signed_density
    source = earned.element
    nodes = tuple(
        tuple(Fraction(float(value)) for value in row)
        for row in source.reference_nodes
    )
    points = tuple(
        (1 - x, y) if earned.sign < 0 else (x / 2, y / 2)
        for x, y in nodes
    )
    with coordinate_reference_budget() as preparation:
        element = _polynomial_chart(source, source, points, budget=preparation)
    with coordinate_reference_budget() as independent:
        expected = _expanded_mapped_density(element, earned.local, budget=independent)
        assert enclosure.expression_bounds(expected, "box", 2, budget=independent)[0] > 0
    numeric_calls: list[tuple[Expression, ...]] = []
    original = measure._numeric_mapped_density

    def numeric(
        coordinates: tuple[Expression, ...], dimension: int,
        *, budget: CoordinateEnclosureBudget,
    ) -> Expression:
        numeric_calls.append(coordinates)
        return original(coordinates, dimension, budget=budget)

    monkeypatch.setattr(measure, "_numeric_mapped_density", numeric)
    with _signed_density_consumer_budget(earned) as budget:
        earned.prepared.require(
            earned.mesh, earned.geometry, earned.receipt, budget,
            coordinate_contract=earned.contract,
        )
        assert mapped_density_expression(element, earned.local, budget=budget) == expected
        # Only the new reference chart needs numerical differentiation; the
        # negative source operand must not be rejected or made absolute first.
        assert len(numeric_calls) == 1


def test_paid_signed_measure_loop_preserves_fresh_science(
    earned_signed_density: _EarnedSignedDensity, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.discretization import _mapped_geometry_measure as measure

    earned = earned_signed_density
    with coordinate_reference_budget() as independent:
        expected, expected_error = measure.integrate_mapped_polynomial(
            earned.polynomial, "quadrilateral", budget=independent,
        )
    certified: list[Expression] = []
    original = measure._certify_mapped_density

    def certify(
        element: ScalarCoordinateElement, density: Expression, axis_plane: bool,
        *, budget: CoordinateEnclosureBudget,
    ) -> Expression:
        certified.append(density)
        return original(element, density, axis_plane, budget=budget)

    def repeated_numeric_work(*args: object, **kwargs: object) -> Expression:
        raise AssertionError("paid measure operands must not be differentiated again")

    monkeypatch.setattr(measure, "_certify_mapped_density", certify)
    monkeypatch.setattr(measure, "expression_derivative", repeated_numeric_work)
    monkeypatch.setattr(measure, "expression_determinant", repeated_numeric_work)
    with _signed_density_consumer_budget(earned) as budget:
        earned.prepared.require(
            earned.mesh, earned.geometry, earned.receipt, budget,
            coordinate_contract=earned.contract,
        )
        if earned.sign > 0:
            values, errors, exact = measure.certified_cell_measures(
                earned.mesh, earned.geometry, budget=budget,
            )
            assert values.shape == errors.shape == (1,)
            assert exact
            assert values[0] > errors[0] >= 0
            assert abs(values[0] - expected) <= errors[0] + expected_error
        else:
            with pytest.raises(ValueError, match="strictly positive certified bound"):
                measure.certified_cell_measures(
                    earned.mesh, earned.geometry, budget=budget,
                )
            assert not budget.density_cache
        assert certified == [earned.polynomial]
        assert 0 < budget.work_units <= budget.maximum_work_units
        assert 0 < budget.peak_bytes_upper <= budget.maximum_memory_bytes


def test_closed_signed_batch_binds_elements_and_expires(
    earned_signed_density: _EarnedSignedDensity,
) -> None:
    earned = earned_signed_density
    with _signed_density_consumer_budget(earned) as budget:
        earned.prepared.require(
            earned.mesh, earned.geometry, earned.receipt, budget,
            coordinate_contract=earned.contract,
        )
        _, key = enclosure.coordinate_preparation_key(
            earned.element, earned.local, budget=budget,
        )
        with _signed_determinant_operand_batch(
            earned.mesh, earned.geometry, budget,
        ) as batch:
            assert batch is not None and batch.for_element(earned.element)
            actual = _signed_determinant_operands_for(
                key, "quadrilateral", 2, budget, element=earned.element, batch=batch,
            )
            assert actual == earned.polynomial
            assert actual is not earned.polynomial
        assert not batch.for_element(earned.element)
        assert _signed_determinant_operands_for(
            key, "quadrilateral", 2, budget, element=earned.element, batch=batch,
        ) is None


@pytest.mark.parametrize("retire", ("foreign-element", "nested", "working-atom"))
def test_closed_signed_batch_refuses_foreign_reentrant_or_mutated_use(
    earned_signed_density: _EarnedSignedDensity, retire: str,
) -> None:
    earned = earned_signed_density
    with _signed_density_consumer_budget(earned) as budget:
        earned.prepared.require(
            earned.mesh, earned.geometry, earned.receipt, budget,
            coordinate_contract=earned.contract,
        )
        _, key = enclosure.coordinate_preparation_key(
            earned.element, earned.local, budget=budget,
        )
        with _signed_determinant_operand_batch(
            earned.mesh, earned.geometry, budget,
        ) as batch:
            assert batch is not None
            if retire == "foreign-element":
                assert not batch.for_element(coordinate_lagrange_element("triangle", 1))
            elif retire == "nested":
                with _signed_determinant_operand_batch(
                    earned.mesh, earned.geometry, budget,
                ) as nested:
                    assert nested is None
            else:
                polynomial = _signed_determinant_operands_for(
                    key, "quadrilateral", 2, budget, element=earned.element, batch=batch,
                )
                assert polynomial is not None
                coefficient = next(iter(polynomial.values()))
                assert isinstance(coefficient, Fraction)
                numerator = coefficient.numerator
                object.__setattr__(coefficient, "_numerator", numerator + 1)
                with pytest.raises(ValueError, match="changed after.*numeric issue"):
                    _signed_determinant_operands_for(
                        key, "quadrilateral", 2, budget,
                        element=earned.element, batch=batch,
                    )
                # Restore the deliberately exposed working atom so the enclosing
                # current-source check can complete independently of this refusal.
                object.__setattr__(coefficient, "_numerator", numerator)
            assert not batch.for_element(earned.element)


@pytest.mark.parametrize("change", ("source", "frame", "unsupported"))
def test_closed_signed_batch_checks_current_owners_before_return(
    earned_signed_density: _EarnedSignedDensity, change: str,
) -> None:
    earned = earned_signed_density
    with _signed_density_consumer_budget(earned) as budget:
        earned.prepared.require(
            earned.mesh, earned.geometry, earned.receipt, budget,
            coordinate_contract=earned.contract,
        )
        with pytest.raises(ValueError, match="changed|current source|frame"):
            with _signed_determinant_operand_batch(
                earned.mesh, earned.geometry, budget,
            ) as batch:
                assert batch is not None
                if change == "frame":
                    object.__setattr__(earned.contract, "reference_frame", "foreign")
                else:
                    if change == "unsupported":
                        assert not batch.for_element(object())
                    object.__setattr__(
                        earned.geometry, "coordinates",
                        earned.geometry.coordinates.at[-1, -1].add(0.125),
                    )


def test_signed_operands_refuse_foreign_frame_and_ended_request(
    earned_signed_density: _EarnedSignedDensity,
) -> None:
    earned = earned_signed_density
    with _signed_density_consumer_budget(earned) as budget:
        with pytest.raises(ValueError, match="current source"):
            earned.prepared.require(
                earned.mesh, earned.geometry, earned.receipt, budget,
                coordinate_contract=SpatialCoordinateContract.si(),
            )
        earned.prepared.require(
            earned.mesh, earned.geometry, earned.receipt, budget,
            coordinate_contract=earned.contract,
        )
    with pytest.raises(ValueError, match="live native request"):
        earned.prepared.require(
            earned.mesh, earned.geometry, earned.receipt, budget,
            coordinate_contract=earned.contract,
        )
    with _signed_density_consumer_budget(earned):
        with pytest.raises(ValueError, match="live native request"):
            earned.prepared.require(
                earned.mesh, earned.geometry, earned.receipt, budget,
                coordinate_contract=earned.contract,
            )


@pytest.mark.parametrize("resource", ("work", "storage"))
def test_closed_signed_numeric_operation_exact_resource_boundary(
    earned_signed_density: _EarnedSignedDensity, resource: str,
) -> None:
    from phydrax.discretization._mapped_geometry_measure import (
        integrate_mapped_polynomial,
    )

    earned = earned_signed_density

    def operation(budget: CoordinateEnclosureBudget) -> tuple[float, float]:
        earned.prepared.require(
            earned.mesh, earned.geometry, earned.receipt, budget,
            coordinate_contract=earned.contract,
        )
        _, key = enclosure.coordinate_preparation_key(
            earned.element, earned.local, budget=budget,
        )
        with _signed_determinant_operand_batch(
            earned.mesh, earned.geometry, budget,
        ) as batch:
            assert batch is not None
            polynomial = _signed_determinant_operands_for(
                key, "quadrilateral", 2, budget, element=earned.element, batch=batch,
            )
            assert polynomial is not None
            return integrate_mapped_polynomial(
                polynomial, "quadrilateral", budget=budget,
            )

    with _signed_density_consumer_budget(earned) as observed:
        expected = operation(observed)
    assert abs(Fraction(expected[0]) - earned.sign * Fraction(217, 192)) <= Fraction(
        expected[1]
    )
    required = observed.work_units if resource == "work" else observed.peak_bytes_upper
    for slack in (0, -1):
        with _signed_density_consumer_budget(
            earned,
            work=required + slack if resource == "work" else 1_000_000,
            storage=required + slack if resource == "storage" else 32 * 1024**2,
        ) as bounded:
            if slack == 0:
                assert operation(bounded) == expected
            else:
                with pytest.raises(CoordinateEnclosureResourceError):
                    operation(bounded)


def _completed_coordinate_source() -> tuple[
    CellMeshingResult, PreparedCoordinateOperands, MeshingLimits,
]:
    """Publish a tiny real audited source and its same-root numerical bank."""
    import phydrax as phx
    from phydrax.meshing._native_preparation import (
        prepare_native_completed_source_issue,
        stage_native_source_publication,
    )
    from tests._support.cell_meshes import triangle_grid

    if not meshcore_available():
        pytest.skip("native meshcore required for completed numerical input")
    limits = MeshingLimits(
        maximum_work_units=1_000_000,
        maximum_scratch_bytes=32 * 1024**2,
        maximum_wall_seconds=120.0,
    )
    collector = CoordinateOperandCollector(
        capture_polynomials=True, capture_signed_determinants=True,
    )
    with NativeExecutionScope(limits.native_limits()) as scope:
        collector.begin(scope, None, native_preparation_ledger(scope).overlap(scope, None))
        with scope.ticket_view() as ticket:
            budget = CoordinateEnclosureBudget(
                ticket, maximum_work_units=limits.maximum_work_units,
                maximum_memory_bytes=limits.maximum_scratch_bytes,
            )
            budget.capture_coordinate_operands = True
            budget.capture_polynomial_operands = True
            budget.capture_signed_determinant_operands = True
            source = phx.meshing.certify_cell_mesh(
                triangle_grid(1, 1), SpatialCoordinateContract.si(), budget=budget,
            )
            enclosure.prepared_coordinate_source_bank(source.geometry, budget=budget)
            collector.capture(
                source.mesh, source.geometry, budget, scope,
                coordinate_contract=source.coordinate_contract,
            )
            stage_native_source_publication(scope, source, budget=budget)
            issue = prepare_native_completed_source_issue(scope, source, budget=budget)
    receipt = NativeExecutionRecord(scope.completed_evidence(), owner_id=source.result_id)
    source = source.with_execution_evidence(receipt)
    issue.complete(source)
    collector.complete(receipt)
    assert collector.prepared is not None
    return source, collector.prepared, limits


def test_completed_ancestor_promotes_numeric_bank_without_recharging_history() -> None:
    from phydrax.discretization._mapped_geometry_measure import certified_cell_measures

    source, operands, limits = _completed_coordinate_source()
    receipt = source.execution_evidence
    assert receipt is not None
    with coordinate_reference_budget() as independent:
        expected = certified_cell_measures(
            source.mesh, source.geometry, budget=independent,
        )
    with NativeExecutionScope(limits.native_limits()) as receiving:
        ledger = native_preparation_ledger(receiving)
        ledger.record_history(receiving, receipt, completed_source=source)
        assert not ledger.receipt_is_accounted(receiving, receipt)
        assert ledger.imported_receipt_ids(receiving) == ()
        with receiving.ticket_view() as ticket:
            budget = CoordinateEnclosureBudget(
                ticket, maximum_work_units=limits.maximum_work_units,
                maximum_memory_bytes=limits.maximum_scratch_bytes,
            )
            operands.require(
                source.mesh, source.geometry, receipt, budget,
                coordinate_contract=source.coordinate_contract,
            )
            actual = certified_cell_measures(
                source.mesh, source.geometry, budget=budget,
            )
            assert budget.work_units > 0 and budget.retained_basis_bytes > 0
            assert budget.source_bank_cache
            assert not ledger.receipt_is_accounted(receiving, receipt)
            assert ledger.imported_receipt_ids(receiving) == ()
        with pytest.raises(ValueError, match="history"):
            ledger.import_receipt(receiving, receipt)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2] == expected[2]
    evidence = NativeExecutionRecord(
        receiving.completed_evidence(), owner_id="coordinate-consumer",
        history_evidence=(receipt,),
    )
    assert evidence.history_evidence == (receipt,)
    assert int(np.asarray(evidence.total_work_units)) == int(
        np.asarray(evidence.work_counters.construction_work)
    )


@pytest.mark.parametrize(
    "invalid", ("no-history", "receipt-only", "clone", "frame", "mutation"),
)
def test_completed_coordinate_input_requires_actual_current_source_history(
    invalid: str,
) -> None:
    source, operands, limits = _completed_coordinate_source()
    receipt = source.execution_evidence
    assert receipt is not None
    with NativeExecutionScope(limits.native_limits()) as receiving:
        ledger = native_preparation_ledger(receiving)
        if invalid == "clone":
            with pytest.raises(ValueError, match="actual issued weak owner"):
                ledger.record_history(
                    receiving, receipt,
                    completed_source=source.with_execution_evidence(receipt),
                )
        elif invalid != "no-history":
            ledger.record_history(
                receiving, receipt,
                completed_source=None if invalid == "receipt-only" else source,
            )
        contract = source.coordinate_contract
        if invalid == "frame":
            contract = SpatialCoordinateContract.si()
        elif invalid == "mutation":
            object.__setattr__(
                source.mesh, "coordinates", source.mesh.coordinates.at[0, 0].add(0.125),
            )
        with receiving.ticket_view() as ticket:
            budget = CoordinateEnclosureBudget(
                ticket, maximum_work_units=limits.maximum_work_units,
                maximum_memory_bytes=limits.maximum_scratch_bytes,
            )
            with pytest.raises(ValueError, match="already be paid|current source|changed"):
                operands.require(
                    source.mesh, source.geometry, receipt, budget,
                    coordinate_contract=contract,
                )
            assert budget._prepared_coordinate_operands is None


@pytest.mark.parametrize("change", ("readonly-bank", "fraction", "container", "policy"))
def test_signed_numeric_bank_reauthenticates_all_public_input_owners(
    earned_signed_density: _EarnedSignedDensity, change: str,
) -> None:
    earned = earned_signed_density
    with _signed_density_consumer_budget(earned) as budget:
        earned.prepared.require(
            earned.mesh, earned.geometry, earned.receipt, budget,
            coordinate_contract=earned.contract,
        )
        if change == "readonly-bank":
            bank = earned.inputs["bank"]
            assert isinstance(bank, np.ndarray)
            bank.flags.writeable = True
            bank[0, -1] = np.nextafter(bank[0, -1], np.inf)
            bank.flags.writeable = False
        elif change == "fraction":
            fraction = earned.inputs["nonbinary"]
            assert isinstance(fraction, Fraction)
            object.__setattr__(fraction, "_numerator", 2)
        elif change == "container":
            policy = earned.inputs["policy"]
            assert isinstance(policy, list)
            policy.append("changed")
        else:
            earned.inputs["policy"] = ["signed", 2]
        with pytest.raises(ValueError, match="changed after issue"):
            enclosure.prepared_coordinate_source_bank(earned.geometry, budget=budget)


def test_refused_source_admission_does_not_leave_signed_numeric_authority(
    earned_signed_density: _EarnedSignedDensity,
) -> None:
    earned = earned_signed_density
    with _signed_density_consumer_budget(earned) as budget:
        earned.prepared.require(
            earned.mesh, earned.geometry, earned.receipt, budget,
            coordinate_contract=earned.contract,
        )
        if earned.sign > 0:
            assert mapped_density_expression(
                earned.element, earned.local, budget=budget,
            ) == earned.polynomial
        with pytest.raises(ValueError, match="current source"):
            earned.prepared.require(
                earned.mesh, earned.geometry, earned.receipt, budget,
                coordinate_contract=SpatialCoordinateContract.si(),
            )
        with pytest.raises(ValueError, match="current source admission"):
            mapped_density_expression(earned.element, earned.local, budget=budget)


@pytest.mark.parametrize("mutated", (False, True))
def test_foreign_equal_source_independently_prepares_without_replacing_issued_bank(
    earned_signed_density: _EarnedSignedDensity, mutated: bool,
) -> None:
    from copy import copy

    from phydrax.discretization._coordinate_identity import (
        coordinate_source_bank_key,
    )

    earned = earned_signed_density
    foreign = copy(earned.geometry)
    key = coordinate_source_bank_key(earned.geometry)
    assert foreign is not earned.geometry
    assert coordinate_source_bank_key(foreign) == key
    if mutated:
        object.__setattr__(
            foreign, "coordinates", foreign.coordinates.at[-1, -1].add(0.125),
        )
    with coordinate_reference_budget() as independent:
        expected = foreign.source_coordinates(budget=independent)
    with _signed_density_consumer_budget(earned) as budget:
        earned.prepared.require(
            earned.mesh, earned.geometry, earned.receipt, budget,
            coordinate_contract=earned.contract,
        )
        admitted = budget.source_bank_cache[key]
        before = budget.work_units
        actual = enclosure.prepared_coordinate_source_bank(foreign, budget=budget)
        assert actual == expected
        if mutated:
            assert actual != admitted
        assert actual is not admitted
        assert budget.work_units > before
        assert budget.source_bank_cache[key] is admitted
        assert enclosure.prepared_coordinate_source_bank(
            earned.geometry, budget=budget,
        ) is admitted
