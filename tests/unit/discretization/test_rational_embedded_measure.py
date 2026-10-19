#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from fractions import Fraction

import numpy as np
import pytest

from phydrax.discretization._cell_geometry import (
    RestrictedCellGeometryElement,
)
from phydrax.discretization._cell_geometry_transition_types import (
    CellGeometryTransitionError,
)
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.discretization._coordinate_enclosure import (
    add,
    axes,
    constant,
    Polynomial,
    RationalPolynomial,
)
from phydrax.discretization._mapped_geometry_measure import (
    certified_cell_measures,
    certified_sqrt_polynomial_integral,
    embedded_squared_density,
    prepare_mapped_edge_arc_length,
)
from phydrax.discretization.fem import (
    discontinuous_element,
    FiniteElementFieldSpec,
    FiniteElementPlan,
)
from phydrax.discretization.fem._sphere_chart_transfer import (
    projective_form,
    projective_rational_density,
)
from phydrax.discretization.fem._surface_chart_transfer import (
    integral,
    IntegrationBudgets,
    restricted,
)
from phydrax.discretization.fem._topology_transfer import (
    prepare_mapped_nested_dg_transfer,
)
from phydrax.geometry._embedding_state import MeshCertificateLimits
from tests._support.coordinate_references import (
    coordinate_reference_budget,
)
from tests._support.rational_mapped_cells import cylinder_source, six_quad_source


# gamma_5*(pi/2) plus one rounded pi/2 constant is below 2e-15: six
# nonnegative cell areas require at most five summation roundings. The same
# allowance covers the unit-scale sqrt/asinh and quadrature reference formulas.
_REFERENCE_PUBLICATION_ERROR = 2e-15
# sqrt(2)*(1-1.5*log(2)) uses five rounded elementary operations; its
# first-order binary64 propagation is below 3e-16 at these arguments.
_LOG_REFERENCE_ERROR = 3e-16
# Preserve the requested certified integration budget; errors are checked
# against the same request rather than an empirical reference tolerance.
_INTEGRATION_ERROR_BUDGET = 1e-10
# Unit-scale coefficient/adjoint accuracy allocations are 2048 and 128 eps;
# retain the stricter pre-wave decimal thresholds for these independent checks.
_DG_COEFFICIENT_ACCURACY = 3e-13
_ADJOINT_ACCURACY = 2e-14


def test_original_and_six_rational_quads_enclose_the_cylinder_not_the_carrier(
    budget: CoordinateEnclosureBudget,
) -> None:
    mesh, geometry, root, controls = cylinder_source()
    quads, restricted = six_quad_source(mesh, geometry, root, controls)
    values, errors, exact = certified_cell_measures(
        mesh, geometry, relative_tolerance=0, budget=budget
    )
    pieces, bounds, child_exact = certified_cell_measures(
        quads, restricted, relative_tolerance=0, budget=budget
    )
    assert not exact and not child_exact
    assert abs(values.sum() - math.pi / 2) <= errors.sum() + _REFERENCE_PUBLICATION_ERROR
    assert abs(pieces.sum() - math.pi / 2) <= bounds.sum() + _REFERENCE_PUBLICATION_ERROR
    assert abs(pieces.sum() - values.sum()) <= bounds.sum() + errors.sum()
    assert abs(values.sum() - math.sqrt(2)) > 0.1
    np.testing.assert_array_equal(geometry.coordinates, controls)
    gram = embedded_squared_density(root, controls, budget=budget)
    weighted, error = certified_sqrt_polynomial_integral(
        gram, {(1, 0): Fraction(1)}, "quadrilateral", relative_tolerance=0, budget=budget
    )
    assert abs(weighted - math.pi / 4) <= error + _REFERENCE_PUBLICATION_ERROR


def test_full_source_arc_enclosure_and_restricted_edges_agree(
    budget: CoordinateEnclosureBudget,
) -> None:
    _, _, root, controls = cylinder_source()
    start, end = (Fraction(0), Fraction(0)), (Fraction(0), Fraction(1))
    prepared = prepare_mapped_edge_arc_length(root, controls, start, end, budget=budget)
    result = prepared.integrate(relative_tolerance=0, budget=budget)
    cold = prepare_mapped_edge_arc_length(
        root, controls, start, end, budget=budget
    ).integrate(relative_tolerance=0, budget=budget)
    assert result == cold
    assert result.lower <= math.pi / 2 <= result.upper
    assert result.lower > math.sqrt(2)
    straight = prepare_mapped_edge_arc_length(
        root, controls, start, (Fraction(1), Fraction(0)), budget=budget
    ).integrate(relative_tolerance=0, budget=budget)
    assert straight.value == straight.lower == straight.upper == 1 and straight.error == 0
    pieces = []
    for offset in (0.0, 0.5):
        element = RestrictedCellGeometryElement(
            root,
            "quadrilateral",
            np.diag((1.0, 0.5)),
            np.asarray((0.0, offset), dtype=np.float64),
        )
        pieces.append(
            prepare_mapped_edge_arc_length(
                element, controls, start, end, budget=budget
            ).integrate(relative_tolerance=0, budget=budget)
        )
    assert (
        abs(sum(piece.value for piece in pieces) - result.value)
        <= sum(piece.error for piece in pieces) + result.error
    )


def test_signed_rational_weight_retains_whole_integral_relative_error(
    budget: CoordinateEnclosureBudget,
) -> None:
    weight = RationalPolynomial(
        {(1, 0): Fraction(1), (0, 0): Fraction(-1, 2)},
        {(0, 0): Fraction(1), (1, 0): Fraction(1)},
    )
    value, error = certified_sqrt_polynomial_integral(
        constant(2, 2, budget=budget),
        weight,
        "quadrilateral",
        absolute_tolerance=0,
        relative_tolerance=_INTEGRATION_ERROR_BUDGET,
        budget=budget,
    )
    oracle = math.sqrt(2) * (1 - 1.5 * math.log(2))
    assert abs(value - oracle) <= error + _LOG_REFERENCE_ERROR
    assert error <= _INTEGRATION_ERROR_BUDGET * (abs(value) - error)
    bad = RationalPolynomial(
        constant(1, 2, budget=budget), {(0, 0): Fraction(1), (1, 0): Fraction(-2)}
    )
    with pytest.raises(ValueError, match="denominator"):
        certified_sqrt_polynomial_integral(
            bad, constant(1, 2, budget=budget), "quadrilateral", budget=budget
        )


def test_rational_measure_one_under_work_and_insufficient_series_refuse_atomically(
    budget: CoordinateEnclosureBudget,
    coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget],
) -> None:
    mesh, geometry, root, controls = cylinder_source()
    ledger = coordinate_budget(100_000_000, 100_000_000)
    with ledger.activate():
        certified_cell_measures(mesh, geometry, relative_tolerance=0, budget=ledger)
    with pytest.raises(CellGeometryTransitionError) as refusal:
        certified_cell_measures(
            mesh,
            geometry,
            relative_tolerance=0,
            maximum_work=ledger.work_units - 1,
            budget=budget,
        )
    assert refusal.value.reason == "resource_limit"
    assert refusal.value.measured > refusal.value.limit
    gram = embedded_squared_density(root, controls, budget=budget)
    for limits in (
        {"maximum_subcells": 1},
        {"maximum_binomial_terms": 1, "maximum_subcells": 2},
    ):
        with pytest.raises(CellGeometryTransitionError) as exhausted:
            certified_sqrt_polynomial_integral(
                gram,
                constant(1, 2, budget=budget),
                "quadrilateral",
                relative_tolerance=0,
                **limits,
                budget=budget,
            )
        assert exhausted.value.reason == "resource_limit"
    np.testing.assert_array_equal(geometry.coordinates, controls)


def test_rational_removable_corner_uses_exact_cone_and_refuses_unbounded_poles(
    budget: CoordinateEnclosureBudget,
) -> None:
    denominator: Polynomial = {
        (2, 0): Fraction(1),
        (1, 1): Fraction(2),
        (0, 2): Fraction(1),
    }
    gram = RationalPolynomial({**denominator, (3, 0): Fraction(1)}, denominator)
    value, error = certified_sqrt_polynomial_integral(
        gram,
        constant(1, 2, budget=budget),
        "triangle",
        relative_tolerance=0,
        budget=budget,
    )
    points, weights = np.polynomial.legendre.leggauss(64)
    u, v = np.meshgrid((points + 1) / 2, (points + 1) / 2, indexing="ij")
    oracle = np.sum(weights[:, None] * weights[None, :] * v * np.sqrt(1 + u**3 * v)) / 4
    assert abs(value - oracle) <= error + _REFERENCE_PUBLICATION_ERROR
    pole = RationalPolynomial(constant(1, 2, budget=budget), denominator)
    with pytest.raises(ValueError, match="denominator|embedded_measure.cone"):
        certified_sqrt_polynomial_integral(
            pole, constant(1, 2, budget=budget), "triangle", budget=budget
        )


def test_bounded_directional_apex_gram_is_integrated_without_faking_a_corner_value(
    budget: CoordinateEnclosureBudget,
) -> None:
    gram = RationalPolynomial(
        {(2, 0): Fraction(1), (0, 2): Fraction(1)},
        {(2, 0): Fraction(1), (1, 1): Fraction(2), (0, 2): Fraction(1)},
    )
    value, error = certified_sqrt_polynomial_integral(
        gram,
        constant(1, 2, budget=budget),
        "triangle",
        relative_tolerance=0,
        budget=budget,
    )
    oracle = (math.sqrt(2) + math.asinh(1)) / (4 * math.sqrt(2))
    assert abs(value - oracle) <= error + _REFERENCE_PUBLICATION_ERROR


def test_actual_nonlinear_six_quad_dg_content_and_component_adjoint(
    budget: CoordinateEnclosureBudget,
) -> None:

    mesh, geometry, root, controls = cylinder_source()
    quads, restricted = six_quad_source(mesh, geometry, root, controls)
    source = FiniteElementPlan(
        mesh,
        FiniteElementFieldSpec(
            "u", {"root": discontinuous_element("quadrilateral", 0)}, component_shape=(2,)
        ),
        coordinate_spec=geometry,
    ).prepare()
    target = FiniteElementPlan(
        quads,
        FiniteElementFieldSpec(
            "u",
            {
                block.name: discontinuous_element("quadrilateral", 0)
                for block in quads.blocks
            },
            component_shape=(2,),
        ),
        coordinate_spec=restricted,
    ).prepare()
    prepared = prepare_mapped_nested_dg_transfer(
        source,
        target,
        geometry,
        restricted,
        field_name="u",
        parent_cells=np.zeros(6, dtype=np.int64),
        budget=budget,
    )
    coefficients = np.asarray(((2.7, -0.8),), dtype=np.float64)
    transferred = np.asarray(prepared.transfer.apply(coefficients))
    dual = np.asarray(
        ((0.3, -0.2), (-0.7, 0.6), (0.4, 0.9), (-0.8, 0.1), (0.2, -0.5), (0.6, 0.8)),
        dtype=np.float64,
    )
    if prepared.target_measures is None:
        raise ValueError("prepared.target_measures must be present; got None")
    np.testing.assert_allclose(
        np.asarray(prepared.target_measures) @ transferred,
        (math.pi / 2) * coefficients[0],
        rtol=0.0,
        atol=_DG_COEFFICIENT_ACCURACY,
    )
    np.testing.assert_allclose(
        np.vdot(transferred, dual),
        np.vdot(coefficients, prepared.transfer.pullback(dual)),
        rtol=0.0,
        atol=_ADJOINT_ACCURACY,
    )
    assert prepared.evidence.passed


def test_rational_surface_and_projective_chart_content_keeps_the_original_source(
    budget: CoordinateEnclosureBudget,
) -> None:

    _, _, root, controls = cylinder_source()
    gram = embedded_squared_density(root, controls, budget=budget)
    reference = np.asarray(
        (
            (Fraction(0), Fraction(0)),
            (Fraction(1), Fraction(0)),
            (Fraction(0), Fraction(1)),
        ),
        dtype=object,
    )
    restricted_gram, _ = restricted(gram, reference, budget=budget)
    assert isinstance(restricted_gram, RationalPolynomial)
    budgets: IntegrationBudgets = dict(
        absolute_tolerance=_INTEGRATION_ERROR_BUDGET,
        relative_tolerance=0.0,
        maximum_work=100_000_000,
        maximum_subcells=10000,
        maximum_binomial_terms=32,
    )
    value, error = integral(
        restricted_gram, constant(1, 2, budget=budget), budgets, budget=budget
    )
    assert abs(value - math.pi / 4) <= error + _REFERENCE_PUBLICATION_ERROR
    u, v = axes(2, budget=budget)
    denominator = add(constant(1, 2, budget=budget), u, budget=budget)
    density, measure_weight, exponent = projective_rational_density(
        restricted_gram, (u, v), denominator, Fraction(1), budget=budget
    )
    values, errors = projective_form(
        (measure_weight,),
        (constant(1, 2, budget=budget),),
        density,
        denominator,
        exponent,
        budgets,
        budget=budget,
    )
    # (u,v)/(1+u) maps this triangle to 2*u+v<=1. Its physical
    # cylinder content is half the source chart, not a corner-chord area.
    assert abs(values[0, 0] - math.pi / 8) <= errors[0, 0] + _REFERENCE_PUBLICATION_ERROR
    assert errors[0, 0] <= budgets["absolute_tolerance"]


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
