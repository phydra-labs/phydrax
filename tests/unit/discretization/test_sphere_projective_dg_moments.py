#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Independent proofs for the factorized affine-source projective DG kernel."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from fractions import Fraction
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from phydrax.discretization._cell_geometry import coordinate_lagrange_element
from phydrax.discretization._cell_geometry_transition_types import (
    CellGeometryTransitionError,
)
from phydrax.discretization._chart_geometry import area, projective_point
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.discretization._coordinate_enclosure import compose, Polynomial
from phydrax.discretization._mapped_geometry_measure import (
    embedded_squared_density,
    mapped_coordinate_expressions,
)
from phydrax.discretization.fem._sphere_chart_transfer import (
    _affine_conservative_form,
    _affine_scaled_integral,
    _dg_reference_forms,
    _dyadic,
    _factorized_float_form,
    _homogeneous,
    _projective_arguments,
    _reference_triangle_integral,
    _vertex_values,
    projective_form,
)
from phydrax.discretization.fem._surface_chart_transfer import (
    integral,
    IntegrationBudgets,
    restricted,
)
from phydrax.linalg._hermitian_spectral import fraction_sqrt_interval
from tests._support.coordinate_references import coordinate_reference_budget


_BARYCENTRIC: tuple[Polynomial, ...] = (
    {(0, 0): Fraction(1), (1, 0): Fraction(-1), (0, 1): Fraction(-1)},
    {(1, 0): Fraction(1)},
    {(0, 1): Fraction(1)},
)
# A degree-one basis with a constant member and non-unit coefficients.
_SCALED: tuple[Polynomial, ...] = (
    {(0, 0): Fraction(1)},
    {(0, 0): Fraction(-1, 3), (1, 0): Fraction(2)},
    {(1, 0): Fraction(1, 2), (0, 1): Fraction(-5, 4)},
)
_SQUARED = Fraction(3, 7)
type _Values = tuple[tuple[Fraction, ...], ...]


@pytest.fixture
def budget() -> Iterator[CoordinateEnclosureBudget]:
    with coordinate_reference_budget() as ledger:
        yield ledger


def _source() -> tuple[tuple[Fraction, Fraction], ...]:
    """Fresh caller-owned vertices; some tests mutate them in place."""
    return (
        (Fraction(1, 10), Fraction(1, 10)),
        (Fraction(1, 2), Fraction(1, 5)),
        (Fraction(1, 5), Fraction(3, 5)),
    )


def _piece(
    coefficients: tuple[tuple[Fraction, ...], ...],
    source: tuple[tuple[Fraction, Fraction], ...] | None = None,
) -> Any:
    source = _source() if source is None else source
    target = tuple(
        projective_point(coefficients, point, invalid_message="outside") for point in source
    )
    return SimpleNamespace(
        exact_source_reference_vertices=source,
        exact_target_reference_vertices=target,
        source_reference_area=area(source),
        projective_map=SimpleNamespace(
            exact_coefficients=coefficients,
            exact_denominator=tuple(
                sum((row[column] for row in coefficients), Fraction(0))
                for column in range(3)
            ),
        ),
    )


def _map(spread: Fraction) -> tuple[tuple[Fraction, ...], ...]:
    return (
        (Fraction(1), Fraction(-9, 10), Fraction(-4, 5)),
        (Fraction(1, 20), Fraction(1) + spread, Fraction(1, 50)),
        (Fraction(3, 100), Fraction(1, 100), Fraction(97, 100) + spread),
    )


def _evaluate(polynomial: Polynomial, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return sum(
        float(value) * x ** index[0] * y ** index[1] for index, value in polynomial.items()
    )


def _quadrature(
    piece: SimpleNamespace, new: Sequence[Polynomial], old: Sequence[Polynomial]
) -> np.ndarray:
    """Collapsed Gauss-Legendre reference, independent of the series kernel."""
    nodes, weights = np.polynomial.legendre.leggauss(40)
    s, t = (nodes + 1) / 2, weights / 2
    u, v = np.meshgrid(s, s, indexing="ij")
    weight = np.outer(t, t) * (1 - u)
    a, b = u, v * (1 - u)
    vertices = np.asarray(piece.exact_source_reference_vertices, dtype=np.float64)
    x = vertices[0, 0] + a * (vertices[1, 0] - vertices[0, 0]) + b * (vertices[2, 0] - vertices[0, 0])
    y = vertices[0, 1] + a * (vertices[1, 1] - vertices[0, 1]) + b * (vertices[2, 1] - vertices[0, 1])
    jacobian = 2 * abs(float(piece.source_reference_area))
    rows = np.asarray(piece.projective_map.exact_coefficients, dtype=np.float64)
    homogeneous = [rows[i, 0] + rows[i, 1] * x + rows[i, 2] * y for i in range(3)]
    total = sum(homogeneous)
    tx, ty = homogeneous[1] / total, homogeneous[2] / total
    root = float(_SQUARED) ** 0.5
    return np.asarray(
        [
            [
                root
                * jacobian
                * np.sum(weight * _evaluate(first, tx, ty) * _evaluate(second, x, y))
                for second in old
            ]
            for first in new
        ]
    )


@pytest.mark.parametrize("spread", [Fraction(0), Fraction(1, 40), Fraction(3, 20)])
@pytest.mark.parametrize("bases", [(_BARYCENTRIC, _BARYCENTRIC), (_SCALED, _BARYCENTRIC), (_BARYCENTRIC, _SCALED)])
def test_factorized_projective_dg_matches_generic_and_quadrature(
    budget: CoordinateEnclosureBudget,
    spread: Fraction,
    bases: tuple[tuple[Polynomial, ...], tuple[Polynomial, ...]],
) -> None:
    new, old = bases
    piece = _piece(_map(spread))
    root = fraction_sqrt_interval(_SQUARED)
    absolute = Fraction(1, 10**16)
    form, error = _affine_conservative_form(
        piece, old, new, root, absolute, 64, budget=budget
    )
    assert 0 < error <= 9 * absolute
    # The generic rational-series owner integrates the same mixed form.
    gram, arguments = restricted(
        {(0, 0): _SQUARED},
        np.asarray(piece.exact_source_reference_vertices, dtype=object),
        budget=budget,
    )
    numerators, denominator = _projective_arguments(piece, arguments, budget=budget)
    generic, generic_errors = projective_form(
        tuple(_homogeneous(term, numerators, denominator, 1, budget=budget) for term in new),
        tuple(compose(term, arguments, budget=budget) for term in old),
        gram,
        denominator,
        1,
        IntegrationBudgets(
            absolute_tolerance=1e-17,
            relative_tolerance=1e-17,
            maximum_work=10**9,
            maximum_subcells=10000,
            maximum_binomial_terms=64,
        ),
        budget=budget,
    )
    assert np.all(np.abs(form - generic) <= float(absolute) + generic_errors + 1e-30)
    reference = _quadrature(piece, new, old)
    assert np.allclose(form, reference, rtol=1e-13, atol=1e-16)


def test_factorized_projective_dg_conserves_the_old_basis_content(
    budget: CoordinateEnclosureBudget,
) -> None:
    piece = _piece(_map(Fraction(1, 10)))
    root = fraction_sqrt_interval(_SQUARED)
    form, error = _affine_conservative_form(
        piece, _SCALED, _BARYCENTRIC, root, Fraction(1, 10**16), 64, budget=budget
    )
    # The projective barycentric target basis is a partition of unity, so each
    # column sum is the old-basis content sqrt(G) * |T| * mean vertex value.
    lower, upper = root
    size = abs(piece.source_reference_area)
    for column, term in enumerate(_SCALED):
        content = size * sum(
            (
                sum(
                    (
                        value * point[0] ** index[0] * point[1] ** index[1]
                        for index, value in term.items()
                    ),
                    Fraction(0),
                )
                for point in piece.exact_source_reference_vertices
            ),
            Fraction(0),
        ) / 3
        total = sum((Fraction(float(value)) for value in form[:, column]), Fraction(0))
        assert abs(total - content * (lower + upper) / 2) <= (
            error + abs(content) * (upper - lower) + Fraction(1, 10**30)
        )


def test_constant_target_moments_annihilate_the_projective_series_error(
    budget: CoordinateEnclosureBudget,
) -> None:
    piece = _piece(_map(Fraction(3, 20)))
    root = fraction_sqrt_interval(_SQUARED)
    constant_basis: tuple[Polynomial, ...] = ({(0, 0): Fraction(1)},)
    values, error = _affine_conservative_form(
        piece, _BARYCENTRIC, constant_basis, root, Fraction(1, 10**16), 1,
        budget=budget,
    )
    # No reciprocal-series terms are needed for a constant target test: W D
    # fixes this integral exactly, independently of the projective denominator.
    size = abs(piece.source_reference_area)
    for column, term in enumerate(_BARYCENTRIC):
        moment = size * sum(
            (
                sum(
                    (value * point[0] ** index[0] * point[1] ** index[1]
                     for index, value in term.items()), Fraction(0),
                )
                for point in piece.exact_source_reference_vertices
            ), Fraction(0),
        ) / 3
        central = moment * (root[0] + root[1]) / 2
        assert abs(Fraction(float(values[0, column])) - central) <= (
            error + abs(moment) * (root[1] - root[0]) / 2
        )


def test_factorized_projective_dg_refuses_exhausted_terms_and_bad_denominators(
    budget: CoordinateEnclosureBudget,
) -> None:
    piece = _piece(_map(Fraction(3, 20)))
    root = fraction_sqrt_interval(_SQUARED)
    with pytest.raises(CellGeometryTransitionError):
        _affine_conservative_form(
            piece, _BARYCENTRIC, _BARYCENTRIC, root, Fraction(1, 10**16), 2, budget=budget
        )
    negative: Any = SimpleNamespace(
        **{
            **vars(piece),
            "projective_map": SimpleNamespace(
                exact_coefficients=piece.projective_map.exact_coefficients,
                exact_denominator=(Fraction(-1), Fraction(0), Fraction(0)),
            ),
        }
    )
    with pytest.raises(ValueError, match="nonpositive projective denominator"):
        _affine_conservative_form(
            negative, _BARYCENTRIC, _BARYCENTRIC, root, Fraction(1, 10**16), 64, budget=budget
        )


def test_affine_scaled_integral_matches_the_certified_sqrt_owner(
    budget: CoordinateEnclosureBudget,
) -> None:
    budgets = IntegrationBudgets(
        absolute_tolerance=1e-15,
        relative_tolerance=1e-15,
        maximum_work=10**8,
        maximum_subcells=1000,
        maximum_binomial_terms=32,
    )
    term: Polynomial = {(0, 0): Fraction(2, 3), (1, 0): Fraction(-1), (1, 1): Fraction(5)}
    value, error, _ = _affine_scaled_integral(
        _SQUARED,
        fraction_sqrt_interval(_SQUARED),
        _reference_triangle_integral(term, budget=budget),
        budgets,
        budget=budget,
    )
    owner, owner_error = integral({(0, 0): _SQUARED}, term, budgets, budget=budget)
    assert abs(value - owner) <= error + owner_error
    assert error <= 1e-15


@pytest.mark.parametrize("basis", [_BARYCENTRIC, _SCALED])
def test_shared_affine_reference_forms_preserve_exact_moments(
    budget: CoordinateEnclosureBudget, basis: tuple[Polynomial, ...]
) -> None:
    moments, mass = _dg_reference_forms(basis, budget=budget)
    for i, polynomial in enumerate(basis):
        assert moments[i] == _reference_triangle_integral(polynomial, budget=budget)
        for j, second in enumerate(basis):
            expected: Polynomial = {}
            for a, first_value in polynomial.items():
                for b, second_value in second.items():
                    index = a[0] + b[0], a[1] + b[1]
                    expected[index] = expected.get(index, Fraction(0)) + first_value * second_value
            assert mass[i][j] == _reference_triangle_integral(expected, budget=budget)
            assert mass[i][j] == mass[j][i]


@pytest.mark.parametrize("degenerate", [False, True])
def test_affine_embedded_density_matches_full_declared_polynomial(
    budget: CoordinateEnclosureBudget, degenerate: bool,
) -> None:
    element = coordinate_lagrange_element("triangle", 1)
    controls = (
        (Fraction(1, 7), Fraction(-2, 5), Fraction(3, 11)),
        (Fraction(8, 7), Fraction(1, 5), Fraction(9, 11)),
        (Fraction(-2, 7), Fraction(6, 5), Fraction(5, 11)),
    )
    if degenerate:
        controls = controls[0], controls[1], controls[1]
    fast = embedded_squared_density(element, controls, budget=budget)
    coordinates = mapped_coordinate_expressions(element, controls, budget=budget)
    full = embedded_squared_density(
        element, controls, prepared_coordinates=coordinates, budget=budget
    )
    assert fast == full
    assert isinstance(fast, dict) and set(fast) <= {(0, 0)}
    assert (fast.get((0, 0), Fraction(0)) == 0) is degenerate


@pytest.mark.parametrize("exponent", [-1070, -30, 0, 20])
def test_binary64_factorized_action_encloses_exact_signed_contraction(
    budget: CoordinateEnclosureBudget, exponent: int
) -> None:
    tau = (
        (Fraction(1, 3), Fraction(-2, 7), Fraction(5, 11)),
        (Fraction(-5, 13), Fraction(3, 17), Fraction(1, 19)),
        (Fraction(2), Fraction(-2), Fraction(1, 2**40)),
    )
    phi = tuple(tuple(reversed(row)) for row in tau)
    weighted = tuple(
        tuple(Fraction(value) * Fraction(2) ** exponent for value in row)
        for row in ((1, 2, -3), (2, -5, 7), (-3, 7, 11))
    )
    root = (Fraction(2, 3), Fraction(2, 3))
    result = _factorized_float_form(
        tau, weighted, phi, root, Fraction(0), Fraction(1), budget=budget
    )
    assert result is not None
    values, error = result
    for i in range(3):
        for j in range(3):
            exact = root[0] * sum(
                (tau[i][k] * weighted[k][l] * phi[j][l] for k in range(3) for l in range(3)),
                Fraction(0),
            )
            assert abs(Fraction(float(values[i, j])) - exact) <= error / 9
    assert _factorized_float_form(
        tau, weighted, phi, root, Fraction(0),
        Fraction(1, 2**1200), budget=budget,
    ) is None


def test_affine_vertex_values_eliminate_identity_arithmetic() -> None:
    points = (
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
    )
    values, operations = _vertex_values(_BARYCENTRIC, points)
    assert values == (
        (Fraction(1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(1)),
    )
    # The first row needs two additions at each point; coordinate basis
    # members are aliases, not multiplication by one or addition to zero.
    assert operations == 6


@pytest.mark.parametrize("exponent", [-1200, -100, 0, 100, 1200])
@pytest.mark.parametrize("numerator", [-17, -1, 1, 17])
def test_integer_shift_dyadic_rounding_has_its_declared_relative_error(
    exponent: int, numerator: int,
) -> None:
    exact = Fraction(numerator, 7) * Fraction(2) ** exponent
    rounded = _dyadic(exact)
    assert abs(rounded - exact) <= abs(exact) / 2**96
    assert rounded.denominator & (rounded.denominator - 1) == 0


def _independent_unit_diagonal(
    delta: tuple[Fraction, ...], terms: int, diagonal: int,
) -> Fraction:
    """Integrate the finite geometric polynomial by barycentric monomials."""
    from math import factorial

    result = Fraction(0)
    for order in range(terms + 1):
        for first in range(order + 1):
            for second in range(order - first + 1):
                powers = (first, second, order - first - second)
                multiplier = (
                    2 * factorial(order)
                    * (powers[diagonal] + 1) * (powers[diagonal] + 2)
                )
                result += (
                    Fraction((-1) ** order * multiplier, factorial(order + 4))
                    * delta[0] ** powers[0]
                    * delta[1] ** powers[1]
                    * delta[2] ** powers[2]
                )
    return result


@pytest.mark.parametrize(
    "delta",
    [
        (Fraction(0), Fraction(0), Fraction(0)),
        (Fraction(1, 8), Fraction(1, 8), Fraction(1, 8)),
        (Fraction(-1, 8), Fraction(0), Fraction(1, 4)),
        (Fraction(-1, 4), Fraction(-1, 4), Fraction(1, 8)),
    ],
)
@pytest.mark.parametrize("terms", [0, 1, 9, 20])
def test_canonical_unit_moments_match_independent_exact_finite_series(
    delta: tuple[Fraction, Fraction, Fraction], terms: int
) -> None:
    from phydrax.discretization.fem._sphere_chart_transfer import (
        _DYADIC_ROUNDOFF,
        _unit_reciprocal_moments,
    )

    with coordinate_reference_budget() as budget:
        weighted, bound = _unit_reciprocal_moments(delta, terms, {}, budget=budget)
    expected = tuple(_independent_unit_diagonal(delta, terms, axis) for axis in range(3))
    assert bound == max(map(abs, expected))
    denominators = tuple(1 + value for value in delta)
    for axis, exact in enumerate(expected):
        actual = weighted[axis][axis] / denominators[axis]
        assert actual == _dyadic(exact)
        assert abs(actual - exact) <= _DYADIC_ROUNDOFF * abs(exact)
        assert sum(
            (weighted[axis][column] * denominators[column] for column in range(3)),
            Fraction(0),
        ) == denominators[axis] / 3


def test_unit_moment_reuse_is_exact_operand_and_order_bound() -> None:
    from phydrax.discretization.fem._sphere_chart_transfer import _unit_reciprocal_moments

    delta = (Fraction(-1, 8), Fraction(0), Fraction(1, 4))
    cache = {}
    with coordinate_reference_budget() as budget:
        expected = _unit_reciprocal_moments(delta, 9, cache, budget=budget)
        first_work = budget.work_units
        assert first_work > 0
        assert _unit_reciprocal_moments(delta, 9, cache, budget=budget) is expected
        assert budget.work_units == first_work
        _unit_reciprocal_moments(delta, 10, cache, budget=budget)
        assert budget.work_units > first_work
        changed_work = budget.work_units
        _unit_reciprocal_moments(
            (delta[0], delta[1], Fraction(3, 8)), 9, cache, budget=budget
        )
        assert budget.work_units > changed_work


def test_canonical_unit_moments_exact_observed_work_boundary() -> None:
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureResourceError
    from phydrax.discretization.fem._sphere_chart_transfer import _unit_reciprocal_moments

    delta = (Fraction(-1, 8), Fraction(0), Fraction(1, 4))
    with coordinate_reference_budget() as budget:
        expected = _unit_reciprocal_moments(delta, 9, {}, budget=budget)
        required = budget.required_work_units
    with coordinate_reference_budget() as admitted:
        with admitted.bound_stage(required, 10**8):
            assert _unit_reciprocal_moments(delta, 9, {}, budget=admitted) == expected
    with coordinate_reference_budget() as refused:
        with pytest.raises(CoordinateEnclosureResourceError):
            with refused.bound_stage(required - 1, 10**8):
                _unit_reciprocal_moments(delta, 9, {}, budget=refused)


def test_affine_form_canonical_moments_reuse_cyclic_triangle_structure() -> None:
    piece = _piece(_map(Fraction(3, 20)))
    # Cyclically rotated corners rebuild the same triangle and projective map.
    source = piece.exact_source_reference_vertices
    rotated = _piece(_map(Fraction(3, 20)), (*source[1:], source[0]))
    root = fraction_sqrt_interval(_SQUARED)
    cache = {}
    with coordinate_reference_budget() as budget:
        first, first_error = _affine_conservative_form(
            piece, _SCALED, _BARYCENTRIC, root, Fraction(1, 10**16), 64,
            budget=budget, moment_cache=cache,
        )
        first_work = budget.work_units
        assert cache
        second, second_error = _affine_conservative_form(
            rotated, _SCALED, _BARYCENTRIC, root, Fraction(1, 10**16), 64,
            budget=budget, moment_cache=cache,
        )
        assert budget.work_units - first_work < first_work
    assert np.all(np.abs(first - second) <= float(first_error + second_error))
    np.testing.assert_allclose(
        second, _quadrature(rotated, _BARYCENTRIC, _SCALED), rtol=1e-13, atol=1e-16
    )


def test_affine_operand_cache_binds_actual_values_and_isolates_caller_fractions() -> None:
    from phydrax.discretization.fem._sphere_chart_transfer import (
        _affine_basis_operand,
        _affine_vertex_values,
    )

    def fresh(row: Polynomial) -> Polynomial:
        copied: Polynomial = {}
        for index, value in row.items():
            assert isinstance(value, Fraction)
            copied[index] = Fraction(value.numerator, value.denominator)
        return copied

    basis = tuple(fresh(row) for row in _SCALED)
    points = _piece(_map(Fraction(1, 8))).exact_source_reference_vertices
    cache = {}
    with coordinate_reference_budget() as budget:
        operand = _affine_basis_operand(basis, cache, budget=budget)
        before = budget.work_units
        exact, rounded = _affine_vertex_values(operand, points, budget=budget)
        first_work = budget.work_units - before
        reference, _ = _vertex_values(basis, points)
        assert exact == reference
        assert rounded == tuple(tuple(_dyadic(value) for value in row) for row in reference)
        before = budget.work_units
        assert _affine_vertex_values(operand, points, budget=budget) == (exact, rounded)
        assert budget.work_units - before < first_work
        exposed = basis[0][(0, 0)]
        assert isinstance(exposed, Fraction)
        original = operand.coefficients[0][0].as_integer_ratio()
        object.__setattr__(exposed, "_numerator", exposed.numerator + exposed.denominator)
        changed = _affine_basis_operand(basis, cache, budget=budget)
        assert changed is not operand
        assert operand.coefficients[0][0].as_integer_ratio() == original
        assert _affine_vertex_values(changed, points, budget=budget)[0] != exact
        exposed_point = points[0][0]
        first_value = exact[1][0].as_integer_ratio()
        object.__setattr__(
            exposed_point, "_numerator", exposed_point.numerator + exposed_point.denominator
        )
        assert exact[1][0].as_integer_ratio() == first_value
        assert _affine_vertex_values(operand, points, budget=budget)[0] != exact


def test_affine_operand_copying_and_point_rounding_observed_work_boundary() -> None:
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureResourceError
    from phydrax.discretization.fem._sphere_chart_transfer import (
        _affine_basis_operand,
        _affine_vertex_values,
    )

    points = _piece(_map(Fraction(1, 8))).exact_source_reference_vertices

    def prepare(budget: CoordinateEnclosureBudget) -> tuple[_Values, _Values]:
        operand = _affine_basis_operand(_SCALED, {}, budget=budget)
        return _affine_vertex_values(operand, points, budget=budget)

    with coordinate_reference_budget() as observed:
        expected = prepare(observed)
        required = observed.required_work_units
    with coordinate_reference_budget() as admitted:
        with admitted.bound_stage(required, 10**8):
            assert prepare(admitted) == expected
    with coordinate_reference_budget() as refused:
        with pytest.raises(CoordinateEnclosureResourceError):
            with refused.bound_stage(required - 1, 10**8):
                prepare(refused)
