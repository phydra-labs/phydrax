#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Host source snapshots preserve coordinate values and declared preparation bounds."""

import math
import sys
from fractions import Fraction
from itertools import permutations, product

import jax
import numpy as np
import pytest
from jax import Array

from phydrax.discretization._cell_geometry import (
    BarycentricCellGeometryElement,
    coordinate_lagrange_element,
    coordinate_source_snapshot_preparation_bounds,
    owned_coordinate_preparation,
    prepare_barycentric_source_snapshot,
    prepare_spline_source_snapshot,
    RestrictedCellGeometryElement,
    SplineCellGeometryElement,
)
from phydrax.discretization._coordinate_budget import CoordinateEnclosureResourceError
from phydrax.discretization._coordinate_enclosure import (
    _accumulate,
    _bernstein_preparation_key,
    _linear_combination_profiled,
    _prepare_bernstein_coefficients,
    _prepare_coordinate_polynomials,
    _prepare_reference_denominators,
    _reference_basis_values,
    _retained_coordinate_point,
    add,
    bernstein_coefficients,
    BernsteinDomain,
    coefficient_profile,
    compose,
    constant,
    coordinate_corner_images,
    coordinate_polynomials,
    coordinate_polynomials_estimate,
    coordinate_preparation_key,
    coordinate_source_signature,
    determinant,
    evaluate,
    expression_control_net,
    expression_derivative,
    expression_evaluate,
    expression_physical_jacobian,
    expression_reference_evaluate,
    ExpressionComposition,
    linear_combinations,
    math_product_polynomials,
    multiply,
    Polynomial,
    power,
    RationalEnclosureError,
    RationalPolynomial,
    scale,
    source_basis,
    source_basis_estimate,
    source_expressions,
)
from phydrax.discretization._exact_algebraic_geometry import (
    RadicalElement,
    rational_coefficient,
)
from tests._support.coordinate_references import coordinate_reference_budget


@pytest.fixture
def barycentric_source() -> BarycentricCellGeometryElement:
    weights = np.eye(3, dtype=np.float64)
    weights[0, 1] = float(Fraction(1, 2**52))
    return BarycentricCellGeometryElement(
        coordinate_lagrange_element("triangle", 1), weights
    )


@pytest.fixture
def spline_source() -> SplineCellGeometryElement:
    knots = np.asarray((0.0, 0.0, 1.0, 1.0), dtype=np.float64)
    return SplineCellGeometryElement(
        knots,
        knots,
        np.ones((2, 2), dtype=np.float64),
        1,
        1,
        (1, 1),
        "coordinate-snapshot-test",
        "source",
    )


def test_barycentric_source_basis_estimate_admits_snapshot_and_exact_action(
    barycentric_source: BarycentricCellGeometryElement,
) -> None:
    estimate = source_basis_estimate(barycentric_source)
    assert estimate is not None
    snapshot_work, snapshot_bytes = coordinate_source_snapshot_preparation_bounds(
        barycentric_source
    )
    assert snapshot_work > 0 and snapshot_bytes > 0
    assert estimate.work_upper >= snapshot_work
    assert estimate.bytes_upper >= snapshot_bytes
    with owned_coordinate_preparation(
        estimate.work_upper, estimate.bytes_upper
    ) as budget:
        basis = source_basis(barycentric_source, budget=budget)
        assert basis is not None
    assert budget.work_units <= estimate.work_upper
    assert budget.peak_bytes_upper <= estimate.bytes_upper
    with coordinate_reference_budget() as evaluation:
        values = tuple(
            expression_evaluate(term, (Fraction(0), Fraction(0)), budget=evaluation)
            for term in basis
        )
    assert values == (Fraction(1), Fraction(1, 2**52), Fraction(0))


def test_barycentric_coordinate_estimate_admits_snapshot_and_reverse_action(
    barycentric_source: BarycentricCellGeometryElement,
) -> None:
    coefficients = (
        (Fraction(2), Fraction(3)),
        (Fraction(5), Fraction(7)),
        (Fraction(11), Fraction(13)),
    )
    signature = coordinate_source_signature(barycentric_source)
    estimate = coordinate_polynomials_estimate(barycentric_source, coefficients)
    assert estimate is not None
    with owned_coordinate_preparation(
        estimate.work_upper, estimate.bytes_upper
    ) as budget:
        coordinates = coordinate_polynomials(
            barycentric_source, coefficients, budget=budget
        )
        assert coordinates is not None
    assert budget.work_units <= estimate.work_upper
    assert budget.peak_bytes_upper <= estimate.bytes_upper
    with coordinate_reference_budget() as evaluation:
        image = tuple(
            expression_evaluate(term, (Fraction(0), Fraction(0)), budget=evaluation)
            for term in coordinates
        )
    delta = Fraction(1, 2**52)
    assert image == (2 + 5 * delta, 3 + 7 * delta)
    assert coordinate_source_signature(barycentric_source) == signature


def test_spline_source_basis_estimate_admits_host_bank_and_bilinear_basis(
    spline_source: SplineCellGeometryElement,
) -> None:
    estimate = source_basis_estimate(spline_source)
    assert estimate is not None
    with owned_coordinate_preparation(
        estimate.work_upper, estimate.bytes_upper
    ) as budget:
        basis = source_expressions(spline_source, budget=budget)
        assert basis is not None
    assert budget.work_units <= estimate.work_upper
    assert budget.peak_bytes_upper <= estimate.bytes_upper
    with coordinate_reference_budget() as evaluation:
        values = tuple(
            expression_evaluate(term, (Fraction(1, 2), Fraction(1, 2)), budget=evaluation)
            for term in basis
        )
    assert values == (Fraction(1, 4),) * 4


@pytest.mark.strict_jax
def test_barycentric_snapshot_reuse_preserves_dynamic_source_leaf(
    barycentric_source: BarycentricCellGeometryElement,
) -> None:
    leaves = jax.tree_util.tree_leaves(barycentric_source)
    signature = coordinate_source_signature(barycentric_source)
    work, memory = coordinate_source_snapshot_preparation_bounds(barycentric_source)
    with owned_coordinate_preparation(work, memory) as budget:
        snapshot = prepare_barycentric_source_snapshot(barycentric_source, budget=budget)
        completed = budget.work_units
        assert (
            prepare_barycentric_source_snapshot(barycentric_source, budget=budget)
            is snapshot
        )
        assert budget.work_units == completed
        assert not snapshot.barycentric_weights.flags.writeable
    restored_leaves = jax.tree_util.tree_leaves(barycentric_source)
    assert len(restored_leaves) == len(leaves)
    assert all(
        after is before for before, after in zip(leaves, restored_leaves, strict=True)
    )
    assert isinstance(barycentric_source.barycentric_weights, Array)
    assert coordinate_source_signature(barycentric_source) == signature


@pytest.mark.strict_jax
def test_spline_snapshot_reuse_preserves_dynamic_source_leaves(
    spline_source: SplineCellGeometryElement,
) -> None:
    leaves = jax.tree_util.tree_leaves(spline_source)
    signature = coordinate_source_signature(spline_source)
    work, memory = coordinate_source_snapshot_preparation_bounds(spline_source)
    with owned_coordinate_preparation(work, memory) as budget:
        snapshot = prepare_spline_source_snapshot(spline_source, budget=budget)
        completed = budget.work_units
        assert prepare_spline_source_snapshot(spline_source, budget=budget) is snapshot
        assert budget.work_units == completed
        assert not snapshot.u_knots.flags.writeable
        assert not snapshot.v_knots.flags.writeable
        assert not snapshot.weights.flags.writeable
    restored_leaves = jax.tree_util.tree_leaves(spline_source)
    assert len(restored_leaves) == len(leaves)
    assert all(
        after is before for before, after in zip(leaves, restored_leaves, strict=True)
    )
    assert isinstance(spline_source.u_knots, Array)
    assert isinstance(spline_source.v_knots, Array)
    assert isinstance(spline_source.weights, Array)
    assert coordinate_source_signature(spline_source) == signature


def test_restricted_affine_simplex_reuses_full_source_law_under_original_budget() -> None:
    root = coordinate_lagrange_element("triangle", 1)
    restriction = RestrictedCellGeometryElement(
        root,
        "triangle",
        np.diag(np.asarray((0.5, 0.25), dtype=np.float64)),
        np.asarray((0.125, 0.25), dtype=np.float64),
    )
    coefficients = (
        (Fraction(1, 3), Fraction(1, 5)),
        (Fraction(2, 7), Fraction(3, 11)),
        (Fraction(5, 13), Fraction(7, 17)),
    )
    with coordinate_reference_budget() as expanded:
        assert (
            coordinate_corner_images(restriction, coefficients, budget=expanded)
            is not None
        )
        expected = _prepare_coordinate_polynomials(
            restriction, coefficients, budget=expanded
        )
    with coordinate_reference_budget() as reduced:
        assert (
            coordinate_corner_images(restriction, coefficients, budget=reduced)
            is not None
        )
        actual = coordinate_polynomials(restriction, coefficients, budget=reduced)
    assert actual == expected and actual is not None
    assert reduced.work_units < expanded.work_units
    with coordinate_reference_budget() as cold:
        unprimed = coordinate_polynomials(restriction, coefficients, budget=cold)
    assert unprimed == expected
    assert cold.work_units <= reduced.work_units
    curved_root = coordinate_lagrange_element("triangle", 2)
    curved = RestrictedCellGeometryElement(
        curved_root,
        "triangle",
        np.eye(2, dtype=np.float64),
        np.zeros((2,), dtype=np.float64),
    )
    # This map has the same reference corners as the identity and a curved edge.
    curved_coefficients = (
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
        (Fraction(1, 2), Fraction(1, 8)),
        (Fraction(1, 2), Fraction(1, 2)),
        (Fraction(0), Fraction(1, 2)),
    )
    with coordinate_reference_budget() as nonlinear:
        preserved = coordinate_polynomials(curved, curved_coefficients, budget=nonlinear)
    assert preserved is not None
    assert any(sum(index) > 1 for polynomial in preserved for index in polynomial)
    reference = (Fraction(1, 4), Fraction(1, 3))
    material = (
        Fraction(1, 8) + reference[0] / 2,
        Fraction(1, 4) + reference[1] / 4,
    )
    with coordinate_reference_budget() as evaluation:
        image = tuple(
            expression_evaluate(value, reference, budget=evaluation) for value in actual
        )
    assert image == tuple(
        coefficients[0][axis]
        + material[0] * (coefficients[1][axis] - coefficients[0][axis])
        + material[1] * (coefficients[2][axis] - coefficients[0][axis])
        for axis in range(2)
    )


def test_affine_polynomial_reuse_preserves_exact_content_and_retained_storage() -> None:
    polynomial: Polynomial = {
        (2, 0): Fraction(3, 7),
        (1, 1): Fraction(-2, 5),
        (0, 0): Fraction(11, 13),
    }
    offset, factor = Fraction(1, 4), Fraction(1, 2)
    other_offset, other_factor = Fraction(-1, 8), Fraction(3, 4)
    arguments: tuple[Polynomial, ...] = (
        {(0, 0): offset, (1, 0): factor},
        {(0, 0): other_offset, (0, 1): other_factor},
    )
    a, b, c = polynomial.values()
    expected: Polynomial = {
        (0, 0): a * offset**2 + b * offset * other_offset + c,
        (1, 0): 2 * a * offset * factor + b * factor * other_offset,
        (0, 1): b * offset * other_factor,
        (2, 0): a * factor**2,
        (1, 1): b * factor * other_factor,
    }
    with coordinate_reference_budget() as budget:
        first = compose(polynomial, arguments, budget=budget)
        assert first == expected
        first_work = budget.work_units
        copied: Polynomial = {
            index: Fraction(
                *rational_coefficient(
                    value, role="fixture coefficient"
                ).as_integer_ratio()
            )
            for index, value in polynomial.items()
        }
        repeated = compose(copied, arguments, budget=budget)
        assert repeated == expected and repeated is not first
        assert 0 < budget.work_units - first_work < first_work
        profile_work = budget.work_units
        retained = budget.retained_basis_bytes
        budget.retain_basis((repeated,))
        assert budget.work_units - profile_work == 2
        assert budget.retained_basis_bytes > retained
        first[(2, 0)] = Fraction(999)
        assert compose(copied, arguments, budget=budget) == expected
        copied[(0, 0)] += 1
        changed_expected = expected.copy()
        changed_expected[(0, 0)] += 1
        assert compose(copied, arguments, budget=budget) == changed_expected
        changed_arguments = (arguments[0].copy(), arguments[1])
        changed_arguments[0][(0, 0)] += 1
        changed = compose(polynomial, changed_arguments, budget=budget)
        assert changed != expected
    with coordinate_reference_budget() as independent:
        assert compose(polynomial, changed_arguments, budget=independent) == changed


@pytest.mark.parametrize("radical_first", (False, True))
def test_bernstein_cache_preserves_equal_scalar_representations(
    radical_first: bool,
) -> None:
    coefficients = {
        (0,): Fraction(3, 7),
        (1,): Fraction(2, 5),
        (2,): Fraction(5, 11),
    }
    rational: Polynomial = {index: value for index, value in coefficients.items()}
    radical: Polynomial = {
        index: RadicalElement(value) for index, value in coefficients.items()
    }
    assert rational == radical
    ordered = (radical, rational) if radical_first else (rational, radical)
    expected = (
        coefficients[(0,)],
        coefficients[(0,)] + coefficients[(1,)] / 2,
        sum(coefficients.values(), start=Fraction(0)),
    )
    with coordinate_reference_budget() as budget:
        for polynomial in (*ordered, *ordered):
            controls = bernstein_coefficients(polynomial, "simplex", 1, budget=budget)
            assert controls == expected
            scalar_type = RadicalElement if polynomial is radical else Fraction
            assert all(type(value) is scalar_type for value in controls)


@pytest.mark.parametrize("radical_first", (False, True))
def test_coordinate_preparation_and_corner_caches_preserve_equal_scalar_representations(
    radical_first: bool,
) -> None:
    element = coordinate_lagrange_element("triangle", 1)
    rational = ((Fraction(2, 7),), (Fraction(3, 5),), (Fraction(5, 11),))
    radical = tuple(tuple(RadicalElement(value) for value in row) for row in rational)
    assert rational == radical
    ordered = (radical, rational) if radical_first else (rational, radical)
    with coordinate_reference_budget() as budget:
        _, first_key = coordinate_preparation_key(element, ordered[0], budget=budget)
        _, second_key = coordinate_preparation_key(element, ordered[1], budget=budget)
        assert first_key != second_key
        for bank in (*ordered, *ordered):
            prepared, _ = coordinate_preparation_key(element, bank, budget=budget)
            assert prepared == rational
            scalar_type = RadicalElement if bank is radical else Fraction
            assert all(type(value) is scalar_type for row in prepared for value in row)
            corners = coordinate_corner_images(element, bank, budget=budget)
            assert corners is not None
            assert corners == rational
            assert all(type(value) is scalar_type for row in corners for value in row)


@pytest.mark.parametrize("domain", ("simplex", "prism", "box"))
def test_bernstein_reuse_profiles_only_new_exact_content(
    domain: BernsteinDomain,
) -> None:
    polynomial: Polynomial = {
        (2, 0, 0): Fraction(3, 7),
        (0, 1, 0): Fraction(-2, 5),
        (0, 0, 1): Fraction(5, 11),
        (0, 0, 0): Fraction(11, 13),
    }
    with coordinate_reference_budget() as budget:
        original = bernstein_coefficients(polynomial, domain, 3, budget=budget)
        first_work = budget.work_units
        copied: Polynomial = {
            index: Fraction(
                *rational_coefficient(
                    value, role="fixture coefficient"
                ).as_integer_ratio()
            )
            for index, value in polynomial.items()
        }
        repeated = bernstein_coefficients(copied, domain, 3, budget=budget)
        assert repeated == original
        assert 0 < budget.work_units - first_work < first_work
        retained = budget.retained_basis_bytes
        profile_work = budget.work_units
        budget.retain_basis(repeated)
        assert budget.work_units == profile_work
        assert budget.retained_basis_bytes == retained
        copied[(0, 0, 0)] += 1
        changed = bernstein_coefficients(copied, domain, 3, budget=budget)
        assert changed == tuple(value + 1 for value in original)
    with coordinate_reference_budget() as independent:
        cold = bernstein_coefficients(polynomial, domain, 3, budget=independent)
        assert cold == original
        assert independent.work_units == first_work


def test_scoped_bernstein_controls_release_storage_and_reuse_request_preparation() -> (
    None
):
    polynomial: Polynomial = {
        (0, 0): Fraction(11, 13),
        (1, 0): Fraction(3, 7),
        (0, 1): Fraction(-2, 5),
    }
    with coordinate_reference_budget() as budget:
        source = bernstein_coefficients(polynomial, "simplex", 2, budget=budget)
        retained = budget.retained_basis_bytes
        changed = polynomial.copy()
        changed[(0, 0)] += 1
        with budget.polynomial_preparation_scope():
            assert (
                bernstein_coefficients(polynomial, "simplex", 2, budget=budget) is source
            )
            before = budget.work_units
            controls = bernstein_coefficients(changed, "simplex", 2, budget=budget)
            prepared_work = budget.work_units - before
            assert controls == tuple(value + 1 for value in source)
            before = budget.work_units
            assert (
                bernstein_coefficients(changed, "simplex", 2, budget=budget) is controls
            )
            assert 0 < budget.work_units - before < prepared_work
            assert budget.retained_basis_bytes == retained
            live = budget.temporary_bytes_upper
            assert live > 0
            with budget.polynomial_preparation_scope():
                nested = changed.copy()
                nested[(0, 0)] += 1
                assert bernstein_coefficients(
                    nested, "simplex", 2, budget=budget
                ) == tuple(value + 2 for value in source)
                assert budget.temporary_bytes_upper > live
            assert budget.temporary_bytes_upper == live
            del controls
        assert budget.temporary_bytes_upper == 0
        assert budget.retained_basis_bytes == retained
        assert bernstein_coefficients(polynomial, "simplex", 2, budget=budget) is source


def test_scoped_affine_snapshots_release_storage_without_changing_coefficients() -> None:
    polynomial: Polynomial = {(1,): Fraction(3, 7), (0,): Fraction(11, 13)}
    arguments: tuple[Polynomial, ...] = ({(0,): Fraction(1, 4), (1,): Fraction(1, 2)},)
    expected = {(0,): Fraction(11, 13) + Fraction(3, 28), (1,): Fraction(3, 14)}
    with coordinate_reference_budget() as budget:
        retained = budget.retained_basis_bytes
        with budget.polynomial_preparation_scope():
            before = budget.work_units
            first = compose(polynomial, arguments, budget=budget)
            prepared_work = budget.work_units - before
            assert first == expected
            before = budget.work_units
            repeated = compose(polynomial.copy(), arguments, budget=budget)
            assert repeated == expected and repeated is not first
            assert 0 < budget.work_units - before < prepared_work
            assert budget.retained_basis_bytes == retained
            assert budget.temporary_bytes_upper > 0
            first[(0,)] += 1
            assert compose(polynomial, arguments, budget=budget) == expected
            del first, repeated
        assert budget.temporary_bytes_upper == 0
        assert budget.retained_basis_bytes == retained
        with budget.polynomial_preparation_scope():
            assert compose(polynomial, arguments, budget=budget) == expected
        assert budget.temporary_bytes_upper == 0
        assert budget.retained_basis_bytes == retained


def test_polynomial_preparation_scope_restores_storage_after_refusal() -> None:
    polynomial: Polynomial = {(1,): Fraction(3, 7), (0,): Fraction(11, 13)}
    with owned_coordinate_preparation(10_000, 4096) as budget:
        with pytest.raises(CoordinateEnclosureResourceError) as caught:
            with budget.polynomial_preparation_scope():
                bernstein_coefficients(polynomial, "simplex", 1, budget=budget)
        assert caught.value.resource == "polynomial_storage"
        assert caught.value.limit == 4096
        assert caught.value.requested > caught.value.limit
        assert budget.retained_basis_bytes == budget.temporary_bytes_upper == 0
        assert budget.ticket.managed_bytes == budget.peak_bytes_upper
        with budget.ticket.managed_storage() as reusable:
            reusable.set_bound(4096)
            assert reusable.bound == 4096
        budget.reserve(1)
        assert budget.ticket.work_units == budget.work_units


def test_request_graph_retained_inside_polynomial_scope_outlives_temporary_memos() -> (
    None
):
    polynomial: Polynomial = {(1,): Fraction(3, 7), (0,): Fraction(11, 13)}
    arguments: tuple[Polynomial, ...] = ({(0,): Fraction(1, 4), (1,): Fraction(1, 2)},)
    expected = {(0,): Fraction(11, 13) + Fraction(3, 28), (1,): Fraction(3, 14)}
    with coordinate_reference_budget() as budget:
        with budget.polynomial_preparation_scope():
            result = compose(polynomial, arguments, budget=budget)
            budget.retain_basis((result,))
            retained = budget.retained_basis_bytes
            assert retained > 0
        assert result == expected
        assert budget.temporary_bytes_upper == 0
        assert budget.retained_basis_bytes == retained
        before = budget.work_units
        budget.retain_basis((result,))
        assert budget.work_units - before == 1


def test_coordinate_value_interning_preserves_true_retained_graph_accounting() -> None:
    point = (Fraction(3, 7), Fraction(5, 11), Fraction(13, 17))
    with coordinate_reference_budget() as budget:
        original = _retained_coordinate_point(point, budget=budget)
        assert original == point
        retained = budget.retained_basis_bytes
        before = budget.work_units
        copied = tuple(Fraction(value.numerator, value.denominator) for value in point)
        repeated = _retained_coordinate_point(copied, budget=budget)
        assert repeated == point
        assert budget.work_units - before == len(point)
        assert budget.retained_basis_bytes == retained
        assert budget.temporary_bytes_upper == 0
        before = budget.work_units
        budget.retain_basis(repeated)
        assert budget.work_units == before
        # Equal values in a separately retained allocation still own its bytes.
        budget.retain_basis(copied)
        assert budget.retained_basis_bytes > retained
        changed = (point[0], point[1], point[2] + 1)
        assert _retained_coordinate_point(changed, budget=budget) == changed
        assert changed != original
    with coordinate_reference_budget() as independent:
        _retained_coordinate_point(point, budget=independent)
        assert independent.retained_basis_bytes == retained


def test_coordinate_value_interning_compares_complete_content_on_hash_collision() -> None:
    first = (Fraction(1), Fraction(3, 7))
    second = (Fraction(1 + sys.hash_info.modulus), Fraction(3, 7))
    assert first != second and hash(first) == hash(second)
    with coordinate_reference_budget() as budget:
        original = _retained_coordinate_point(first, budget=budget)
        distinct = _retained_coordinate_point(second, budget=budget)
        assert original == first
        assert distinct == second
        assert distinct != original
        copied = tuple(Fraction(value.numerator, value.denominator) for value in first)
        assert _retained_coordinate_point(copied, budget=budget) == first


@pytest.mark.parametrize("offset", (Fraction(0), Fraction(5, 11)))
@pytest.mark.parametrize(
    ("domain", "degrees", "alpha"),
    (
        ("simplex", (16,), (4, 5)),
        ("prism", (16, 4), (4, 5, 2)),
        ("box", (16, 16), (4, 5)),
    ),
)
def test_bernstein_control_net_matches_independent_falling_factorial_weights(
    domain: BernsteinDomain,
    degrees: tuple[int, ...],
    alpha: tuple[int, ...],
    offset: Fraction,
) -> None:
    coefficient = Fraction(3, 7)
    polynomial: Polynomial = {alpha: coefficient}
    if offset:
        polynomial[(0,) * len(alpha)] = offset
    with coordinate_reference_budget() as budget:
        actual = expression_control_net(
            polynomial, domain, degrees, len(alpha), budget=budget
        )
    if domain == "box":
        lattice = tuple(product(*(range(degree + 1) for degree in degrees)))
    else:
        base = tuple(
            index
            for index in product(range(degrees[0] + 1), repeat=2)
            if sum(index) <= degrees[0]
        )
        lattice = (
            tuple((*index, height) for index in base for height in range(degrees[1] + 1))
            if domain == "prism"
            else base
        )

    def independent_weight(beta: tuple[int, ...]) -> Fraction:
        if any(b < a for a, b in zip(alpha, beta, strict=True)):
            return Fraction(0)
        if domain == "box":
            return Fraction(
                math.prod(math.comb(b, a) for a, b in zip(alpha, beta, strict=True)),
                math.prod(math.comb(n, a) for a, n in zip(alpha, degrees, strict=True)),
            )
        numerator = math.prod(
            math.factorial(b) // math.factorial(b - a)
            for a, b in zip(alpha[:2], beta[:2], strict=True)
        )
        order = sum(alpha[:2])
        denominator = math.factorial(degrees[0]) // math.factorial(degrees[0] - order)
        if domain == "prism":
            numerator *= math.comb(beta[2], alpha[2])
            denominator *= math.comb(degrees[1], alpha[2])
        return Fraction(numerator, denominator)

    expected = tuple(offset + coefficient * independent_weight(beta) for beta in lattice)
    assert actual == expected


@pytest.mark.parametrize(
    ("domain", "alpha", "zero_controls"),
    (
        ("simplex", (2, 0), 5),
        ("prism", (2, 0, 0), 5),
        ("box", (2, 0), 2),
    ),
)
def test_natural_bernstein_controls_seed_constant_before_quadratic_accumulation(
    domain: BernsteinDomain, alpha: tuple[int, ...], zero_controls: int
) -> None:
    offset, coefficient = Fraction(5, 11), Fraction(3, 7)
    polynomial: Polynomial = {alpha: coefficient, (0,) * len(alpha): offset}
    with coordinate_reference_budget() as budget:
        actual = bernstein_coefficients(polynomial, domain, len(alpha), budget=budget)
    # In degree two, x squared has only its x=1 Bernstein coefficient nonzero.
    assert actual == (offset,) * zero_controls + (offset + coefficient,)


@pytest.mark.parametrize("maximum_work", (7, 6))
def test_reference_rational_evaluation_uses_earned_denominator_once(
    maximum_work: int,
) -> None:
    numerator: Polynomial = {
        (0, 0): Fraction(2, 3),
        (1, 0): Fraction(3, 5),
        (0, 1): Fraction(-2, 7),
    }
    denominator: Polynomial = {
        (0, 0): Fraction(1),
        (1, 0): Fraction(1, 3),
        (0, 1): Fraction(-1, 5),
        (1, 1): Fraction(1, 11),
    }
    value = RationalPolynomial(numerator, denominator)
    point = (Fraction(1, 4), Fraction(1, 6))
    expected = (Fraction(2, 3) + Fraction(3, 20) - Fraction(1, 21)) / (
        1 + Fraction(1, 12) - Fraction(1, 30) + Fraction(1, 264)
    )
    with owned_coordinate_preparation(maximum_work, 8192) as budget:
        if maximum_work == 7:
            actual = expression_reference_evaluate(value, point, "simplex", budget=budget)
            assert actual == expected
            assert budget.work_units == 7
        else:
            with pytest.raises(CoordinateEnclosureResourceError) as caught:
                expression_reference_evaluate(value, point, "simplex", budget=budget)
            assert caught.value.requested == 7
            assert caught.value.completed == 4


@pytest.mark.parametrize("removable", (True, False))
def test_reference_zero_denominator_preserves_corner_limit_proof(
    removable: bool,
) -> None:
    numerator: Polynomial = {(1,): Fraction(2)} if removable else {(0,): Fraction(1)}
    value = RationalPolynomial(numerator, {(1,): Fraction(1)})
    with coordinate_reference_budget() as budget:
        if removable:
            assert expression_reference_evaluate(
                value, (Fraction(0),), "box", budget=budget
            ) == Fraction(2)
        else:
            with pytest.raises(RationalEnclosureError):
                expression_reference_evaluate(value, (Fraction(0),), "box", budget=budget)


@pytest.mark.parametrize("changed_last_denominator", (False, True))
def test_reference_bank_shares_only_equal_complete_denominator_content(
    changed_last_denominator: bool,
) -> None:
    denominator: Polynomial = {
        (0, 0): Fraction(1),
        (1, 0): Fraction(1, 3),
        (0, 1): Fraction(-1, 5),
        (1, 1): Fraction(1, 11),
    }
    denominators = tuple(denominator.copy() for _ in range(4))
    if changed_last_denominator:
        denominators[-1][(0, 0)] += 1
    basis = tuple(
        RationalPolynomial(
            {
                (0, 0): Fraction(index + 1, 3),
                (1, 0): Fraction(3, 5),
                (0, 1): Fraction(-2, 7),
            },
            current,
        )
        for index, current in enumerate(denominators)
    )
    points = (
        (Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(1)),
        (Fraction(1), Fraction(0)),
        (Fraction(1), Fraction(1)),
    )
    expected = tuple(
        tuple(
            (Fraction(index + 1, 3) + 3 * x / 5 - 2 * y / 7)
            / (
                1
                + int(changed_last_denominator and index == 3)
                + x / 3
                - y / 5
                + x * y / 11
            )
            for index in range(4)
        )
        for x, y in points
    )
    required_work = 82 if changed_last_denominator else 69
    with owned_coordinate_preparation(required_work, 64_000) as budget:
        prepared = _prepare_reference_denominators(basis, budget=budget)
        actual = tuple(
            _reference_basis_values(
                basis, points, "quadrilateral", prepared, budget=budget
            )
        )
        assert actual == expected
        assert budget.work_units == required_work


@pytest.mark.parametrize("removable", (True, False))
def test_reference_bank_keeps_shared_zero_denominator_corner_proofs(
    removable: bool,
) -> None:
    denominator: Polynomial = {(1,): Fraction(1)}
    second: Polynomial = {(1,): Fraction(4)} if removable else {(0,): Fraction(1)}
    basis = (
        RationalPolynomial({(1,): Fraction(2)}, denominator),
        RationalPolynomial(second, denominator.copy()),
    )
    with coordinate_reference_budget() as budget:
        prepared = _prepare_reference_denominators(basis, budget=budget)
        if removable:
            assert tuple(
                _reference_basis_values(
                    basis, ((Fraction(0),),), "interval", prepared, budget=budget
                )
            ) == ((Fraction(2), Fraction(4)),)
        else:
            with pytest.raises(RationalEnclosureError):
                tuple(
                    _reference_basis_values(
                        basis, ((Fraction(0),),), "interval", prepared, budget=budget
                    )
                )


def test_rational_component_pullback_reuses_only_complete_denominator_snapshots() -> None:
    arguments: tuple[Polynomial, ...] = (
        {(0, 0): Fraction(1, 5), (1, 0): Fraction(1), (0, 1): Fraction(1, 3)},
        {(1, 0): Fraction(-1, 7), (0, 1): Fraction(1)},
    )
    denominator: Polynomial = {
        (0, 0): Fraction(1),
        (1, 0): Fraction(1, 3),
        (0, 1): Fraction(1, 5),
        (2, 0): Fraction(1, 11),
        (1, 1): Fraction(1, 13),
        (0, 2): Fraction(1, 17),
    }
    with coordinate_reference_budget() as reference:
        expected = compose(denominator, arguments, budget=reference)
    with coordinate_reference_budget() as budget:
        composition = ExpressionComposition(arguments, budget=budget)
        before = budget.work_units
        first = composition(
            RationalPolynomial({(0, 0): Fraction(1)}, denominator), budget=budget
        )
        cold_work = budget.work_units - before
        assert isinstance(first, RationalPolynomial)
        assert first.denominator == expected
        first.denominator[(0, 0)] += 100
        before = budget.work_units
        second = composition(
            RationalPolynomial({(0, 0): Fraction(2)}, denominator.copy()), budget=budget
        )
        assert isinstance(second, RationalPolynomial)
        assert second.denominator == expected
        assert 0 < budget.work_units - before < cold_work
        changed = denominator.copy()
        changed[(0, 0)] += 1
        third = composition(
            RationalPolynomial({(0, 0): Fraction(3)}, changed), budget=budget
        )
        assert isinstance(third, RationalPolynomial)
        changed_expected = expected.copy()
        changed_expected[(0, 0)] += 1
        assert third.denominator == changed_expected


@pytest.mark.parametrize("changed_last_denominator", (False, True))
def test_rational_jacobian_reuses_only_equal_complete_denominator_jets(
    changed_last_denominator: bool,
) -> None:
    denominator: Polynomial = {
        (0, 0): Fraction(1),
        (1, 0): Fraction(1, 3),
        (0, 1): Fraction(1, 5),
        (1, 1): Fraction(1, 7),
    }
    denominators = tuple(denominator.copy() for _ in range(3))
    if changed_last_denominator:
        denominators[-1][(0, 0)] += 1
    coordinates = tuple(
        RationalPolynomial(
            {
                (0, 0): Fraction(index + 1),
                (1, 0): Fraction(index + 2),
                (0, 1): Fraction(2 * index + 3),
            },
            current,
        )
        for index, current in enumerate(denominators)
    )
    with coordinate_reference_budget() as cold:
        expected = tuple(
            tuple(expression_derivative(value, axis, budget=cold) for axis in range(2))
            for value in coordinates
        )
    with coordinate_reference_budget() as shared:
        actual = expression_physical_jacobian(
            coordinates, "quadrilateral", 2, budget=shared
        )
    assert 0 < shared.work_units < cold.work_units
    point = (Fraction(1, 4), Fraction(1, 6))
    x, y = point
    with coordinate_reference_budget() as evaluation:
        for index, (actual_row, expected_row) in enumerate(
            zip(actual, expected, strict=True)
        ):
            denominator_value = (
                1
                + int(changed_last_denominator and index == 2)
                + x / 3
                + y / 5
                + x * y / 7
            )
            numerator_value = index + 1 + (index + 2) * x + (2 * index + 3) * y
            oracle = (
                (
                    (index + 2) * denominator_value
                    - numerator_value * (Fraction(1, 3) + y / 7)
                )
                / denominator_value**2,
                (
                    (2 * index + 3) * denominator_value
                    - numerator_value * (Fraction(1, 5) + x / 7)
                )
                / denominator_value**2,
            )
            for axis, (actual_value, expected_value) in enumerate(
                zip(actual_row, expected_row, strict=True)
            ):
                assert (
                    expression_evaluate(actual_value, point, budget=evaluation)
                    == oracle[axis]
                )
                assert (
                    expression_evaluate(expected_value, point, budget=evaluation)
                    == oracle[axis]
                )


def test_source_bound_denominator_groups_reuse_across_distinct_corner_charts() -> None:
    knots = np.asarray((0.0, 0.0, 1.0, 1.0), dtype=np.float64)
    weights = np.asarray(((1.0, 0.9), (1.1, 1.0)), dtype=np.float64)
    source = SplineCellGeometryElement(
        knots, knots, weights, 1, 1, (1, 1), "denominator-bank-test", "source"
    )
    matrix = np.diag(np.asarray((0.5, 0.5), dtype=np.float64))
    first = RestrictedCellGeometryElement(
        source, "quadrilateral", matrix, np.asarray((0.0, 0.0), dtype=np.float64)
    )
    second = RestrictedCellGeometryElement(
        source, "quadrilateral", matrix, np.asarray((0.5, 0.5), dtype=np.float64)
    )
    coefficients = (
        (Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(1)),
        (Fraction(1), Fraction(0)),
        (Fraction(1), Fraction(1)),
    )
    changed_weights = weights.copy()
    changed_weights[0, 1] = np.nextafter(changed_weights[0, 1], np.inf)
    changed_source = SplineCellGeometryElement(
        knots,
        knots,
        changed_weights,
        1,
        1,
        (1, 1),
        "denominator-bank-test",
        "source",
    )
    changed_chart = RestrictedCellGeometryElement(
        changed_source,
        "quadrilateral",
        matrix,
        np.asarray((0.5, 0.5), dtype=np.float64),
    )
    with coordinate_reference_budget() as shared:
        first_images = coordinate_corner_images(first, coefficients, budget=shared)
        second_images = coordinate_corner_images(second, coefficients, budget=shared)
        original_work = shared.work_units
        changed_shared = coordinate_corner_images(
            changed_chart, coefficients, budget=shared
        )
    with coordinate_reference_budget() as cold_first:
        assert (
            coordinate_corner_images(first, coefficients, budget=cold_first)
            == first_images
        )
    with coordinate_reference_budget() as cold_second:
        assert (
            coordinate_corner_images(second, coefficients, budget=cold_second)
            == second_images
        )
    assert original_work < cold_first.work_units + cold_second.work_units
    with coordinate_reference_budget() as changed:
        changed_images = coordinate_corner_images(
            changed_chart, coefficients, budget=changed
        )
    assert changed_shared == changed_images
    assert changed_images != second_images


@pytest.mark.parametrize("size", (2, 3))
def test_determinant_accumulates_ordered_signed_products_without_replaying_sums(
    size: int,
) -> None:
    u: Polynomial = {(1, 0): Fraction(1)}
    v: Polynomial = {(0, 1): Fraction(1)}
    one: Polynomial = {(0, 0): Fraction(1)}
    zero: Polynomial = {}
    matrix: tuple[tuple[Polynomial, ...], ...]
    if size == 2:
        matrix = ((u, one), (one, v))
        expected: Polynomial = {(1, 1): Fraction(1), (0, 0): Fraction(-1)}
    else:
        matrix = (
            (u, one, zero),
            (zero, v, one),
            (one, zero, {(1, 0): Fraction(1), (0, 1): Fraction(1)}),
        )
        expected = {(2, 1): Fraction(1), (1, 2): Fraction(1), (0, 0): Fraction(1)}
    with coordinate_reference_budget() as shared:
        actual = determinant(matrix, budget=shared)
    with coordinate_reference_budget() as repeated:
        legacy: Polynomial = {}
        for permutation in permutations(range(size)):
            factors = tuple(matrix[row][permutation[row]] for row in range(size))
            if any(not factor for factor in factors):
                continue
            sign = (-1) ** sum(
                permutation[row] > permutation[column]
                for row in range(size)
                for column in range(row + 1, size)
            )
            term = math_product_polynomials(factors, 2, budget=repeated)
            legacy = add(legacy, scale(term, sign, budget=repeated), budget=repeated)
    assert actual == legacy == expected
    assert tuple(actual.items()) == tuple(legacy.items())
    assert 0 < shared.work_units < repeated.work_units


@pytest.mark.parametrize("offset", (None, Fraction(3, 7)))
def test_sparse_origin_evaluation_visits_only_its_constant_coefficient(
    offset: Fraction | None,
) -> None:
    polynomial: Polynomial = {(7, 3): Fraction(-2, 3), (2, 11): Fraction(5, 7)}
    if offset is not None:
        polynomial[(0, 0)] = offset
    with owned_coordinate_preparation(1, 0) as budget:
        actual = evaluate(polynomial, (Fraction(0), Fraction(0)), budget=budget)
        assert actual == (Fraction(0) if offset is None else offset)
        assert budget.work_units == 1
        assert budget.temporary_bytes_upper == 0


def test_sparse_origin_evaluation_admits_its_constant_lookup_before_work() -> None:
    polynomial: Polynomial = {(7, 3): Fraction(-2, 3), (0, 0): Fraction(5, 7)}
    with owned_coordinate_preparation(0, 0) as budget:
        with pytest.raises(CoordinateEnclosureResourceError) as caught:
            evaluate(polynomial, (Fraction(0), Fraction(0)), budget=budget)
        assert caught.value.requested == 1
        assert caught.value.completed == 0
        assert budget.work_units == 0


def test_empty_origin_evaluation_does_not_admit_unneeded_work() -> None:
    with owned_coordinate_preparation(0, 0) as budget:
        assert evaluate({}, (Fraction(0), Fraction(0)), budget=budget) == 0
        assert budget.work_units == 0


def test_empty_accumulation_does_not_reprofile_unchanged_target() -> None:
    target: Polynomial = {(2, 3): Fraction(17, 29)}
    coefficient = target[(2, 3)]
    with owned_coordinate_preparation(0, 0) as budget:
        _accumulate(target, {}, budget=budget)
        assert target == {(2, 3): Fraction(17, 29)}
        assert target[(2, 3)] is coefficient
        assert budget.work_units == 0


@pytest.mark.parametrize("first_is_empty", (False, True))
def test_zero_operand_addition_precharges_only_its_fresh_dictionary(
    first_is_empty: bool,
) -> None:
    source: Polynomial = {
        (2, 3): Fraction(17, 29),
        (1, 0): Fraction(-7, 11),
        (0, 0): Fraction(0),
    }
    expected = {
        index: value for index, value in source.items() if value or not first_is_empty
    }
    with owned_coordinate_preparation(3, 448) as budget:
        actual = (
            add({}, source, budget=budget)
            if first_is_empty
            else add(source, {}, budget=budget)
        )
        assert budget.work_units == 3
        assert actual == expected
        assert actual is not source
        assert all(value is source[index] for index, value in actual.items())
        actual[(2, 3)] = Fraction(1)
        assert source[(2, 3)] == Fraction(17, 29)


@pytest.mark.parametrize("exponent", (1, 2, 3, 4, 8))
@pytest.mark.parametrize("cancellation", (False, True))
def test_power_reuses_its_base_without_changing_legacy_order(
    exponent: int, cancellation: bool
) -> None:
    polynomial: Polynomial = (
        {
            (2, 0): Fraction(1),
            (1, 0): Fraction(2),
            (0, 0): Fraction(-2),
            (0, 1): Fraction(3, 5),
        }
        if cancellation
        else {
            (0, 1): Fraction(-2, 3),
            (0, 0): Fraction(5, 7),
            (1, 0): Fraction(3, 11),
            (2, 0): Fraction(1, 5),
        }
    )
    with coordinate_reference_budget() as reused:
        actual = power(polynomial, exponent, 2, budget=reused)
    with coordinate_reference_budget() as repeated:
        legacy = constant(1, 2, budget=repeated)
        for _ in range(exponent):
            legacy = multiply(legacy, polynomial, budget=repeated)
    assert tuple(actual.items()) == tuple(legacy.items())
    assert actual is not polynomial
    assert 0 < reused.work_units < repeated.work_units
    point = (Fraction(2, 7), Fraction(3, 11))
    with coordinate_reference_budget() as reference:
        assert evaluate(actual, point, budget=reference) == (
            evaluate(polynomial, point, budget=reference) ** exponent
        )


@pytest.mark.parametrize("refined", (False, True))
def test_p1_exact_corner_images_preserve_all_three_corners_including_origin(
    refined: bool,
) -> None:
    root = coordinate_lagrange_element("triangle", 1)
    identity = tuple(
        tuple(Fraction(int(row == column)) for column in range(3)) for row in range(3)
    )
    weights = (
        (
            (Fraction(1), Fraction(0), Fraction(0)),
            (Fraction(1, 2), Fraction(1, 2), Fraction(0)),
            (Fraction(0), Fraction(0), Fraction(1)),
        )
        if refined
        else identity
    )
    element = (
        BarycentricCellGeometryElement(root, np.asarray(weights, dtype=np.float64))
        if refined
        else root
    )
    last = (
        Fraction(3152519739159347, 4503599627370496),
        Fraction(5404319552844595, 18014398509481984),
    )
    coefficients = (
        (Fraction(1), Fraction(0)),
        (Fraction(1), Fraction(1)),
        last,
    )
    expected = (
        coefficients[0],
        (Fraction(1), Fraction(1, 2)) if refined else coefficients[1],
        coefficients[2],
    )
    with coordinate_reference_budget() as budget:
        assert coordinate_corner_images(element, identity, budget=budget) == weights
        assert coordinate_corner_images(element, coefficients, budget=budget) == expected
        polynomials = coordinate_polynomials(element, coefficients, budget=budget)
        assert polynomials is not None
        for point, image in zip(
            (
                (Fraction(0), Fraction(0)),
                (Fraction(1), Fraction(0)),
                (Fraction(0), Fraction(1)),
            ),
            expected,
            strict=True,
        ):
            assert (
                tuple(
                    expression_evaluate(polynomial, point, budget=budget)
                    for polynomial in polynomials
                )
                == image
            )


@pytest.mark.parametrize(
    ("domain", "dimension"),
    (("simplex", 2), ("simplex", 3), ("box", 2), ("box", 3), ("prism", 3)),
)
@pytest.mark.parametrize(
    "coefficient", (Fraction(0), Fraction(17, 29), Fraction(-(2**200), 31))
)
def test_constant_control_net_needs_only_its_earned_coefficient(
    domain: BernsteinDomain, dimension: int, coefficient: Fraction
) -> None:
    polynomial: Polynomial = {(0,) * dimension: coefficient}
    with owned_coordinate_preparation(1, 256) as budget:
        actual = bernstein_coefficients(polynomial, domain, dimension, budget=budget)
        assert actual == (coefficient,)
        assert actual[0] is coefficient
        assert budget.work_units == 1
        assert budget.retained_basis_bytes == 0
        assert not budget.bernstein_coefficient_cache
    with coordinate_reference_budget() as reference:
        expected = _prepare_bernstein_coefficients(
            polynomial, domain, dimension, budget=reference
        )
    assert actual == expected


def test_constant_control_net_still_admits_its_only_visit() -> None:
    with owned_coordinate_preparation(0, 256) as budget:
        with pytest.raises(CoordinateEnclosureResourceError) as caught:
            bernstein_coefficients(
                {(0, 0): Fraction(17, 29)}, "simplex", 2, budget=budget
            )
        assert caught.value.requested == 1
        assert caught.value.completed == 0
        assert budget.work_units == budget.retained_basis_bytes == 0


def test_constant_control_net_borrows_immutable_values_without_stale_source_reuse() -> (
    None
):
    polynomial: Polynomial = {(0, 0): Fraction(17, 29)}
    with owned_coordinate_preparation(2, 512) as budget:
        first = bernstein_coefficients(polynomial, "simplex", 2, budget=budget)
        polynomial[(0, 0)] = Fraction(17, 29) + Fraction(1, 2**200)
        second = bernstein_coefficients(polynomial, "simplex", 2, budget=budget)
        assert first == (Fraction(17, 29),)
        assert second == (polynomial[(0, 0)],)
        assert first != second
        assert budget.work_units == 2
        assert budget.retained_basis_bytes == 0


@pytest.mark.parametrize("dimension", (0, 1, 2))
def test_constant_control_net_does_not_bypass_prism_dimension_failure(
    dimension: int,
) -> None:
    polynomial: Polynomial = {(0,) * dimension: Fraction(17, 29)}
    with coordinate_reference_budget() as reference:
        with pytest.raises(IndexError):
            _prepare_bernstein_coefficients(
                polynomial, "prism", dimension, budget=reference
            )
    with coordinate_reference_budget() as budget:
        with pytest.raises(IndexError):
            bernstein_coefficients(polynomial, "prism", dimension, budget=budget)


def test_structural_bernstein_key_reuses_full_ordered_fraction_content() -> None:
    polynomial: Polynomial = {(0, 0): Fraction(1), (0, 1): Fraction(-1, 64)}
    collision = Fraction(1 + sys.hash_info.modulus)
    assert hash(collision) == hash(Fraction(1)) and collision != 1
    with coordinate_reference_budget() as budget:
        first = bernstein_coefficients(polynomial, "simplex", 2, budget=budget)
        before = budget.work_units
        assert (
            bernstein_coefficients(dict(polynomial), "simplex", 2, budget=budget) is first
        )
        assert budget.work_units - before == len(polynomial)
        assert len(budget.bernstein_coefficient_cache) == 1
        first_key = _bernstein_preparation_key(polynomial, "simplex", 2, budget=budget)
        changed = dict(polynomial)
        changed[(0, 0)] = collision
        changed_key = _bernstein_preparation_key(changed, "simplex", 2, budget=budget)
        assert hash(first_key) == hash(changed_key) and first_key != changed_key
        second = bernstein_coefficients(changed, "simplex", 2, budget=budget)
        assert second == tuple(value + sys.hash_info.modulus for value in first)
        assert len(budget.bernstein_coefficient_cache) == 2
        reordered = dict(reversed(tuple(polynomial.items())))
        assert bernstein_coefficients(reordered, "simplex", 2, budget=budget) == first
        assert len(budget.bernstein_coefficient_cache) == 3


@pytest.mark.parametrize("constant", (False, True))
@pytest.mark.parametrize("coefficient", (1, True, 1.0, np.int64(1)))
def test_bernstein_cache_refuses_noncanonical_equal_coefficient_types(
    constant: bool, coefficient: object
) -> None:
    polynomial: Polynomial = {(0, 0): Fraction(1)}
    if not constant:
        polynomial[(0, 1)] = Fraction(-1, 64)
    with coordinate_reference_budget() as budget:
        first = bernstein_coefficients(polynomial, "simplex", 2, budget=budget)
        entries = len(budget.bernstein_coefficient_cache)
        malformed = dict(polynomial)
        dict.__setitem__(malformed, (0, 0), coefficient)
        with pytest.raises(TypeError, match="Fraction values without subclasses"):
            bernstein_coefficients(malformed, "simplex", 2, budget=budget)
        assert len(budget.bernstein_coefficient_cache) == entries
        assert bernstein_coefficients(polynomial, "simplex", 2, budget=budget) == first


@pytest.mark.parametrize("index", ((True, 0), (1.0, 0), (np.int64(1), 0)))
def test_bernstein_cache_refuses_noncanonical_equal_exponent_types(
    index: tuple[object, object],
) -> None:
    polynomial: Polynomial = {(1, 0): Fraction(1), (0, 0): Fraction(2)}
    with coordinate_reference_budget() as budget:
        first = bernstein_coefficients(polynomial, "simplex", 2, budget=budget)
        entries = len(budget.bernstein_coefficient_cache)
        malformed: Polynomial = {}
        dict.__setitem__(malformed, index, Fraction(1))
        dict.__setitem__(malformed, (0, 0), Fraction(2))
        with pytest.raises(TypeError, match="built-in tuples"):
            bernstein_coefficients(malformed, "simplex", 2, budget=budget)
        assert len(budget.bernstein_coefficient_cache) == entries
        assert bernstein_coefficients(polynomial, "simplex", 2, budget=budget) is first


@pytest.mark.parametrize("work_limit", (1, 2))
def test_disjoint_accumulation_admits_probe_and_copy_before_mutation(
    work_limit: int,
) -> None:
    target: Polynomial = {(0, 0): Fraction(1, 2)}
    second: Polynomial = {(0, 1): Fraction(1, 2)}
    with owned_coordinate_preparation(work_limit, 384) as budget:
        if work_limit == 1:
            with pytest.raises(CoordinateEnclosureResourceError) as caught:
                _accumulate(target, second, budget=budget)
            assert caught.value.requested == 2
            assert caught.value.completed == 1
            assert target == {(0, 0): Fraction(1, 2)}
        else:
            _accumulate(target, second, budget=budget)
            assert target == {(0, 0): Fraction(1, 2), (0, 1): Fraction(1, 2)}
            assert target[(0, 1)] is second[(0, 1)]
            assert budget.work_units == 2
        assert budget.retained_basis_bytes == 0


def test_disjoint_accumulation_preserves_zero_filtering_and_existing_zero_entries() -> (
    None
):
    target: Polynomial = {(0, 0): Fraction(2, 3), (1, 0): Fraction(0)}
    second: Polynomial = {(0, 1): Fraction(17, 29), (1, 1): Fraction(0)}
    with owned_coordinate_preparation(4, 512) as budget:
        _accumulate(target, second, budget=budget)
        assert target == {
            (0, 0): Fraction(2, 3),
            (1, 0): Fraction(0),
            (0, 1): Fraction(17, 29),
        }
        assert target[(0, 1)] is second[(0, 1)]
        assert budget.work_units == 4


@pytest.mark.parametrize("prepared", (False, True))
@pytest.mark.parametrize(
    ("coefficient", "expected"),
    ((Fraction(1, 3), Fraction(5, 6)), (Fraction(-1, 2), None)),
)
def test_overlapping_accumulation_keeps_exact_arithmetic_after_its_charged_probe(
    prepared: bool, coefficient: Fraction, expected: Fraction | None
) -> None:
    target: Polynomial = {(0, 0): Fraction(1, 2)}
    second: Polynomial = {(0, 0): coefficient}
    with owned_coordinate_preparation(4, 8192) as budget:
        _accumulate(
            target,
            second,
            target_profile=(1, 2) if prepared else None,
            budget=budget,
        )
        assert target == ({} if expected is None else {(0, 0): expected})
        assert budget.work_units == 4


@pytest.mark.parametrize("negative_second", (False, True))
def test_owned_signed_reduction_consumes_earned_weight_bounds_once(
    negative_second: bool,
) -> None:
    bank: tuple[Polynomial, ...] = (
        {(0, 0): Fraction(3, 7), (1, 0): Fraction(1, 2)},
        {(0, 0): Fraction(3, 7), (0, 1): Fraction(-2, 5)},
    )
    weights = (Fraction(1), Fraction(-1 if negative_second else 1))
    active_terms = sum(map(len, bank))
    scaled_terms = len(bank[1]) if negative_second else 0
    with coordinate_reference_budget() as profiled:
        numerator, denominator = coefficient_profile(bank, budget=profiled)
        actual = _linear_combination_profiled(
            bank,
            weights,
            dimension=2,
            coefficient_bits=numerator
            + denominator
            + 2
            + max(active_terms, 1).bit_length(),
            active_terms=active_terms,
            scaled_terms=scaled_terms,
            budget=profiled,
        )
    with coordinate_reference_budget() as repeated:
        expected = linear_combinations(bank, (weights,), budget=repeated)[0]
    assert tuple(actual.items()) == tuple(expected.items())
    assert repeated.work_units - profiled.work_units == len(weights)
    assert bank == (
        {(0, 0): Fraction(3, 7), (1, 0): Fraction(1, 2)},
        {(0, 0): Fraction(3, 7), (0, 1): Fraction(-2, 5)},
    )


@pytest.mark.parametrize("dimension", (0, 2, 3))
@pytest.mark.parametrize("value", (0, Fraction(0)))
def test_zero_constant_has_no_coefficient_or_index_preparation(
    dimension: int, value: int | Fraction
) -> None:
    with owned_coordinate_preparation(0, 0) as budget:
        assert constant(value, dimension, budget=budget) == {}
        assert budget.work_units == budget.required_work_units == 0
        assert budget.temporary_bytes_upper == budget.retained_basis_bytes == 0


@pytest.mark.parametrize("value", (1, -1, Fraction(1, 7)))
def test_nonzero_constant_still_admits_its_coefficient_visit(
    value: int | Fraction,
) -> None:
    with owned_coordinate_preparation(0, 0) as budget:
        with pytest.raises(CoordinateEnclosureResourceError) as caught:
            constant(value, 2, budget=budget)
        assert caught.value.requested == 1
        assert caught.value.completed == 0
        assert budget.work_units == 0


def test_zero_constant_preserves_noncanonical_subclass_and_dimension_admission() -> None:
    class FractionSubtype(Fraction):
        pass

    class IntegerSubtype(int):
        pass

    for value, dimension in (
        (FractionSubtype(0), 2),
        (IntegerSubtype(0), 2),
        (False, 2),
        (Fraction(0), True),
    ):
        with owned_coordinate_preparation(1, 256) as budget:
            assert constant(value, dimension, budget=budget) == {}
            assert budget.work_units == 1
