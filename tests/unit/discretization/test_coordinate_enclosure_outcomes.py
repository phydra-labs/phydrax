#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Rational enclosure outcomes preserve values, refusal categories, and accounting."""

from collections.abc import Iterator
from contextlib import contextmanager
from fractions import Fraction
from math import factorial

import numpy as np
import pytest

from phydrax._resource_ticket import HostBatchTicket
from phydrax.discretization import (
    CellGeometrySpec,
    CellMesh,
    CellValidityPolicy,
    CellValidityStatus,
    certify_cell_geometry_validity,
)
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
    RationalEnclosureError,
)
from phydrax.discretization._coordinate_enclosure import (
    determinant,
    divide_polynomial,
    Expression,
    expression_bernstein_coefficients,
    expression_bernstein_coefficients_or_none,
    expression_evaluate,
    expression_evaluate_or_none,
    expression_gram_determinant,
    Polynomial,
    RationalPolynomial,
)
from tests._support.coordinate_references import coordinate_reference_budget


@contextmanager
def _zero_work_budget() -> Iterator[CoordinateEnclosureBudget]:
    """Admit no arithmetic work, so the first coefficient charge refuses."""
    ticket = HostBatchTicket(0, bytes_upper=1 << 20)
    budget = CoordinateEnclosureBudget(
        ticket, maximum_work_units=0, maximum_memory_bytes=1 << 20
    )
    with ticket, budget.activate():
        yield budget


@pytest.mark.parametrize(
    ("value", "point", "expected"),
    (
        pytest.param(
            {(0,): Fraction(1), (1,): Fraction(2)},
            (Fraction(1, 2),),
            Fraction(2),
            id="polynomial",
        ),
        pytest.param(
            RationalPolynomial(
                {(0,): Fraction(2)}, {(0,): Fraction(1), (1,): Fraction(1)}
            ),
            (Fraction(1),),
            Fraction(1),
            id="rational",
        ),
        pytest.param(
            RationalPolynomial({}, {(0,): Fraction(1)}),
            (Fraction(0),),
            Fraction(0),
            id="zero-value",
        ),
    ),
)
def test_evaluation_outcome_preserves_fraction_and_charge_order(
    value: Expression, point: tuple[Fraction, ...], expected: Fraction
) -> None:
    with coordinate_reference_budget() as strict:
        assert expression_evaluate(value, point, budget=strict) == expected
    with coordinate_reference_budget() as outcome:
        assert expression_evaluate_or_none(value, point, budget=outcome) == expected
    assert outcome.work_units == strict.work_units
    assert outcome.peak_bytes_upper == strict.peak_bytes_upper


def test_zero_denominator_outcome_preserves_throwing_refusal_work() -> None:
    value = RationalPolynomial({(0,): Fraction(1)}, {(1,): Fraction(1)})
    point = (Fraction(0),)
    with coordinate_reference_budget() as strict:
        with pytest.raises(
            RationalEnclosureError, match="coordinate_enclosure.evaluation"
        ):
            expression_evaluate(value, point, budget=strict)
    with coordinate_reference_budget() as outcome:
        assert expression_evaluate_or_none(value, point, budget=outcome) is None
    assert outcome.work_units == strict.work_units == 1
    assert outcome.peak_bytes_upper == strict.peak_bytes_upper == 0


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        pytest.param(
            {(0,): Fraction(1), (1,): Fraction(1)},
            (Fraction(1), Fraction(2)),
            id="polynomial",
        ),
        pytest.param(
            RationalPolynomial(
                {(0,): Fraction(2), (1,): Fraction(1)},
                {(0,): Fraction(1), (1,): Fraction(1)},
            ),
            (Fraction(2), Fraction(3, 2)),
            id="positive-denominator",
        ),
        pytest.param(
            RationalPolynomial(
                {(0,): Fraction(-2), (1,): Fraction(-1)},
                {(0,): Fraction(-1), (1,): Fraction(-1)},
            ),
            (Fraction(2), Fraction(3, 2)),
            id="negative-denominator",
        ),
        pytest.param(
            RationalPolynomial({(1,): Fraction(1)}, {(1,): Fraction(1)}),
            (Fraction(1),),
            id="coincident-zero-control",
        ),
    ),
)
def test_control_outcome_preserves_bounds_and_charge_order(
    value: Expression, expected: tuple[Fraction, ...]
) -> None:
    with coordinate_reference_budget() as strict:
        assert (
            expression_bernstein_coefficients(value, "box", 1, budget=strict) == expected
        )
    with coordinate_reference_budget() as outcome:
        assert (
            expression_bernstein_coefficients_or_none(value, "box", 1, budget=outcome)
            == expected
        )
    assert outcome.work_units == strict.work_units
    assert outcome.peak_bytes_upper == strict.peak_bytes_upper


@pytest.mark.parametrize(
    "value",
    (
        pytest.param(
            RationalPolynomial(
                {(0,): Fraction(1)}, {(0,): Fraction(-1), (1,): Fraction(2)}
            ),
            id="mixed-control-signs",
        ),
        pytest.param(
            RationalPolynomial({(0,): Fraction(1)}, {(1,): Fraction(1)}),
            id="nonzero-numerator-at-zero-weight",
        ),
        pytest.param(
            RationalPolynomial({(0,): Fraction(1)}, {(0,): Fraction(0)}),
            id="all-zero-controls",
        ),
    ),
)
def test_unresolved_control_outcome_preserves_throwing_refusal_work(
    value: RationalPolynomial,
) -> None:
    with coordinate_reference_budget() as strict:
        with pytest.raises(
            RationalEnclosureError, match="coordinate_enclosure.denominator"
        ):
            expression_bernstein_coefficients(value, "box", 1, budget=strict)
    with coordinate_reference_budget() as outcome:
        assert (
            expression_bernstein_coefficients_or_none(value, "box", 1, budget=outcome)
            is None
        )
    assert outcome.work_units == strict.work_units
    assert outcome.peak_bytes_upper == strict.peak_bytes_upper


def test_evaluation_outcome_propagates_resource_refusal() -> None:
    value = RationalPolynomial({(0,): Fraction(1)}, {(1,): Fraction(1)})
    with _zero_work_budget() as budget:
        with pytest.raises(CoordinateEnclosureResourceError) as caught:
            expression_evaluate_or_none(value, (Fraction(0),), budget=budget)
    assert caught.value.resource == "coefficient_work"
    assert caught.value.requested == 1
    assert caught.value.completed == 0


def test_control_outcome_propagates_resource_refusal() -> None:
    value = RationalPolynomial({(0,): Fraction(1)}, {(1,): Fraction(1)})
    with _zero_work_budget() as budget:
        with pytest.raises(CoordinateEnclosureResourceError) as caught:
            expression_bernstein_coefficients_or_none(value, "box", 1, budget=budget)
    assert caught.value.resource == "coefficient_work"
    assert caught.value.requested > 0
    assert caught.value.completed == 0


def test_determinant_parameter_rank_is_independent_of_matrix_size() -> None:
    # Cofactor expansion of [[1+x,2s,0],[0,2+y,3s],[4s,0,3+z]]
    # gives (1+x)(2+y)(3+z) + 24s^3 in four independent variables.
    zero = (0, 0, 0, 0)
    matrix: tuple[tuple[Polynomial, ...], ...] = (
        ({zero: Fraction(1), (1, 0, 0, 0): Fraction(1)}, {(0, 0, 0, 1): Fraction(2)}, {}),
        ({}, {zero: Fraction(2), (0, 1, 0, 0): Fraction(1)}, {(0, 0, 0, 1): Fraction(3)}),
        ({(0, 0, 0, 1): Fraction(4)}, {}, {zero: Fraction(3), (0, 0, 1, 0): Fraction(1)}),
    )
    expected = {
        (0, 0, 0, 0): Fraction(6),
        (1, 0, 0, 0): Fraction(6),
        (0, 1, 0, 0): Fraction(3),
        (0, 0, 1, 0): Fraction(2),
        (1, 1, 0, 0): Fraction(3),
        (1, 0, 1, 0): Fraction(2),
        (0, 1, 1, 0): Fraction(1),
        (1, 1, 1, 0): Fraction(1),
        (0, 0, 0, 3): Fraction(24),
    }
    with coordinate_reference_budget() as budget:
        assert determinant(matrix, budget=budget) == expected


@pytest.mark.parametrize("rational", (False, True), ids=("polynomial", "shared-quotient"))
def test_embedded_gram_determinant_retains_exact_minor_identity(rational: bool) -> None:
    # Cauchy--Binet for columns (1,v,1) and (u,2,1):
    # (2-uv)^2 + (1-u)^2 + (v-2)^2, divided by D^4 for J/D.
    polynomial_jacobian: tuple[tuple[Polynomial, ...], ...] = (
        ({(0, 0): Fraction(1)}, {(1, 0): Fraction(1)}),
        ({(0, 1): Fraction(1)}, {(0, 0): Fraction(2)}),
        ({(0, 0): Fraction(1)}, {(0, 0): Fraction(1)}),
    )
    denominator: Polynomial = {
        (0, 0): Fraction(1),
        (1, 0): Fraction(1),
        (0, 1): Fraction(1),
    }
    jacobian: tuple[tuple[Expression, ...], ...] = (
        tuple(
            tuple(RationalPolynomial(value, denominator) for value in row)
            for row in polynomial_jacobian
        )
        if rational
        else polynomial_jacobian
    )
    expected = {
        (0, 0): Fraction(9),
        (1, 0): Fraction(-2),
        (0, 1): Fraction(-4),
        (1, 1): Fraction(-4),
        (2, 0): Fraction(1),
        (0, 2): Fraction(1),
        (2, 2): Fraction(1),
    }
    with coordinate_reference_budget() as budget:
        result = expression_gram_determinant(jacobian, budget=budget)
    if rational:
        assert isinstance(result, RationalPolynomial)
        assert result.numerator == expected
        assert result.denominator == {
            (i, j): Fraction(
                factorial(4), factorial(i) * factorial(j) * factorial(4 - i - j)
            )
            for i in range(5)
            for j in range(5 - i)
        }
    else:
        assert result == expected


@pytest.mark.parametrize(
    ("numerator", "denominator", "expected"),
    (
        pytest.param(
            {(2,): Fraction(2), (1,): Fraction(3), (0,): Fraction(1)},
            {(1,): Fraction(1), (0,): Fraction(1)},
            {(1,): Fraction(2), (0,): Fraction(1)},
            id="exact-nonmonomial",
        ),
        pytest.param(
            {(2,): Fraction(2), (0,): Fraction(1)},
            {(1,): Fraction(1), (0,): Fraction(1)},
            None,
            id="nonzero-remainder",
        ),
        pytest.param(
            {(3,): Fraction(3, 7), (2,): Fraction(-2, 5)},
            {(1,): Fraction(3, 11)},
            {(2,): Fraction(11, 7), (1,): Fraction(-22, 15)},
            id="rational-monomial",
        ),
        pytest.param(
            {(2,): Fraction(1), (0,): Fraction(1)},
            {(1,): Fraction(1)},
            None,
            id="negative-exponent-refusal",
        ),
    ),
)
def test_sparse_division_retains_exact_quotient_and_remainder_decision(
    numerator: Polynomial, denominator: Polynomial, expected: Polynomial | None
) -> None:
    original = dict(numerator)
    with coordinate_reference_budget() as budget:
        quotient = divide_polynomial(numerator, denominator, budget=budget)
    assert quotient == expected
    assert numerator == original
    if quotient is not None:
        assert expected is not None
        assert tuple(quotient) == tuple(expected)
        assert budget.work_units > 0


def test_validity_certificate_reuses_only_request_owned_complete_proof() -> None:
    mesh = CellMesh.from_triangles(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)), dtype=np.float64),
        np.asarray(((0, 1, 2),), dtype=np.int32),
    )
    with coordinate_reference_budget() as budget:
        geometry = CellGeometrySpec.affine(mesh, budget=budget)
        first = certify_cell_geometry_validity(geometry, mesh=mesh, budget=budget)
        earned_work = budget.work_units
        retained = budget.retained_basis_bytes
        assert budget.logical_array_bytes_upper >= (
            first.status.nbytes
            + first.determinant_lower.nbytes
            + first.determinant_upper.nbytes
            + first.depth.nbytes
        )
        repeated = certify_cell_geometry_validity(geometry, mesh=mesh, budget=budget)
        assert budget.work_units == earned_work
        assert budget.retained_basis_bytes == retained
    assert first.certificate_id == repeated.certificate_id
    assert int(first.status[0]) == CellValidityStatus.CERTIFIED_VALID
    with coordinate_reference_budget() as independent:
        cold = certify_cell_geometry_validity(geometry, mesh=mesh, budget=independent)
        assert independent.work_units > 0
    assert cold.certificate_id == first.certificate_id


def test_validity_certificate_reuse_rechecks_current_coordinate_content() -> None:
    mesh = CellMesh.from_triangles(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)), dtype=np.float64),
        np.asarray(((0, 1, 2),), dtype=np.int32),
    )
    inverted = CellMesh.from_triangles(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, -1.0)), dtype=np.float64),
        np.asarray(((0, 1, 2),), dtype=np.int32),
    )
    assert inverted.topology_id == mesh.topology_id
    with coordinate_reference_budget() as budget:
        geometry = CellGeometrySpec.affine(mesh, budget=budget)
        changed_geometry = CellGeometrySpec.affine(inverted, budget=budget)
        first = certify_cell_geometry_validity(geometry, mesh=mesh, budget=budget)
        earned_work = budget.work_units
        changed = certify_cell_geometry_validity(
            changed_geometry, mesh=inverted, budget=budget
        )
        assert budget.work_units > earned_work
    assert int(first.status[0]) == CellValidityStatus.CERTIFIED_VALID
    assert int(changed.status[0]) == CellValidityStatus.INVALID
    assert changed.certificate_id != first.certificate_id


def test_validity_certificate_reuse_binds_every_policy_field() -> None:
    mesh = CellMesh.from_triangles(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)), dtype=np.float64),
        np.asarray(((0, 1, 2),), dtype=np.int32),
    )
    with coordinate_reference_budget() as budget:
        geometry = CellGeometrySpec.affine(mesh, budget=budget)
        first = certify_cell_geometry_validity(geometry, mesh=mesh, budget=budget)
        earned_work = budget.work_units
        changed = certify_cell_geometry_validity(
            geometry,
            mesh=mesh,
            policy=CellValidityPolicy(relative_determinant_floor=0.9),
            budget=budget,
        )
        assert budget.work_units > earned_work
    assert int(first.status[0]) == CellValidityStatus.CERTIFIED_VALID
    assert int(changed.status[0]) == CellValidityStatus.INVALID


def test_vertex_images_bind_one_exact_image_per_shared_vertex() -> None:
    from phydrax.discretization._coordinate_enclosure import coordinate_vertex_images

    controls = (
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
        (Fraction(1), Fraction(1, 3)),
        (Fraction(1), Fraction(0)),
        (Fraction(1), Fraction(1, 7)),
    )
    vertices = np.asarray(((0, 1, 2), (1, 3, 2)), dtype=np.int64)
    with coordinate_reference_budget() as budget:
        element = budget.degree_one_coordinate_element("triangle")
        images = coordinate_vertex_images(
            ((element, vertices, vertices),),
            controls,
            5,
            mismatch="shared",
            budget=budget,
        )
        assert images == (*controls[:4], None)
        # A distinct control row with an equal value still names one image.
        agreeing = np.asarray(((0, 1, 2), (4, 3, 2)), dtype=np.int64)
        assert coordinate_vertex_images(
            ((element, agreeing, vertices),), controls, 4, mismatch="shared", budget=budget
        ) == controls[:4]
        conflicting = np.asarray(((0, 1, 2), (5, 3, 2)), dtype=np.int64)
        with pytest.raises(ValueError, match="shared"):
            coordinate_vertex_images(
                ((element, conflicting, vertices),),
                controls,
                4,
                mismatch="shared",
                budget=budget,
            )
