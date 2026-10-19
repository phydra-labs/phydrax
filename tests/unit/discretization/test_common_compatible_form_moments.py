# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Compare physical circulation and flux on independent rational patch charts."""

from collections.abc import Iterator
from fractions import Fraction

import numpy as np
import pytest
from numpy.typing import NDArray

from phydrax import config as pxconfig
from phydrax._resource_ticket import HostBatchTicket
from phydrax.discretization import _coordinate_enclosure as algebra, FormBasis
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.discretization.fem._exact_form_moments import ExactFormMoments
from phydrax.discretization.fem._form_elements import form_entity_chart
from phydrax.exterior._form_type import FormTwist


pxconfig.configure(pxconfig.RuntimeConfig(enable_x64=True))
# Regression envelopes for the independently inverted projective chart below:
# 48-point Legendre/Duffy integration of denominators bounded below by 8.
# Keep the fixture's existing thresholds; no acceptance bound is widened.
PROJECTIVE_MOMENT_TOLERANCE = 2e-11
PROJECTIVE_ROUNDOFF_FLOOR = 2e-13


def _evaluate(
    value: algebra.Expression,
    points: NDArray[np.float64],
    *,
    budget: CoordinateEnclosureBudget,
) -> NDArray[np.float64]:
    numerator, denominator = algebra.expression_parts(
        value, points.shape[1], budget=budget
    )

    def polynomial(terms: algebra.Polynomial) -> NDArray[np.float64]:
        result = np.zeros(points.shape[0], dtype=np.float64)
        for powers, coefficient in terms.items():
            result += float(coefficient) * np.prod(points ** np.asarray(powers), axis=1)
        return result

    return polynomial(numerator) / polynomial(denominator)


@pytest.mark.parametrize(
    "degree,twist,reversed_source",
    (
        (1, "untwisted", False),
        (2, "twisted", False),
        (1, "untwisted", True),
        (1, "twisted", True),
        (2, "twisted", True),
    ),
)
def test_projective_common_moments_against_physical_inverse_chart(
    degree: int,
    twist: FormTwist,
    reversed_source: bool,
    *,
    ledger: CoordinateEnclosureBudget,
) -> None:
    u, v = algebra.axes(2, budget=ledger)
    source_arguments = (
        (v, u)
        if reversed_source
        else (
            v,
            algebra.add(
                algebra.constant(1, 2, budget=ledger),
                algebra.scale(algebra.add(u, v, budget=ledger), -1, budget=ledger),
                budget=ledger,
            ),
        )
    )
    denominator = algebra.add(
        algebra.constant(1, 2, budget=ledger),
        algebra.scale(v, Fraction(1, 8), budget=ledger),
        budget=ledger,
    )
    target_arguments = (
        algebra.rational_expression(u, denominator, budget=ledger),
        algebra.rational_expression(
            algebra.scale(v, Fraction(9, 8), budget=ledger), denominator, budget=ledger
        ),
    )
    basis = FormBasis(2, degree, 2, "trimmed", twist)
    constant = np.asarray([1.0, 2.0] if degree == 1 else [1.0])
    coefficients = np.asarray(
        basis.interpolate(
            np.broadcast_to(constant, (basis.functional_points.shape[0], constant.size))
        )
    )
    nodes, weights = np.polynomial.legendre.leggauss(48)
    nodes, weights = (nodes + 1) / 2, weights / 2
    matrix = np.zeros((basis.local_dof_count, basis.local_dof_count), dtype=np.float64)
    errors = np.zeros_like(matrix)
    expected = np.zeros(basis.local_dof_count, dtype=np.float64)
    with ledger.activate():
        moments = ExactFormMoments(ledger)
        for entities in basis.entity_vertices:
            for entity in entities:
                rows = [
                    row
                    for row, label in enumerate(basis.dof_labels)
                    if label[0] == entity
                ]
                if not rows:
                    continue
                with ledger.temporary_scope():
                    result = moments.common_entity(
                        basis,
                        basis,
                        entity,
                        "triangle",
                        entity,
                        source_arguments,
                        target_arguments,
                    )
                assert result is not None
                contribution, _ = result
                matrix += contribution.value
                errors += contribution.error
                _, origin, chart = form_entity_chart(2, basis.family, entity)
                if chart.shape[1] == 1:
                    parameters, quadrature = nodes[:, None], weights
                else:
                    first, second = np.meshgrid(nodes, nodes, indexing="ij")
                    first_weight, second_weight = np.meshgrid(
                        weights, weights, indexing="ij"
                    )
                    parameters = np.column_stack(
                        (first.ravel(), ((1 - first) * second).ravel())
                    )
                    quadrature = (first_weight * second_weight * (1 - first)).ravel()
                points = origin + parameters @ chart.T
                y = points[:, 1]
                # The inverse projective chart is (9*x/(9-y), 8*y/(9-y)).
                # A reversed source chart instead pulls the one-form back to
                # (2*du + dv); twisted flux includes the full orientation
                # bundle sign, while a twisted top density uses |det|.
                field = (
                    np.column_stack(
                        (-18 / (9 - y), (-18 * points[:, 0] - 72) / (9 - y) ** 2)
                    )
                    if degree == 1
                    else (648 / (9 - y) ** 3)[:, None]
                )
                if reversed_source and degree == 1 and twist == "untwisted":
                    field = -field
                densities = basis.functional_density_expressions(entity, budget=ledger)
                for row in rows:
                    density = np.column_stack(
                        tuple(
                            _evaluate(value, parameters, budget=ledger)
                            for value in densities[row]
                        )
                    )
                    expected[row] = np.sum(quadrature * np.sum(density * field, axis=1))
    actual = matrix @ coefficients
    np.testing.assert_allclose(
        actual,
        expected,
        atol=PROJECTIVE_MOMENT_TOLERANCE,
        rtol=PROJECTIVE_MOMENT_TOLERANCE,
    )
    # The certificate must carry the nonzero rational integration uncertainty;
    # a corner-affine substitute would fail the independent physical moments.
    np.testing.assert_array_less(
        np.abs(actual - expected),
        errors @ np.abs(coefficients) + PROJECTIVE_ROUNDOFF_FLOOR,
    )


@pytest.fixture
def ledger() -> Iterator[CoordinateEnclosureBudget]:
    with HostBatchTicket(100_000_000, bytes_upper=256_000_000) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=100_000_000,
            maximum_memory_bytes=256_000_000,
        )
        with budget.activate():
            yield budget
