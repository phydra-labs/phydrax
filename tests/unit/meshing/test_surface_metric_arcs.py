# Copyright © 2026 PHYDRA, Inc. All rights reserved.

from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
import pytest
from numpy.typing import ArrayLike

from phydrax._resource_ticket import HostBatchTicket
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from phydrax.geometry._meshing_domain import (
    MeshingDomain,
    MeshingDomainCurve,
    MeshingSurfacePatch,
    PatchCurveUse,
)
from phydrax.geometry.brep._patches import (
    CylinderPatch,
    LineCurve,
    PlanePatch,
    surface_value_bounds,
)
from phydrax.meshing import _surface_metric as owner
from phydrax.meshing._tetra_metric import MetricRemeshingEvidence, MetricRemeshingStatus
from tests._support.coordinate_references import coordinate_reference_budget


# Exact binary64 operands retained by the capped-cylinder first-evidence observer.
_CASES = (
    (
        (
            (-0.48210770910340917, -0.6384137818241886),
            (-0.29750821700101054, -0.5356737105404833),
        ),
        (
            (-0.4821077091034098, -0.6384137818241882, 1.5),
            (-0.29750821700101054, -0.5356737105404833, 1.5),
        ),
        1.4084269354589953,
        1.9984014443252818e-15,
    ),
    (
        (
            (0.566636528065334, 0.18248737709293764),
            (0.745977783523485, 0.28899333294972157),
        ),
        (
            (0.566636528065334, 0.18248737709293764, 1.5),
            (0.7459777835234849, 0.2889933329497219, 1.5),
        ),
        1.390552320998514,
        2.220446049250313e-16,
    ),
)


def _matrix(scale: float = 1.0) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(Fraction(float(value)) for value in row) for row in np.eye(3) * scale
    )


def _evidence(bounds: np.ndarray) -> MetricRemeshingEvidence:
    return MetricRemeshingEvidence(
        MetricRemeshingStatus.COMPLETE,
        passes=1,
        counts=(0, 0, 0, 0),
        rejected_operations=0,
        work_units=0,
        lengths=np.mean(bounds, axis=1),
        quality=np.ones(1),
        metric_arc_bounds=bounds,
        metric_arc_edge_ids=np.asarray(((23, 342),), dtype=np.int64),
        metric_arc_binding="source-chart-regression",
    )


@pytest.mark.parametrize("charts,stored,old_upper,inversion", _CASES)
@pytest.mark.parametrize("reverse", (False, True))
def test_source_chord_not_rounded_mesh_chord(
    charts: ArrayLike,
    stored: ArrayLike,
    old_upper: float,
    inversion: float,
    reverse: bool,
) -> None:
    charts, stored = np.asarray(charts), np.asarray(stored)
    if reverse:
        charts, stored = charts[::-1], stored[::-1]
    matrix = _matrix(44.44444444444444)
    patch = PlanePatch((0, 0, 1.5), (1, 0, 0), (0, 1, 0))
    with coordinate_reference_budget() as budget:
        old_squared = sum(
            (
                matrix[i][i]
                * (Fraction(float(stored[1, i])) - Fraction(float(stored[0, i]))) ** 2
                for i in range(3)
            ),
            Fraction(0),
        )
        assert (
            float(owner._fraction_sqrt_lower(old_squared, budget=budget)) - old_upper
            == inversion
        )
        lower, upper = surface_value_bounds(
            patch, np.stack((charts, charts), axis=1), budget=budget
        )
        low = owner._source_metric_chord_lower(lower, upper, matrix, budget=budget)
        box = np.stack((charts.min(axis=0), charts.max(axis=0)))[None]
        derivative_low, derivative_high = patch.derivative_bounds_batch(
            box, budget=budget
        )
        direction = (
            Fraction(float(charts[1, 0])) - Fraction(float(charts[0, 0])),
            Fraction(float(charts[1, 1])) - Fraction(float(charts[0, 1])),
        )
        high = owner._constant_metric_tangent_upper(
            direction, derivative_low[0], derivative_high[0], matrix, None, budget=budget
        )
    with localcontext() as context:
        context.prec = 100
        exact_squared = sum(matrix[i][i] * direction[i] ** 2 for i in range(2))
        exact = (
            Decimal(exact_squared.numerator) / Decimal(exact_squared.denominator)
        ).sqrt()
        assert 0 < Decimal(low.numerator) / Decimal(low.denominator) <= exact
        assert exact <= Decimal(high.numerator) / Decimal(high.denominator)
    bounds = np.asarray(((float(low), float(high)),))
    evidence = _evidence(bounds)
    evidence.validate_restored()
    assert evidence.evidence_id == _evidence(bounds.copy()).evidence_id
    np.testing.assert_array_equal(evidence.metric_arc_bounds, bounds)


def _trig_enclosure(angle: Fraction, *, sine: bool) -> tuple[Fraction, Fraction]:
    """Independent alternating Taylor bounds, exact rational coefficients."""
    term = angle if sine else Fraction(1)
    total = term
    for index in range(1, 81):
        degree = 2 * index + int(sine)
        term *= -angle * angle / (degree * (degree - 1))
        total += term
    degree = 162 + int(sine)
    following = term * -angle * angle / (degree * (degree - 1))
    return min(total, total + following), max(total, total + following)


@pytest.mark.parametrize("scale", (1e-100, 1.0, 1e100))
@pytest.mark.parametrize("angle", (1e-8, 0.125, 3.0))
def test_source_chord_curved_arc_high_precision(scale: float, angle: float) -> None:
    patch = CylinderPatch((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), scale)
    charts = np.asarray(((0.0, 0.0), (angle, 0.0)))
    matrix = _matrix()
    with coordinate_reference_budget() as budget:
        lower, upper = surface_value_bounds(
            patch, np.stack((charts, charts), axis=1), budget=budget
        )
        low = owner._source_metric_chord_lower(lower, upper, matrix, budget=budget)
        derivatives = patch.derivative_bounds_batch(charts[None], budget=budget)
        high = owner._constant_metric_tangent_upper(
            (Fraction(angle), Fraction(0)),
            derivatives[0][0],
            derivatives[1][0],
            matrix,
            None,
            budget=budget,
        )
    # The exact circular arc is radius * angle; no floating quadrature oracle.
    exact_arc = Fraction(scale) * Fraction(angle)
    assert 0 < low < exact_arc <= high
    assert np.isfinite(float(low)) and np.isfinite(float(high))
    for axis, sine in ((0, False), (1, True)):
        exact_low, exact_high = _trig_enclosure(Fraction(angle), sine=sine)
        assert Fraction(float(lower[1, axis])) <= Fraction(scale) * exact_low
        assert Fraction(scale) * exact_high <= Fraction(float(upper[1, axis]))


def test_source_chord_overlapping_boxes_and_resource_boundary() -> None:
    lower = np.asarray(((0, 0, 0), (2, 0, 0)), dtype=np.float64)
    upper = np.asarray(((1, 1, 1), (3, 1, 1)), dtype=np.float64)
    matrix = _matrix()
    with coordinate_reference_budget() as budget:
        assert owner._source_metric_chord_lower(lower, upper, matrix, budget=budget) == 1
        required = budget.work_units
    for allowance in (required, required - 1):
        ticket = HostBatchTicket(allowance, bytes_upper=1_000_000)
        ledger = CoordinateEnclosureBudget(
            ticket, maximum_work_units=allowance, maximum_memory_bytes=1_000_000
        )
        with ticket, ledger.activate():
            if allowance == required:
                assert (
                    owner._source_metric_chord_lower(lower, upper, matrix, budget=ledger)
                    == 1
                )
            else:
                with pytest.raises(CoordinateEnclosureResourceError):
                    owner._source_metric_chord_lower(lower, upper, matrix, budget=ledger)
    with coordinate_reference_budget() as budget:
        assert (
            owner._source_metric_chord_lower(
                lower[:1].repeat(2, axis=0),
                upper[:1].repeat(2, axis=0),
                matrix,
                budget=budget,
            )
            == 0
        )


@pytest.mark.parametrize(
    "bounds",
    (
        np.asarray([1.0, 2.0]),
        np.asarray([[np.nan, 2.0]]),
        np.asarray([[1.0, np.inf]]),
        np.asarray([[-1.0, 2.0]]),
        np.asarray([[2.0, 1.0]]),
    ),
)
def test_metric_evidence_retains_strict_outward_bound_validation(
    bounds: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        _evidence(bounds)


@pytest.mark.parametrize("case", (0, 1))
def test_complete_constant_arc_kernel_uses_source_endpoint_owner(case: int) -> None:
    charts, stored, _, _ = _CASES[case]
    patch = PlanePatch((0, 0, 1.5), (1, 0, 0), (0, 1, 0))
    corners = np.asarray(((-1, -1), (1, -1), (1, 1), (-1, 1)), dtype=np.float64)
    loop = tuple(
        PatchCurveUse(
            index,
            LineCurve(corners[index], corners[(index + 1) % 4] - corners[index]),
            0.0,
            1.0,
        )
        for index in range(4)
    )
    domain = MeshingDomain(
        (MeshingSurfacePatch(patch, (loop,)),),
        tuple(MeshingDomainCurve(index, (index + 1) % 4) for index in range(4)),
        4,
        source_id="arc-endpoint-regression",
        source_revision="1",
    )
    triangle = np.asarray((*charts, (0.0, 0.0)))
    points = np.asarray((*stored, (0.0, 0.0, 1.5)))
    metric = np.eye(3) * 44.44444444444444
    state = owner.SurfaceState(
        points,
        np.broadcast_to(metric, (3, 3, 3)),
        np.asarray(((0, 1, 2),)),
        np.asarray((0,)),
        triangle[None],
        np.zeros(3, dtype=np.int8),
        np.asarray((97,)),
        np.asarray((23, 342, 7)),
        [{97}],
        [{23}, {342}, {7}],
        98,
        343,
    )
    with coordinate_reference_budget(queries_upper=100) as budget:
        bounds, _ = owner._constant_surface_metric_arc_bounds(
            domain, state, metric, np.ones(3, dtype=np.bool_), budget=budget
        )
    exact_squared = sum(
        Fraction(float(metric[i, i]))
        * (Fraction(charts[1][i]) - Fraction(charts[0][i])) ** 2
        for i in range(2)
    )
    assert Fraction(float(bounds[0, 0])) ** 2 <= exact_squared
    assert exact_squared <= Fraction(float(bounds[0, 1])) ** 2
    assert np.all(np.isfinite(bounds))
    assert np.all((bounds[:, 0] > 0) & (bounds[:, 0] <= bounds[:, 1]))
