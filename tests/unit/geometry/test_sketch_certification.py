from collections.abc import Iterator
from contextlib import contextmanager

import equinox as eqx
import numpy as np
import pytest
from jax import Array

from phydrax._resource_ticket import HostBatchTicket
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from phydrax.geometry.design import (
    AbstractSketchConstraint,
    certify_sketch_solution,
    FixedPoint,
    Horizontal,
    LineAngle,
    PointDistance,
    Radius,
    Sketch,
    sketch_dimension_derivatives,
    solve_priority_sketch,
)


# Empirical `test_distance_enclosure_and_dimension_derivative_closed_form`: matched max |delta|=2.22045e-16; joint allowance fraction=2.22045e-09, margin=100%.
_SKETCH_DISTANCE_DIMENSION_DERIVATIVE_ATOL = 1e-7
# Empirical `test_distance_enclosure_and_dimension_derivative_closed_form`: matched max |delta|=2.22045e-16; joint allowance fraction=2.22045e-11, margin=100%.
_SKETCH_DISTANCE_FINITE_DIFFERENCE_ATOL = 1e-5
# Empirical `test_radius_dimension_derivative_and_underconstraint_refusal`: matched max |delta|=0; joint allowance fraction=0, margin=100%.
_SKETCH_RADIUS_DIMENSION_DERIVATIVE_ATOL = 1e-7
# Empirical `test_radius_dimension_derivative_and_underconstraint_refusal`: matched max |delta|=8.47033e-22; joint allowance fraction=8.47033e-17, margin=100%.
_SKETCH_RADIUS_FINITE_DIFFERENCE_ATOL = 1e-5
# Empirical `test_priority_tiers_report_unsatisfied_and_rejected_ids`: matched max |delta|=0; joint allowance fraction=0, margin=100%.
_SKETCH_PRIORITY_RADIUS_ATOL = 1e-8
# Empirical `test_angle_dimension_derivative_matches_closed_form_and_finite_difference`: matched max |delta|=5.0713e-17; joint allowance fraction=5.0713e-10, margin=100%.
_SKETCH_ANGLE_DIMENSION_DERIVATIVE_ATOL = 1e-7
# Empirical `test_angle_dimension_derivative_matches_closed_form_and_finite_difference`: matched max |delta|=2.97707e-08; joint allowance fraction=0.00297707, margin=99.7023%.
_SKETCH_ANGLE_FINITE_DIFFERENCE_ATOL = 1e-5


@contextmanager
def _budget() -> Iterator[CoordinateEnclosureBudget]:
    with HostBatchTicket(100_000_000, bytes_upper=1_000_000_000) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket, maximum_work_units=100_000_000, maximum_memory_bytes=1_000_000_000
        )
        with budget.activate():
            yield budget


def _circle(
    *, fixed: bool = True, duplicate: bool = False, conflict: bool = False
) -> Sketch:
    constraints = (
        (FixedPoint(0, (0.0, 0.0)), Radius(0, 2.0)) if fixed else (Radius(0, 2.0),)
    )
    identities = ("center", "radius") if fixed else ("radius",)
    if duplicate:
        constraints += (Radius(0, 2.0),)
        identities += ("duplicate-radius",)
    if conflict:
        constraints += (Radius(0, 3.0),)
        identities += ("conflicting-radius",)
    return Sketch(
        [[0.2, -0.1]],
        circle_centers=[0],
        circle_radii=[1.0],
        constraints=constraints,
        constraint_ids=identities,
        feature_id="circle",
    )


def test_component_under_over_and_redundant_ids() -> None:
    with _budget() as budget:
        under = certify_sketch_solution(_circle(fixed=False).solve(), budget=budget)
        redundant = certify_sketch_solution(
            _circle(duplicate=True).solve(), budget=budget
        )
        over = certify_sketch_solution(_circle(conflict=True).solve(), budget=budget)
    assert under.dof.components[0].underconstrained_ids == ("radius",)
    assert under.dof.nullity == 2
    assert redundant.dof.components[0].redundant_ids == ("duplicate-radius",)
    assert set(over.dof.components[0].overconstrained_ids) == {
        "radius",
        "conflicting-radius",
    }
    assert over.status == "not_fully_constrained"


def test_unique_enclosure_contains_closed_form_circle() -> None:
    with _budget() as budget:
        certificate = certify_sketch_solution(_circle().solve(), budget=budget)
    assert certificate.status == "certified", certificate.reason
    reference = np.asarray([0.0, 0.0, 2.0])
    assert np.all(np.asarray(certificate.lower) <= reference)
    assert np.all(reference <= np.asarray(certificate.upper))
    assert np.all(np.asarray(certificate.image_lower) > np.asarray(certificate.lower))
    assert np.all(np.asarray(certificate.image_upper) < np.asarray(certificate.upper))


def test_distance_enclosure_and_dimension_derivative_closed_form() -> None:
    sketch = Sketch(
        [[0.1, 0.1], [1.8, 0.1]],
        lines=[[0, 1]],
        constraints=(FixedPoint(0, (0.0, 0.0)), Horizontal(0), PointDistance(0, 1, 2.0)),
        constraint_ids=("anchor", "horizontal", "length"),
        feature_id="dimension-line",
    )
    solution = sketch.solve()
    with _budget() as budget:
        certificate = certify_sketch_solution(solution, budget=budget)
        (derivative,) = sketch_dimension_derivatives(certificate, budget=budget)
    assert certificate.status == "certified", certificate.reason
    reference = np.asarray([0.0, 0.0, 2.0, 0.0])
    assert np.all(np.asarray(certificate.lower) <= reference)
    assert np.all(reference <= np.asarray(certificate.upper))
    np.testing.assert_allclose(
        derivative.derivative,
        [0.0, 0.0, 1.0, 0.0],
        atol=_SKETCH_DISTANCE_DIMENSION_DERIVATIVE_ATOL,
        rtol=0.0,
    )
    dimension = sketch.constraints[2]
    assert isinstance(dimension, PointDistance)
    step = 2.0**-12
    roots = []
    for offset in (-step, step):
        revised_dimension = eqx.tree_at(
            lambda item: item.distance, dimension, dimension.distance + offset
        )
        revised = eqx.tree_at(
            lambda owner: owner.constraints[2], sketch, revised_dimension
        )
        roots.append(revised.solve().points.reshape(-1))
    finite_difference = (roots[1] - roots[0]) / (2 * step)
    np.testing.assert_allclose(
        derivative.derivative,
        finite_difference,
        atol=_SKETCH_DISTANCE_FINITE_DIFFERENCE_ATOL,
        rtol=0.0,
    )


def test_radius_dimension_derivative_and_underconstraint_refusal() -> None:
    with _budget() as budget:
        certificate = certify_sketch_solution(_circle().solve(), budget=budget)
        (derivative,) = sketch_dimension_derivatives(certificate, budget=budget)
        under = certify_sketch_solution(_circle(fixed=False).solve(), budget=budget)
        with pytest.raises(ValueError, match="fully constrained"):
            sketch_dimension_derivatives(under, budget=budget)
    np.testing.assert_allclose(
        derivative.derivative,
        [0.0, 0.0, 1.0],
        atol=_SKETCH_RADIUS_DIMENSION_DERIVATIVE_ATOL,
        rtol=0.0,
    )
    sketch = certificate.solution.sketch
    dimension = sketch.constraints[1]
    assert isinstance(dimension, Radius)
    step = 2.0**-12
    roots = []
    for offset in (-step, step):
        revised_dimension = eqx.tree_at(
            lambda item: item.radius, dimension, dimension.radius + offset
        )
        revised = eqx.tree_at(
            lambda owner: owner.constraints[1], sketch, revised_dimension
        ).solve()
        roots.append(
            np.concatenate(
                (np.asarray(revised.points).reshape(-1), np.asarray(revised.circle_radii))
            )
        )
    finite_difference = (roots[1] - roots[0]) / (2 * step)
    np.testing.assert_allclose(
        derivative.derivative,
        finite_difference,
        atol=_SKETCH_RADIUS_FINITE_DIFFERENCE_ATOL,
        rtol=0.0,
    )


def test_priority_tiers_report_unsatisfied_and_rejected_ids() -> None:
    sketch = Sketch(
        [[0.1, 0.2]],
        circle_centers=[0],
        circle_radii=[1.0],
        constraints=(FixedPoint(0, (0.0, 0.0)), Radius(0, 2.0), Radius(0, 3.0)),
        constraint_ids=("center", "preferred-radius", "lower-radius"),
        constraint_priorities=(0, 1, 2),
        feature_id="priority-circle",
    )
    with _budget() as budget:
        result = solve_priority_sketch(sketch, budget=budget)
    assert result.lowest_satisfied_tier == 1
    assert result.unsatisfied_constraint_ids == ("lower-radius",)
    assert result.rejected_constraint_ids == ("lower-radius",)
    assert tuple(certificate.status for certificate in result.tier_certificates) == (
        "not_fully_constrained",
        "certified",
        "not_fully_constrained",
    )
    np.testing.assert_allclose(
        result.solution.circle_radii, [2.0], atol=_SKETCH_PRIORITY_RADIUS_ATOL, rtol=0.0
    )


def test_rank_ambiguity_refuses_and_budget_precedes_evaluation() -> None:
    sketch = Sketch(
        [[0.0, 0.0]],
        circle_centers=[0],
        circle_radii=[2.0],
        constraints=(
            FixedPoint(0, (0.0, 0.0)),
            Radius(0, 2.0, weight=(8 * 3 * np.finfo(np.float64).eps) ** 2),
        ),
        constraint_ids=("anchor", "weak-radius"),
        feature_id="weak",
    )
    with _budget() as budget:
        certificate = certify_sketch_solution(sketch.solve(), budget=budget)
    assert not certificate.dof.rank_certified
    assert certificate.status == "rank_unresolved"
    solution = _circle().solve()
    with HostBatchTicket(0, bytes_upper=0) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket, maximum_work_units=0, maximum_memory_bytes=0
        )
        with pytest.raises(CoordinateEnclosureResourceError):
            certify_sketch_solution(solution, budget=budget)


def test_disconnected_components_have_separate_constraint_ids() -> None:
    sketch = Sketch(
        [[0.0, 0.0], [1.0, 1.0]],
        constraints=(FixedPoint(0, (0.0, 0.0)),),
        constraint_ids=("first-anchor",),
        feature_id="disconnected",
    )
    with _budget() as budget:
        certificate = certify_sketch_solution(sketch.solve(), budget=budget)
    assert certificate.dof.components[0].constraint_ids == ("first-anchor",)
    assert certificate.dof.components[0].nullity == 0
    assert certificate.dof.components[1].constraint_ids == ()
    assert certificate.dof.components[1].nullity == 2


def test_lowered_profile_retains_unique_root_source_fidelity() -> None:
    with _budget() as budget:
        certificate = certify_sketch_solution(_circle().solve(), budget=budget)
    lowered = certificate.to_profile()
    assert lowered.certificate is certificate
    assert np.all(np.asarray(lowered.certificate.radius) > 0.0)
    with _budget() as budget:
        under = certify_sketch_solution(_circle(fixed=False).solve(), budget=budget)
    with pytest.raises(ValueError, match="unique-root enclosure"):
        under.to_profile()


def test_angle_dimension_derivative_matches_closed_form_and_finite_difference() -> None:
    angle = 0.4
    sketch = Sketch(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [np.cos(angle), np.sin(angle)]],
        lines=[[0, 1], [2, 3]],
        constraints=(
            FixedPoint(0, (0.0, 0.0)),
            FixedPoint(1, (1.0, 0.0)),
            FixedPoint(2, (0.0, 0.0)),
            PointDistance(2, 3, 1.0),
            LineAngle(0, 1, angle),
        ),
        constraint_ids=("origin", "baseline", "pivot", "length", "angle"),
        feature_id="angle-dimension",
    )
    with _budget() as budget:
        certificate = certify_sketch_solution(sketch.solve(), budget=budget)
        derivatives = sketch_dimension_derivatives(certificate, budget=budget)
    assert certificate.status == "certified", certificate.reason
    reference_root = np.asarray(sketch.points).reshape(-1)
    assert np.all(np.asarray(certificate.lower) <= reference_root)
    assert np.all(reference_root <= np.asarray(certificate.upper))
    derivative = derivatives[1].derivative
    reference = np.zeros(8, dtype=np.float64)
    reference[-2:] = (-np.sin(angle), np.cos(angle))
    np.testing.assert_allclose(
        derivative, reference, atol=_SKETCH_ANGLE_DIMENSION_DERIVATIVE_ATOL, rtol=0.0
    )
    dimension = sketch.constraints[-1]
    assert isinstance(dimension, LineAngle)
    step = 2.0**-12
    roots = []
    for offset in (-step, step):
        revised_dimension = eqx.tree_at(
            lambda item: item.angle, dimension, dimension.angle + offset
        )
        revised = eqx.tree_at(
            lambda owner: owner.constraints[-1], sketch, revised_dimension
        )
        roots.append(revised.solve().points.reshape(-1))
    finite_difference = (roots[1] - roots[0]) / (2 * step)
    np.testing.assert_allclose(
        derivative, finite_difference, atol=_SKETCH_ANGLE_FINITE_DIFFERENCE_ATOL, rtol=0.0
    )


class _CustomConstraint(AbstractSketchConstraint):
    def __init__(self) -> None:
        super().__init__()

    def residual(
        self, points: Array, lines: Array, circle_centers: Array, circle_radii: Array, /
    ) -> Array:
        del lines, circle_centers, circle_radii
        return points[0]


def test_custom_constraint_family_refuses_before_certification_evaluation() -> None:
    sketch = Sketch(
        [[0.0, 0.0]],
        constraints=(_CustomConstraint(),),
        constraint_ids=("custom",),
        feature_id="custom-refusal",
    )
    with _budget() as budget:
        certificate = certify_sketch_solution(sketch.solve(), budget=budget)
    assert certificate.status == "unsupported"
    assert certificate.dof.rank == -1
    assert not certificate.dof.rank_certified
