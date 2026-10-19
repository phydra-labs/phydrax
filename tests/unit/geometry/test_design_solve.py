#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native design solves preserve admissibility, residual decrease, and evidence.

Circle area and scalar polynomial residuals provide independent references.
The solve tolerance is an explicit weighted-residual acceptance bound; geometry
checks use machine-rounding error or the corresponding propagated area bound.
"""

import math
from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax.geometry.design import AbstractSketchConstraint
from phydrax.nonlinear import SensitivityStatus
from phydrax.optim import OptimizationStatus
from phydrax.units import METER


# Empirical `test_converged_sketch_fixed_point_target_has_implicit_jvp`: matched max |delta|=0; joint allowance fraction=0, margin=100%.
_SKETCH_TARGET_IMPLICIT_TANGENT_ATOL = 1e-7


_TOLERANCE = 1e-9


class _CubicRadius(AbstractSketchConstraint):
    def residual(
        self,
        points: Array,
        lines: Array,
        circle_centers: Array,
        circle_radii: Array,
        /,
    ) -> Array:
        del points, lines, circle_centers
        return self._weighted(circle_radii**3 - 1.0)


class _NegativeRadius(AbstractSketchConstraint):
    def residual(
        self,
        points: Array,
        lines: Array,
        circle_centers: Array,
        circle_radii: Array,
        /,
    ) -> Array:
        del points, lines, circle_centers
        return self._weighted(circle_radii + 1.0)


def _circle(*, inconsistent: bool = False) -> phx.geometry.Sketch:
    targets = (phx.geometry.Radius(0, 1.25),)
    if inconsistent:
        targets += (phx.geometry.Radius(0, 2.0),)
    return phx.geometry.Sketch(
        [[0.2, -0.1]],
        circle_centers=[0],
        circle_radii=[1.0],
        constraints=(
            phx.geometry.FixedPoint(0, jnp.zeros((2,), dtype=jnp.float64)),
            *targets,
        ),
        feature_id="design-solve-circle",
    )


@pytest.mark.parametrize("lowering", ["source", "profile", "model"])
def test_unconverged_sketch_refuses_lowering(lowering: str) -> None:
    sketch = _circle(inconsistent=True)
    solution = sketch.solve(tolerance=_TOLERANCE)
    assert not bool(solution.converged)
    assert int(solution.status) != int(OptimizationStatus.SUCCESS)
    assert float(solution.residual_norm) > float(solution.residual_bound)
    with pytest.raises(ValueError, match="Sketch.solution"):
        match lowering:
            case "source":
                sketch.to_source(solution)
            case "profile":
                sketch.to_profile(solution)
            case "model":
                sketch.to_model(
                    solution, coordinate_contract=phx.SpatialCoordinateContract.si()
                )
            case _:
                raise AssertionError(lowering)


@pytest.mark.parametrize("lowering", ["source", "profile", "model"])
def test_lowering_requires_a_solution(lowering: str) -> None:
    sketch = _circle()
    with pytest.raises(TypeError, match="solution"):
        match lowering:
            case "source":
                sketch.to_source(None)  # ty: ignore[invalid-argument-type]
            case "profile":
                sketch.to_profile(None)  # ty: ignore[invalid-argument-type]
            case "model":
                sketch.to_model(
                    None,  # ty: ignore[invalid-argument-type]
                    coordinate_contract=phx.SpatialCoordinateContract.si(),
                )
            case _:
                raise AssertionError(lowering)


@pytest.mark.parametrize("binding", ["geometry", "constraint-weight"])
def test_solution_from_a_different_sketch_binding_refuses(binding: str) -> None:
    sketch = _circle()
    solution = sketch.solve()
    assert bool(solution.converged)
    match binding:
        case "geometry":
            selected = eqx.tree_at(
                lambda value: value.points,
                sketch,
                jnp.asarray([[9.0, 9.0]], dtype=jnp.float64),
            )
        case "constraint-weight":
            selected = eqx.tree_at(
                lambda value: value.constraints[1].weight,
                sketch,
                jnp.asarray(2.0, dtype=jnp.float64),
            )
        case _:
            raise AssertionError(binding)
    with pytest.raises(ValueError, match="Sketch.solution.sketch"):
        selected.to_profile(solution)


def test_converged_sketch_lowers_with_residual_evidence() -> None:
    sketch = _circle()
    solution = sketch.solve(tolerance=_TOLERANCE)
    assert bool(solution.converged)
    assert int(solution.status) == int(OptimizationStatus.SUCCESS)
    assert int(solution.solver_result.diagnostics.accepted_steps) > 0
    assert float(solution.residual_norm) <= _TOLERANCE
    profile = sketch.to_profile(solution)
    assert profile.residual_bound == float(solution.residual_bound) == _TOLERANCE
    # Radius error <= the Radius residual bound, hence |delta area| <=
    # pi * (2 R tolerance + tolerance**2), plus binary area evaluation error.
    area_bound = math.pi * (2.0 * 1.25 * _TOLERANCE + _TOLERANCE**2)
    area = float(sketch.to_source(solution).compile(coordinate_unit=METER).measure)
    assert abs(area - math.pi * 1.25**2) <= area_bound + np.spacing(area)
    model = sketch.to_model(
        solution, coordinate_contract=phx.SpatialCoordinateContract.si()
    )
    assert model.source_id == sketch.feature_id


@pytest.mark.parametrize("radius", [0.0, -1.0])
def test_nonpositive_initial_sketch_radius_refuses(radius: float) -> None:
    with pytest.raises(ValueError, match="Sketch.circle_radii"):
        phx.geometry.Sketch([[0.0, 0.0]], circle_centers=[0], circle_radii=[radius])


@pytest.mark.parametrize("radius", [0.0, -1.0])
def test_nonpositive_restored_sketch_radius_refuses_solve(radius: float) -> None:
    sketch = _circle()
    sketch = eqx.tree_at(
        lambda value: value.circle_radii, sketch, jnp.asarray([radius], dtype=jnp.float64)
    )
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError), match="Sketch.circle_radii"
    ):
        sketch.solve()


def test_nonpositive_trial_radius_is_rejected_without_clamping() -> None:
    sketch = phx.geometry.Sketch(
        [[0.0, 0.0]],
        circle_centers=[0],
        circle_radii=[1.0],
        constraints=(_NegativeRadius(),),
    )
    solution = sketch.solve(max_iterations=1, damping=1e-8)
    assert not bool(solution.converged)
    np.testing.assert_array_equal(solution.circle_radii, sketch.circle_radii)
    assert int(solution.solver_result.diagnostics.accepted_steps) == 0
    assert int(solution.solver_result.diagnostics.rejected_steps) > 0


def test_sketch_solver_commits_only_residual_decrease() -> None:
    sketch = phx.geometry.Sketch(
        [[0.0, 0.0]],
        circle_centers=[0],
        circle_radii=[0.1],
        constraints=(_CubicRadius(),),
    )
    initial = abs(0.1**3 - 1.0)
    solution = sketch.solve(max_iterations=1)
    assert float(solution.residual_norm) <= initial
    if int(solution.solver_result.diagnostics.accepted_steps):
        assert float(solution.residual_norm) < initial
    assert not solution.solver_result.provenance.implicit_differentiation


def test_local_design_constraints_publish_native_status_and_bound() -> None:
    geometry = phx.geometry.Sphere(
        (0.0, 0.0, 0.0), 1.0, feature_id="solve-sphere"
    ).compile(coordinate_unit=METER)
    radius = phx.geometry.ParameterId("solve-sphere", "radius")
    system = phx.geometry.DesignConstraintSystem(
        geometry, (phx.geometry.ParameterTarget(radius, 1.5),)
    )
    result = system.solve(tolerance=_TOLERANCE)
    assert bool(result.converged)
    assert int(result.status) == int(OptimizationStatus.SUCCESS)
    assert float(result.residual_norm) <= float(result.residual_bound) == _TOLERANCE
    assert (
        abs(float(result.state.values[geometry.schema.index(radius)]) - 1.5) <= _TOLERANCE
    )
    assert int(result.solver_result.diagnostics.linear_solves) > 0
    assert not result.solver_result.provenance.implicit_differentiation


def test_inconsistent_design_constraints_do_not_report_convergence() -> None:
    geometry = phx.geometry.Sphere(
        (0.0, 0.0, 0.0), 1.0, feature_id="solve-sphere"
    ).compile(coordinate_unit=METER)
    radius = phx.geometry.ParameterId("solve-sphere", "radius")
    system = phx.geometry.DesignConstraintSystem(
        geometry,
        (
            phx.geometry.ParameterTarget(radius, 1.25),
            phx.geometry.ParameterTarget(radius, 2.0),
        ),
    )
    result = system.solve(tolerance=_TOLERANCE)
    assert not bool(result.converged)
    assert int(result.status) != int(OptimizationStatus.SUCCESS)
    assert float(result.residual_norm) > float(result.residual_bound)


@pytest.mark.parametrize("radius_value", [0.0, -1.0])
def test_nonpositive_initial_design_radius_refuses(radius_value: float) -> None:
    geometry = phx.geometry.Sphere(
        (0.0, 0.0, 0.0), 1.0, feature_id="solve-sphere"
    ).compile(coordinate_unit=METER)
    radius = phx.geometry.ParameterId("solve-sphere", "radius")
    system = phx.geometry.DesignConstraintSystem(
        geometry, (phx.geometry.ParameterTarget(radius, 1.5),)
    )
    invalid = geometry.state.replace_at(
        geometry.schema.index(radius), jnp.asarray(radius_value, dtype=jnp.float64)
    )
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError), match="design.solve.initial"
    ):
        system.solve(initial_state=invalid)


def test_nonpositive_design_radius_trials_are_rejected_without_projection() -> None:
    geometry = phx.geometry.Sphere(
        (0.0, 0.0, 0.0), 1.0, feature_id="solve-sphere"
    ).compile(coordinate_unit=METER)
    radius = phx.geometry.ParameterId("solve-sphere", "radius")
    system = phx.geometry.DesignConstraintSystem(
        geometry, (phx.geometry.ParameterTarget(radius, -1.0),)
    )
    result = system.solve(max_iterations=1)
    assert not bool(result.converged)
    np.testing.assert_array_equal(result.state.values[geometry.schema.index(radius)], 1.0)
    assert int(result.solver_result.diagnostics.accepted_steps) == 0
    assert int(result.solver_result.diagnostics.rejected_steps) > 0


def test_unresolved_initial_design_validity_refuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = phx.geometry.Sphere(
        (0.0, 0.0, 0.0), 1.0, feature_id="solve-sphere"
    ).compile(coordinate_unit=METER)
    radius = phx.geometry.ParameterId("solve-sphere", "radius")
    system = phx.geometry.DesignConstraintSystem(
        geometry, (phx.geometry.ParameterTarget(radius, 1.5),)
    )
    restricted = replace(geometry.field_certificate, validity_region="fixed_topology")
    monkeypatch.setattr(
        type(geometry.kernel), "field_certificate", property(lambda _self: restricted)
    )
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError), match="design.solve.initial"
    ):
        system.solve()


def test_converged_sketch_radius_is_independent_of_initial_seed() -> None:
    sketch = _circle()

    def radius(value: Array) -> Array:
        selected = eqx.tree_at(lambda owner: owner.circle_radii, sketch, value)
        return selected.solve().circle_radii

    _, tangent = jax.jvp(
        radius, (sketch.circle_radii,), (jnp.ones_like(sketch.circle_radii),)
    )
    np.testing.assert_array_equal(tangent, jnp.zeros_like(tangent))


def test_converged_sketch_fixed_point_target_has_implicit_jvp() -> None:
    sketch = _circle()
    constraint = sketch.constraints[0]
    assert isinstance(constraint, phx.geometry.FixedPoint)

    def solve_target(target: Array) -> phx.geometry.SketchSolution:
        revised_constraint = eqx.tree_at(lambda value: value.target, constraint, target)
        revised_sketch = eqx.tree_at(
            lambda value: value.constraints[0], sketch, revised_constraint
        )
        return revised_sketch.solve(tolerance=_TOLERANCE)

    def solved_center(target: Array) -> Array:
        solution = solve_target(target)
        return eqx.error_if(
            solution.points[0],
            ~solution.converged,
            "target sensitivity requires a converged sketch",
        )

    target = constraint.target
    direction = jnp.asarray((1.0, 0.0), dtype=jnp.float64)
    solution = solve_target(target)
    assert bool(solution.converged)
    assert int(solution.status) == int(OptimizationStatus.SUCCESS)
    _, tangent = jax.jvp(solved_center, (target,), (direction,))
    np.testing.assert_allclose(
        tangent, direction, rtol=0.0, atol=_SKETCH_TARGET_IMPLICIT_TANGENT_ATOL
    )

    step = 2.0**-12
    difference = (
        solved_center(target + step * direction)
        - solved_center(target - step * direction)
    ) / (2.0 * step)
    # Each coordinate's solve error is at most the weighted residual bound.
    # The two-sided quotient therefore adds at most tolerance / step.
    difference_bound = _TOLERANCE / step
    difference_bound += 8.0 * np.finfo(np.float64).eps / step
    np.testing.assert_allclose(tangent, difference, rtol=0.0, atol=difference_bound)


def test_underconstrained_sketch_reports_undefined_implicit_derivative() -> None:
    constraint = phx.geometry.Radius(0, 1.25)
    sketch = phx.geometry.Sketch(
        [[0.2, -0.1]],
        circle_centers=[0],
        circle_radii=[1.0],
        constraints=(constraint,),
        feature_id="underconstrained-circle-sensitivity",
    )
    solution = sketch.solve(tolerance=_TOLERANCE)
    assert bool(solution.converged)
    assert int(solution.status) == int(OptimizationStatus.SUCCESS)
    assert int(solution.derivative_rank) == 1
    assert int(solution.derivative_evidence.status) == int(SensitivityStatus.SINGULAR)

    def solved_radius(target: Array) -> Array:
        revised_constraint = eqx.tree_at(lambda value: value.radius, constraint, target)
        revised_sketch = eqx.tree_at(
            lambda value: value.constraints[0], sketch, revised_constraint
        )
        return revised_sketch.solve(tolerance=_TOLERANCE).circle_radii[0]

    target = constraint.radius
    _, tangent = jax.jvp(solved_radius, (target,), (jnp.ones_like(target),))
    cotangent = jax.grad(solved_radius)(target)
    assert bool(jnp.isnan(tangent))
    assert bool(jnp.isnan(cotangent))
