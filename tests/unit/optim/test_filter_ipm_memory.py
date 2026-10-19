#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax import optim
from phydrax.linalg import prepare_linearization
from phydrax.optim._constrained_model import prepare_constrained_model_evaluation
from phydrax.optim._filter_ipm import _select_derivative_trial
from phydrax.optim._primal_dual import _build_kkt_system, PrimalDualNewtonKrylov


@jax.custom_jvp
def _trial_objective(point: Array) -> Array:
    return jnp.sum((point - 1.0) ** 2)


@_trial_objective.defjvp
def _trial_objective_jvp(
    primals: tuple[Array], tangents: tuple[Array]
) -> tuple[Array, Array]:
    (point,), (tangent,) = primals, tangents
    gradient = jnp.where(point > 0.9, jnp.nan, 2.0 * (point - 1.0))
    return _trial_objective(point), jnp.vdot(gradient, tangent).real


def _problem() -> optim.MinimizationProblem:
    return optim.MinimizationProblem(
        lambda point, _: _trial_objective(point),
        bounds=optim.Bounds(-jnp.inf, jnp.inf),
        problem_id="ordered-nonfinite-derivative",
    )


@pytest.mark.parametrize("batch_size", [1, 2, 3, 8])
def test_filter_rejects_first_value_eligible_nonfinite_derivative(
    batch_size: int,
) -> None:
    result = optim.minimize(
        _problem(),
        jnp.zeros((1,), dtype=jnp.float64),
        method=optim.PrimalDualInteriorPoint(
            mode="dense-filter",
            maximum_line_search_steps=3,
            derivative_batch_size=batch_size,
        ),
        termination=optim.OptimizationTermination(
            maximum_steps=1,
            maximum_evaluations=4,
            absolute_optimality=1e-8,
            relative_optimality=0.0,
        ),
    )
    # Independent Newton reference: H=2, grad=-2, alpha=1 is poisoned,
    # alpha=1/2 is the first derivative-finite improvement. Reversing the
    # alpha order would instead accept x=1/4.
    np.testing.assert_allclose(result.parameters, [0.5], atol=1e-12)
    assert int(result.diagnostics.accepted_steps) == 1
    assert int(result.diagnostics.rejected_steps) == 1
    assert int(result.status) == int(optim.OptimizationStatus.MAXIMUM_EVALUATIONS_REACHED)
    assert int(result.diagnostics.objective_evaluations) == 4
    expected_derivatives = 3 if batch_size == 3 else 2
    assert int(result.diagnostics.gradient_evaluations) == 1 + expected_derivatives
    assert int(result.diagnostics.jacobian_evaluations) == 1 + expected_derivatives


def test_filter_rejects_finite_values_with_nonfinite_constraint_jacobian() -> None:
    problem = optim.MinimizationProblem(
        lambda point, _: jnp.sum((point - 1.0) ** 2),
        constraints=(
            optim.NonlinearConstraint(
                lambda point, _: jnp.reshape(_trial_objective(point), (1,)),
                lower=-jnp.inf,
                upper=jnp.inf,
                constraint_id="poisoned-derivative",
            ),
        ),
        bounds=optim.Bounds(-jnp.inf, jnp.inf),
        problem_id="finite-value-nonfinite-jacobian",
    )
    result = optim.minimize(
        problem,
        jnp.zeros((1,), dtype=jnp.float64),
        method=optim.PrimalDualInteriorPoint(
            mode="dense-filter",
            maximum_line_search_steps=3,
        ),
        termination=optim.OptimizationTermination(
            maximum_steps=1,
            maximum_evaluations=4,
            absolute_optimality=1e-8,
            relative_optimality=0.0,
        ),
    )
    np.testing.assert_allclose(result.parameters, [0.5], atol=1e-12)
    assert int(result.diagnostics.rejected_steps) == 1
    assert int(result.diagnostics.jacobian_evaluations) == 3


@pytest.mark.parametrize("batch_size", [1, 2, 8])
def test_filter_all_value_rejected_avoids_derivative_work(batch_size: int) -> None:
    model, initial = prepare_constrained_model_evaluation(
        _problem(), jnp.zeros((1,), dtype=jnp.float64)
    )
    points = jnp.asarray([[1.0], [0.5], [0.25]], dtype=jnp.float64)
    values = jax.vmap(lambda point: model.evaluate_values(point, None))(points)
    selected, evaluation, derivative_count = _select_derivative_trial(
        model,
        values,
        jnp.zeros((3,), dtype=jnp.bool_),
        initial,
        None,
        batch_size,
    )
    assert int(selected) == 3
    assert int(derivative_count) == 0
    np.testing.assert_array_equal(evaluation.coordinates, initial.coordinates)
    np.testing.assert_array_equal(
        evaluation.constraint_jacobian, initial.constraint_jacobian
    )


def test_filter_initial_gradient_and_jacobian_are_reused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No outer steps or final certificate budget: only preparation evaluates
    # derivatives. Counting transformation invocations avoids relying on tracing
    # side effects inside a compiled callback.
    original_grad = jax.value_and_grad
    original_jacobian = jax.jacfwd
    counts = {"gradient": 0, "jacobian": 0}

    def value_and_grad(*args: Any, **kwargs: Any) -> Any:
        counts["gradient"] += 1
        return original_grad(*args, **kwargs)

    def jacfwd(*args: Any, **kwargs: Any) -> Any:
        counts["jacobian"] += 1
        return original_jacobian(*args, **kwargs)

    monkeypatch.setattr(jax, "value_and_grad", value_and_grad)
    monkeypatch.setattr(jax, "jacfwd", jacfwd)
    result = optim.minimize(
        _problem(),
        jnp.zeros((1,), dtype=jnp.float64),
        method=optim.PrimalDualInteriorPoint(mode="dense-filter"),
        termination=optim.OptimizationTermination(maximum_steps=1, maximum_evaluations=1),
    )
    assert counts == {"gradient": 1, "jacobian": 1}
    assert int(result.diagnostics.gradient_evaluations) == 1
    assert int(result.diagnostics.jacobian_evaluations) == 1


@pytest.mark.parametrize("invalid", [0, -1, True, 1.5])
def test_filter_derivative_batch_limit_is_validated(invalid: Any) -> None:
    error = (
        ValueError
        if isinstance(invalid, int) and not isinstance(invalid, bool)
        else TypeError
    )
    with pytest.raises(error, match="derivative_batch_size"):
        optim.PrimalDualInteriorPoint(mode="dense-filter", derivative_batch_size=invalid)


@pytest.mark.parametrize("coordinates", [(0.3, 0.4), (0.7, 0.2)])
def test_prepared_kkt_actions_match_point_specific_analytic_curvature(
    coordinates: tuple[float, float],
) -> None:
    weights = jnp.asarray([3.0, 5.0], dtype=jnp.float64)
    point = jnp.asarray(coordinates, dtype=jnp.float64)
    equality_dual = jnp.asarray([0.4], dtype=jnp.float64)
    inequality_dual = jnp.asarray([0.6, 0.7], dtype=jnp.float64)
    slacks = jnp.asarray([1.2, 1.4], dtype=jnp.float64)
    problem = optim.MinimizationProblem(
        lambda value, _: 0.5 * jnp.sum(weights * value**2),
        constraints=(
            optim.NonlinearConstraint(
                lambda value, _: jnp.asarray([jnp.sum(value**2)]),
                lower=1.0,
                upper=1.0,
                constraint_id="squared-sum",
            ),
            optim.NonlinearConstraint(
                lambda value, _: value**3,
                upper=2.0,
                constraint_id="cubic-upper",
            ),
        ),
        problem_id="analytic-prepared-kkt",
    )

    def constraints(value: Array) -> tuple[Array, Array]:
        return jnp.asarray([jnp.sum(value**2) - 1.0]), value**3 - 2.0

    derivative = prepare_linearization(constraints, point)
    method = PrimalDualNewtonKrylov()
    system, _, _ = _build_kkt_system(
        method,
        problem,
        optim.constraint_layout(problem, point, None),
        lambda value: value,
        None,
        point,
        equality_dual,
        derivative.primal[0],
        inequality_dual,
        slacks,
        constraints,
        derivative,
    )
    primal_tangent = jnp.asarray([0.2, -0.1], dtype=jnp.float64)
    dual_tangent = jnp.asarray([0.3], dtype=jnp.float64)
    hessian_diagonal = weights + 2.0 * equality_dual[0] + 6.0 * point * inequality_dual
    hessian_diagonal += (3.0 * point**2) ** 2 * inequality_dual / slacks
    hessian_diagonal += method.kkt_regularization
    expected_primal = hessian_diagonal * primal_tangent + 2.0 * point * dual_tangent[0]
    expected_dual = jnp.asarray([jnp.vdot(2.0 * point, primal_tangent).real])
    action = system.operator.mv((primal_tangent, dual_tangent))
    np.testing.assert_allclose(action[0], expected_primal, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(action[1], expected_dual, rtol=1e-12, atol=1e-12)
    repeated = system.operator.mv((2.0 * primal_tangent, -dual_tangent))
    np.testing.assert_allclose(
        repeated[0],
        2.0 * hessian_diagonal * primal_tangent - 2.0 * point * dual_tangent[0],
        rtol=1e-12,
        atol=1e-12,
    )
