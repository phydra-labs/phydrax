from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax import linalg, optim
from phydrax.optim._constrained_sensitivity import (
    constrained_solution_jvp,
    constrained_solution_vjp,
)


def _graph() -> optim.ResidualGraphProblem:
    x = optim.ParameterBlock(
        lambda value: value[:1],
        lambda value, replacement: value.at[:1].set(replacement),
        block_id="x",
    )
    y = optim.ParameterBlock(
        lambda value: value[1:],
        lambda value, replacement: value.at[1:].set(replacement),
        block_id="y",
    )
    coupled = optim.ResidualBlock(
        lambda values, _: values[0] + 2.0 * values[1] - 1.0,
        ("x", "y"),
        block_id="coupled",
    )
    anchor = optim.ResidualBlock(
        lambda values, _: values[0] - 0.25, ("x",), block_id="anchor"
    )
    return optim.ResidualGraphProblem(
        (x, y), (coupled, anchor), problem_id="sparse-curvature-oracle"
    )


def test_curvature_preserves_cross_blocks_and_owner_coordinates() -> None:
    model = optim.linearize_residual_graph(_graph(), jnp.zeros((2,), dtype=jnp.float64))
    reference = np.asarray([[2.0, 2.0], [2.0, 4.0]], dtype=np.float64)
    dense = linalg.materialize(
        model.curvature, linalg.MaterializationPolicy(max_entries=4)
    )
    np.testing.assert_allclose(dense, reference, atol=1e-12)
    direction = jnp.asarray([2.0, -3.0], dtype=jnp.float64)
    np.testing.assert_allclose(
        model.curvature.mv(direction), reference @ np.asarray(direction), atol=1e-12
    )
    np.testing.assert_allclose(
        model.curvature.adjoint_mv(direction),
        reference.T @ np.asarray(direction),
        atol=1e-12,
    )
    damped = model.curvature + jnp.asarray(
        0.3, dtype=jnp.float64
    ) * linalg.IdentityLinearOperator(model.curvature.source)
    np.testing.assert_allclose(
        damped.mv(direction),
        (reference + 0.3 * np.eye(2)) @ np.asarray(direction),
        atol=1e-12,
    )


def test_dense_route_refuses_byte_budget() -> None:
    with pytest.raises(linalg.LinearCapabilityError, match="bytes"):
        optim.solve_residual_graph(
            _graph(),
            jnp.zeros((2,), dtype=jnp.float64),
            route_policy=optim.LeastSquaresRoutePolicy(
                dense_dimension=2, dense_max_bytes=8
            ),
        )


def test_lsmr_solves_cross_factor_graph() -> None:
    result = optim.solve_residual_graph(
        _graph(),
        jnp.zeros((2,), dtype=jnp.float64),
        route_policy=optim.LeastSquaresRoutePolicy(dense_dimension=1),
        termination=optim.OptimizationTermination(
            absolute_optimality=1e-8, relative_optimality=0.0, maximum_steps=20
        ),
    )
    assert result.method_evidence.route == "lsmr"
    assert bool(result.successful)
    np.testing.assert_allclose(result.parameters, [0.25, 0.375], atol=1e-6)


def _problem(*, singular: bool = False) -> optim.MinimizationProblem:
    constraint = optim.NonlinearConstraint(
        lambda x, _: jnp.asarray([jnp.sum(x)]), lower=1.0, upper=1.0, constraint_id="sum"
    )

    def objective(x: jax.Array, args: jax.Array) -> jax.Array:
        return (
            0.5 * (jnp.sum(x) - jnp.sum(args)) ** 2
            if singular
            else 0.5 * jnp.sum((x - args) ** 2)
        )

    return optim.MinimizationProblem(
        objective, constraints=(constraint,), problem_id="sensitivity-memory-oracle"
    )


@pytest.mark.parametrize("route", ["matrix-free", "dense-svd"])
def test_sensitivity_matches_independent_projection(route: Any) -> None:
    point = jnp.asarray([0.2, 0.8], dtype=jnp.float64)
    tangent = jnp.asarray([2.0, -1.0], dtype=jnp.float64)
    cotangent = jnp.asarray([0.3, 1.7], dtype=jnp.float64)
    forward = constrained_solution_jvp(_problem(), point, point, tangent, route=route)
    reverse = constrained_solution_vjp(_problem(), point, point, cotangent, route=route)
    assert bool(forward.regular) and bool(reverse.regular)
    projection = np.eye(2) - np.ones((2, 2)) / 2.0
    np.testing.assert_allclose(forward.value, projection @ tangent, atol=1e-9)
    np.testing.assert_allclose(reverse.value, projection @ cotangent, atol=1e-9)
    np.testing.assert_allclose(
        jnp.vdot(cotangent, forward.value), jnp.vdot(reverse.value, tangent), atol=1e-9
    )
    assert forward.route == route


@pytest.mark.parametrize("route", ["matrix-free", "dense-svd"])
def test_sensitivity_refuses_hidden_singular_direction(route: Any) -> None:
    point = jnp.asarray([0.2, 0.8], dtype=jnp.float64)
    result = constrained_solution_jvp(
        _problem(singular=True), point, point, jnp.ones_like(point), route=route
    )
    assert not bool(result.regular)
    assert bool(jnp.all(jnp.isnan(result.value)))


def test_sensitivity_dense_svd_requires_byte_admission() -> None:
    point = jnp.asarray([0.2, 0.8], dtype=jnp.float64)
    with pytest.raises(linalg.LinearCapabilityError, match="bytes"):
        constrained_solution_jvp(
            _problem(),
            point,
            point,
            jnp.ones_like(point),
            route="dense-svd",
            materialization=linalg.MaterializationPolicy(max_bytes=8),
        )


def test_matrix_free_sensitivity_uses_active_bound_identity_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Sensitivity must not build a dense Jacobian.")

    monkeypatch.setattr(jax, "jacfwd", forbidden)
    problem = optim.MinimizationProblem(
        lambda x, target: 0.5 * jnp.sum((x - target) ** 2),
        bounds=optim.Bounds(0.0, jnp.inf),
        problem_id="active-bound-identity",
    )
    point = jnp.asarray([0.0, 1.0], dtype=jnp.float64)
    target = jnp.asarray([-1.0, 1.0], dtype=jnp.float64)
    tangent = jnp.asarray([2.0, 3.0], dtype=jnp.float64)
    result = constrained_solution_jvp(problem, point, target, tangent)
    assert bool(result.regular)
    assert int(result.active_constraints) == 1
    assert result.regularity_evidence is not None
    assert int(result.regularity_evidence.rank) == 3
    np.testing.assert_allclose(result.value, [0.0, 3.0], atol=1e-9)


def test_curvature_preserves_opaque_global_retraction() -> None:
    def replace_x(value: jax.Array, replacement: jax.Array) -> jax.Array:
        change = replacement[0] - value[0]
        return value.at[0].set(replacement[0]).at[1].add(change)

    x = optim.ParameterBlock(lambda value: value[:1], replace_x, block_id="x")
    y = optim.ParameterBlock(
        lambda value: value[1:],
        lambda value, replacement: value.at[1:].set(replacement),
        block_id="y",
    )
    graph = optim.ResidualGraphProblem(
        (x, y), _graph().residual_blocks, problem_id="opaque-retraction"
    )
    point = jnp.asarray([0.1, 0.3], dtype=jnp.float64)
    model = optim.linearize_residual_graph(graph, point)
    policy = linalg.MaterializationPolicy(max_entries=4)
    expected = np.asarray([[10.0, 6.0], [6.0, 4.0]], dtype=np.float64)
    np.testing.assert_allclose(
        linalg.materialize(model.curvature, policy), expected, atol=1e-12
    )
    np.testing.assert_allclose(model.gradient, [-1.05, -0.6], atol=1e-12)
    np.testing.assert_allclose(model.objective, 0.05625, atol=1e-12)
    np.testing.assert_allclose(model.residuals[0], [-0.3], atol=1e-12)


def test_sensitivity_dense_route_preserves_mixed_precision() -> None:
    from phydrax._nonlinear_precision import NonlinearPrecisionPolicy

    precision = NonlinearPrecisionPolicy(
        state_dtype="float32",
        residual_dtype="float32",
        direction_dtype="float32",
        accumulation_dtype="float64",
        decision_dtype="float64",
        output_dtype="float32",
    )
    point = jnp.asarray([0.2, 0.8], dtype=jnp.float32)
    result = constrained_solution_jvp(
        _problem(),
        point,
        point,
        jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        precision=precision,
        route="dense-svd",
    )
    assert bool(result.regular)
    assert result.value.dtype == jnp.float32
    assert result.condition_estimate.dtype == jnp.float64
    np.testing.assert_allclose(result.value, [0.5, -0.5], atol=1e-6)


def test_schur_operator_preserves_explicit_budgeted_dense_pivot() -> None:
    graph = _graph()
    x, y = graph.parameter_blocks
    retained = optim.ParameterBlock(
        y.extract, y.replace, block_id=y.block_id, elimination_group=1
    )
    prepared = optim.prepare_residual_graph(
        optim.ResidualGraphProblem((x, retained), graph.residual_blocks),
        jnp.zeros((2,), dtype=jnp.float64),
    )
    plan = optim.prepare_schur_plan(prepared)
    matrix = np.asarray([[2.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    gradient = np.asarray([-1.0, -2.0], dtype=np.float64)
    result = optim.solve_schur_system(
        linalg.DenseLinearOperator(jnp.asarray(matrix)),
        jnp.asarray(gradient),
        plan,
        linear=linalg.LinearSolvePolicy(
            linalg.DenseLU(),
            materialization=linalg.MaterializationPolicy(max_entries=4, max_bytes=32),
        ),
    )
    np.testing.assert_allclose(result, np.linalg.solve(matrix, -gradient), atol=1e-10)
