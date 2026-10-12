#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from numpy.typing import NDArray

import phydrax as phx
from phydrax._strict import StrictModule


class _PreparedNonlinearResidual(StrictModule):
    scale: Array

    def residual(self, parameters: Array, target: Array) -> Array:
        return self.scale * (parameters**2 - target)

    def __call__(
        self, parameters: Array, target: Array
    ) -> phx.linalg.PreparedLinearization:
        return phx.linalg.prepare_linearization(
            lambda point: self.residual(point, target), parameters
        )


def test_prepared_nonlinear_signed_model_has_known_optimum_and_dynamic_arrays() -> None:
    def solve(scale: Array, target: Array, initial: Array) -> Array:
        provider = _PreparedNonlinearResidual(scale)
        problem = phx.optim.CompositeLeastSquaresProblem(
            provider.residual,
            lambda point, _: -0.5 * jnp.sum(point**2),
            residual_linearization=provider,
        )
        return phx.optim.composite_least_squares(
            problem, initial, args=target, termination=_termination(maximum_steps=80)
        ).parameters

    compiled = eqx.filter_jit(solve)
    for scale, target, initial in ((2.0, 1.0, 0.8), (1.5, 2.0, 1.2)):
        expected = np.sqrt(target + 0.5 / scale**2)
        actual = compiled(jnp.array([scale]), jnp.array([target]), jnp.array([initial]))
        np.testing.assert_allclose(actual, [expected], atol=2e-7)

    derivative = jax.jvp(
        lambda scale: solve(scale, jnp.array([1.0]), jnp.array([0.8])),
        (jnp.array([2.0]),),
        (jnp.ones(1),),
    )[1]
    np.testing.assert_allclose(derivative, [-0.5 / (2.0**3 * np.sqrt(1.125))], atol=2e-6)


def test_prepared_residual_constructor_refuses_noncallable_provider() -> None:
    with pytest.raises(TypeError):
        phx.optim.CompositeLeastSquaresProblem(
            lambda point, _: point,
            residual_linearization=3,  # ty: ignore[invalid-argument-type]
        )


@pytest.mark.parametrize(
    ("failure", "error_type"),
    [("kind", TypeError), ("source", ValueError), ("complex", TypeError)],
    ids=["artifact-kind", "scientific-source", "real-residual"],
)
def test_prepared_residual_refuses_invalid_artifacts(
    failure: str, error_type: type[Exception]
) -> None:
    def provider(point: Array, _: None) -> phx.linalg.PreparedLinearization:
        if failure == "kind":
            return point  # ty: ignore[invalid-return-type]
        source = (
            phx.linalg.PyTreeSpace(point, space_id="unrelated-scientific-space")
            if failure == "source"
            else None
        )
        return phx.linalg.prepare_linearization(
            lambda value: value.astype(jnp.complex128) if failure == "complex" else value,
            point,
            source=source,
        )

    problem = phx.optim.CompositeLeastSquaresProblem(
        lambda point, _: point, residual_linearization=provider
    )
    with pytest.raises(error_type):
        phx.optim.composite_least_squares(
            problem, jnp.ones(1), termination=_termination()
        )


def _termination(*, maximum_steps: int = 40) -> phx.optim.OptimizationTermination:
    return phx.optim.OptimizationTermination(
        absolute_optimality=1e-10,
        relative_optimality=0.0,
        maximum_steps=maximum_steps,
    )


@pytest.mark.parametrize(
    "materialization",
    [None, phx.linalg.MaterializationPolicy(max_entries=64, max_bytes=512)],
    ids=["matrix-free", "bounded-curvature"],
)
def test_signed_scalar_is_not_a_squared_residual(
    materialization: phx.linalg.MaterializationPolicy | None,
) -> None:
    problem = phx.optim.CompositeLeastSquaresProblem(
        lambda parameters, _: jnp.array([parameters[0] - 1.0]),
        lambda parameters, _: -0.25 * parameters[0] ** 2,
    )
    result = phx.optim.composite_least_squares(
        problem,
        jnp.array([0.0]),
        method=phx.optim.GeneralizedGaussNewton(
            curvature_materialization=materialization
        ),
        termination=_termination(),
    )

    np.testing.assert_allclose(result.parameters, jnp.array([2.0]), atol=1e-7)
    np.testing.assert_allclose(result.residual_objective, 0.5, atol=1e-7)
    np.testing.assert_allclose(result.scalar_objective, -1.0, atol=1e-7)
    np.testing.assert_allclose(result.objective, -0.5, atol=1e-7)
    assert int(result.status) == int(phx.optim.OptimizationStatus.SUCCESS)


def test_vector_valued_scalar_objective_is_refused() -> None:
    problem = phx.optim.CompositeLeastSquaresProblem(
        lambda parameters, _: parameters,
        lambda parameters, _: parameters,
    )

    with pytest.raises(TypeError):
        problem.scalar_value(jnp.ones(2, dtype=jnp.float64))


def test_indefinite_scalar_curvature_uses_descent_fallback() -> None:
    problem = phx.optim.CompositeLeastSquaresProblem(
        lambda parameters, _: jnp.array([parameters[0] - 1.0]),
        lambda parameters, _: -2.0 * parameters[0] ** 2 + 0.25 * parameters[0] ** 4,
    )
    method = phx.optim.GeneralizedGaussNewton(
        initial_damping=1e-8,
        damping_increase=10.0,
        maximum_trials=3,
    )
    initial = jnp.array([0.1])
    initial_objective = problem.objective(initial)
    result = phx.optim.composite_least_squares(
        problem,
        initial,
        method=method,
        termination=_termination(maximum_steps=60),
    )

    assert result.objective < initial_objective
    assert int(result.diagnostics.direction_fallbacks) > 0
    assert jnp.all(jnp.isfinite(result.parameters))


def test_compiled_composite_solve_accepts_dynamic_observations() -> None:
    problem = phx.optim.CompositeLeastSquaresProblem(
        lambda parameters, target: parameters - target,
        lambda parameters, _: 0.5 * jnp.sum(parameters**2),
    )
    solve = eqx.filter_jit(
        lambda target: phx.optim.composite_least_squares(
            problem,
            jnp.zeros(2),
            args=target,
            termination=_termination(),
        )
    )

    result = solve(jnp.array([2.0, -4.0]))
    np.testing.assert_allclose(result.parameters, jnp.array([1.0, -2.0]), atol=1e-7)
    assert int(result.status) == int(phx.optim.OptimizationStatus.SUCCESS)


@pytest.mark.parametrize(
    "materialization",
    [None, phx.linalg.MaterializationPolicy(max_entries=64, max_bytes=512)],
    ids=["matrix-free", "bounded-curvature"],
)
def test_composite_quadratic_matches_independent_dense_solution(
    materialization: phx.linalg.MaterializationPolicy | None,
) -> None:
    design = jnp.array([[1.0, 2.0], [2.0, -1.0], [1.0, 1.0]])
    target = jnp.array([1.0, -2.0, 0.5])
    regularizer = jnp.array([[2.0, 0.25], [0.25, 1.0]])
    linear = jnp.array([0.5, -1.0])
    problem = phx.optim.CompositeLeastSquaresProblem(
        lambda parameters, _: design @ parameters - target,
        lambda parameters, _: (
            0.5 * parameters @ regularizer @ parameters - linear @ parameters
        ),
    )
    expected = np.linalg.solve(
        np.asarray(design).T @ np.asarray(design) + np.asarray(regularizer),
        np.asarray(design).T @ np.asarray(target) + np.asarray(linear),
    )

    result = phx.optim.composite_least_squares(
        problem,
        jnp.zeros(2),
        method=phx.optim.GeneralizedGaussNewton(
            curvature_materialization=materialization
        ),
        termination=_termination(),
    )

    np.testing.assert_allclose(result.parameters, expected, atol=2e-7)
    assert int(result.status) == int(phx.optim.OptimizationStatus.SUCCESS)


def _signed_residual(parameters: Array, args: tuple[Array, Array]) -> Array:
    weights, target = args
    return weights * (parameters - target)


def _signed_scalar(parameters: Array, args: tuple[Array, Array]) -> Array:
    return 0.2 * jnp.sum(jnp.cos(parameters)) + 0.05 * jnp.sum(parameters) ** 2


def _signed_point_reference(
    point: NDArray[np.float64],
    weights: NDArray[np.float64],
    target: NDArray[np.float64],
    damping: float,
) -> NDArray[np.float64]:
    gradient = weights**2 * (point - target) - 0.2 * np.sin(point) + 0.1 * point.sum()
    curvature = np.diag(weights**2 - 0.2 * np.cos(point)) + 0.1
    return point - np.linalg.solve(
        curvature + damping * np.eye(point.size, dtype=np.float64), gradient
    )


def test_bounded_curvature_refreshes_at_each_accepted_point() -> None:
    problem = phx.optim.CompositeLeastSquaresProblem(_signed_residual, _signed_scalar)
    method = phx.optim.GeneralizedGaussNewton(
        curvature_materialization=phx.linalg.MaterializationPolicy(
            max_entries=4, max_bytes=32
        )
    )
    weights = np.array([1.0, 1.4], dtype=np.float64)
    target = np.array([-0.3, 0.1], dtype=np.float64)
    args = (jnp.asarray(weights), jnp.asarray(target))
    point = jnp.array([0.2, 0.7], dtype=jnp.float64)
    state = method.init(point)
    step = eqx.filter_jit(method.step)
    first, first_state, _ = step(problem, point, state, termination=None, args=args)
    np.testing.assert_allclose(
        first,
        _signed_point_reference(np.asarray(point), weights, target, float(state.damping)),
        rtol=1e-8,
        atol=1e-9,
    )
    assert bool(first_state.metrics.accepted)
    assert int(first_state.metrics.linear_status) == int(
        phx.linalg.LinearSolveStatus.SUCCESS
    )
    second, second_state, _ = step(
        problem, first, first_state, termination=None, args=args
    )
    np.testing.assert_allclose(
        second,
        _signed_point_reference(
            np.asarray(first), weights, target, float(first_state.damping)
        ),
        rtol=1e-8,
        atol=1e-9,
    )
    assert bool(second_state.metrics.accepted)
    assert int(second_state.metrics.linear_status) == int(
        phx.linalg.LinearSolveStatus.SUCCESS
    )


def test_bounded_indefinite_curvature_is_reused_across_damping_trials() -> None:
    problem = phx.optim.CompositeLeastSquaresProblem(
        lambda parameters, _: parameters - 1.0,
        lambda parameters, _: -jnp.sum(parameters**2),
    )
    method = phx.optim.GeneralizedGaussNewton(
        curvature_materialization=phx.linalg.MaterializationPolicy(
            max_entries=1, max_bytes=8
        ),
        maximum_trials=3,
    )
    point = jnp.array([0.0], dtype=jnp.float64)
    next_point, state, objective = method.step(
        problem, point, method.init(point), termination=None, args=None
    )
    np.testing.assert_allclose(next_point, np.array([1.0], dtype=np.float64))
    np.testing.assert_allclose(objective, -1.0)
    assert bool(state.metrics.accepted)
    assert bool(state.metrics.direction_fallback)
    assert int(state.metrics.linear_status) == int(phx.linalg.LinearSolveStatus.SUCCESS)
    assert int(state.linear_solves) == 3
    assert int(state.direction_fallbacks) == 1
    assert int(state.jvp_evaluations) == 1
    assert int(state.scalar_hvp_evaluations) == 1
    assert int(state.vjp_evaluations) == 2


@pytest.mark.parametrize(
    ("maximum_entries", "maximum_bytes"),
    [(63, 512), (64, 511)],
    ids=["entry-budget", "byte-budget"],
)
def test_bounded_curvature_refuses_insufficient_parameter_capacity(
    maximum_entries: int, maximum_bytes: int
) -> None:
    method = phx.optim.GeneralizedGaussNewton(
        curvature_materialization=phx.linalg.MaterializationPolicy(
            max_entries=maximum_entries, max_bytes=maximum_bytes
        )
    )
    with pytest.raises(phx.linalg.LinearCapabilityError):
        method.init(jnp.ones(8, dtype=jnp.float64))


def test_bounded_curvature_exact_budget_preserves_parameter_pytree_coordinates() -> None:
    def residual(
        parameters: dict[str, Array], target: dict[str, Array]
    ) -> dict[str, Array]:
        return {
            "shared": parameters["shared"] - target["shared"],
            "sensor": parameters["sensor"] - target["sensor"],
        }

    def scalar(parameters: dict[str, Array], target: dict[str, Array]) -> Array:
        return -0.25 * (
            jnp.sum(parameters["shared"] ** 2) + jnp.sum(parameters["sensor"] ** 2)
        )

    target = {
        "shared": jnp.array([1.0, -2.0], dtype=jnp.float64),
        "sensor": jnp.linspace(0.1, 0.6, 6, dtype=jnp.float64),
    }
    result = phx.optim.composite_least_squares(
        phx.optim.CompositeLeastSquaresProblem(residual, scalar),
        jax.tree.map(jnp.zeros_like, target),
        args=target,
        method=phx.optim.GeneralizedGaussNewton(
            curvature_materialization=phx.linalg.MaterializationPolicy(
                max_entries=64, max_bytes=512
            )
        ),
        termination=_termination(),
    )
    np.testing.assert_allclose(
        result.parameters["shared"], 2.0 * target["shared"], atol=1e-8
    )
    np.testing.assert_allclose(
        result.parameters["sensor"], 2.0 * target["sensor"], atol=1e-8
    )
    np.testing.assert_allclose(
        result.objective,
        -0.5 * (jnp.sum(target["shared"] ** 2) + jnp.sum(target["sensor"] ** 2)),
        atol=1e-8,
    )
    assert int(result.status) == int(phx.optim.OptimizationStatus.SUCCESS)


def test_bounded_curvature_step_jvp_matches_independent_directional_difference() -> None:
    problem = phx.optim.CompositeLeastSquaresProblem(_signed_residual, _signed_scalar)
    method = phx.optim.GeneralizedGaussNewton(
        curvature_materialization=phx.linalg.MaterializationPolicy(
            max_entries=4, max_bytes=32
        )
    )
    weights = np.array([1.0, 1.4], dtype=np.float64)
    target = np.array([-0.3, 0.1], dtype=np.float64)
    args = (jnp.asarray(weights), jnp.asarray(target))
    host_point = np.array([0.2, 0.7], dtype=np.float64)
    host_direction = np.array([0.3, -0.2], dtype=np.float64)

    def step(point: Array) -> Array:
        return method.step(
            problem, point, method.init(point), termination=None, args=args
        )[0]

    value, tangent = jax.jvp(
        step, (jnp.asarray(host_point),), (jnp.asarray(host_direction),)
    )
    epsilon = 1e-5
    expected_tangent = (
        _signed_point_reference(
            host_point + epsilon * host_direction, weights, target, method.initial_damping
        )
        - _signed_point_reference(
            host_point - epsilon * host_direction, weights, target, method.initial_damping
        )
    ) / (2.0 * epsilon)
    np.testing.assert_allclose(
        value,
        _signed_point_reference(host_point, weights, target, method.initial_damping),
        rtol=1e-8,
        atol=1e-9,
    )
    np.testing.assert_allclose(tangent, expected_tangent, rtol=1e-6, atol=1e-8)
