from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import pytest
from jax import Array

import phydrax as phx


def linear_residual(time: Array, state: Array, rate: Array, parameter: Array) -> Array:
    del time
    return rate + parameter * state


def solve(
    reuse: bool,
    *,
    residual: Callable[[Array, Array, Array, Array], Array] = linear_residual,
    times: Array | None = None,
    maximum_steps: int = 12,
) -> phx.solver.DifferentialAlgebraicSolution:
    system = phx.dynamics.DifferentialAlgebraicSystem(
        residual,
        state_shape=(1,),
        structure=phx.dynamics.DAEStructure(("differential",)),
        system_id="reuse-reference",
    )
    problem = phx.solver.DifferentialAlgebraicProblem(
        system,
        jnp.ones(1, dtype=jnp.float64),
        initial_state_rate=-jnp.ones(1, dtype=jnp.float64),
        args=jnp.asarray(1.0),
        problem_id="reuse-reference",
    )
    grid = phx.dynamics.TimeGrid(
        jnp.arange(9, dtype=jnp.float64) * 0.125 if times is None else times,
        time_id="reuse-reference",
    )
    policy = phx.solver.DAESolvePolicy(
        method=phx.solver.BDFMethod(1),
        temporal_reuse=phx.solver.DAETemporalReusePolicy(
            enabled=reuse,
            refresh_after_iterations=None,
        ),
        nonlinear_termination=phx.nonlinear.NonlinearTermination(
            absolute_residual=1e-11, relative_residual=1e-11, maximum_steps=maximum_steps
        ),
        regularity=phx.solver.DAERegularityPolicy("periodic"),
    )
    return phx.solver.solve_dae(problem, grid, policy=policy)


def test_fixed_grid_reuse_halves_refreshes_with_identical_certificates() -> None:
    reused, refreshed = solve(True), solve(False)
    assert reused.successful & refreshed.successful
    assert jnp.allclose(reused.states[:, 0], (1 / 1.125) ** jnp.arange(9), atol=1e-11)
    assert jnp.array_equal(reused.status, refreshed.status)
    assert jnp.array_equal(
        reused.regularity.stage_status, refreshed.regularity.stage_status
    )
    assert jnp.array_equal(reused.regularity.stage_rank, refreshed.regularity.stage_rank)
    assert jnp.array_equal(
        reused.regularity.stage_condition_estimate,
        refreshed.regularity.stage_condition_estimate,
    )
    assert jnp.allclose(reused.residual_norm, refreshed.residual_norm, atol=1e-11)
    assert jnp.sum(reused.attempt_history.jacobian_preparations) <= 0.5 * jnp.sum(
        refreshed.attempt_history.jacobian_preparations
    )
    assert jnp.sum(reused.attempt_history.stale_jacobian_retries) == 0


def nonlinear_residual(time: Array, state: Array, rate: Array, parameter: Array) -> Array:
    del parameter
    return rate + (1 + 15 * time) * state**3


def test_fixed_grid_stale_failure_mandates_fresh_retry() -> None:
    reused = solve(
        True,
        residual=nonlinear_residual,
        times=jnp.asarray([0.0, 0.125, 0.25]),
        maximum_steps=4,
    )
    refreshed = solve(
        False,
        residual=nonlinear_residual,
        times=jnp.asarray([0.0, 0.125, 0.25]),
        maximum_steps=4,
    )
    assert refreshed.successful
    assert reused.successful
    assert jnp.sum(reused.attempt_history.stale_jacobian_retries) > 0
    assert jnp.allclose(reused.states, refreshed.states, atol=1e-10)


@pytest.mark.parametrize("reuse", [False, True])
def test_fixed_grid_reuse_preserves_implicit_derivative(reuse: bool) -> None:
    system = phx.dynamics.DifferentialAlgebraicSystem(
        linear_residual,
        state_shape=(1,),
        structure=phx.dynamics.DAEStructure(("differential",)),
        system_id="reuse-gradient",
    )
    problem = phx.solver.DifferentialAlgebraicProblem(
        system,
        jnp.ones(1),
        initial_state_rate=-jnp.ones(1),
        args=jnp.asarray(1.0),
        problem_id="reuse-gradient",
    )
    grid = phx.dynamics.TimeGrid(
        jnp.arange(5, dtype=jnp.float64) * 0.125, time_id="reuse-gradient"
    )
    prepared = phx.solver.prepare_dae(
        problem,
        grid,
        policy=phx.solver.DAESolvePolicy(
            method=phx.solver.BDFMethod(1),
            temporal_reuse=phx.solver.DAETemporalReusePolicy(enabled=reuse),
        ),
    )
    derivative = jax.grad(
        lambda parameter: phx.solver.solve_dae(prepared, args=parameter).states[-1, 0]
    )(jnp.asarray(1.0))
    assert jnp.allclose(derivative, -4 * 0.125 / 1.125**5, atol=1e-9)


def test_varying_stage_shift_requires_fresh_numerics() -> None:
    times = jnp.asarray([0.0, 0.125, 0.375, 0.5], dtype=jnp.float64)
    reused = solve(True, times=times)
    refreshed = solve(False, times=times)
    assert reused.successful & refreshed.successful
    assert jnp.allclose(reused.states, refreshed.states, atol=1e-11)
    assert jnp.allclose(
        reused.states[1:, 0],
        jnp.cumprod(1 / (1 + jnp.diff(times))),
        atol=1e-11,
    )
    assert jnp.array_equal(
        reused.attempt_history.jacobian_preparations,
        refreshed.attempt_history.jacobian_preparations,
    )


@pytest.mark.parametrize("reuse", [False, True])
def test_runtime_scalar_arguments_rebind_temporal_static_context(reuse: bool) -> None:
    system = phx.dynamics.DifferentialAlgebraicSystem(
        linear_residual,
        state_shape=(1,),
        structure=phx.dynamics.DAEStructure(("differential",)),
        system_id="runtime-static-reuse",
    )
    problem = phx.solver.DifferentialAlgebraicProblem(
        system,
        jnp.ones(1, dtype=jnp.float64),
        initial_state_rate=-jnp.ones(1, dtype=jnp.float64),
        args=1.0,
        problem_id="runtime-static-reuse",
    )
    grid = phx.dynamics.TimeGrid(
        jnp.arange(5, dtype=jnp.float64) * 0.125, time_id="runtime-static-reuse"
    )
    prepared = phx.solver.prepare_dae(
        problem,
        grid,
        policy=phx.solver.DAESolvePolicy(
            method=phx.solver.BDFMethod(1),
            temporal_reuse=phx.solver.DAETemporalReusePolicy(enabled=reuse),
        ),
    )
    result = phx.solver.solve_dae(prepared, args=2.0)
    assert bool(result.successful)
    assert jnp.allclose(result.states[:, 0], 1.25 ** -jnp.arange(5), atol=1e-10)
    assert prepared.problem.args == 1.0
