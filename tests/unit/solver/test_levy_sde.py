from typing import Any

import jax.numpy as jnp
import jax.random as jr
import opt_einsum as oe
import pytest

import phydrax as phx


def _driver(*, dimension=2, scale=0.2, drift=0.0):
    return phx.stochastic.SymmetricStableLevyProcess(
        1.25,
        scale,
        dimension=dimension,
        drift=drift,
        process_id=f"stable-driver-{dimension}",
    )


def test_levy_euler_reproduces_additive_truncated_driver_path():
    driver = _driver(drift=jnp.asarray([0.1, -0.2]))
    initial = jnp.asarray([0.4, -0.7])
    deterministic_drift = jnp.asarray([0.3, 0.15])
    problem = phx.solver.LevySDEProblem(
        lambda time, state, args: deterministic_drift,
        initial,
        driver,
        t0=0.0,
        t1=0.4,
    )
    realization = phx.stochastic.LevyProcessRealization.from_process(
        driver,
        jr.key(50),
        support=(0.0, 0.4),
        max_terms=128,
        sample_shape=(16,),
        gaussian_tolerance=1e-5,
    )
    cutoff = 0.04
    solution = phx.solver.solve_levy_sde(
        problem,
        realization,
        save_times=jnp.asarray([0.0, 0.2, 0.4]),
        dt=0.05,
        cutoff=cutoff,
    )
    driver_increment = realization.truncated_increments(
        driver,
        jnp.asarray([0.0]),
        jnp.asarray([0.4]),
        cutoff=cutoff,
    )[:, 0]
    expected = initial + 0.4 * deterministic_drift + driver_increment
    trajectory = solution.to_stochastic_trajectory(
        realization_axes=("path",),
        state_axes=("component",),
    )

    assert solution.solver_name == "LevyEuler"
    assert jnp.all(solution.successful)
    assert jnp.allclose(solution.states[:, -1], expected, rtol=0.0, atol=1e-11)
    assert trajectory.states.shape == (16, 3, 2)
    assert trajectory.realizations == (realization,)


def test_gaussian_small_jump_closure_uses_reserved_global_wiener_path():
    driver = _driver()
    problem = phx.solver.LevySDEProblem(
        lambda time, state, args: jnp.zeros_like(state),
        jnp.zeros((2,)),
        driver,
        t0=0.0,
        t1=0.25,
    )
    realization = phx.stochastic.LevyProcessRealization.from_process(
        driver,
        jr.key(51),
        support=(0.0, 0.25),
        max_terms=128,
        sample_shape=(32,),
        gaussian_tolerance=1e-5,
    )
    kwargs: dict[str, Any] = dict(
        save_times=jnp.asarray([0.25]),
        dt=0.25,
        cutoff=0.05,
    )
    truncated = phx.solver.solve_levy_sde(
        problem,
        realization,
        small_jumps="truncate",
        **kwargs,
    )
    gaussian = phx.solver.solve_levy_sde(
        problem,
        realization,
        small_jumps="gaussian",
        **kwargs,
    )
    replay = phx.solver.solve_levy_sde(
        problem,
        realization,
        small_jumps="gaussian",
        **kwargs,
    )
    covariance = driver.small_jump_covariance(0.05)
    eigenvalues, eigenvectors = jnp.linalg.eigh(covariance)
    factor = eigenvectors * jnp.sqrt(eigenvalues)[None, :]
    brownian = realization.gaussian_realization().increments(
        jnp.asarray([0.0]),
        jnp.asarray([0.25]),
    )[:, 0]
    expected_closure = oe.contract("ij,...j->...i", factor, brownian)

    assert jnp.array_equal(gaussian.states, replay.states)
    assert jnp.allclose(
        gaussian.states[:, 0] - truncated.states[:, 0],
        expected_closure,
        rtol=0.0,
        atol=1e-11,
    )
    assert gaussian.diagnostics.small_jump_approximation == "gaussian"


def test_levy_solver_reports_insufficient_series_capacity_without_fabrication():
    driver = _driver(dimension=1, scale=0.5)
    problem = phx.solver.LevySDEProblem(
        lambda time, state, args: jnp.zeros_like(state),
        jnp.zeros((1,)),
        driver,
        t0=0.0,
        t1=1.0,
    )
    realization = phx.stochastic.LevyProcessRealization.from_process(
        driver,
        jr.key(52),
        support=(0.0, 1.0),
        max_terms=1,
        sample_shape=(4,),
    )

    with pytest.raises(RuntimeError, match="extend the realization"):
        phx.solver.solve_levy_sde(
            problem,
            realization,
            save_times=jnp.asarray([1.0]),
            dt=0.1,
            cutoff=1e-3,
        )

    result = phx.solver.solve_levy_sde(
        problem,
        realization,
        save_times=jnp.asarray([1.0]),
        dt=0.1,
        cutoff=1e-3,
        throw=False,
    )
    assert not jnp.any(result.successful)
    assert not jnp.any(result.diagnostics.capacity_sufficient)


def test_tamed_levy_euler_bounds_only_the_deterministic_drift_update():
    driver = _driver(dimension=1, scale=0.1)
    initial = jnp.asarray([2.0])
    problem = phx.solver.LevySDEProblem(
        lambda time, state, args: state**3,
        initial,
        driver,
        t0=0.0,
        t1=0.1,
        dispersion=lambda time, state, args: jnp.zeros((1, 1)),
    )
    realization = phx.stochastic.LevyProcessRealization.from_process(
        driver,
        jr.key(53),
        support=(0.0, 0.1),
        max_terms=128,
    )
    solution = phx.solver.solve_levy_sde(
        problem,
        realization,
        save_times=jnp.asarray([0.1]),
        dt=0.1,
        cutoff=0.05,
        scheme="tamed_euler",
    )
    drift = initial**3
    expected = initial + 0.1 * drift / (1.0 + 0.1 * jnp.linalg.norm(drift))

    assert solution.solver_name == "LevyTamedEuler"
    assert jnp.allclose(solution.states[0], expected, rtol=0.0, atol=1e-12)
