import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def _geometric_problem(rate=0.7):
    return phx.solver.DifferentialProblem(
        lambda t, state, value: value * state,
        jnp.asarray([2.0]),
        t0=0.0,
        t1=1.0,
        args=jnp.asarray(rate),
    )


def _brownian_problem(*, interpretation="ito"):
    return phx.solver.DifferentialProblem(
        lambda t, state, args: jnp.zeros_like(state),
        jnp.asarray([0.0]),
        t0=0.0,
        t1=1.0,
        diffusion=lambda t, state, args: jnp.ones((1, 1)),
        interpretation=interpretation,
    )


def test_solve_diffrax_ode_is_accurate_differentiable_and_jittable():
    times = jnp.linspace(0.0, 1.0, 11)
    solution = phx.solver.solve_diffrax(_geometric_problem(), save_times=times)
    expected = 2.0 * jnp.exp(0.7 * times)

    def terminal(rate):
        solved = phx.solver.solve_diffrax(
            _geometric_problem(rate),
            save_times=jnp.asarray([1.0]),
        )
        return solved.states[-1, 0]

    compiled = jax.jit(terminal)(jnp.asarray(0.7))
    derivative = jax.grad(terminal)(jnp.asarray(0.7))

    assert solution.states.shape == (11, 1)
    assert solution.times.shape == (11,)
    assert solution.sample_shape == ()
    assert solution.solver_name == "Tsit5"
    assert bool(solution.successful)
    assert jnp.allclose(solution.states[:, 0], expected, rtol=2e-5, atol=2e-6)
    assert jnp.allclose(compiled, 2.0 * jnp.exp(0.7), rtol=2e-5)
    assert jnp.allclose(derivative, 2.0 * jnp.exp(0.7), rtol=3e-5)


def test_solve_diffrax_sde_replays_driver_and_changes_with_key():
    problem = _brownian_problem()
    times = jnp.asarray([0.25, 0.5, 1.0])
    first_driver = phx.solver.WienerDriver(
        jr.key(3),
        (1,),
        tolerance=1e-3,
        basis_id="scalar",
        realization_id="first",
    )
    first = phx.solver.solve_diffrax(
        problem,
        save_times=times,
        driver=first_driver,
        dt0=0.02,
    )
    replay = phx.solver.solve_diffrax(
        problem,
        save_times=times,
        driver=first_driver,
        dt0=0.02,
    )
    changed = phx.solver.solve_diffrax(
        problem,
        save_times=times,
        driver=phx.solver.WienerDriver(jr.key(4), (1,), tolerance=1e-3),
        dt0=0.02,
    )

    assert first.solver_name == "Euler"
    assert first.driver is first_driver
    assert first.driver.basis_id == "scalar"
    assert first.driver.realization_id == "first"
    assert jnp.array_equal(first.states, replay.states)
    assert not jnp.array_equal(first.states, changed.states)
    assert jnp.array_equal(
        jr.key_data(first.realization_keys), jr.key_data(first_driver.key)
    )


def test_solve_diffrax_ensemble_has_process_axis_and_brownian_moments():
    solution = phx.solver.solve_diffrax_ensemble(
        _brownian_problem(),
        save_times=jnp.asarray([1.0]),
        driver=phx.solver.WienerDriver(jr.key(8), (1,), tolerance=1e-3),
        num_paths=256,
        dt0=0.02,
    )
    terminal = solution.states[:, -1, 0]

    assert solution.states.shape == (256, 1, 1)
    assert solution.times.shape == (256, 1)
    assert solution.valid.shape == (256, 1)
    assert solution.sample_shape == (256,)
    assert solution.realization_keys.shape == (256,)
    assert jnp.all(solution.successful)
    assert abs(float(jnp.mean(terminal))) < 0.15
    assert jnp.allclose(jnp.var(terminal), 1.0, rtol=0.2, atol=0.1)


def test_stratonovich_defaults_to_euler_heun_and_accepts_explicit_solver():
    problem = _brownian_problem(interpretation="stratonovich")
    driver = phx.solver.WienerDriver(jr.key(5), (1,), tolerance=1e-3)
    default = phx.solver.solve_diffrax(
        problem,
        save_times=jnp.asarray([1.0]),
        driver=driver,
        dt0=0.02,
    )
    explicit = phx.solver.solve_diffrax(
        problem,
        save_times=jnp.asarray([1.0]),
        driver=driver,
        solver=dfx.EulerHeun(),
        dt0=0.02,
    )

    assert default.solver_name == "EulerHeun"
    assert jnp.array_equal(default.states, explicit.states)


def test_diffrax_contract_rejects_invalid_problem_driver_and_save_configuration():
    deterministic = _geometric_problem()
    stochastic = _brownian_problem()
    driver = phx.solver.WienerDriver(jr.key(0), (1,))

    with pytest.raises(ValueError, match="requires t1 > t0"):
        phx.solver.DifferentialProblem(
            lambda t, state, args: state,
            jnp.asarray([1.0]),
            t0=1.0,
            t1=0.0,
        )
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError), match="strictly increasing"
    ):
        phx.solver.solve_diffrax(
            deterministic,
            save_times=jnp.asarray([0.5, 0.5]),
        )
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="time interval"):
        phx.solver.solve_diffrax(deterministic, save_times=jnp.asarray([1.1]))
    with pytest.raises(ValueError, match="require a WienerDriver"):
        phx.solver.solve_diffrax(
            stochastic,
            save_times=jnp.asarray([1.0]),
            dt0=0.01,
        )
    with pytest.raises(ValueError, match="explicit dt0"):
        phx.solver.solve_diffrax(
            stochastic,
            save_times=jnp.asarray([1.0]),
            driver=driver,
        )
    with pytest.raises(ValueError, match="do not accept a WienerDriver"):
        phx.solver.solve_diffrax(
            deterministic,
            save_times=jnp.asarray([1.0]),
            driver=driver,
        )
    with pytest.raises(ValueError, match="requires a stochastic problem"):
        phx.solver.solve_diffrax_ensemble(
            deterministic,
            save_times=jnp.asarray([1.0]),
            driver=driver,
            num_paths=2,
            dt0=0.01,
        )
