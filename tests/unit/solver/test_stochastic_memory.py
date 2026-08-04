import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def test_volterra_left_convolution_recovers_deterministic_integral_equation():
    rate = 0.4
    times = jnp.linspace(0.0, 1.0, 101)
    problem = phx.solver.StochasticVolterraProblem(
        lambda time, state, args: rate * state,
        jnp.asarray([1.0]),
        t0=0.0,
        t1=1.0,
    )
    solution = phx.solver.solve_stochastic_volterra(problem, times=times)

    assert solution.states.shape == (101, 1)
    assert jnp.all(solution.successful)
    assert jnp.allclose(solution.states[-1, 0], jnp.exp(rate), atol=1.3e-3)


def test_volterra_stochastic_convolution_replays_global_wiener_path():
    times = jnp.linspace(0.0, 1.0, 17)
    realization = phx.stochastic.WienerRealization(
        jr.key(80),
        (1,),
        support=(0.0, 1.0),
        sample_shape=(8,),
        tolerance=1e-5,
        noise_id="volterra-noise",
    )
    problem = phx.solver.StochasticVolterraProblem(
        lambda time, state, args: jnp.zeros_like(state),
        jnp.asarray([0.2]),
        t0=0.0,
        t1=1.0,
        diffusion=lambda time, state, args: jnp.ones((1, 1)),
        diffusion_kernel=lambda target, source, args: jnp.sqrt(target - source),
        noise_shape=(1,),
        noise_id="volterra-noise",
    )
    solution = phx.solver.solve_stochastic_volterra(
        problem,
        times=times,
        realization=realization,
    )
    increments = realization.increments(times[:-1], times[1:])[:, :, 0]
    expected = 0.2 + jnp.sum(jnp.sqrt(1.0 - times[:-1]) * increments, axis=-1)
    trajectory = solution.to_stochastic_trajectory(
        realization_axes=("path",),
        state_axes=("state",),
    )

    assert jnp.allclose(solution.states[:, -1, 0], expected, rtol=0.0, atol=1e-12)
    assert trajectory.realizations == (realization,)
    assert trajectory.states.shape == (8, 17, 1)


def test_delay_solver_interpolates_causal_history_and_resolved_states():
    delay = 0.4
    times = jnp.linspace(0.0, 0.8, 81)
    problem = phx.solver.StochasticDelayProblem(
        lambda time, state, delayed, args: delayed[0],
        lambda time, args: jnp.asarray([1.0]),
        jnp.asarray([delay]),
        t0=0.0,
        t1=0.8,
    )
    solution = phx.solver.solve_stochastic_delay(problem, times=times)
    exact = 1.0 + times + 0.5 * jnp.maximum(times - delay, 0.0) ** 2

    assert solution.states.shape == (81, 1)
    assert jnp.all(solution.successful)
    assert jnp.max(jnp.abs(solution.states[:, 0] - exact)) < 2.1e-3


def test_multi_delay_sde_uses_history_and_one_global_increment():
    times = jnp.asarray([0.0, 0.1])
    delays = jnp.asarray([0.2, 0.4])
    history = lambda time, args: jnp.asarray([1.0 + time])
    realization = phx.stochastic.WienerRealization(
        jr.key(81),
        (1,),
        support=(0.0, 0.1),
        sample_shape=(6,),
        tolerance=1e-5,
        noise_id="delay-noise",
    )
    problem = phx.solver.StochasticDelayProblem(
        lambda time, state, delayed, args: jnp.sum(delayed, axis=0),
        history,
        delays,
        t0=0.0,
        t1=0.1,
        diffusion=lambda time, state, delayed, args: jnp.asarray([[0.3]]),
        noise_shape=(1,),
        noise_id="delay-noise",
    )
    solution = phx.solver.solve_stochastic_delay(
        problem,
        times=times,
        realization=realization,
    )
    delayed = history(-0.2, None) + history(-0.4, None)
    increment = realization.increments(times[:-1], times[1:])[:, 0, 0]
    expected = 1.0 + 0.1 * delayed[0] + 0.3 * increment

    assert solution.states.shape == (6, 2, 1)
    assert jnp.allclose(solution.states[:, 1, 0], expected, rtol=0.0, atol=1e-12)
