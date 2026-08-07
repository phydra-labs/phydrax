import diffrax as dfx
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
    assert solution.solver_id == "solver:volterra:left-convolution-euler:v1"
    assert solution.resolved_method == "explicit-left-convolution"
    assert solution.stats["num_accepted_steps"] == 100


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
    problem = phx.solver.DelayDifferentialProblem(
        lambda time, state, delayed, args: delayed[0],
        lambda time, args: jnp.asarray([1.0]),
        (phx.solver.ConstantDelay("lag", delay),),
        t0=0.0,
        t1=0.8,
    )
    solution = phx.solver.solve_diffrax_delay(
        problem,
        save_times=times,
        solver=dfx.Euler(),
        stepsize_controller=dfx.ConstantStepSize(),
        dt0=0.01,
        max_steps=96,
    )
    exact = 1.0 + times + 0.5 * jnp.maximum(times - delay, 0.0) ** 2

    assert solution.states.shape == (81, 1)
    assert jnp.all(solution.successful)
    assert jnp.max(jnp.abs(solution.states[:, 0] - exact)) < 2.1e-3
    assert solution.solver_id == "solver:diffrax-delay:Euler:retarded-v1"
    assert solution.resolved_method == "Euler:causal-retarded-method-of-steps"
    assert solution.stats["num_rejected_steps"] == 0


def test_multi_delay_sde_uses_one_global_wiener_increment():
    times = jnp.asarray([0.0, 0.1])
    history = lambda time, args: jnp.asarray([1.0 + time])
    problem = phx.solver.DelayDifferentialProblem(
        lambda time, state, memory, args: memory["short"] + memory["long"],
        history,
        (
            phx.solver.ConstantDelay("short", 0.2),
            phx.solver.ConstantDelay("long", 0.4),
        ),
        t0=0.0,
        t1=0.1,
        wiener_terms=(
            phx.solver.DelayWienerTerm(
                "noise",
                lambda time, state, memory, args: jnp.asarray([[0.3]]),
                (1,),
                structure="additive",
                basis_id="delay-noise",
            ),
        ),
    )
    realization = phx.stochastic.WienerRealization(
        jr.key(81),
        problem.noise_shape,
        support=(0.0, 0.1),
        sample_shape=(6,),
        tolerance=1e-5,
        noise_id=problem.noise_id,
    )
    solution = phx.solver.solve_diffrax_delay(
        problem,
        save_times=times,
        realization=realization,
        solver=dfx.Euler(),
        dt0=0.1,
        max_steps=8,
    )
    delayed = history(-0.2, None) + history(-0.4, None)
    increment = realization.increments(times[:-1], times[1:])[:, 0, 0]
    expected = 1.0 + 0.1 * delayed[0] + 0.3 * increment

    assert solution.states.shape == (6, 2, 1)
    assert jnp.allclose(solution.states[:, 1, 0], expected, rtol=0.0, atol=1e-12)


def test_obsolete_fixed_grid_delay_api_is_not_public():
    assert not hasattr(phx.solver, "StochasticDelayProblem")
    assert not hasattr(phx.solver, "solve_stochastic_delay")
