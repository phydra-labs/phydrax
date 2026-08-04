import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def _brownian_residuals(*, generator_offset=0.0):
    num_paths = 512
    times = jnp.linspace(0.0, 1.0, 9)
    dt = jnp.diff(times)
    increments = jr.normal(jr.key(10), (num_paths, 8)) * jnp.sqrt(dt)
    states = jnp.concatenate(
        (jnp.zeros((num_paths, 1)), jnp.cumsum(increments, axis=1)), axis=1
    )[..., None]
    realization = phx.stochastic.WienerRealization(
        jr.key(11),
        (1,),
        support=(0.0, 1.0),
        sample_shape=(num_paths,),
        noise_id="brownian",
    )
    trajectory = phx.stochastic.StochasticTrajectory(
        times,
        states,
        realization_axes=("process",),
        realization_shape=(num_paths,),
        state_axes=("state",),
        realizations=(realization,),
    )
    problem = phx.stochastic.MartingaleProblem(
        lambda state: state,
        lambda state, time: jnp.full_like(state, generator_offset),
        observable_shape=(1,),
        bracket_density=lambda state, time: jnp.asarray([[1.0]]),
    )
    return phx.stochastic.martingale_increments(trajectory, problem)


def test_brownian_martingale_and_quadratic_variation_pass():
    residuals = _brownian_residuals()
    moments = phx.uq.martingale_diagnostics(
        residuals,
        {"constant": lambda state, time: 1.0, "state": lambda state, time: state},
        confidence=0.999,
    )
    bracket = phx.stochastic.predictable_bracket_increments(residuals)
    variation = phx.uq.quadratic_variation_diagnostics(
        residuals, bracket, confidence=0.999
    )
    report = phx.uq.martingale_validation_report(moments, variation)

    assert moments.moments.shape == (2, 8, 1)
    assert moments.independent_clusters.shape == (2, 8)
    assert moments.passed
    assert variation.passed
    assert report.passed


def test_wrong_brownian_generator_is_rejected():
    residuals = _brownian_residuals(generator_offset=2.0)
    diagnostics = phx.uq.martingale_diagnostics(residuals)

    assert not diagnostics.passed
    assert jnp.max(jnp.abs(diagnostics.standardized)) > 10.0


def test_poisson_compensator_uses_complete_event_paths():
    process = phx.stochastic.JumpProcess(
        lambda time, state, args: jnp.asarray([2.0]),
        lambda state, channel, mark, args: state + jnp.asarray([1.0]),
        state_shape=(1,),
        num_channels=1,
        process_id="counting",
    )
    realization = phx.stochastic.PoissonClockRealization(
        jr.key(20),
        1,
        support=(0.0, 1.0),
        max_events_per_channel=16,
        sample_shape=(512,),
        process_id="counting",
    )
    solution = phx.solver.solve_next_reaction(
        process,
        realization,
        jnp.asarray([0.0]),
        t0=0.0,
        t1=1.0,
        save_times=jnp.asarray([0.0, 1.0]),
        max_events=16,
    )
    diagnostics = phx.uq.jump_compensator_diagnostics(
        solution.events,
        process,
        jnp.asarray([0.0]),
        t0=0.0,
        t1=1.0,
        independence_labels=phx.stochastic.realization_independence_labels(
            realization, realization.sample_shape
        ),
    )

    assert diagnostics.channel_counts.shape == (512, 1)
    assert jnp.allclose(diagnostics.integrated_intensities, 2.0)
    assert diagnostics.passed
