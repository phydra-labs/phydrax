import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def test_mean_attraction_preserves_empirical_mean_and_contracts_deviations():
    initial = jnp.asarray([[-1.0], [0.0], [2.0], [3.0]])
    times = jnp.linspace(0.0, 1.0, 11)
    problem = phx.solver.InteractingParticleProblem(
        lambda time, state, law, args: law.mean - state,
        initial,
        t0=0.0,
        t1=1.0,
        mean_field_id="mean-attraction-law",
    )
    solution = phx.solver.solve_interacting_particles(problem, times=times)
    initial_mean = jnp.mean(initial, axis=0)
    expected_terminal = initial_mean + (1.0 - 0.1) ** 10 * (initial - initial_mean)
    mean_field = solution.empirical_mean_field()

    assert solution.particles.shape == (11, 4, 1)
    assert jnp.all(solution.successful)
    assert jnp.allclose(solution.means, initial_mean)
    assert jnp.allclose(solution.particles[-1], expected_terminal, atol=1e-12)
    assert jnp.allclose(mean_field.snapshot(1.0).mean, initial_mean)
    assert mean_field.num_particles == 4


def test_idiosyncratic_particle_noise_replays_independent_driver_components():
    initial = jnp.asarray([[-0.3], [0.2], [0.7]])
    times = jnp.asarray([0.0, 0.25, 0.5])
    realization = phx.stochastic.WienerRealization(
        jr.key(90),
        (3, 1),
        support=(0.0, 0.5),
        sample_shape=(5,),
        tolerance=1e-5,
        noise_id="particle-noise",
    )
    problem = phx.solver.InteractingParticleProblem(
        lambda time, state, law, args: jnp.zeros_like(state),
        initial,
        t0=0.0,
        t1=0.5,
        diffusion=lambda time, state, law, args: jnp.ones((1, 1)),
        noise_shape=(1,),
        noise_id="particle-noise",
    )
    solution = phx.solver.solve_interacting_particles(
        problem,
        times=times,
        realization=realization,
    )
    increments = realization.increments(
        jnp.asarray([0.0]),
        jnp.asarray([0.5]),
    )[:, 0, :, 0]
    expected = initial[:, 0] + increments
    trajectory = solution.to_stochastic_trajectory(
        realization_axes=("system",),
        state_axes=("state",),
    )

    assert solution.particles.shape == (5, 3, 3, 1)
    assert jnp.allclose(solution.particles[:, -1, :, 0], expected, atol=1e-12)
    assert trajectory.states.shape == (5, 3, 3, 1)
    assert trajectory.realizations == (realization,)


def test_common_noise_translates_population_without_changing_pairwise_offsets():
    initial = jnp.asarray([[-1.0], [0.5], [2.0]])
    times = jnp.linspace(0.0, 0.4, 5)
    common = phx.stochastic.WienerRealization(
        jr.key(91),
        (1,),
        support=(0.0, 0.4),
        sample_shape=(7,),
        tolerance=1e-5,
        noise_id="common-particle-noise",
    )
    problem = phx.solver.InteractingParticleProblem(
        lambda time, state, law, args: jnp.zeros_like(state),
        initial,
        t0=0.0,
        t1=0.4,
        common_diffusion=lambda time, state, law, args: jnp.asarray([[0.6]]),
        common_noise_shape=(1,),
        common_noise_id="common-particle-noise",
    )
    solution = phx.solver.solve_interacting_particles(
        problem,
        times=times,
        common_realization=common,
    )
    terminal_offsets = solution.particles[:, -1, :, 0] - solution.particles[:, -1, :1, 0]
    expected_offsets = initial[:, 0] - initial[0, 0]

    assert jnp.allclose(terminal_offsets, expected_offsets)
    assert jnp.allclose(solution.covariances[:, -1], solution.covariances[:, 0])
