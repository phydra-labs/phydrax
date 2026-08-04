import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def _linear_gaussian_problem():
    observations = phx.stochastic.ObservationSequence(
        jnp.asarray([0.2, 0.4, 0.6, 0.8, 1.0]),
        jnp.asarray([[0.1], [0.35], [0.55], [0.8], [1.0]]),
        case_ids=("analytic",),
        sequence_id="analytic-filter-sequence",
    )
    prior = phx.stochastic.GaussianStatePrior(
        jnp.asarray([0.0]),
        jnp.asarray([[1.0]]),
        state_shape=(1,),
        prior_id="analytic-prior",
    )
    transition = phx.stochastic.LinearGaussianTransitionKernel(
        jnp.asarray([[1.0]]),
        jnp.asarray([[0.05]]),
        state_shape=(1,),
        process_id="analytic-random-walk",
    )
    observation = phx.stochastic.LinearGaussianObservationModel(
        jnp.asarray([[1.0]]),
        jnp.asarray([[0.1]]),
        state_shape=(1,),
        observation_shape=(1,),
    )
    model = phx.stochastic.StateSpaceModel(
        prior, transition, observation, model_id="analytic-state-space"
    )
    return phx.stochastic.StateSpaceProblem(
        model,
        observations,
        initial_time=0.0,
        problem_id="analytic-filter",
    )


def test_particle_and_ensemble_filters_recover_analytic_kalman_moments():
    problem = _linear_gaussian_problem()
    exact = phx.uq.kalman_filter(problem)
    particles = phx.uq.bootstrap_particle_filter(
        jr.key(80),
        problem,
        num_particles=1024,
        resampling_method="systematic",
        resampling_policy="ess",
    )
    ensemble = phx.uq.ensemble_transform_kalman_filter(
        jr.key(81),
        problem,
        ensemble_size=256,
    )

    exact_mean = exact.final_state.mean[0]
    exact_variance = exact.final_state.covariance[0, 0]
    particle_weights = jnp.exp(particles.final_state.log_weights)
    particle_values = particles.final_state.particles[:, 0]
    particle_mean = jnp.sum(particle_weights * particle_values)
    particle_variance = jnp.sum(particle_weights * (particle_values - particle_mean) ** 2)
    ensemble_values = ensemble.final_state.ensemble[:, 0]
    ensemble_mean = jnp.mean(ensemble_values)
    ensemble_variance = jnp.var(ensemble_values, ddof=1)

    assert particles.successful
    assert ensemble.successful
    assert jnp.abs(particle_mean - exact_mean) < 0.05
    assert jnp.abs(particle_variance - exact_variance) < 0.03
    assert jnp.abs(ensemble_mean - exact_mean) < 0.05
    assert jnp.abs(ensemble_variance - exact_variance) < 0.03


def _heat_bsde_evaluation(realization, num_steps):
    times = jnp.linspace(0.0, 1.0, num_steps + 1)
    increments = realization.increments(times[:-1], times[1:])
    states = jnp.concatenate(
        (
            jnp.zeros(realization.sample_shape + (1, 1)),
            jnp.cumsum(increments, axis=-2),
        ),
        axis=-2,
    )
    paths = phx.stochastic.BSDEPathBatch(
        times,
        states,
        increments,
        sample_shape=realization.sample_shape,
        state_shape=(1,),
        noise_shape=(1,),
        path_id=f"heat-{num_steps}",
        process_id="heat-wiener",
        realization=realization,
    )
    problem = phx.stochastic.BSDEProblem(
        lambda key: paths,
        lambda time, state, args: jnp.zeros((1,)),
        lambda time, state, args: jnp.ones((1, 1)),
        lambda time, state, value, control, args: jnp.zeros((1,)),
        lambda state, args: jnp.asarray([state[0] ** 2]),
        state_shape=(1,),
        noise_shape=(1,),
        output_shape=(1,),
        problem_id="analytic-heat-bsde",
        process_id="heat-wiener",
    )
    value = lambda time, state: jnp.asarray([state[0] ** 2 + 1.0 - time])
    return (
        problem,
        value,
        phx.stochastic.evaluate_bsde(
            problem,
            paths,
            value,
            control_mode="autodiff",
        ),
    )


def test_heat_bsde_discrete_residual_refines_at_analytic_rate():
    realization = phx.stochastic.WienerRealization(
        jr.key(82),
        (1,),
        support=(0.0, 1.0),
        sample_shape=(512,),
        tolerance=1e-4,
        noise_id="heat-wiener",
        label="analytic-heat",
    )
    coarse_problem, coarse_value, coarse = _heat_bsde_evaluation(realization, 8)
    _, _, fine = _heat_bsde_evaluation(realization, 32)
    coarse_local_mse = jnp.mean(coarse.local_residuals**2)
    fine_local_mse = jnp.mean(fine.local_residuals**2)
    coarse_global_mse = jnp.mean(coarse.global_residual**2)
    fine_global_mse = jnp.mean(fine.global_residual**2)

    assert fine_local_mse < 0.2 * coarse_local_mse
    assert fine_global_mse < 0.45 * coarse_global_mse
    assert jnp.allclose(
        phx.stochastic.semilinear_pde_residual(
            coarse_problem,
            coarse_value,
            jnp.asarray(0.3),
            jnp.asarray([0.4]),
        ),
        0.0,
    )
