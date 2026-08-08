#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def test_nonlinear_inverse_compares_pathfinder_nuts_and_laplace():
    x = jnp.linspace(0.0, 2.0, 30)
    query = jnp.linspace(0.0, 2.5, 37)
    true_amplitude = 1.7
    true_rate = 0.8
    observation_scale = 0.03
    observations = true_amplitude * jnp.exp(-true_rate * x)
    likelihood = phx.uq.GaussianLikelihood(observation_scale)
    space = phx.uq.ParameterSpace(
        {"amplitude": jnp.asarray(1.4), "rate": jnp.log(jnp.asarray(0.6))},
        priors={
            "amplitude": phx.uq.Normal(0.0, 3.0),
            "rate": phx.uq.LogNormal(jnp.log(0.8), 0.5),
        },
        bijectors={
            "amplitude": phx.uq.IdentityBijector(),
            "rate": phx.uq.ExpBijector(),
        },
    )
    term = phx.uq.FixedObservationLikelihood(
        lambda parameters: parameters["amplitude"] * jnp.exp(-parameters["rate"] * x),
        observations,
        likelihood,
        label="decay_sensors",
    )
    problem = phx.uq.PosteriorProblem.from_terms(
        space,
        (term,),
        predict=lambda parameters, locations: cx.Field(
            parameters["amplitude"] * jnp.exp(-parameters["rate"] * locations),
            dims=("x",),
        ),
    )

    mode = phx.uq.find_map(problem, gradient_tolerance=1e-7)
    laplace = phx.uq.fit_laplace(problem, mode.position)
    pathfinder = phx.uq.fit_pathfinder(
        problem,
        key=jr.key(102),
        num_samples=2_000,
        num_elbo_samples=100,
        max_steps=60,
    )
    nuts = phx.uq.sample_nuts(
        problem,
        key=jr.key(100),
        num_chains=4,
        num_warmup=180,
        num_samples=250,
        initial_step_size=0.05,
        target_acceptance_rate=0.9,
        max_num_doublings=8,
    )
    report = nuts.convergence_report(
        max_rhat=1.05,
        min_bulk_ess=80,
        min_tail_ess=50,
    )
    laplace_samples = laplace.sample(jr.key(101), num_samples=4096)
    nuts_prediction = nuts.predict(query, batch_size=127)
    pathfinder_prediction = pathfinder.predict(query, batch_size=127)

    nuts_amplitude = jnp.mean(nuts.samples["amplitude"])
    nuts_rate = jnp.mean(nuts.samples["rate"])
    pathfinder_amplitude = jnp.mean(pathfinder.samples["amplitude"])
    pathfinder_rate = jnp.mean(pathfinder.samples["rate"])
    assert report.passed
    assert jnp.abs(mode.parameters["amplitude"] - true_amplitude) < 2e-3
    assert jnp.abs(mode.parameters["rate"] - true_rate) < 2e-3
    assert jnp.abs(nuts_amplitude - true_amplitude) < 5e-3
    assert jnp.abs(nuts_rate - true_rate) < 5e-3
    assert jnp.abs(jnp.mean(laplace_samples["amplitude"]) - nuts_amplitude) < 3e-3
    assert jnp.abs(jnp.mean(laplace_samples["rate"]) - nuts_rate) < 3e-3
    assert jnp.abs(pathfinder_amplitude - nuts_amplitude) < 3e-3
    assert jnp.abs(pathfinder_rate - nuts_rate) < 3e-3
    assert pathfinder.duration_seconds > 0.0
    assert pathfinder.sample_memory_bytes > 0
    assert jnp.all(jnp.isfinite(pathfinder.importance_log_weights))
    exact = true_amplitude * jnp.exp(-true_rate * query)
    assert jnp.sqrt(jnp.mean((nuts_prediction.mean().data - exact) ** 2)) < 2e-3
    assert jnp.sqrt(jnp.mean((pathfinder_prediction.mean().data - exact) ** 2)) < 2e-3


def test_fixed_physics_residual_likelihood_identifies_hidden_source():
    sensor_x = jnp.linspace(0.05, 0.95, 20)
    basis_values = 0.5 * sensor_x * (1.0 - sensor_x)
    true_source = 1.5
    observation_scale = 0.02
    residual_scale = 0.04
    observations = true_source * basis_values
    geometry = phx.domain.Interval1d(0.0, 1.0)

    @geometry.Function("x")
    def poisson_basis(x):
        return 0.5 * x[0] * (1.0 - x[0])

    residual_points = {
        "x": cx.Field(jnp.linspace(0.1, 0.9, 11)[:, None], dims=("point", None))
    }

    def pde_residual(parameters):
        state = parameters["amplitude"] * poisson_basis
        residual = -phx.operators.laplacian(state, var="x") - parameters["source"]
        return residual(residual_points).data

    space = phx.uq.ParameterSpace(
        {"amplitude": jnp.asarray(1.0), "source": jnp.asarray(0.0)},
        priors={
            "amplitude": phx.uq.Normal(0.0, 3.0),
            "source": phx.uq.Normal(0.0, 3.0),
        },
    )
    data_term = phx.uq.FixedObservationLikelihood(
        lambda parameters: parameters["amplitude"] * basis_values,
        observations,
        phx.uq.GaussianLikelihood(observation_scale),
        label="sensors",
    )
    physics_term = phx.uq.FixedResidualLikelihood(
        pde_residual,
        phx.uq.GaussianLikelihood(residual_scale),
        label="poisson_residual",
    )
    data_problem = phx.uq.PosteriorProblem.from_terms(space, (data_term,))
    physics_problem = phx.uq.PosteriorProblem.from_terms(
        space,
        (data_term, physics_term),
    )

    data_mode = phx.uq.find_map(data_problem)
    physics_mode = phx.uq.find_map(physics_problem)
    physics_laplace = phx.uq.fit_laplace(physics_problem, physics_mode.position)
    nuts = phx.uq.sample_nuts(
        physics_problem,
        key=jr.key(110),
        num_chains=4,
        num_warmup=160,
        num_samples=220,
        initial_step_size=0.05,
        target_acceptance_rate=0.9,
        max_num_doublings=8,
    )
    report = nuts.convergence_report(
        max_rhat=1.05,
        min_bulk_ess=50,
        min_tail_ess=40,
    )

    assert jnp.array_equal(
        physics_term.per_case_log_prob(physics_mode.parameters),
        physics_term.per_case_log_prob(physics_mode.parameters),
    )
    assert jnp.abs(data_mode.parameters["source"] - true_source) > 1.0
    assert jnp.abs(physics_mode.parameters["source"] - true_source) < 3e-3
    assert jnp.abs(jnp.mean(nuts.samples["source"]) - true_source) < 0.02
    assert physics_laplace.covariance.shape == (2, 2)
    assert report.passed


def test_repeated_omitted_physics_discrepancy_improves_predictions_and_scores():
    observation_x = jnp.linspace(0.0, 1.0, 18)
    test_x = jnp.linspace(0.025, 0.975, 25)
    true_parameter = 1.2
    observation_scale = 0.03
    fixed_amplitude = 0.25
    fixed_length_scale = 0.22

    def truth(x):
        return true_parameter * x + 0.3 * jnp.sin(jnp.pi * x)

    no_discrepancy_rmse = []
    fixed_gp_rmse = []
    joint_gp_rmse = []
    no_discrepancy_nll = []
    fixed_gp_nll = []
    no_discrepancy_crps = []
    fixed_gp_crps = []
    no_discrepancy_parameters = []
    fixed_gp_parameters = []
    joint_gp_parameters = []
    fixed_gp_coverage = []
    joint_parameter_gp_correlations = []
    observation_likelihood = phx.uq.GaussianLikelihood(observation_scale)
    def gp_state(amplitude, length_scale, noise_scale):
        return phx.uq.GaussianProcessLikelihoodState(
            kernel=phx.kernels.AmplitudeKernel(
                phx.kernels.Matern32Kernel(length_scale=length_scale),
                amplitude,
            ),
            noise_scale=noise_scale,
        )

    fixed_state = gp_state(
        fixed_amplitude,
        fixed_length_scale,
        observation_scale,
    )

    for repeat in range(6):
        observations = truth(observation_x) + observation_scale * jr.normal(
            jr.fold_in(jr.key(120), repeat),
            observation_x.shape,
        )
        discrepancy = phx.uq.ExactGaussianProcessDiscrepancy(
            observation_x,
            observations,
        )
        physical_space = phx.uq.ParameterSpace(
            {"parameter": jnp.asarray(1.0)},
            priors={"parameter": phx.uq.Normal(0.0, 3.0)},
        )
        no_discrepancy = phx.uq.PosteriorProblem(
            physical_space,
            lambda parameters: jnp.sum(
                observation_likelihood.log_prob(
                    parameters["parameter"] * observation_x,
                    observations,
                )
            ),
        )
        fixed_gp = phx.uq.PosteriorProblem(
            physical_space,
            lambda parameters: discrepancy.log_marginal_likelihood(
                parameters["parameter"] * observation_x,
                state=fixed_state,
            ),
        )
        joint_space = phx.uq.ParameterSpace(
            {
                "parameter": jnp.asarray(1.0),
                "amplitude": jnp.log(jnp.asarray(fixed_amplitude)),
                "length_scale": jnp.log(jnp.asarray(fixed_length_scale)),
                "noise_scale": jnp.log(jnp.asarray(observation_scale)),
            },
            priors={
                "parameter": phx.uq.Normal(0.0, 3.0),
                "amplitude": phx.uq.LogNormal(jnp.log(fixed_amplitude), 0.5),
                "length_scale": phx.uq.LogNormal(jnp.log(fixed_length_scale), 0.5),
                "noise_scale": phx.uq.LogNormal(jnp.log(observation_scale), 0.3),
            },
            bijectors={
                "parameter": phx.uq.IdentityBijector(),
                "amplitude": phx.uq.ExpBijector(),
                "length_scale": phx.uq.ExpBijector(),
                "noise_scale": phx.uq.ExpBijector(),
            },
        )
        joint_gp = phx.uq.PosteriorProblem(
            joint_space,
            lambda parameters: discrepancy.log_marginal_likelihood(
                parameters["parameter"] * observation_x,
                state=gp_state(
                    parameters["amplitude"],
                    parameters["length_scale"],
                    parameters["noise_scale"],
                ),
            ),
        )

        no_mode = phx.uq.find_map(no_discrepancy)
        fixed_mode = phx.uq.find_map(fixed_gp)
        joint_mode = phx.uq.find_map(joint_gp, gradient_tolerance=1e-5)
        joint_laplace = phx.uq.fit_laplace(
            joint_gp,
            joint_mode.position,
            damping=1e-6,
            stationarity_tolerance=1e-4,
        )
        no_parameter = no_mode.parameters["parameter"]
        fixed_parameter = fixed_mode.parameters["parameter"]
        joint_parameters = joint_mode.parameters
        fixed_condition = discrepancy.condition(
            fixed_parameter * observation_x,
            test_x,
            state=fixed_state,
        )
        joint_condition = discrepancy.condition(
            joint_parameters["parameter"] * observation_x,
            test_x,
            state=gp_state(
                joint_parameters["amplitude"],
                joint_parameters["length_scale"],
                joint_parameters["noise_scale"],
            ),
        )
        target = truth(test_x)
        no_mean = no_parameter * test_x
        fixed_mean = fixed_parameter * test_x + fixed_condition.mean
        joint_mean = joint_parameters["parameter"] * test_x + joint_condition.mean
        fixed_scale = jnp.sqrt(fixed_condition.variance + observation_scale**2)
        interval_radius = 1.645 * jnp.sqrt(fixed_condition.variance)

        no_discrepancy_rmse.append(jnp.sqrt(jnp.mean((no_mean - target) ** 2)))
        fixed_gp_rmse.append(jnp.sqrt(jnp.mean((fixed_mean - target) ** 2)))
        joint_gp_rmse.append(jnp.sqrt(jnp.mean((joint_mean - target) ** 2)))
        no_discrepancy_nll.append(
            jnp.mean(-observation_likelihood.log_prob(no_mean, target))
        )
        fixed_gp_nll.append(
            jnp.mean(-phx.uq.GaussianLikelihood(fixed_scale).log_prob(fixed_mean, target))
        )
        no_discrepancy_crps.append(
            jnp.mean(phx.uq.gaussian_crps(no_mean, observation_scale, target))
        )
        fixed_gp_crps.append(
            jnp.mean(phx.uq.gaussian_crps(fixed_mean, fixed_scale, target))
        )
        no_discrepancy_parameters.append(no_parameter)
        fixed_gp_parameters.append(fixed_parameter)
        joint_gp_parameters.append(joint_parameters["parameter"])
        paths = phx.uq.ParameterSubspace.array_leaf_paths(joint_mode.parameters)
        parameter_index = paths.index("['parameter']")
        gp_indices = tuple(
            index for index, path in enumerate(paths) if path != "['parameter']"
        )
        joint_parameter_gp_correlations.append(
            joint_laplace.physical_correlation()[parameter_index, jnp.asarray(gp_indices)]
        )
        fixed_gp_coverage.append(
            jnp.mean(
                (target >= fixed_mean - interval_radius)
                & (target <= fixed_mean + interval_radius)
            )
        )

    no_rmse = jnp.mean(jnp.stack(no_discrepancy_rmse))
    fixed_rmse = jnp.mean(jnp.stack(fixed_gp_rmse))
    joint_rmse = jnp.mean(jnp.stack(joint_gp_rmse))
    no_bias = jnp.abs(jnp.mean(jnp.stack(no_discrepancy_parameters)) - true_parameter)
    joint_bias = jnp.abs(jnp.mean(jnp.stack(joint_gp_parameters)) - true_parameter)
    identifiability = phx.uq.discrepancy_identifiability_report(
        true_parameters=true_parameter,
        baseline_parameter_estimates=jnp.stack(no_discrepancy_parameters),
        fixed_gp_parameter_estimates=jnp.stack(fixed_gp_parameters),
        joint_gp_parameter_estimates=jnp.stack(joint_gp_parameters),
        baseline_nll=jnp.stack(no_discrepancy_nll),
        fixed_gp_nll=jnp.stack(fixed_gp_nll),
        baseline_crps=jnp.stack(no_discrepancy_crps),
        fixed_gp_crps=jnp.stack(fixed_gp_crps),
        fixed_gp_coverage=jnp.stack(fixed_gp_coverage),
        joint_parameter_gp_correlations=jnp.stack(joint_parameter_gp_correlations),
    )

    assert fixed_rmse < 0.25 * no_rmse
    assert joint_rmse < 0.25 * no_rmse
    assert joint_bias < 0.5 * no_bias
    assert jnp.mean(jnp.stack(fixed_gp_nll)) < jnp.mean(jnp.stack(no_discrepancy_nll))
    assert jnp.mean(jnp.stack(fixed_gp_crps)) < jnp.mean(jnp.stack(no_discrepancy_crps))
    assert jnp.mean(jnp.stack(fixed_gp_coverage)) >= 0.9
    assert identifiability.passed
