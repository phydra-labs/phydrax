#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def test_fixed_observation_likelihood_matches_manual_gaussian_sum():
    target = jnp.asarray([[1.0, -0.5], [0.2, 0.7], [-1.0, 0.4]])
    likelihood = phx.uq.GaussianLikelihood(jnp.asarray([0.2, 0.4]))
    term = phx.uq.FixedObservationLikelihood(
        lambda parameters: parameters["offset"] + target,
        target,
        likelihood,
        label="sensors",
    )
    parameters = {"offset": jnp.asarray([0.1, -0.2])}

    expected = likelihood.log_prob(parameters["offset"] + target, target).sum(axis=1)
    assert jnp.allclose(term.per_case_log_prob(parameters), expected)
    assert jnp.allclose(term.log_prob(parameters), expected.sum())
    assert jnp.allclose(
        jax.grad(term.log_prob)(parameters)["offset"],
        jnp.asarray([-7.5, 3.75]),
    )


def test_fixed_heteroscedastic_observation_extracts_explicit_parameters():
    target = jnp.asarray([0.0, 1.0, 2.0])
    likelihood = phx.uq.GaussianLocationScaleLikelihood(min_scale=1e-4)
    term = phx.uq.FixedObservationLikelihood(
        lambda parameters: parameters["location"],
        target,
        likelihood,
        parameters=lambda parameters: {"raw_scale": parameters["raw_scale"]},
    )
    parameters = {
        "location": jnp.asarray([0.1, 0.9, 2.2]),
        "raw_scale": jnp.asarray([-2.0, -1.0, -0.5]),
    }

    expected = likelihood.log_prob(
        parameters["location"],
        target,
        raw_scale=parameters["raw_scale"],
    )
    assert jnp.allclose(term.per_case_log_prob(parameters), expected)


def test_fixed_residual_likelihood_is_deterministic_and_normalized_by_scale():
    likelihood = phx.uq.GaussianLikelihood(0.25)
    term = phx.uq.FixedResidualLikelihood(
        lambda coefficient: jnp.asarray(
            [[coefficient - 2.0, 2.0 * coefficient - 4.0], [coefficient - 2.0, 0.0]]
        ),
        likelihood,
        label="pde",
    )

    first = term.per_case_log_prob(jnp.asarray(2.1))
    second = term.per_case_log_prob(jnp.asarray(2.1))
    expected = likelihood.log_prob(
        jnp.asarray([[0.1, 0.2], [0.1, 0.0]]),
        jnp.zeros((2, 2)),
    ).sum(axis=1)
    assert jnp.array_equal(first, second)
    assert jnp.allclose(first, expected)


def test_structured_gp_marginal_term_matches_direct_likelihood_and_gradients():
    points = jnp.linspace(0.0, 1.0, 16)
    observations = 0.8 * points + 0.2 * jnp.sin(2.0 * jnp.pi * points)
    discrepancy = phx.uq.ExactGaussianProcessDiscrepancy(
        points,
        observations,
    )
    physical_mean = lambda parameters: parameters["coefficient"] * points
    state = lambda parameters: phx.uq.GaussianProcessLikelihoodState(
        kernel=phx.kernels.AmplitudeKernel(
            phx.kernels.Matern32Kernel(
                length_scale=parameters["length_scale"],
            ),
            parameters["amplitude"],
        ),
        noise_scale=parameters["noise_scale"],
    )
    term = phx.uq.GaussianProcessMarginalLikelihood(
        discrepancy,
        physical_mean,
        state=state,
        label="model_discrepancy",
    )
    parameters = {
        "coefficient": jnp.asarray(0.75),
        "amplitude": jnp.asarray(0.25),
        "length_scale": jnp.asarray(0.2),
        "noise_scale": jnp.asarray(0.01),
    }
    expected = discrepancy.log_marginal_likelihood(
        physical_mean(parameters),
        state=state(parameters),
    )

    assert term.label == "model_discrepancy"
    assert term.per_case_log_prob(parameters).shape == (1,)
    assert jnp.allclose(term.log_prob(parameters), expected)
    gradient = jax.grad(term.log_prob)(parameters)
    assert all(jnp.isfinite(value) for value in gradient.values())

    malformed = phx.uq.GaussianProcessMarginalLikelihood(
        discrepancy,
        physical_mean,
        state=lambda _: {"amplitude": 0.25},
    )
    with pytest.raises(TypeError, match="GaussianProcessLikelihoodState"):
        malformed.log_prob(parameters)


def test_computation_aware_gp_elbo_term_matches_direct_bound_and_gradients():
    points = jnp.linspace(0.0, 1.0, 12)
    observations = 0.75 * points + 0.1 * jnp.sin(2.0 * jnp.pi * points)
    discrepancy = phx.uq.ComputationAwareGaussianProcessDiscrepancy(
        points,
        observations,
    )
    physical_mean = lambda parameters: parameters["coefficient"] * points
    state = lambda parameters: phx.uq.GaussianProcessLikelihoodState(
        kernel=phx.kernels.AmplitudeKernel(
            phx.kernels.Matern32Kernel(
                length_scale=parameters["length_scale"],
            ),
            parameters["amplitude"],
        ),
        noise_scale=parameters["noise_scale"],
    )
    actions = lambda parameters: phx.uq.BlockSparseGaussianProcessActionPolicy(
        parameters["actions"],
        4,
    )
    term = phx.uq.ComputationAwareGaussianProcessELBO(
        discrepancy,
        physical_mean,
        state=state,
        actions=actions,
        label="computation_aware_discrepancy",
    )
    parameters = {
        "coefficient": jnp.asarray(0.7),
        "amplitude": jnp.asarray(0.25),
        "length_scale": jnp.asarray(0.2),
        "noise_scale": jnp.asarray(0.04),
        "actions": jnp.linspace(0.5, 1.5, points.size),
    }
    expected = discrepancy.elbo(
        physical_mean(parameters),
        state=state(parameters),
        actions=actions(parameters),
    )

    assert term.label == "computation_aware_discrepancy"
    assert term.per_case_log_prob(parameters).shape == (1,)
    assert jnp.allclose(term.log_prob(parameters), expected)
    gradient = jax.grad(term.log_prob)(parameters)
    assert all(jnp.all(jnp.isfinite(value)) for value in gradient.values())

    fixed_actions = actions(parameters)
    fixed_term = phx.uq.ComputationAwareGaussianProcessELBO(
        discrepancy,
        physical_mean,
        state=state,
        actions=fixed_actions,
    )
    assert jnp.allclose(fixed_term.log_prob(parameters), expected)


def test_computation_aware_gp_elbo_term_rejects_malformed_callbacks():
    points = jnp.linspace(0.0, 1.0, 6)
    discrepancy = phx.uq.ComputationAwareGaussianProcessDiscrepancy(
        points,
        jnp.sin(points),
    )
    state = lambda _: phx.uq.GaussianProcessLikelihoodState(noise_scale=0.1)
    malformed = phx.uq.ComputationAwareGaussianProcessELBO(
        discrepancy,
        lambda _: jnp.zeros_like(points),
        state=state,
        actions=lambda _: jnp.eye(points.size),
    )
    with pytest.raises(TypeError, match="AbstractGaussianProcessActionPolicy"):
        malformed.log_prob({})


def test_fixed_supervised_likelihood_preserves_operator_and_ignores_training_weight():
    rows = jnp.linspace(0.0, 1.0, 6)[:, None]
    domain = phx.domain.DatasetDomain(rows)

    @domain.Function("data")
    def base(row):
        return row[0]

    likelihood = phx.uq.GaussianLikelihood(0.1)
    target = 4.0 * rows[:, 0]
    supervised = phx.terms.SupervisedLikelihoodTerm(
        "u",
        domain.component(),
        target,
        likelihood,
        sampling=phx.domain.PointSampling(3, design="uniform"),
        observation_operator=lambda function: 2.0 * function,
        weight=1000.0,
        sample_weight=jnp.arange(1.0, 7.0),
        reduction="mean",
        label="flux_sensors",
    )
    term = phx.uq.FixedSupervisedLikelihood(
        supervised,
        lambda coefficient: {"u": coefficient * base},
    )

    observed = term.per_case_log_prob(jnp.asarray(2.0))
    expected = likelihood.log_prob(target, target)
    assert term.label == "flux_sensors"
    assert observed.shape == (6,)
    assert jnp.allclose(observed, expected)
    assert jnp.allclose(term.log_prob(jnp.asarray(2.0)), expected.sum())


def test_fixed_supervised_likelihood_accepts_classification_sibling():
    logits = jnp.asarray([[2.0, -1.0, 0.2], [-0.5, 1.4, 0.1], [0.0, -0.3, 1.7]])
    targets = jnp.asarray([0, 1, 2], dtype=jnp.int32)
    domain = phx.domain.DatasetDomain(logits)

    @domain.Function("data")
    def field(row):
        return row

    supervised = phx.terms.SupervisedClassificationTerm(
        "phase",
        domain.component(),
        targets,
        phx.ml.TargetSchema(
            "multiclass",
            class_labels=("solid", "liquid", "gas"),
        ),
        sampling=phx.domain.PointSampling(3, design="uniform"),
        weight=50.0,
        sample_weight=jnp.asarray([3.0, 2.0, 1.0]),
        label="phase_sensors",
    )
    term = phx.uq.FixedSupervisedLikelihood(
        supervised,
        lambda _: {"phase": field},
    )
    expected = jax.nn.log_softmax(logits, axis=-1)[jnp.arange(targets.size), targets]

    assert term.label == "phase_sensors"
    assert jnp.allclose(term.per_case_log_prob(None), expected)
    assert jnp.allclose(term.log_prob(None), jnp.sum(expected))


def test_composite_terms_construct_problem_without_hidden_reweighting():
    likelihood = phx.uq.GaussianLikelihood(0.2)
    observations = phx.uq.FixedObservationLikelihood(
        lambda value: value * jnp.asarray([1.0, 2.0, 3.0]),
        jnp.asarray([1.0, 2.0, 3.0]),
        likelihood,
        label="data",
    )
    residuals = phx.uq.FixedResidualLikelihood(
        lambda value: jnp.asarray([value - 1.0, 2.0 * (value - 1.0)]),
        likelihood,
        label="physics",
    )
    composite = phx.uq.CompositePosteriorLikelihood((observations, residuals))
    space = phx.uq.ParameterSpace(jnp.asarray(0.8), priors=phx.uq.Normal(0.0, 2.0))
    problem = phx.uq.PosteriorProblem.from_terms(space, (observations, residuals))

    values = composite.term_values(jnp.asarray(0.8))
    assert tuple(values) == ("data", "physics")
    assert jnp.allclose(composite(jnp.asarray(0.8)), sum(values.values()))
    expected = (
        composite(jnp.asarray(0.8))
        + space.log_prior(jnp.asarray(0.8))
        + space.log_abs_det_jacobian(jnp.asarray(0.8))
    )
    assert jnp.allclose(problem.log_density(jnp.asarray(0.8)), expected)

    mode = phx.uq.find_map(problem)
    assert mode.converged
    assert jnp.allclose(mode.position, 0.9995, atol=1e-3)


def test_fixed_terms_reject_shape_changes_and_duplicate_labels():
    likelihood = phx.uq.GaussianLikelihood(1.0)
    malformed = phx.uq.FixedObservationLikelihood(
        lambda _: jnp.ones((2,)),
        jnp.ones((3,)),
        likelihood,
        label="same",
    )
    other = phx.uq.FixedResidualLikelihood(
        lambda _: jnp.ones((3,)),
        likelihood,
        label="same",
    )

    with pytest.raises(ValueError, match="shapes are incompatible"):
        malformed.log_prob(None)
    with pytest.raises(ValueError, match="labels must be unique"):
        phx.uq.CompositePosteriorLikelihood((malformed, other))
