#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import coordax as cx
import jax.numpy as jnp
import jax.random as jr
import optax

import phydrax as phx


ISHIGAMI_FIRST_ORDER = jnp.asarray([0.3139, 0.4424, 0.0])
ISHIGAMI_TOTAL_ORDER = jnp.asarray([0.5576, 0.4424, 0.2437])


def _poisson_basis(x):
    return 0.5 * x * (1.0 - x)


def _uniform_exponential_moment(rate, lower: float, upper: float):
    width = upper - lower
    rate_width = rate * width
    regular = jnp.exp(-rate * lower) * (-jnp.expm1(-rate_width)) / rate_width
    return jnp.where(rate == 0.0, 1.0, regular)


def test_inverse_poisson_likelihood_and_posterior_benchmark():
    """Infer a Poisson source, then propagate its posterior to the solution field."""
    true_source = 4.0
    observation_scale = 0.02
    sensor_x = jnp.linspace(0.05, 0.95, 24)
    basis = _poisson_basis(sensor_x)

    # Keep a noisy, deterministic realization while removing its component along the
    # one-dimensional inverse design. The benchmark then measures implementation error,
    # not whether one unlucky finite dataset misses its nominal posterior interval.
    raw_noise = jr.normal(jr.key(10), sensor_x.shape)
    raw_noise = raw_noise - basis * jnp.vdot(basis, raw_noise) / jnp.vdot(basis, basis)
    observations = true_source * basis + observation_scale * raw_noise

    sensor_domain = phx.domain.DatasetDomain(sensor_x[:, None])
    source_parameter = sensor_domain.Parameter(1.0)

    @sensor_domain.Function("data")
    def sensor_basis(row):
        return _poisson_basis(row[0])

    state = source_parameter * sensor_basis
    likelihood = phx.uq.GaussianLikelihood(observation_scale)
    constraint = phx.terms.SupervisedLikelihoodTerm(
        "u",
        sensor_domain.component(),
        observations,
        likelihood,
        sampling=phx.domain.PointSampling(96, design="uniform"),
    )
    solver = phx.solver.FunctionalSolver(functions={"u": state}, terms=[constraint])
    initial_nll = solver.loss(key=jr.key(11))
    trained = solver.solve(
        num_iter=250,
        optim=optax.adam(5e-2),
        seed=12,
        jit=True,
        keep_best=False,  # Per-step minibatch losses are not a stable selector.
    )

    sensor_points = {
        "data": cx.Field(sensor_x[:, None], dims=("sensor", None)),
    }
    fitted_values = jnp.asarray(trained["u"](sensor_points).data)
    fitted_source = jnp.vdot(basis, fitted_values) / jnp.vdot(basis, basis)
    fitted_nll = phx.uq.negative_log_likelihood(
        likelihood,
        fitted_source * basis,
        observations,
    )

    prior_mean = 0.0
    prior_scale = 3.0
    posterior_variance = 1.0 / (
        1.0 / prior_scale**2 + jnp.vdot(basis, basis) / observation_scale**2
    )
    posterior_mean = posterior_variance * (
        prior_mean / prior_scale**2 + jnp.vdot(basis, observations) / observation_scale**2
    )
    posterior_scale = jnp.sqrt(posterior_variance)

    query_x = jnp.linspace(0.0, 1.0, 65)
    source_samples = phx.uq.sample_joint(
        {"source": phx.uq.Normal(posterior_mean, posterior_scale)},
        num_samples=2048,
        key=jr.key(13),
    )
    prediction = phx.uq.propagate(
        lambda source: cx.Field(
            source * _poisson_basis(query_x),
            dims=("x",),
        ),
        source_samples,
        batch_size=257,
    )
    interval = prediction.interval(0.05, 0.95)
    exact = true_source * _poisson_basis(query_x)

    geometry = phx.domain.Interval1d(0.0, 1.0)

    @geometry.Function("x")
    def posterior_solution(x):
        return posterior_mean * _poisson_basis(x[0])

    residual = -phx.operators.laplacian(posterior_solution, var="x") - posterior_mean
    residual_points = {
        "x": cx.Field(jnp.linspace(0.1, 0.9, 9)[:, None], dims=("point", None)),
    }

    assert trained.loss(key=jr.key(11)) < initial_nll
    assert jnp.abs(fitted_source - true_source) < 0.02
    assert fitted_nll < phx.uq.negative_log_likelihood(
        likelihood, jnp.ones_like(observations) * jnp.mean(observations), observations
    )
    assert posterior_scale < 0.05
    assert jnp.max(jnp.abs(prediction.mean().data - exact)) < 2e-3
    assert jnp.all(exact >= interval.lower.data)
    assert jnp.all(exact <= interval.upper.data)
    assert jnp.allclose(jnp.asarray(prediction.samples.data)[:, (0, -1)], 0.0, atol=1e-7)
    assert jnp.max(jnp.abs(jnp.asarray(residual(residual_points).data))) < 1e-6


def test_uncertain_heat_joint_qmc_propagation_benchmark():
    """Match analytic heat-solution moments under joint input uncertainty."""
    amplitude_mean = 1.0
    amplitude_scale = 0.15
    diffusivity_lower = 0.05
    diffusivity_upper = 0.15
    x = jnp.linspace(0.0, 1.0, 17)
    t = jnp.linspace(0.0, 1.0, 9)
    spatial = jnp.sin(jnp.pi * x)[:, None]
    decay_rate = jnp.pi**2 * t[None, :]

    distributions = {
        "amplitude": phx.uq.Normal(amplitude_mean, amplitude_scale),
        "diffusivity": phx.uq.Uniform(diffusivity_lower, diffusivity_upper),
    }
    samples = phx.uq.sample_joint(
        distributions,
        num_samples=4096,
        key=jr.key(20),
    )

    def solve_heat(amplitude, diffusivity):
        return cx.Field(
            amplitude * spatial * jnp.exp(-diffusivity * decay_rate),
            dims=("x", "t"),
        )

    prediction = phx.uq.propagate(solve_heat, samples, batch_size=513)
    first_decay_moment = _uniform_exponential_moment(
        decay_rate, diffusivity_lower, diffusivity_upper
    )
    second_decay_moment = _uniform_exponential_moment(
        2.0 * decay_rate, diffusivity_lower, diffusivity_upper
    )
    exact_mean = amplitude_mean * spatial * first_decay_moment
    exact_second = (
        (amplitude_mean**2 + amplitude_scale**2) * spatial**2 * second_decay_moment
    )
    exact_variance = exact_second - exact_mean**2

    qmc_small = phx.uq.sample_joint(
        distributions,
        num_samples=512,
        key=jr.key(21),
    )
    mc_small = phx.uq.sample_joint(
        distributions,
        num_samples=512,
        key=jr.key(21),
        sampler="uniform",
    )

    def terminal_midpoint(amplitude, diffusivity):
        return amplitude * jnp.exp(-(jnp.pi**2) * diffusivity)

    exact_terminal_mean = amplitude_mean * _uniform_exponential_moment(
        jnp.pi**2, diffusivity_lower, diffusivity_upper
    )
    qmc_error = jnp.abs(
        phx.uq.propagate(terminal_midpoint, qmc_small).mean().data - exact_terminal_mean
    )
    mc_error = jnp.abs(
        phx.uq.propagate(terminal_midpoint, mc_small).mean().data - exact_terminal_mean
    )

    domain = phx.domain.Interval1d(0.0, 1.0) @ phx.domain.TimeInterval(0.0, 1.0)
    reference_amplitude = 1.2
    reference_diffusivity = 0.08

    @domain.Function("x", "t")
    def reference_solution(x_value, time_value):
        return (
            reference_amplitude
            * jnp.sin(jnp.pi * x_value[0])
            * jnp.exp(-(jnp.pi**2) * reference_diffusivity * time_value)
        )

    pde_residual = phx.operators.dt(
        reference_solution, var="t"
    ) - reference_diffusivity * phx.operators.laplacian(reference_solution, var="x")
    residual_points = {
        "x": cx.Field(jnp.linspace(0.1, 0.9, 9)[:, None], dims=("point", None)),
        "t": cx.Field(jnp.linspace(0.05, 0.95, 9), dims=("point",)),
    }

    assert jnp.max(jnp.abs(prediction.mean().data - exact_mean)) < 8e-4
    assert jnp.max(jnp.abs(prediction.variance().data - exact_variance)) < 8e-4
    assert qmc_error < mc_error
    assert jnp.allclose(
        jnp.asarray(prediction.samples.data)[:, (0, -1), :], 0.0, atol=2e-7
    )
    assert jnp.max(jnp.abs(jnp.asarray(pde_residual(residual_points).data))) < 2e-6


def test_functional_conformal_simultaneous_coverage_benchmark():
    """Calibrate one score per trajectory and verify held-out simultaneous coverage."""
    num_cases = 12_000
    x = jnp.linspace(0.0, 1.0, 33)
    mean = jnp.sin(jnp.pi * x)
    scale = 0.04 + 0.08 * (0.25 + x)
    coefficients = jr.normal(jr.key(30), (num_cases, 4))
    standardized_error = (
        0.65 * coefficients[:, 0, None]
        + 0.45 * coefficients[:, 1, None] * jnp.cos(2.0 * jnp.pi * x)[None, :]
        + 0.35 * coefficients[:, 2, None] * jnp.sin(3.0 * jnp.pi * x)[None, :]
        + 0.20 * coefficients[:, 3, None] * (2.0 * x - 1.0)[None, :]
    )
    trajectories = mean[None, :] + scale[None, :] * standardized_error

    train_indices, calibration_indices, test_indices = (
        phx.data_utils.train_calibration_test_split_indices(
            num_cases,
            calibration_fraction=0.2,
            test_fraction=0.2,
            key=jr.key(31),
        )
    )
    fitted_center = jnp.mean(trajectories[train_indices], axis=0)
    fitted_scale = jnp.std(trajectories[train_indices], axis=0, ddof=1)
    calibration_center = jnp.broadcast_to(
        fitted_center, (calibration_indices.size, x.size)
    )
    calibration_scale = jnp.broadcast_to(fitted_scale, (calibration_indices.size, x.size))
    calibrator = phx.uq.FunctionalConformal.calibrate(
        cx.Field(calibration_center, dims=("case", "x")),
        cx.Field(trajectories[calibration_indices], dims=("case", "x")),
        alpha=0.1,
        case_dim="case",
        scale=cx.Field(calibration_scale, dims=("case", "x")),
    )

    test_center = cx.Field(
        jnp.broadcast_to(fitted_center, (test_indices.size, x.size)),
        dims=("case", "x"),
    )
    test_scale = cx.Field(
        jnp.broadcast_to(fitted_scale, (test_indices.size, x.size)),
        dims=("case", "x"),
    )
    test_target = trajectories[test_indices]
    interval = calibrator.interval(test_center, test_scale)
    lower = jnp.asarray(interval.lower.data)
    upper = jnp.asarray(interval.upper.data)
    coordinate_coverage = (test_target >= lower) & (test_target <= upper)
    simultaneous_coverage = jnp.mean(jnp.all(coordinate_coverage, axis=1))
    pointwise_coverage = jnp.mean(coordinate_coverage)

    assert interval.simultaneous
    assert interval.calibrated
    assert 0.87 <= simultaneous_coverage <= 0.93
    assert pointwise_coverage >= simultaneous_coverage
    assert jnp.isfinite(phx.uq.interval_width(lower, upper))


def test_ishigami_sobol_sensitivity_benchmark():
    """Recover reference first- and total-order Ishigami sensitivity indices."""

    def ishigami(x1, x2, x3):
        return jnp.sin(x1) + 7.0 * jnp.sin(x2) ** 2 + 0.1 * x3**4 * jnp.sin(x1)

    distributions = {
        "x1": phx.uq.Uniform(-jnp.pi, jnp.pi),
        "x2": phx.uq.Uniform(-jnp.pi, jnp.pi),
        "x3": phx.uq.Uniform(-jnp.pi, jnp.pi),
    }
    result = phx.uq.sobol_indices(
        ishigami,
        distributions,
        num_samples=8192,
        key=jr.key(40),
        batch_size=1025,
    )
    exact_variance = (
        7.0**2 / 8.0 + 0.1 * jnp.pi**4 / 5.0 + 0.1**2 * jnp.pi**8 / 18.0 + 0.5
    )

    output_variance = jnp.asarray(result.output_variance.data)
    first_order = jnp.asarray(result.first_order.data)
    total_order = jnp.asarray(result.total_order.data)
    assert result.parameter_names == ("x1", "x2", "x3")
    assert jnp.allclose(
        jnp.asarray(result.first_order.data), ISHIGAMI_FIRST_ORDER, atol=0.025
    )
    assert jnp.allclose(
        jnp.asarray(result.total_order.data), ISHIGAMI_TOTAL_ORDER, atol=0.025
    )
    assert jnp.abs(output_variance - exact_variance) < 0.1
    assert total_order[0] - first_order[0] > 0.2
    assert total_order[2] - first_order[2] > 0.2
