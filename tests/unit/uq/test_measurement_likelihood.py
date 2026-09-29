#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _normal_log_density(residual: Any, covariance: Any) -> Any:
    sign, log_determinant = jnp.linalg.slogdet(covariance)
    assert sign > 0.0
    return -0.5 * (
        residual @ jnp.linalg.solve(covariance, residual)
        + log_determinant
        + residual.size * jnp.log(2.0 * jnp.pi)
    )


def test_measurement_likelihood_scenario_1() -> None:
    inputs = jnp.asarray([[0.5], [1.0], [2.0]])
    targets = jnp.asarray([1.4, 2.1, 3.8])
    parameters = {"intercept": jnp.asarray(0.4), "slope": jnp.asarray(1.8)}
    input_variance = 0.09
    observation_variance = 0.04
    term = phx.uq.LinearizedGaussianMeasurementLikelihood(
        lambda current, value: current["intercept"] + current["slope"] * value[0],
        inputs,
        targets,
        input_covariance=jnp.asarray([[input_variance]]),
        observation_covariance=jnp.asarray([[observation_variance]]),
    )
    effective_variance = observation_variance + parameters["slope"] ** 2 * input_variance
    residuals = targets - (parameters["intercept"] + parameters["slope"] * inputs[:, 0])
    expected = -0.5 * (
        residuals**2 / effective_variance + jnp.log(2.0 * jnp.pi * effective_variance)
    )

    assert jnp.allclose(term.per_case_log_prob(parameters), expected)
    assert jnp.allclose(term.log_prob(parameters), jnp.sum(expected))
    assert jnp.allclose(eqx.filter_jit(term.log_prob)(parameters), jnp.sum(expected))
    matrix = jnp.asarray([[1.2, -0.4], [0.3, 0.8]])
    inputs = jnp.asarray([[0.2, -0.1], [1.0, 0.5], [-0.4, 0.7]])
    targets = jnp.asarray([[0.4, -0.2], [0.9, 0.8], [-0.7, 0.3]])
    input_covariance = jnp.asarray([[0.20, 0.07], [0.07, 0.10]])
    observation_covariance = jnp.asarray([[0.08, -0.02], [-0.02, 0.12]])
    offset = jnp.asarray([0.1, -0.2])
    term = phx.uq.LinearizedGaussianMeasurementLikelihood(
        lambda parameters, value: parameters["matrix"] @ value + parameters["offset"],
        inputs,
        targets,
        input_covariance=input_covariance,
        observation_covariance=observation_covariance,
    )
    parameters = {"matrix": matrix, "offset": offset}
    effective_covariance = observation_covariance + matrix @ input_covariance @ matrix.T
    expected = jnp.asarray(
        [
            _normal_log_density(target - (matrix @ value + offset), effective_covariance)
            for value, target in zip(inputs, targets, strict=True)
        ]
    )

    assert jnp.allclose(term.per_case_log_prob(parameters), expected)
    assert jnp.allclose(
        jax.grad(term.log_prob)(parameters)["matrix"],
        jax.grad(
            lambda value: jnp.sum(
                jnp.asarray(
                    [
                        _normal_log_density(
                            target - (value @ measured + offset),
                            observation_covariance + value @ input_covariance @ value.T,
                        )
                        for measured, target in zip(inputs, targets, strict=True)
                    ]
                )
            )
        )(matrix),
        rtol=1e-10,
        atol=1e-10,
    )
    inputs = jnp.zeros((4, 1))
    targets = jnp.zeros((4,))
    term = phx.uq.LinearizedGaussianMeasurementLikelihood(
        lambda parameters, value: parameters["slope"] * value[0],
        inputs,
        targets,
        input_covariance=lambda parameters: jnp.asarray(
            [[parameters["input_scale"] ** 2]]
        ),
        observation_covariance=lambda parameters: jnp.asarray(
            [[parameters["observation_scale"] ** 2]]
        ),
    )
    narrow = {
        "slope": jnp.asarray(0.5),
        "input_scale": jnp.asarray(0.2),
        "observation_scale": jnp.asarray(0.1),
    }
    broad = {**narrow, "slope": jnp.asarray(2.0)}
    narrow_variance = 0.1**2 + 0.5**2 * 0.2**2
    broad_variance = 0.1**2 + 2.0**2 * 0.2**2

    narrow_value = eqx.filter_jit(term.log_prob)(narrow)
    broad_value, broad_gradient = jax.value_and_grad(term.log_prob)(broad)

    assert jnp.allclose(
        narrow_value,
        -0.5 * inputs.shape[0] * jnp.log(2.0 * jnp.pi * narrow_variance),
    )
    assert jnp.allclose(
        broad_value,
        -0.5 * inputs.shape[0] * jnp.log(2.0 * jnp.pi * broad_variance),
    )
    assert broad_value < narrow_value
    assert all(
        bool(jnp.all(jnp.isfinite(value)))
        for value in jax.tree_util.tree_leaves(broad_gradient)
    )


def test_measurement_likelihood_scenario_2() -> None:
    term = phx.uq.LinearizedGaussianMeasurementLikelihood(
        lambda slope, value: slope * value[0],
        jnp.ones((2, 1)),
        jnp.ones((2,)),
        input_covariance=jnp.asarray([[1.0]]),
        observation_covariance=lambda _: jnp.asarray([[-0.5]]),
    )

    assert jnp.isneginf(term.log_prob(jnp.asarray(1.0)))
    inputs = jnp.arange(5.0)[:, None]
    targets = 1.3 * inputs[:, 0] + jnp.asarray([0.1, -0.2, 0.0, 0.3, -0.1])
    input_covariances = jnp.asarray([[[value]] for value in jnp.linspace(0.01, 0.05, 5)])
    observation_covariances = jnp.asarray(
        [[[value]] for value in jnp.linspace(0.02, 0.06, 5)]
    )
    term = phx.uq.LinearizedGaussianMeasurementLikelihood(
        lambda slope, value: slope * value[0],
        inputs,
        targets,
        input_covariance=input_covariances,
        observation_covariance=observation_covariances,
        input_covariance_batching="per_case",
        observation_covariance_batching="per_case",
    )
    full = term.per_case_log_prob(jnp.asarray(1.3))
    indices = jnp.asarray([4, 1, 3])
    selected = term.log_prob_cases(
        jnp.asarray(1.3),
        inputs[indices],
        targets[indices],
        case_indices=indices,
    )

    assert jnp.allclose(selected, full[indices])
    invalid = term.log_prob_cases(
        jnp.asarray(1.3),
        inputs[:1],
        targets[:1],
        case_indices=jnp.asarray([8]),
    )
    assert jnp.isneginf(invalid[0])
    for input_covariance, observation_covariance, message in [
        (jnp.asarray([[-0.1]]), jnp.asarray([[0.1]]), "positive semidefinite"),
        (jnp.asarray([[0.1]]), jnp.asarray([[0.0]]), "positive definite"),
        (
            jnp.asarray([[0.1, 0.0], [0.0, 0.1]]),
            jnp.asarray([[0.1]]),
            "shape",
        ),
    ]:
        with pytest.raises(ValueError, match=message):
            phx.uq.LinearizedGaussianMeasurementLikelihood(
                lambda parameter, value: parameter * value[0],
                jnp.ones((3, 1)),
                jnp.ones((3,)),
                input_covariance=input_covariance,
                observation_covariance=observation_covariance,
            )
    term = phx.uq.LinearizedGaussianMeasurementLikelihood(
        lambda parameter, value: parameter * value[0],
        jnp.ones((2, 1)),
        jnp.ones((2,)),
        input_covariance=jnp.asarray([[0.0]]),
        observation_covariance=jnp.asarray([[0.0]]),
        stabilization=1.0e-3,
    )

    assert jnp.all(jnp.isfinite(term.per_case_log_prob(jnp.asarray(1.0))))
    with pytest.raises(ValueError, match="max_output_dimension"):
        phx.uq.LinearizedGaussianMeasurementLikelihood(
            lambda parameter, value: jnp.asarray([parameter, value[0]]),
            jnp.ones((2, 1)),
            jnp.ones((2, 2)),
            input_covariance=jnp.asarray([[0.1]]),
            observation_covariance=jnp.eye(2),
            max_output_dimension=1,
        )


def test_measurement_likelihood_reuses_the_native_minibatch_posterior_contract() -> None:
    inputs = jnp.linspace(0.2, 1.4, 7)[:, None]
    targets = 1.5 * inputs[:, 0] + 0.1
    term = phx.uq.LinearizedGaussianMeasurementLikelihood(
        lambda slope, value: slope * value[0],
        inputs,
        targets,
        input_covariance=jnp.asarray([[0.03]]),
        observation_covariance=jnp.asarray([[0.02]]),
    )
    data = {
        "inputs": inputs,
        "targets": targets,
        "case_indices": jnp.arange(inputs.shape[0]),
    }
    source = phx.uq.ArrayMinibatchSource(data, batch_size=3, seed=5)

    def factors(slope: Any, batch: Any) -> Any:
        return term.log_prob_cases(
            slope,
            batch.data["inputs"],
            batch.data["targets"],
            case_indices=batch.data["case_indices"],
        )

    space = phx.uq.ParameterSpace(
        jnp.asarray(1.4),
        priors=phx.uq.Normal(0.0, 2.0),
    )
    problem = phx.uq.MinibatchPosteriorProblem(
        space,
        factors,
        num_factors=inputs.shape[0],
        full_log_likelihood=term.log_prob,
    )
    diagnostics = phx.uq.diagnose_minibatch_posterior(problem, source)

    assert diagnostics.passed
    assert diagnostics.full_log_density_matches
    assert diagnostics.full_gradient_matches


def test_parameter_dependent_correlated_noise_gradient_includes_the_log_determinant() -> (
    None
):
    correlation = np.asarray([[1.0, 0.6], [0.6, 1.0]])
    inputs = np.asarray([[0.4], [1.0], [-0.7]])
    targets = np.asarray([[0.5, 0.1], [1.3, 0.2], [-0.6, -0.5]])
    input_variance = 0.02
    gain = 1.1
    direction = np.asarray([1.0, 0.5])
    term = phx.uq.LinearizedGaussianMeasurementLikelihood(
        lambda parameters, value: parameters["gain"] * jnp.asarray(direction) * value[0],
        jnp.asarray(inputs),
        jnp.asarray(targets),
        input_covariance=jnp.asarray([[input_variance]]),
        observation_covariance=lambda parameters: (
            parameters["noise"] ** 2 * jnp.asarray(correlation)
        ),
    )
    noise = 0.3
    # Host reference for d/d(noise) of -1/2 sum(r^T C^-1 r + log det C), with
    # C = noise^2 R + gain^2 input_variance d d^T and dC = 2 noise R.
    covariance = noise**2 * correlation + gain**2 * input_variance * np.outer(
        direction, direction
    )
    derivative = 2.0 * noise * correlation
    quadratic_part = 0.0
    log_determinant_part = 0.0
    for value, target in zip(inputs, targets, strict=True):
        weighted = np.linalg.solve(covariance, target - gain * direction * value[0])
        quadratic_part += 0.5 * weighted @ derivative @ weighted
        log_determinant_part -= 0.5 * np.trace(np.linalg.solve(covariance, derivative))

    def at_noise(value: float) -> Any:
        return term.log_prob({"gain": jnp.asarray(gain), "noise": jnp.asarray(value)})

    step = 1e-6
    central = (float(at_noise(noise + step)) - float(at_noise(noise - step))) / (
        2.0 * step
    )
    gradient = jax.grad(term.log_prob)(
        {"gain": jnp.asarray(gain), "noise": jnp.asarray(noise)}
    )["noise"]

    np.testing.assert_allclose(
        gradient, quadratic_part + log_determinant_part, rtol=1e-10
    )
    np.testing.assert_allclose(gradient, central, rtol=1e-6)
    assert abs(float(gradient) - quadratic_part) > 1.0
