#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

import phydrax as phx


def test_find_map_recovers_correlated_gaussian_mode():
    precision = jnp.asarray([[5.0, 1.0], [1.0, 3.0]])
    target = jnp.asarray([0.4, -0.7])
    initial = jnp.asarray([2.0, 1.0])
    space = phx.uq.ParameterSpace(initial, priors=phx.uq.Normal(0.0, 1.0))
    problem = phx.uq.PosteriorProblem(
        space,
        lambda value: -0.5 * (value - target) @ precision @ (value - target),
    )
    expected = jnp.linalg.solve(precision + jnp.eye(2), precision @ target)

    result = phx.uq.find_map(problem, gradient_tolerance=1e-8)

    assert isinstance(result, phx.uq.MAPResult)
    assert result.converged
    assert result.termination_reason == "gradient_tolerance"
    assert result.gradient_norm <= 1e-8
    assert jnp.allclose(result.position, expected, atol=1e-7)
    assert jnp.array_equal(result.position, result.parameters)
    assert result.objective_history.shape == (result.num_steps + 1,)
    assert result.objective_evaluations >= result.objective_history.size
    assert jnp.all(jnp.diff(result.objective_history) <= 1e-10)
    assert jnp.allclose(
        result.objective_history,
        jnp.asarray(
            [
                17.79287707,
                6.94344864,
                2.10654950,
                2.07847351,
                2.07287707,
                2.07287707,
                2.07287707,
            ]
        ),
        atol=1e-7,
    )
    assert result.num_steps == 6
    assert result.objective_evaluations == 7
    assert result.duration_seconds >= result.compilation_seconds
    assert result.duration_seconds >= result.execution_seconds
    assert result.compilation_seconds >= 0.0
    assert result.execution_seconds >= 0.0
    assert result.mean_step_seconds == pytest.approx(
        result.optimization_seconds / result.num_steps
    )


def test_find_map_optimizes_unconstrained_coordinates_with_jacobian():
    log_location = jnp.log(jnp.asarray(2.5))
    space = phx.uq.ParameterSpace(
        jnp.asarray(-1.0),
        priors=phx.uq.LogNormal(log_location, 0.4),
        bijectors=phx.uq.ExpBijector(),
    )
    problem = phx.uq.PosteriorProblem(space, lambda _: jnp.zeros(()))

    result = phx.uq.find_map(problem, gradient_tolerance=1e-9)

    assert jnp.allclose(result.position, log_location, atol=1e-8)
    assert jnp.allclose(result.parameters, 2.5, atol=1e-8)


def test_find_map_returns_or_raises_with_complete_failure_evidence():
    space = phx.uq.ParameterSpace(
        jnp.asarray([-1.2, 1.0]),
        priors=phx.uq.Normal(0.0, 100.0),
    )
    problem = phx.uq.PosteriorProblem(
        space,
        lambda value: -(100.0 * (value[1] - value[0] ** 2) ** 2 + (1.0 - value[0]) ** 2),
    )

    result = phx.uq.find_map(
        problem,
        max_steps=1,
        gradient_tolerance=1e-14,
        objective_tolerance=None,
        raise_on_failure=False,
    )

    assert not result.converged
    assert result.termination_reason == "max_steps"
    assert result.num_steps == 1
    with pytest.raises(phx.uq.MAPConvergenceError) as error:
        phx.uq.find_map(
            problem,
            max_steps=1,
            gradient_tolerance=1e-14,
            objective_tolerance=None,
        )
    assert error.value.result.termination_reason == "max_steps"


def test_structured_gp_problems_reuse_compiled_map_executable():
    points = jnp.linspace(0.0, 1.0, 16)

    def physical_mean(parameters):
        return parameters["coefficient"] * points

    def state(parameters):
        return phx.uq.GaussianProcessLikelihoodState(
            kernel=phx.kernels.AmplitudeKernel(
                phx.kernels.Matern32Kernel(
                    length_scale=parameters["length_scale"],
                ),
                parameters["amplitude"],
            ),
            noise_scale=parameters["noise_scale"],
        )

    def problem(observations):
        discrepancy = phx.uq.ExactGaussianProcessDiscrepancy(
            points,
            observations,
        )
        term = phx.uq.GaussianProcessMarginalLikelihood(
            discrepancy,
            physical_mean,
            state=state,
        )
        space = phx.uq.ParameterSpace(
            {
                "coefficient": jnp.asarray(0.7),
                "amplitude": jnp.log(jnp.asarray(0.2)),
                "length_scale": jnp.log(jnp.asarray(0.2)),
                "noise_scale": jnp.log(jnp.asarray(0.02)),
            },
            priors={
                "coefficient": phx.uq.Normal(0.0, 3.0),
                "amplitude": phx.uq.LogNormal(jnp.log(0.2), 0.5),
                "length_scale": phx.uq.LogNormal(jnp.log(0.2), 0.5),
                "noise_scale": phx.uq.LogNormal(jnp.log(0.02), 0.3),
            },
            bijectors={
                "coefficient": phx.uq.IdentityBijector(),
                "amplitude": phx.uq.ExpBijector(),
                "length_scale": phx.uq.ExpBijector(),
                "noise_scale": phx.uq.ExpBijector(),
            },
        )
        return phx.uq.PosteriorProblem.from_terms(space, [term])

    signal = 0.8 * points + 0.2 * jnp.sin(jnp.pi * points)
    cold = phx.uq.find_map(problem(signal), gradient_tolerance=1e-5)
    warm = phx.uq.find_map(problem(signal + 0.002), gradient_tolerance=1e-5)

    assert cold.converged
    assert warm.converged
    assert cold.compilation_seconds > 0.0
    assert warm.compilation_seconds < cold.compilation_seconds
    assert warm.initial_compilation_seconds < cold.initial_compilation_seconds
    assert warm.step_compilation_seconds < cold.step_compilation_seconds
    assert cold.execution_seconds > 0.0
    assert warm.execution_seconds > 0.0


def test_find_map_rejects_nonfinite_initial_density():
    space = phx.uq.ParameterSpace(jnp.asarray(0.0), priors=phx.uq.Normal(0.0, 1.0))
    problem = phx.uq.PosteriorProblem(space, lambda _: jnp.asarray(jnp.nan))

    with pytest.raises(FloatingPointError, match="finite scalar"):
        phx.uq.find_map(problem)


def test_find_map_rejects_nonfinite_initial_gradient():
    space = phx.uq.ParameterSpace(jnp.asarray(0.0), priors=phx.uq.Normal(0.0, 1.0))
    problem = phx.uq.PosteriorProblem(space, lambda value: -jnp.sqrt(value))

    with pytest.raises(FloatingPointError, match="gradient must be finite"):
        phx.uq.find_map(problem)
