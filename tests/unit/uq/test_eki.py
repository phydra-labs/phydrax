#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
import phydrax.axes as cx


def _linear_problem():
    design = jnp.asarray(
        [
            [1.0, 0.3],
            [-0.2, 1.2],
            [0.7, -0.4],
        ]
    )
    observations = jnp.asarray([0.5, -0.8, 0.9])
    noise_scale = 0.25
    space = phx.uq.ParameterSpace(
        jnp.zeros(2),
        priors=phx.uq.Normal(0.0, 1.0),
    )
    problem = phx.uq.PosteriorProblem(
        space,
        lambda value: (
            -0.5 * jnp.sum(((design @ value - observations) / noise_scale) ** 2)
        ),
        gauss_newton_residual=lambda value: (design @ value - observations) / noise_scale,
        predict=lambda value, query: cx.AxisArray(value @ query, dims=("point",)),
    )
    precision = jnp.eye(2) + design.T @ design / noise_scale**2
    covariance = jnp.linalg.inv(precision)
    mean = covariance @ (design.T @ observations / noise_scale**2)
    return problem, mean, covariance


def test_tempered_eki_matches_linear_gaussian_mean_covariance_and_replays():
    problem, exact_mean, exact_covariance = _linear_problem()
    settings: dict[str, Any] = {
        "key": jr.key(950),
        "ensemble_size": 512,
        "target_ess": 0.8,
        "max_steps": 20,
    }
    result = phx.uq.fit_eki(problem, **settings)
    replay = phx.uq.fit_eki(problem, **settings)

    assert result.converged
    assert result.termination_reason == "unit_temperature"
    assert result.temperatures[0] == 0.0
    assert result.temperatures[-1] == 1.0
    assert jnp.all(jnp.diff(result.temperatures) > 0.0)
    assert jnp.sum(result.diagnostics.temperature_increments) == pytest.approx(1.0)
    assert result.diagnostics.forward_solve_count == (result.num_steps + 1) * 512
    assert jnp.all(result.diagnostics.effective_ranks <= 2)
    assert not result.diagnostics.collapsed
    assert jnp.array_equal(result.ensemble, replay.ensemble)
    assert jnp.array_equal(result.residuals, replay.residuals)

    empirical_mean = jnp.mean(result.ensemble, axis=0)
    empirical_covariance = jnp.cov(result.ensemble, rowvar=False)
    assert jnp.allclose(empirical_mean, exact_mean, atol=0.04)
    assert jnp.allclose(empirical_covariance, exact_covariance, atol=0.012)

    query = jnp.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]]).T
    prediction = result.predict(query)
    assert isinstance(prediction, phx.uq.PredictiveField)
    assert prediction.samples.dims == ("__phydra_uq_ensemble", "point")
    assert prediction.samples.shape == (512, 3)


def test_eki_respects_bijectors_custom_initial_ensembles_and_no_reverse_mode():
    @jax.custom_vjp
    def forward(value):
        return jnp.asarray([value, 0.5 * value])

    def forward_fwd(value):
        return forward(value), None

    def forward_bwd(residual, cotangent):
        raise AssertionError("EKI requested a reverse-mode derivative")

    forward.defvjp(forward_fwd, forward_bwd)
    space = phx.uq.ParameterSpace(
        jnp.log(jnp.asarray(1.0)),
        priors=phx.uq.LogNormal(0.0, 0.5),
        bijectors=phx.uq.ExpBijector(),
    )
    observations = jnp.asarray([2.0, 1.0])
    problem = phx.uq.PosteriorProblem(
        space,
        lambda physical: -0.5 * jnp.sum(((forward(physical) - observations) / 0.2) ** 2),
        gauss_newton_residual=lambda physical: (forward(physical) - observations) / 0.2,
    )
    initial = jnp.linspace(-0.8, 0.8, 128)
    result = phx.uq.fit_eki(
        problem,
        key=jr.key(951),
        ensemble_size=128,
        initial_ensemble=initial,
        target_ess=0.75,
    )

    assert result.converged
    assert jnp.array_equal(result.initial_unconstrained_ensemble, initial)
    assert jnp.all(result.initial_ensemble > 0.0)
    assert jnp.all(result.ensemble > 0.0)
    assert jnp.mean(result.ensemble) == pytest.approx(2.0, abs=0.12)


def test_eki_reports_collapse_and_rejects_invalid_residuals_and_configuration():
    problem, _, _ = _linear_problem()
    collapsed = phx.uq.fit_eki(
        problem,
        key=jr.key(952),
        ensemble_size=8,
        initial_ensemble=jnp.zeros((8, 2)),
    )
    assert collapsed.diagnostics.collapsed
    assert collapsed.diagnostics.collapse_step == 1
    assert collapsed.diagnostics.effective_ranks[-1] == 0

    nonfinite = phx.uq.PosteriorProblem(
        phx.uq.ParameterSpace(
            jnp.asarray(0.0),
            priors=phx.uq.Normal(0.0, 1.0),
        ),
        lambda value: -0.5 * value**2,
        gauss_newton_residual=lambda value: jnp.asarray([jnp.nan]),
    )
    with pytest.raises(FloatingPointError, match="residuals must be finite"):
        phx.uq.fit_eki(nonfinite, key=jr.key(953), ensemble_size=8)

    with pytest.raises(ValueError, match="mutually exclusive"):
        phx.uq.fit_eki(
            problem,
            key=jr.key(954),
            ensemble_size=8,
            initial_ensemble=jnp.zeros((8, 2)),
            prior_position_sampler=lambda key, count: jr.normal(key, (count, 2)),
        )
    with pytest.raises(phx.uq.EnsembleKalmanConvergenceError) as error:
        phx.uq.fit_eki(
            problem,
            key=jr.key(955),
            ensemble_size=64,
            target_ess=0.95,
            max_steps=1,
            raise_on_failure=True,
        )
    assert error.value.result.termination_reason == "max_steps"


class _ObservedCoefficients(phx.AbstractArrayModel):
    """Linear coefficients from a provider that declares no derivative route."""

    coefficients: jax.Array
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self, coefficients):
        self.coefficients = jnp.asarray(coefficients)
        self.in_size = 2
        self.out_size = 3

    def __call__(self, design, /, *, key=None):
        return design @ self.coefficients

    def model_execution_contract(self):
        return phx.ModelExecutionContract(
            derivative=phx.DerivativeContract(route=phx.DerivativeRoute.STOPPED),
            execution=phx.ExecutionCapabilities("native-jax"),
            randomness=phx.RandomnessContract("deterministic"),
        )


def _observation_objective(accepted=True, accepted_results="reject-attempt"):
    design = jnp.asarray([[1.0, 0.3], [-0.2, 1.2], [0.7, -0.4]])
    observations = jnp.asarray([0.5, -0.8, 0.9])

    def measure(owner, case):
        del case
        (design_, observations_), binding = owner
        return phx.solver.SolverCaseResult(
            residual=(binding.model(design_) - observations_) / 0.25,
            accepted=accepted,
        )

    return phx.solver.SolverObjective(
        (design, observations),
        lambda solve, binding: (solve, binding),
        measure,
        objective_id="linear-observations",
        accepted_results=accepted_results,
    )


def test_eki_consumes_a_solver_objective_likelihood_without_derivatives():
    _, exact_mean, exact_covariance = _linear_problem()
    tree = phx.bind_component(
        _ObservedCoefficients(jnp.zeros(2)), phx.ComponentAuthority.MODEL
    )
    space = phx.uq.ParameterSpace(jnp.zeros(2), priors=phx.uq.Normal(0.0, 1.0))
    problem = phx.uq.posterior_problem_from_solver_objective(
        _observation_objective(), tree, space
    )

    result = phx.uq.fit_eki(problem, key=jr.key(950), ensemble_size=512)
    assert result.converged
    assert jnp.allclose(jnp.mean(result.ensemble, axis=0), exact_mean, atol=0.04)
    assert jnp.allclose(
        jnp.cov(result.ensemble, rowvar=False), exact_covariance, atol=0.012
    )
    # The provider declares no derivative: gradient consumers are refused, never
    # handed a derivative, while EKI above ran on residuals alone.
    with pytest.raises(Exception, match="not differentiable"):
        jax.block_until_ready(jax.grad(problem.log_density)(jnp.zeros(2)))

    failed = phx.uq.posterior_problem_from_solver_objective(
        _observation_objective(accepted=False), tree, space
    )
    assert problem.log_likelihood(jnp.zeros(2)) > -jnp.inf
    assert failed.log_likelihood(jnp.zeros(2)) == -jnp.inf
    with pytest.raises(FloatingPointError, match="residuals must be finite"):
        phx.uq.fit_eki(failed, key=jr.key(951), ensemble_size=8)
    with pytest.raises(ValueError, match="cannot drop failed cases"):
        phx.uq.posterior_problem_from_solver_objective(
            _observation_objective(accepted_results="reduce-support"), tree, space
        )
