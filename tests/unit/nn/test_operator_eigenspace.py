#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

import phydrax as phx


def _problem():
    properties = phx.linalg.OperatorProperties(
        self_adjoint=True,
        evidence={"self_adjoint": "construction"},
    )
    operator = phx.linalg.DenseLinearOperator(
        jnp.diag(jnp.asarray([1.0, 3.0, 7.0])),
        properties=properties,
    )
    return phx.linalg.eigen.Eigenproblem(operator)


def _samples(values, *, mask=None):
    return phx.nn.operator.FunctionSamples(
        values=jnp.asarray(values),
        coordinates=jnp.arange(3.0)[:, None],
        mask=mask,
    )


def test_operator_trial_subspace_lowers_modes_and_honors_mask():
    problem = _problem()
    samples = _samples(
        [[1.0, 0.0], [0.0, 1.0], [5.0, -2.0]],
        mask=jnp.asarray([True, True, False]),
    )

    trial = phx.nn.operator.operator_trial_subspace(
        samples,
        problem.operator.source,
    )
    result = phx.nn.operator.rayleigh_ritz_from_samples(problem, samples)

    assert trial.masked
    assert jnp.array_equal(trial.basis[-1], jnp.zeros((2,)))
    assert jnp.allclose(result.eigenvalues, jnp.asarray([1.0, 3.0]), atol=2e-12)
    assert jnp.allclose(result.relative_residuals, 0.0, atol=2e-12)
    assert bool(result.valid)


def test_warm_started_eigensolve_refines_predicted_trial_space():
    problem = _problem()
    samples = _samples([[1.0, 0.0], [0.0, 1.0], [0.2, 0.0]])
    trial = phx.nn.operator.rayleigh_ritz_from_samples(problem, samples)
    policy = phx.linalg.eigen.EigenSolvePolicy(
        phx.linalg.eigen.LOBPCG(),
        count=2,
        max_steps=12,
    )

    refined = phx.nn.operator.warm_started_eigensolve_from_samples(
        problem,
        samples,
        policy=policy,
    )

    assert bool(refined.successful)
    assert jnp.max(refined.trial.relative_residuals) > 1e-3
    assert jnp.allclose(refined.solve.eigenvalues, jnp.asarray([1.0, 3.0]), atol=2e-9)
    assert jnp.max(refined.solve.diagnostics.relative_residuals) < jnp.max(
        trial.relative_residuals
    )


def test_operator_trial_subspace_rejects_unsliced_case_axes():
    problem = _problem()
    samples = phx.nn.operator.FunctionSamples(
        values=jnp.ones((2, 3, 2)),
        coordinates=jnp.arange(3.0)[:, None],
    )

    with pytest.raises(ValueError, match="exactly one case|space size"):
        phx.nn.operator.operator_trial_subspace(samples, problem.operator.source)
