#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax._data_plane import IndexEpochPlan


def _problem(data, *, num_factors=None, full_shift=0.0):
    values = jnp.asarray(data)
    count = int(values.shape[0]) if num_factors is None else int(num_factors)
    space = phx.uq.ParameterSpace(
        jnp.log(jnp.asarray(1.5)),
        priors=phx.uq.LogNormal(0.0, 1.0),
        bijectors=phx.uq.ExpBijector(),
    )

    def factors(parameter, batch):
        return -0.5 * (batch.data - parameter) ** 2

    return phx.uq.MinibatchPosteriorProblem(
        space,
        factors,
        num_factors=count,
        full_log_likelihood=lambda parameter: (
            jnp.sum(-0.5 * (values - parameter) ** 2) + full_shift
        ),
        predict=lambda parameter, scale: scale * parameter,
        observation_variance=lambda parameter: jnp.ones_like(parameter) * 0.25,
        sample_observation=lambda key, parameter: (
            parameter + 0.5 * jax.random.normal(key)
        ),
    )


def _active_values(source, epoch):
    return jnp.concatenate(
        [batch.data[batch.factor_mask] for batch in source.epoch(epoch)]
    )


def test_array_minibatch_source_is_deterministic_complete_and_padded():
    data = jnp.arange(7)
    source = phx.uq.ArrayMinibatchSource(data, batch_size=3, seed=11)
    duplicate = phx.uq.ArrayMinibatchSource(data, batch_size=3, seed=11)

    first = tuple(source.epoch(2))
    second = tuple(duplicate.epoch(2))
    expected = tuple(IndexEpochPlan(7, 3, True, 11, 2, False).iter_batches())

    assert source.num_factors == 7
    assert source.batch_capacity == 3
    assert source.batches_per_epoch == 3
    assert [int(batch.factor_count) for batch in first] == [3, 3, 1]
    assert all(batch.capacity == 3 for batch in first)
    assert jnp.array_equal(jnp.sort(_active_values(source, 2)), data)
    assert all(
        jnp.array_equal(jnp.asarray(left.data), right.data)
        and jnp.array_equal(left.factor_mask, right.factor_mask)
        for left, right in zip(first, second, strict=True)
    )
    assert all(
        jnp.array_equal(
            batch.data[batch.factor_mask],
            data[jnp.asarray(indices)],
        )
        for batch, (_, indices) in zip(first, expected, strict=True)
    )
    assert source.fingerprint == duplicate.fingerprint
    assert not jnp.array_equal(_active_values(source, 2), _active_values(source, 3))
    assert first[-1].data[-1] == first[-1].data[0]
    assert not bool(first[-1].factor_mask[-1])


def test_array_minibatch_source_fingerprint_covers_data_and_configuration():
    baseline = phx.uq.ArrayMinibatchSource(jnp.arange(6), batch_size=4, seed=2)
    changed_data = phx.uq.ArrayMinibatchSource(
        jnp.arange(6).at[0].set(9), batch_size=4, seed=2
    )
    changed_batch = phx.uq.ArrayMinibatchSource(jnp.arange(6), batch_size=3, seed=2)
    changed_seed = phx.uq.ArrayMinibatchSource(jnp.arange(6), batch_size=4, seed=3)

    assert (
        len(
            {
                baseline.fingerprint,
                changed_data.fingerprint,
                changed_batch.fingerprint,
                changed_seed.fingerprint,
            }
        )
        == 4
    )
    assert baseline.configuration()["data"]["sha256"]
    assert baseline.configuration()["ordering"] == "feistel32-v1"
    assert (
        baseline.fingerprint
        == "5d8818b7363587c82a17a427ce6fe34590b45838611f64fdae8b2091f3e015ab"
    )


@pytest.mark.parametrize(
    ("data", "batch_size", "message"),
    [
        (jnp.asarray(1.0), 2, "positive leading axis"),
        ({"x": jnp.ones((3,)), "y": jnp.ones((2,))}, 2, "share"),
        (jnp.ones((3,)), 0, "batch_size"),
    ],
)
def test_array_minibatch_source_rejects_invalid_contracts(data, batch_size, message):
    with pytest.raises(ValueError, match=message):
        phx.uq.ArrayMinibatchSource(data, batch_size=batch_size)


def test_likelihood_batch_requires_a_nonempty_boolean_factor_mask():
    with pytest.raises(ValueError, match="one-dimensional"):
        phx.uq.LikelihoodBatch(jnp.ones((2,)), jnp.ones((1, 2), dtype=bool))
    with pytest.raises(TypeError, match="boolean"):
        phx.uq.LikelihoodBatch(jnp.ones((2,)), jnp.ones((2,)))
    with pytest.raises(ValueError, match="active factor"):
        phx.uq.LikelihoodBatch(jnp.ones((2,)), jnp.zeros((2,), dtype=bool))


def test_minibatch_posterior_scales_only_active_likelihood_factors():
    data = jnp.asarray([0.5, 1.0, 2.0, 4.0, 8.0])
    problem = _problem(data)
    batch = phx.uq.LikelihoodBatch(
        jnp.asarray([0.5, 2.0, 1.0e20]),
        jnp.asarray([True, True, False]),
    )
    position = problem.initial_position
    physical = problem.parameter_space.constrain(position)
    factors = -0.5 * (batch.data[:2] - physical) ** 2
    expected_likelihood = data.shape[0] / 2 * jnp.sum(factors)
    expected = (
        expected_likelihood
        + problem.parameter_space.log_prior(physical)
        + problem.parameter_space.log_abs_det_jacobian(position)
    )

    assert jnp.allclose(
        problem.log_likelihood_estimate(physical, batch), expected_likelihood
    )
    assert jnp.allclose(problem.log_density_estimate(position, batch), expected)
    assert problem.predict(position, 3.0) == 3.0 * physical
    assert problem.conditional_observation_variance(position) == 0.25
    assert jnp.isfinite(problem.sample_observation(jax.random.key(1), position))


def test_minibatch_posterior_rejects_wrong_factor_shapes():
    space = phx.uq.ParameterSpace(jnp.asarray(0.0), priors=phx.uq.Normal(0.0, 1.0))
    batch = phx.uq.LikelihoodBatch(jnp.ones((3,)), jnp.ones((3,), dtype=bool))
    scalar_problem = phx.uq.MinibatchPosteriorProblem(
        space, lambda parameter, current: jnp.asarray(0.0), num_factors=3
    )
    short_problem = phx.uq.MinibatchPosteriorProblem(
        space, lambda parameter, current: jnp.ones((2,)), num_factors=3
    )

    with pytest.raises(ValueError, match="one scalar"):
        scalar_problem.log_density_estimate(space.initial, batch)
    with pytest.raises(ValueError, match="one scalar"):
        short_problem.log_density_estimate(space.initial, batch)


def test_minibatch_diagnostics_reconstruct_full_density_and_gradient():
    data = jnp.linspace(0.2, 1.4, 7)
    source = phx.uq.ArrayMinibatchSource(data, batch_size=3, seed=4)
    diagnostics = phx.uq.diagnose_minibatch_posterior(_problem(data), source)

    assert diagnostics.passed
    assert diagnostics.epoch_active_factor_count == data.shape[0]
    assert diagnostics.repeated_evaluation_matches
    assert diagnostics.jit_evaluation_matches
    assert diagnostics.full_log_density_matches
    assert diagnostics.full_gradient_matches
    assert diagnostics.capabilities.prediction
    assert diagnostics.capabilities.control_variates


def test_minibatch_diagnostics_report_population_and_full_density_mismatches():
    data = jnp.linspace(-1.0, 1.0, 5)
    source = phx.uq.ArrayMinibatchSource(data, batch_size=2, seed=5)
    diagnostics = phx.uq.diagnose_minibatch_posterior(
        _problem(data, num_factors=6, full_shift=1.0), source
    )

    assert not diagnostics.passed
    assert "source_population_mismatch" in diagnostics.failures
    assert "epoch_factor_count_mismatch" in diagnostics.failures
    assert "full_log_density_mismatch" in diagnostics.failures
