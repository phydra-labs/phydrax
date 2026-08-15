#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


la = phx.linalg


def _positive_properties():
    return la.OperatorProperties(
        self_adjoint=True,
        positive_definite=True,
        evidence={
            "self_adjoint": "construction",
            "positive_definite": "construction",
        },
    )


def test_adaptive_trace_stops_at_first_eligible_batch_for_zero_variance_samples():
    diagonal = jnp.asarray([1.0, 2.0, 3.0, 4.0])
    operator = la.DenseLinearOperator(
        jnp.diag(diagonal),
        properties=_positive_properties(),
    )
    policy = la.AdaptiveStochasticPolicy(
        min_probes=4,
        max_probes=12,
        batch_size=2,
        max_dimension=4,
        relative_tolerance=1e-10,
        absolute_tolerance=1e-12,
    )
    result = la.adaptive_stochastic_trace(
        operator,
        key=jax.random.key(0),
        policy=policy,
    )

    assert result.num_probes == 4
    assert result.converged
    assert result.stopped_early
    assert result.finite
    assert jnp.allclose(result.estimate, jnp.sum(diagonal), atol=1e-12)
    assert result.standard_error == 0
    assert result.numerical_error_estimate == 0
    assert result.total_error_estimate == 0
    assert result.matvec_count == 16
    assert jnp.all(result.probe_statuses[:4] == int(la.StochasticProbeStatus.SUCCESS))
    assert jnp.all(
        result.probe_statuses[4:] == int(la.StochasticProbeStatus.NOT_EVALUATED)
    )
    assert jnp.all(jnp.isnan(result.samples[4:]))
    assert result.cost.first_stopping_matvec_budget == 16
    assert result.cost.maximum_matvec_budget == 48


def test_adaptive_trace_hits_probe_budget_when_statistical_error_is_too_large():
    matrix = jnp.asarray([[3.0, 0.8, 0.2], [0.8, 2.0, 0.4], [0.2, 0.4, 1.5]])
    operator = la.DenseLinearOperator(
        matrix,
        properties=_positive_properties(),
    )
    policy = la.AdaptiveStochasticPolicy(
        min_probes=2,
        max_probes=8,
        batch_size=2,
        max_dimension=3,
        relative_tolerance=0.0,
        absolute_tolerance=1e-15,
    )
    result = la.adaptive_stochastic_trace(
        operator,
        key=jax.random.key(7),
        policy=policy,
    )

    assert result.num_probes == 8
    assert not result.converged
    assert not result.stopped_early
    assert result.finite
    assert result.standard_error > 0
    assert result.total_error_estimate > result.tolerance
    assert result.matvec_count == 24
    assert jnp.all(jnp.isfinite(result.samples))


def test_adaptive_slq_never_claims_success_with_unresolved_krylov_truncation():
    matrix = jnp.asarray([[3.0, 0.8, 0.2], [0.8, 2.0, 0.4], [0.2, 0.4, 1.5]])
    operator = la.DenseLinearOperator(
        matrix,
        properties=_positive_properties(),
    )
    policy = la.AdaptiveStochasticPolicy(
        min_probes=2,
        max_probes=6,
        batch_size=2,
        max_dimension=1,
        relative_tolerance=1.0,
        absolute_tolerance=100.0,
    )
    result = la.adaptive_stochastic_log_determinant(
        operator,
        key=jax.random.key(2),
        policy=policy,
    )

    assert result.finite
    assert not result.converged
    assert result.num_probes == 6
    assert jnp.isinf(result.numerical_error_estimate)
    assert jnp.isinf(result.total_error_estimate)
    assert jnp.all(result.probe_statuses == int(la.StochasticProbeStatus.TRUNCATED))
    assert result.quantity == "log-determinant"


def test_adaptive_log_determinant_is_reproducible_and_jittable():
    diagonal = jnp.asarray([1.0, 2.0, 3.0, 4.0])
    operator = la.DenseLinearOperator(
        jnp.diag(diagonal),
        properties=_positive_properties(),
    )
    policy = la.AdaptiveStochasticPolicy(
        min_probes=4,
        max_probes=8,
        batch_size=2,
        max_dimension=4,
        relative_tolerance=1e-10,
        absolute_tolerance=1e-12,
    )
    key = jax.random.key(11)
    first = la.adaptive_stochastic_log_determinant(
        operator,
        key=key,
        policy=policy,
    )
    second = la.adaptive_stochastic_log_determinant(
        operator,
        key=key,
        policy=policy,
    )
    compiled = jax.jit(
        lambda value: la.adaptive_stochastic_log_determinant(
            operator,
            key=value,
            policy=policy,
        )
    )(key)

    expected = jnp.sum(jnp.log(diagonal))
    assert jnp.array_equal(first.samples, second.samples, equal_nan=True)
    assert first.num_probes == second.num_probes
    assert jnp.allclose(first.estimate, expected, atol=1e-12)
    assert jnp.allclose(compiled.estimate, expected, atol=1e-12)
    assert compiled.num_probes == first.num_probes
    assert compiled.converged


def test_adaptive_slq_supports_complex_hermitian_operators():
    matrix = jnp.asarray([[2.0 + 0.0j, 1.0j], [-1.0j, 3.0 + 0.0j]])
    operator = la.DenseLinearOperator(
        matrix,
        properties=_positive_properties(),
    )
    policy = la.AdaptiveStochasticPolicy(
        min_probes=4,
        max_probes=8,
        batch_size=2,
        max_dimension=2,
        relative_tolerance=1.0,
        absolute_tolerance=10.0,
    )
    result = la.adaptive_stochastic_trace(
        operator,
        key=jax.random.key(4),
        policy=policy,
    )

    assert result.finite
    assert result.converged
    assert result.num_probes == 4
    assert jnp.isrealobj(result.estimate)
    assert jnp.isclose(result.estimate, jnp.trace(matrix).real, atol=2.0)


def test_adaptive_stochastic_policy_rejects_invalid_fixed_capacity_batches():
    with pytest.raises(ValueError, match="divide"):
        la.AdaptiveStochasticPolicy(
            min_probes=3,
            max_probes=8,
            batch_size=2,
        )
    with pytest.raises(ValueError, match="At least one"):
        la.AdaptiveStochasticPolicy(
            relative_tolerance=0.0,
            absolute_tolerance=0.0,
        )
    with pytest.raises(ValueError, match="strictly between"):
        la.AdaptiveStochasticPolicy(confidence_level=1.0)
