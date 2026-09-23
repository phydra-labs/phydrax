import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
import phydrax.axes as cx
from phydrax.domain import BatchEvaluator


class _KeyedBatchValue(BatchEvaluator):
    def __call_batch__(self, batch, /, *, key=jr.key(0), **kwargs):
        del kwargs
        reference = batch["x"]
        return cx.AxisArray(
            jnp.broadcast_to(jr.uniform(key), reference.data.shape),
            dims=reference.dims,
        )


def test_prepared_linear_reduction_is_linear_and_refresh_stable():
    domain = phx.domain.ScalarInterval(0.0, 1.0, label="x")
    target = phx.integration.over(domain.component())
    realization = phx.integration.materialize(
        target,
        phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(8)),
    )
    prepared = phx.integration.prepare_linear_reduction(realization)

    @domain.Function("x")
    def square(x):
        return x**2

    @domain.Function("x")
    def linear(x):
        return x

    left = prepared.apply(2.0 * square + 3.0 * linear)
    right = 2.0 * prepared.apply(square) + 3.0 * prepared.apply(linear)
    assert jnp.allclose(left.data, right.data, atol=1e-10)
    assert jnp.allclose(left.data, 13.0 / 6.0, atol=1e-10)

    refreshed = phx.integration.refresh_linear_reduction(prepared, realization)
    assert refreshed.numeric_version == prepared.numeric_version
    assert refreshed.realization_id == prepared.realization_id


def test_prepared_linear_reduction_requires_declared_domain_functions():
    domain = phx.domain.ScalarInterval(0.0, 1.0, label="x")
    realization = phx.integration.materialize(
        phx.integration.over(domain.component()),
        phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(8)),
    )
    prepared = phx.integration.prepare_linear_reduction(realization)

    def square(x=jnp.asarray(2.0), *, key=None):
        del key
        return x**2

    with pytest.raises(TypeError, match=r"domain\.Function\(\*labels\)\(callable\)"):
        prepared.apply(square)
    declared = prepared.apply(domain.Function("x")(square))
    assert jnp.allclose(declared.data, 1.0 / 3.0, atol=1e-10)


def test_prepared_replicated_qmc_averages_coefficient_actions():
    domain = phx.domain.ScalarInterval(0.0, 1.0, label="x")
    target = phx.integration.over(domain.component())
    realization = phx.integration.materialize(
        target,
        phx.integration.QuasiMonteCarloPlan(16, num_replicates=4),
        key=jr.key(10),
    )
    prepared = phx.integration.prepare_linear_reduction(realization)

    assert len(prepared.batches) == 4
    assert jnp.allclose(prepared.apply(1.0).data, 1.0)


def test_prepared_additive_reduction_splits_evaluation_keys_per_term():
    x = phx.domain.ScalarInterval(0.0, 1.0, label="x")
    t = phx.domain.ScalarInterval(0.0, 1.0, label="t")
    domain = phx.domain.ProductDomain(x, t)
    component = phx.domain.ComponentSum(
        (
            domain.component({"t": phx.domain.FixedStart()}),
            domain.component({"t": phx.domain.FixedEnd()}),
        ),
        assume_disjoint=True,
    )
    realization = phx.integration.materialize(
        phx.integration.over(component),
        phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(4)),
    )
    prepared = phx.integration.prepare_linear_reduction(realization)
    key = jr.key(11)
    term_keys = jr.split(key, 2)
    expected = jr.uniform(term_keys[0]) + jr.uniform(term_keys[1])

    value = prepared.apply(domain.Function("x")(_KeyedBatchValue()), key=key)

    assert jnp.allclose(value.data, expected)
