import jax.numpy as jnp

import phydrax as phx


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
