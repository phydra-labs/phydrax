import jax.numpy as jnp

from phydrax.uq._forecast_comparison import compare_forecasts
from phydrax.uq._multiple_testing import benjamini_hochberg, holm_adjust


def test_holm_and_bh_preserve_family_order_and_monotonic_corrections():
    p_values = jnp.asarray([0.01, 0.04, 0.03])

    holm = holm_adjust(p_values, alpha=0.05)
    bh = benjamini_hochberg(p_values, alpha=0.05)

    assert jnp.allclose(holm.adjusted_p_values, jnp.asarray([0.03, 0.06, 0.06]))
    assert jnp.array_equal(holm.rejected, jnp.asarray([True, False, False]))
    assert jnp.allclose(bh.adjusted_p_values, jnp.asarray([0.03, 0.04, 0.04]))
    assert jnp.array_equal(bh.rejected, jnp.asarray([True, True, True]))
    assert jnp.all(jnp.diff(holm.ordered_adjusted_p_values) >= 0.0)
    assert jnp.all(jnp.diff(bh.ordered_adjusted_p_values) >= 0.0)


def test_forecast_comparison_handles_exact_zero_variance_tie():
    losses = jnp.asarray([1.0, 2.0, 3.0, 4.0])

    result = compare_forecasts(losses, losses, hac_lags=1)

    assert result.successful
    assert result.zero_variance
    assert result.statistic == 0.0
    assert result.p_value == 1.0
    assert result.mean_differential == 0.0


def test_forecast_comparison_mask_excludes_irregular_missing_pairs():
    first = jnp.asarray([1.0, 1000.0, 2.0, 4.0, 5.0])
    second = jnp.asarray([2.0, -1000.0, 3.0, 4.0, 7.0])
    mask = jnp.asarray([True, False, True, True, True])

    result = compare_forecasts(first, second, mask=mask, hac_lags=0)

    assert result.effective_sample_count == 4
    assert jnp.isclose(result.mean_differential, -1.0)
    assert result.first_has_lower_loss
