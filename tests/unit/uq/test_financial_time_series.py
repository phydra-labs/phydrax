import jax.numpy as jnp
import numpy as np

from phydrax.uq._conditional_volatility import fit_har, GARCHModel
from phydrax.uq._linear_time_series import (
    augmented_dickey_fuller,
    fit_arima,
    fit_var,
    test_cointegration as cointegration_test,
)


def test_arima_recovers_stable_ar_coefficient_with_irregular_mask():
    generator = np.random.default_rng(13)
    values = np.zeros(320)
    noise = generator.normal(scale=0.03, size=values.size)
    for index in range(1, values.size):
        values[index] = 0.72 * values[index - 1] + noise[index]
    mask = np.ones(values.size, dtype=bool)
    mask[101] = False

    fit = fit_arima(values, p=1, mask=mask)

    assert fit.successful
    assert jnp.isclose(fit.model.autoregressive[0], 0.72, atol=0.08)
    assert fit.model.stable
    assert not fit.residual_mask[101]
    assert not fit.residual_mask[102]
    assert jnp.isfinite(fit.log_likelihood)


def test_var_recovers_cross_lag_and_stability():
    generator = np.random.default_rng(4)
    transition = np.asarray([[0.55, 0.18], [-0.08, 0.42]])
    values = np.zeros((500, 2))
    noise = generator.normal(scale=0.02, size=values.shape)
    for index in range(1, values.shape[0]):
        values[index] = transition @ values[index - 1] + noise[index]

    fit = fit_var(values, order=1)

    assert fit.successful
    assert jnp.allclose(fit.model.lag_matrices[0], transition, atol=0.06)
    assert fit.model.stable
    assert fit.rank == 3
    assert jnp.isfinite(fit.log_likelihood)


def test_augmented_dickey_fuller_distinguishes_unit_root_from_stationary_series():
    generator = np.random.default_rng(31)
    noise = generator.normal(scale=0.1, size=900)
    random_walk = np.cumsum(noise)
    stationary = np.zeros(noise.size)
    for index in range(1, stationary.size):
        stationary[index] = 0.45 * stationary[index - 1] + noise[index]

    unit_root = augmented_dickey_fuller(random_walk, lag_differences=1)
    mean_reverting = augmented_dickey_fuller(stationary, lag_differences=1)

    assert unit_root.valid
    assert mean_reverting.valid
    assert not unit_root.stationary
    assert mean_reverting.stationary
    assert mean_reverting.statistic < unit_root.statistic


def test_johansen_evidence_separates_common_unit_root_from_spread():
    generator = np.random.default_rng(9)
    common = np.cumsum(generator.normal(scale=0.2, size=700))
    spread = np.zeros(common.size)
    innovations = generator.normal(scale=0.05, size=common.size)
    for index in range(1, spread.size):
        spread[index] = 0.35 * spread[index - 1] + innovations[index]
    levels = np.stack((common + spread, common - spread), axis=-1)

    result = cointegration_test(levels, lag_differences=1)

    assert result.valid
    assert result.eigenvalues[0] > result.eigenvalues[1]
    assert result.selected_rank == 1
    assert jnp.isfinite(result.trace_statistics).all()
    assert result.cointegration_vectors.shape == (2, 2)


def test_garch_and_gjr_variance_recursions_match_hand_oracle():
    residuals = jnp.asarray([1.0, -2.0, 0.5])
    garch = GARCHModel(0.1, 0.2, 0.5)
    gjr = GARCHModel(0.1, 0.2, 0.5, gamma=0.3, kind="gjr-garch")

    plain_variance = garch.conditional_variance(residuals, initial_variance=1.0)
    asymmetric_variance = gjr.conditional_variance(residuals, initial_variance=1.0)

    assert jnp.allclose(plain_variance, jnp.asarray([1.0, 0.8, 1.3]))
    assert jnp.allclose(asymmetric_variance, jnp.asarray([1.0, 0.8, 2.5]))
    assert garch.stable
    assert gjr.stable


def test_har_requires_complete_trailing_windows():
    values = jnp.arange(1.0, 50.0)
    mask = jnp.ones(values.shape, dtype=bool).at[20].set(False)

    fit = fit_har(values, windows=(1, 5), mask=mask, ridge=1e-8)

    # The missing observation invalidates its target and every five-period window
    # that contains it, rather than treating the masked value as zero.
    assert not fit.valid_mask[20 - 5]
    assert not fit.valid_mask[21 - 5]
    assert fit.effective_sample_count < values.size - 5
    assert jnp.isfinite(fit.log_likelihood)
