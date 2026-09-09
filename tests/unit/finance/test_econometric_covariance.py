import jax.numpy as jnp
import numpy as np

from phydrax.finance.core import PhysicalLaw
from phydrax.finance.econometrics._covariance import (
    clean_covariance_rmt,
    CovarianceDefinition,
    FactorModelDefinition,
    fit_covariance,
    fit_factor_model,
    RMTCleaningDefinition,
)
from phydrax.finance.econometrics._returns import ReturnResult


def _factor_returns() -> ReturnResult:
    generator = np.random.default_rng(18)
    factor = generator.normal(size=600)
    loadings = np.asarray([1.0, 0.8, -0.5, 0.3])
    noise = generator.normal(scale=0.12, size=(600, 4))
    values = factor[:, None] * loadings[None, :] + noise
    intervals = np.arange(600, dtype=np.int64)
    return ReturnResult(
        values=jnp.asarray(values.T),
        valid_mask=jnp.ones((4, 600), dtype=bool),
        status=jnp.zeros((4, 600), dtype=jnp.int32),
        interval_start_ns=jnp.broadcast_to(jnp.asarray(intervals), (4, 600)),
        interval_end_ns=jnp.broadcast_to(jnp.asarray(intervals + 1), (4, 600)),
        batches=(),
        batch_channels=(),
        source_panel_id="panel",
        adjustment_id="raw",
        definition_id="returns-definition",
        result_id="returns",
        kind="simple",
    )


def test_shrinkage_factor_and_rmt_results_retain_psd_rank_condition_and_trace():
    returns = _factor_returns()
    law = PhysicalLaw("observed", "synthetic", "four-assets", "historical")

    covariance = fit_covariance(
        returns,
        CovarianceDefinition(method="ledoit-wolf", regularization=1e-8),
        law,
    )
    factor = fit_factor_model(returns, FactorModelDefinition(1), law)
    cleaned = clean_covariance_rmt(
        covariance,
        RMTCleaningDefinition(preserve_trace=True),
        law,
    )

    assert jnp.min(jnp.linalg.eigvalsh(covariance.covariance.matrix)) >= 0.0
    assert covariance.effective_rank == 4
    assert jnp.isfinite(covariance.condition_number)
    assert 0.0 <= covariance.shrinkage_intensity <= 1.0
    assert factor.loadings.shape == (4, 1)
    recovered = factor.loadings[:, 0] / jnp.linalg.norm(factor.loadings[:, 0])
    expected = jnp.asarray([1.0, 0.8, -0.5, 0.3])
    expected = expected / jnp.linalg.norm(expected)
    assert jnp.dot(recovered, expected) > 0.98
    assert jnp.all(factor.idiosyncratic_variance >= 0.0)
    assert jnp.min(jnp.linalg.eigvalsh(cleaned.covariance.matrix)) >= 0.0
    assert jnp.isclose(
        jnp.trace(cleaned.covariance.matrix),
        jnp.trace(covariance.covariance.matrix),
        rtol=1e-5,
    )
    assert cleaned.cleaning.effective_rank == 4
