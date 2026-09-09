import jax.numpy as jnp

from phydrax.finance.calibration._calibration import (
    compile_calibration,
    evaluate_calibration,
    prepare_calibration,
    replay_calibration,
)
from phydrax.finance.calibration._surface import (
    ESSVISurface,
    evaluate_surface_arbitrage,
    SVIParameters,
    SVISlice,
    VolatilityObservationSet,
)


def _inverse_softplus(value):
    return jnp.log(jnp.expm1(value))


def test_svi_calibration_recovers_synthetic_slice_and_replays_exactly():
    expiry = 1.0
    parameters = SVIParameters(0.04, 0.1, -0.3, 0.0, 0.4)
    log_moneyness = jnp.array([-0.8, -0.4, 0.0, 0.4, 0.8, 1.2])
    total_variance = parameters.total_variance(log_moneyness)
    observations = VolatilityObservationSet(
        jnp.full_like(log_moneyness, expiry),
        log_moneyness,
        jnp.sqrt(total_variance / expiry),
    )
    plan = compile_calibration(observations, family="svi")
    initial = jnp.array(
        [
            [
                _inverse_softplus(0.04),
                _inverse_softplus(0.1 - 1.0e-10),
                jnp.arctanh(-0.3 / 0.999),
                0.0,
                _inverse_softplus(0.4 - 1.0e-10),
            ]
        ]
    )
    prepared = prepare_calibration(plan, observations, initial_parameters=initial)
    result = evaluate_calibration(prepared)
    replay = replay_calibration(prepared, result)
    assert result.successful
    assert (
        jnp.max(
            jnp.abs(
                result.fitted_implied_volatilities - observations.implied_volatilities
            )
        )
        < 1.0e-6
    )
    assert replay.matches


def test_surface_arbitrage_evidence_separates_butterfly_calendar_and_wings():
    slice_ = SVISlice(1.0, SVIParameters(0.04, 0.08, -0.2, 0.0, 0.3))
    evidence = evaluate_surface_arbitrage(slice_)
    assert evidence.finite
    assert evidence.butterfly_free
    assert evidence.wing_admissible

    essvi = ESSVISurface(
        jnp.array([0.5, 1.0, 2.0]),
        jnp.array([0.02, 0.04, 0.08]),
        jnp.array([-0.2, -0.25, -0.3]),
        0.2,
        0.5,
    )
    evidence = evaluate_surface_arbitrage(essvi)
    assert evidence.calendar_free
    assert evidence.valid
