#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Callable
from dataclasses import dataclass

import equinox as eqx
import numpy as np
import pytest
from scipy import stats

import phydrax as phx
from phydrax.observation import CovarianceAction


# Host references use SciPy/NumPy dense algebra, independent of the structured
# whitening, Woodbury, Schur-complement, and FFT paths under test.
_LAYOUT = phx.observation.CoordinateLayout(("a", "b", "c", "d"))
_OBSERVED = np.asarray((0.4, -1.1, 0.7, 2.0))
_PREDICTED = np.asarray((0.9, -0.3, 0.1, 1.2))
_DENSE = np.asarray(
    (
        (1.3, 0.4, 0.1, -0.2),
        (0.4, 1.7, 0.3, 0.1),
        (0.1, 0.3, 0.9, 0.25),
        (-0.2, 0.1, 0.25, 1.4),
    )
)
_VARIANCE = np.asarray((0.5, 1.2, 2.0, 0.8))
_FACTORS = np.asarray(((0.3, -0.1), (0.2, 0.4), (-0.5, 0.2), (0.1, 0.3)))
_FIRST = np.asarray(((1.2, 0.2), (0.2, 0.8)))
_SECOND = np.asarray(((0.7, 0.1), (0.1, 1.1)))
_SPECTRUM = np.asarray((1.0, 2.0, 1.5))
_RTOL = 1e-10
_ATOL = 1e-12


def _circulant_covariance() -> np.ndarray:
    first_row = np.fft.irfft(_SPECTRUM, n=_LAYOUT.size)
    index = np.arange(_LAYOUT.size)
    return first_row[(index[:, None] - index[None, :]) % _LAYOUT.size]


def _precision_operator() -> phx.observation.PrecisionOperatorCovarianceAction:
    properties = phx.linalg.OperatorProperties(
        self_adjoint=True,
        positive_definite=True,
        evidence={
            "self_adjoint": "construction",
            "positive_definite": "construction",
            "positive_semidefinite": "construction",
        },
    )
    operator = phx.linalg.DenseLinearOperator(
        np.linalg.inv(_DENSE), properties=properties
    )
    return phx.observation.PrecisionOperatorCovarianceAction(
        operator, np.linalg.slogdet(_DENSE)[1], _LAYOUT
    )


@dataclass(frozen=True)
class CovarianceCase:
    build: Callable[[], CovarianceAction]
    dense: Callable[[], np.ndarray]
    whitening: phx.observation.MeasurementWhitening


_CASES = {
    "diagonal": CovarianceCase(
        lambda: phx.observation.DiagonalCovarianceAction(_VARIANCE, _LAYOUT),
        lambda: np.diag(_VARIANCE),
        "diagonal",
    ),
    "cholesky": CovarianceCase(
        lambda: phx.observation.CholeskyCovarianceAction(
            np.linalg.cholesky(_DENSE), _LAYOUT
        ),
        lambda: _DENSE,
        "cholesky",
    ),
    "kronecker": CovarianceCase(
        lambda: phx.observation.KroneckerCholeskyCovarianceAction(
            (np.linalg.cholesky(_FIRST), np.linalg.cholesky(_SECOND)), _LAYOUT
        ),
        lambda: np.kron(_FIRST, _SECOND),
        "kronecker_cholesky",
    ),
    "circulant": CovarianceCase(
        lambda: phx.observation.CirculantCovarianceAction(_SPECTRUM, _LAYOUT),
        _circulant_covariance,
        "circulant",
    ),
    "low-rank": CovarianceCase(
        lambda: phx.observation.LowRankDiagonalCovarianceAction(
            _VARIANCE, _FACTORS, _LAYOUT
        ),
        lambda: np.diag(_VARIANCE) + _FACTORS @ _FACTORS.T,
        "unavailable",
    ),
    "precision": CovarianceCase(
        lambda: phx.observation.PrecisionCovarianceAction(
            np.linalg.inv(_DENSE), np.linalg.slogdet(_DENSE)[1], _LAYOUT
        ),
        lambda: _DENSE,
        "unavailable",
    ),
    "precision-operator": CovarianceCase(
        _precision_operator, lambda: _DENSE, "unavailable"
    ),
}


def _field(
    values: np.ndarray,
    valid: np.ndarray,
    field_id: str,
    /,
    *,
    uncertainty: np.ndarray | None = None,
) -> phx.measurement.PreparedQuantityField:
    return phx.measurement.PreparedQuantityField(
        values,
        valid,
        standard_uncertainty=uncertainty,
        quantity_id="test.signal",
        compatibility_id="test.signal",
        layout_id="scalar",
        support_id="samples",
        sampling_id="point",
        unit_id="one",
        field_id=field_id,
    )


def _all_valid() -> np.ndarray:
    return np.ones(_LAYOUT.size, dtype=np.bool_)


@pytest.mark.parametrize(
    "case_id",
    ("diagonal", "cholesky", "kronecker", "circulant"),
)
def test_whitened_residual_realizes_the_covariance_quadratic(case_id: str) -> None:
    case = _CASES[case_id]
    dense = case.dense()
    plan = phx.observation.MeasurementComparisonPlan(
        _field(_OBSERVED, _all_valid(), "observed"), covariance=case.build()
    )
    result = eqx.filter_jit(plan.evaluate)(_field(_PREDICTED, _all_valid(), "predicted"))
    residual = _PREDICTED - _OBSERVED
    reference = residual @ np.linalg.solve(dense, residual)

    assert result.whitening == case.whitening
    assert result.whitened_residual is not None and result.quadratic is not None
    assert result.logdet_covariance is not None and result.log_likelihood is not None
    np.testing.assert_allclose(
        np.sum(np.asarray(result.whitened_residual) ** 2),
        reference,
        rtol=_RTOL,
        atol=_ATOL,
    )
    np.testing.assert_allclose(result.quadratic, reference, rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(
        result.logdet_covariance, np.linalg.slogdet(dense)[1], rtol=_RTOL, atol=_ATOL
    )
    np.testing.assert_allclose(
        result.log_likelihood,
        stats.multivariate_normal(_OBSERVED, dense).logpdf(_PREDICTED),
        rtol=_RTOL,
        atol=_ATOL,
    )
    assert bool(result.successful)


@pytest.mark.parametrize("case_id", ("low-rank", "precision", "precision-operator"))
def test_factorless_covariances_keep_quadratic_and_normalization(case_id: str) -> None:
    case = _CASES[case_id]
    dense = case.dense()
    plan = phx.observation.MeasurementComparisonPlan(
        _field(_OBSERVED, _all_valid(), "observed"), covariance=case.build()
    )
    result = plan.evaluate(_field(_PREDICTED, _all_valid(), "predicted"))
    residual = _PREDICTED - _OBSERVED

    assert result.whitening == "unavailable"
    assert result.whitened_residual is None
    assert result.quadratic is not None and result.log_likelihood is not None
    np.testing.assert_allclose(
        result.quadratic,
        residual @ np.linalg.solve(dense, residual),
        rtol=_RTOL,
        atol=_ATOL,
    )
    np.testing.assert_allclose(
        result.log_likelihood,
        stats.multivariate_normal(_OBSERVED, dense).logpdf(_PREDICTED),
        rtol=_RTOL,
        atol=_ATOL,
    )
    assert bool(result.successful)


def test_incomplete_observed_validity_refuses_a_full_correlated_covariance() -> None:
    valid = np.asarray((True, False, True, True))
    observed = _field(_OBSERVED, valid, "observed")

    with pytest.raises(ValueError, match="active observed values"):
        phx.observation.MeasurementComparisonPlan(
            observed, covariance=_CASES["cholesky"].build()
        )


@pytest.mark.parametrize("case_id", ("diagonal", "low-rank", "cholesky", "precision"))
def test_restricted_covariance_is_the_exact_gaussian_marginal(case_id: str) -> None:
    case = _CASES[case_id]
    dense = case.dense()
    valid = np.asarray((True, False, True, True))
    active = np.flatnonzero(valid)
    restricted = phx.observation.restrict_observation_covariance(case.build(), valid)
    plan = phx.observation.MeasurementComparisonPlan(
        _field(_OBSERVED, valid, "observed"), covariance=restricted
    )
    result = plan.evaluate(_field(_PREDICTED, _all_valid(), "predicted"))
    marginal = dense[np.ix_(active, active)]
    residual = (_PREDICTED - _OBSERVED)[active]
    zero_filled = np.where(valid, _PREDICTED - _OBSERVED, 0.0)

    assert restricted.layout.labels == ("a", "c", "d")
    assert int(result.active_value_count) == active.size
    assert result.quadratic is not None and result.logdet_covariance is not None
    assert result.log_likelihood is not None
    np.testing.assert_allclose(
        result.quadratic,
        residual @ np.linalg.solve(marginal, residual),
        rtol=_RTOL,
        atol=_ATOL,
    )
    np.testing.assert_allclose(
        result.logdet_covariance, np.linalg.slogdet(marginal)[1], rtol=_RTOL, atol=_ATOL
    )
    expected = stats.multivariate_normal(_OBSERVED[active], marginal).logpdf(
        _PREDICTED[active]
    )
    np.testing.assert_allclose(result.log_likelihood, expected, rtol=_RTOL, atol=_ATOL)
    # Fault adequacy: zero-filling the masked value under the full covariance is
    # a different, incorrect likelihood.
    wrong = stats.multivariate_normal(np.zeros(_LAYOUT.size), dense).logpdf(zero_filled)
    assert abs(float(expected) - float(wrong)) > 1e-3
    assert bool(result.successful)


@pytest.mark.parametrize("case_id", ("kronecker", "circulant", "precision-operator"))
def test_structured_covariances_refuse_partial_restriction(case_id: str) -> None:
    covariance = _CASES[case_id].build()

    with pytest.raises(ValueError, match="active-set covariance explicitly"):
        phx.observation.restrict_observation_covariance(
            covariance, np.asarray((True, False, True, True))
        )
    complete = phx.observation.restrict_observation_covariance(covariance, _all_valid())
    assert complete.action_id == covariance.action_id


def test_dense_precision_restriction_holds_at_sensor_noise_scale() -> None:
    # Sensor standard deviations near 1e-4 give precision entries near 1e7; the
    # rounded Schur complement is then asymmetric by ~1e-10 in absolute terms.
    size = 12
    factor = np.random.default_rng(0).standard_normal((size, size))
    covariance = (factor @ factor.T + size * np.eye(size)) * 1.0e-8
    precision = np.linalg.inv(covariance)
    layout = phx.observation.CoordinateLayout(tuple(f"s{i}" for i in range(size)))
    action = phx.observation.PrecisionCovarianceAction(
        0.5 * (precision + precision.T), np.linalg.slogdet(covariance)[1], layout
    )
    valid = np.ones(size, dtype=np.bool_)
    valid[[2, 5, 7]] = False
    active = np.flatnonzero(valid)
    restricted = phx.observation.restrict_observation_covariance(action, valid)
    marginal = covariance[np.ix_(active, active)]

    assert isinstance(restricted, phx.observation.PrecisionCovarianceAction)
    np.testing.assert_allclose(
        np.asarray(restricted.precision) @ marginal,
        np.eye(active.size),
        rtol=0.0,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        restricted.logdet_covariance, np.linalg.slogdet(marginal)[1], rtol=1.0e-12
    )


def test_precision_symmetry_is_judged_at_the_precision_scale() -> None:
    layout = phx.observation.CoordinateLayout(("a", "b"))
    base = np.asarray(((2.0, 0.5), (0.5, 1.0)))
    # Rounding-level asymmetry of a large precision is symmetric at its scale...
    rounded = 1.0e8 * base
    rounded[0, 1] += 1.0e-7
    phx.observation.PrecisionCovarianceAction(rounded, 0.0, layout)
    # ... and a percent-level asymmetry of a small precision is not.
    skewed = 1.0e-8 * base
    skewed[0, 1] *= 1.01
    with pytest.raises(eqx.EquinoxRuntimeError, match="symmetric"):
        phx.observation.PrecisionCovarianceAction(skewed, 0.0, layout)


def test_prediction_invalid_on_a_correlated_value_fails_the_comparison() -> None:
    plan = phx.observation.MeasurementComparisonPlan(
        _field(_OBSERVED, _all_valid(), "observed"),
        covariance=_CASES["cholesky"].build(),
    )
    predicted = _field(_PREDICTED, np.asarray((True, True, False, True)), "predicted")
    result = plan.evaluate(predicted)

    assert not bool(result.active_set_consistent)
    assert not bool(result.successful)


def test_independent_uncertainty_normalizes_only_active_values() -> None:
    uncertainty = np.asarray((0.3, 0.5, 0.2, 0.8))
    observed = _field(_OBSERVED, _all_valid(), "observed", uncertainty=uncertainty)
    predicted_valid = np.asarray((True, True, False, True))
    result = phx.observation.MeasurementComparisonPlan(observed).evaluate(
        _field(_PREDICTED, predicted_valid, "predicted")
    )
    active = np.flatnonzero(predicted_valid)

    assert result.noise_model == "independent_uncertainty"
    assert result.whitening == "diagonal"
    assert int(result.active_value_count) == active.size
    assert result.log_likelihood is not None
    np.testing.assert_allclose(
        result.log_likelihood,
        np.sum(
            stats.norm(_OBSERVED[active], uncertainty[active]).logpdf(_PREDICTED[active])
        ),
        rtol=_RTOL,
        atol=_ATOL,
    )
    assert bool(result.successful)


def test_unquantified_data_require_an_explicit_reference_weighting() -> None:
    observed = _field(_OBSERVED, _all_valid(), "observed")
    predicted = _field(_PREDICTED, _all_valid(), "predicted")
    unquantified = phx.observation.MeasurementComparisonPlan(observed).evaluate(predicted)
    scale = np.asarray((0.5, 1.0, 2.0, 4.0))
    weighted = phx.observation.MeasurementComparisonPlan(
        observed, reference_scale=scale
    ).evaluate(predicted)

    assert unquantified.noise_model == "unquantified"
    assert unquantified.whitening == "unavailable"
    assert unquantified.whitened_residual is None
    assert unquantified.quadratic is None
    assert unquantified.log_likelihood is None
    assert bool(unquantified.successful)
    assert weighted.noise_model == "reference_weighting"
    assert weighted.quadratic is not None
    np.testing.assert_allclose(
        weighted.quadratic, np.sum(((_PREDICTED - _OBSERVED) / scale) ** 2)
    )
    assert weighted.logdet_covariance is None and weighted.log_likelihood is None
    with pytest.raises(ValueError, match="already declares a noise model"):
        phx.observation.MeasurementComparisonPlan(
            _field(_OBSERVED, _all_valid(), "observed", uncertainty=scale),
            reference_scale=1.0,
        )
    with pytest.raises(RuntimeError, match="finite and strictly positive"):
        phx.observation.MeasurementComparisonPlan(observed, reference_scale=0.0)
