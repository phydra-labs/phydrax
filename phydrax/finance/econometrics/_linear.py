#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...uq._linear_time_series import (
    ARIMAFit as GenericARIMAFit,
    CointegrationResult as GenericCointegrationResult,
    fit_arima as fit_arima_arrays,
    fit_var as fit_var_arrays,
    fit_vecm as fit_vecm_arrays,
    test_cointegration as test_cointegration_arrays,
    VARFit as GenericVARFit,
    VECMFit as GenericVECMFit,
)
from ..core import FinanceEvidenceBinding, PhysicalLaw
from ._returns import ReturnResult


def _binding(data_id: str, model_id: str, route: str) -> FinanceEvidenceBinding:
    return FinanceEvidenceBinding(
        (canonical_fingerprint({"kind": "econometrics-data-evidence", "data": data_id}),),
        (
            canonical_fingerprint(
                {"kind": "econometrics-model-evidence", "model": model_id}
            ),
        ),
        (
            canonical_fingerprint(
                {"kind": "econometrics-numerical-evidence", "route": route}
            ),
        ),
        (
            canonical_fingerprint(
                {"kind": "econometrics-use-evidence", "trade_emission": False}
            ),
        ),
    )


class ARIMADefinition(StrictModule):
    p: int = eqx.field(static=True)
    d: int = eqx.field(static=True)
    q: int = eqx.field(static=True)
    include_intercept: bool = eqx.field(static=True)
    iterations: int = eqx.field(static=True)
    ridge: float = eqx.field(static=True)
    definition_id: str = eqx.field(static=True)

    def __init__(
        self,
        p: int,
        d: int = 0,
        q: int = 0,
        /,
        *,
        include_intercept: bool = True,
        iterations: int = 4,
        ridge: float = 0.0,
    ):
        orders = (int(p), int(d), int(q))
        if min(orders) < 0 or orders[0] + orders[2] < 1:
            raise ValueError("ARIMA orders must be nonnegative with p + q positive.")
        iterations_ = int(iterations)
        ridge_ = float(ridge)
        if iterations_ < 1 or not jnp.isfinite(ridge_) or ridge_ < 0.0:
            raise ValueError("iterations must be positive and ridge finite/nonnegative.")
        self.p, self.d, self.q = orders
        self.include_intercept = bool(include_intercept)
        self.iterations = iterations_
        self.ridge = ridge_
        self.definition_id = canonical_fingerprint(
            {
                "kind": "finance-arima-definition",
                "orders": orders,
                "include_intercept": bool(include_intercept),
                "iterations": iterations_,
                "ridge": ridge_,
            }
        )


class VARDefinition(StrictModule):
    order: int = eqx.field(static=True)
    include_intercept: bool = eqx.field(static=True)
    ridge: float = eqx.field(static=True)
    definition_id: str = eqx.field(static=True)

    def __init__(
        self,
        order: int,
        /,
        *,
        include_intercept: bool = True,
        ridge: float = 0.0,
    ):
        order_ = int(order)
        ridge_ = float(ridge)
        if order_ < 1 or not jnp.isfinite(ridge_) or ridge_ < 0.0:
            raise ValueError("order must be positive and ridge finite/nonnegative.")
        self.order = order_
        self.include_intercept = bool(include_intercept)
        self.ridge = ridge_
        self.definition_id = canonical_fingerprint(
            {
                "kind": "finance-var-definition",
                "order": order_,
                "include_intercept": bool(include_intercept),
                "ridge": ridge_,
            }
        )


class CointegrationDefinition(StrictModule):
    lag_differences: int = eqx.field(static=True)
    deterministic: Literal["constant", "none"] = eqx.field(static=True)
    critical_values: Array
    definition_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        lag_differences: int = 0,
        deterministic: Literal["constant", "none"] = "constant",
        critical_values: ArrayLike = (),
    ):
        lags = int(lag_differences)
        if lags < 0:
            raise ValueError("lag_differences must be nonnegative.")
        if deterministic not in ("constant", "none"):
            raise ValueError("deterministic must be 'constant' or 'none'.")
        critical = jnp.asarray(critical_values, dtype=float)
        if critical.ndim != 1:
            raise ValueError("critical_values must be a vector (possibly empty).")
        critical = eqx.error_if(
            critical,
            jnp.any(~jnp.isfinite(critical) | (critical <= 0.0)),
            "critical_values must be finite and positive.",
        )
        self.lag_differences = lags
        self.deterministic = deterministic
        self.critical_values = critical
        self.definition_id = canonical_fingerprint(
            {
                "kind": "finance-cointegration-definition",
                "lag_differences": lags,
                "deterministic": deterministic,
                "critical_values": tuple(float(value) for value in critical.tolist()),
            }
        )


class VECMDefinition(StrictModule):
    rank: int = eqx.field(static=True)
    lag_differences: int = eqx.field(static=True)
    include_intercept: bool = eqx.field(static=True)
    ridge: float = eqx.field(static=True)
    definition_id: str = eqx.field(static=True)

    def __init__(
        self,
        rank: int,
        /,
        *,
        lag_differences: int = 0,
        include_intercept: bool = True,
        ridge: float = 1e-10,
    ):
        rank_ = int(rank)
        lags = int(lag_differences)
        ridge_ = float(ridge)
        if rank_ < 1 or lags < 0 or not jnp.isfinite(ridge_) or ridge_ < 0.0:
            raise ValueError("rank must be positive, lags nonnegative, and ridge valid.")
        self.rank = rank_
        self.lag_differences = lags
        self.include_intercept = bool(include_intercept)
        self.ridge = ridge_
        self.definition_id = canonical_fingerprint(
            {
                "kind": "finance-vecm-definition",
                "rank": rank_,
                "lag_differences": lags,
                "include_intercept": bool(include_intercept),
                "ridge": ridge_,
            }
        )


class ARIMAFit(StrictModule):
    fit: GenericARIMAFit
    evidence: FinanceEvidenceBinding = eqx.field(static=True)
    data_id: str = eqx.field(static=True)
    definition_id: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)
    series_index: int = eqx.field(static=True)


class VARFit(StrictModule):
    fit: GenericVARFit
    evidence: FinanceEvidenceBinding = eqx.field(static=True)
    data_id: str = eqx.field(static=True)
    definition_id: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)


class CointegrationFit(StrictModule):
    result: GenericCointegrationResult
    evidence: FinanceEvidenceBinding = eqx.field(static=True)
    data_id: str = eqx.field(static=True)
    definition_id: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)


class VECMFit(StrictModule):
    fit: GenericVECMFit
    evidence: FinanceEvidenceBinding = eqx.field(static=True)
    data_id: str = eqx.field(static=True)
    definition_id: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)


class ARIMAResult(StrictModule):
    forecast: Array
    fit: ARIMAFit
    horizon: int = eqx.field(static=True)


class VARResult(StrictModule):
    forecast: Array
    fit: VARFit
    horizon: int = eqx.field(static=True)


class VECMResult(StrictModule):
    forecast: Array
    fit: VECMFit
    horizon: int = eqx.field(static=True)


def _physical(law: PhysicalLaw) -> None:
    if not isinstance(law, PhysicalLaw):
        raise TypeError("econometric estimation requires an explicit PhysicalLaw.")


def fit_arima(
    returns: ReturnResult,
    definition: ARIMADefinition,
    law: PhysicalLaw,
    /,
    *,
    series_index: int = 0,
) -> ARIMAFit:
    _physical(law)
    if not isinstance(returns, ReturnResult) or not isinstance(
        definition, ARIMADefinition
    ):
        raise TypeError("returns and definition must have their declared finance types.")
    series = int(series_index)
    if not 0 <= series < returns.values.shape[0]:
        raise IndexError("series_index is outside the return panel.")
    generic = fit_arima_arrays(
        returns.values[series],
        p=definition.p,
        d=definition.d,
        q=definition.q,
        mask=returns.valid_mask[series],
        include_intercept=definition.include_intercept,
        iterations=definition.iterations,
        ridge=definition.ridge,
    )
    return ARIMAFit(
        fit=generic,
        evidence=_binding(
            returns.result_id, definition.definition_id, "arima-conditional-likelihood"
        ),
        data_id=returns.result_id,
        definition_id=definition.definition_id,
        law_id=law.law_id,
        series_index=series,
    )


def fit_var(
    returns: ReturnResult,
    definition: VARDefinition,
    law: PhysicalLaw,
    /,
) -> VARFit:
    _physical(law)
    if not isinstance(returns, ReturnResult) or not isinstance(definition, VARDefinition):
        raise TypeError("returns and definition must have their declared finance types.")
    generic = fit_var_arrays(
        returns.values.T,
        order=definition.order,
        mask=returns.valid_mask.T,
        include_intercept=definition.include_intercept,
        ridge=definition.ridge,
    )
    return VARFit(
        fit=generic,
        evidence=_binding(
            returns.result_id, definition.definition_id, "var-least-squares"
        ),
        data_id=returns.result_id,
        definition_id=definition.definition_id,
        law_id=law.law_id,
    )


def test_cointegration(
    levels: ArrayLike,
    mask: ArrayLike,
    data_id: str,
    definition: CointegrationDefinition,
    law: PhysicalLaw,
    /,
) -> CointegrationFit:
    _physical(law)
    if not isinstance(definition, CointegrationDefinition):
        raise TypeError("definition must be a CointegrationDefinition.")
    critical = (
        None if definition.critical_values.size == 0 else definition.critical_values
    )
    generic = test_cointegration_arrays(
        levels,
        lag_differences=definition.lag_differences,
        mask=mask,
        critical_values=critical,
        deterministic=definition.deterministic,
    )
    return CointegrationFit(
        result=generic,
        evidence=_binding(data_id, definition.definition_id, "johansen-trace"),
        data_id=str(data_id),
        definition_id=definition.definition_id,
        law_id=law.law_id,
    )


def fit_vecm(
    levels: ArrayLike,
    mask: ArrayLike,
    data_id: str,
    definition: VECMDefinition,
    law: PhysicalLaw,
    /,
) -> VECMFit:
    _physical(law)
    if not isinstance(definition, VECMDefinition):
        raise TypeError("definition must be a VECMDefinition.")
    generic = fit_vecm_arrays(
        levels,
        rank=definition.rank,
        lag_differences=definition.lag_differences,
        mask=mask,
        include_intercept=definition.include_intercept,
        ridge=definition.ridge,
    )
    return VECMFit(
        fit=generic,
        evidence=_binding(data_id, definition.definition_id, "vecm-least-squares"),
        data_id=str(data_id),
        definition_id=definition.definition_id,
        law_id=law.law_id,
    )


def forecast_arima(fit: ARIMAFit, history: ArrayLike, horizon: int, /) -> ARIMAResult:
    if not isinstance(fit, ARIMAFit):
        raise TypeError("fit must be an ARIMAFit.")
    horizon_ = int(horizon)
    return ARIMAResult(
        forecast=fit.fit.model.forecast(history, horizon_), fit=fit, horizon=horizon_
    )


def forecast_var(fit: VARFit, history: ArrayLike, horizon: int, /) -> VARResult:
    if not isinstance(fit, VARFit):
        raise TypeError("fit must be a VARFit.")
    horizon_ = int(horizon)
    return VARResult(
        forecast=fit.fit.model.forecast(history, horizon_), fit=fit, horizon=horizon_
    )


def forecast_vecm(fit: VECMFit, history: ArrayLike, horizon: int, /) -> VECMResult:
    if not isinstance(fit, VECMFit):
        raise TypeError("fit must be a VECMFit.")
    horizon_ = int(horizon)
    return VECMResult(
        forecast=fit.fit.model.forecast(history, horizon_), fit=fit, horizon=horizon_
    )


__all__ = [
    "ARIMADefinition",
    "ARIMAFit",
    "ARIMAResult",
    "CointegrationDefinition",
    "CointegrationFit",
    "VARDefinition",
    "VARFit",
    "VARResult",
    "VECMDefinition",
    "VECMFit",
    "VECMResult",
    "fit_arima",
    "fit_var",
    "fit_vecm",
    "forecast_arima",
    "forecast_var",
    "forecast_vecm",
    "test_cointegration",
]
