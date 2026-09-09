#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...stochastic._state_space import StateSpaceProblem
from ...uq._conditional_volatility import (
    fit_garch as fit_garch_arrays,
    fit_har as fit_har_arrays,
    GARCHFit as GenericGARCHFit,
    HARFit as GenericHARFit,
)
from ...uq._state_space_inference import exact_state_space_log_likelihood
from ..core import FinanceEvidenceBinding, PhysicalLaw
from ._returns import RealizedMeasureResult, ReturnResult


def _binding(data_id: str, model_id: str, numerical_route: str) -> FinanceEvidenceBinding:
    return FinanceEvidenceBinding(
        (canonical_fingerprint({"kind": "volatility-data-evidence", "data": data_id}),),
        (
            canonical_fingerprint(
                {"kind": "volatility-model-evidence", "model": model_id}
            ),
        ),
        (
            canonical_fingerprint(
                {"kind": "volatility-numerical-evidence", "route": numerical_route}
            ),
        ),
        (
            canonical_fingerprint(
                {"kind": "volatility-use-evidence", "trade_emission": False}
            ),
        ),
    )


def _physical(law: PhysicalLaw) -> None:
    if not isinstance(law, PhysicalLaw):
        raise TypeError("physical-measure volatility estimation requires a PhysicalLaw.")


class GARCHDefinition(StrictModule):
    maximum_steps: int = eqx.field(static=True)
    definition_id: str = eqx.field(static=True)

    def __init__(self, *, maximum_steps: int = 128):
        steps = int(maximum_steps)
        if steps < 1:
            raise ValueError("maximum_steps must be positive.")
        self.maximum_steps = steps
        self.definition_id = canonical_fingerprint(
            {"kind": "finance-garch-definition", "maximum_steps": steps}
        )


class GJRDefinition(StrictModule):
    maximum_steps: int = eqx.field(static=True)
    definition_id: str = eqx.field(static=True)

    def __init__(self, *, maximum_steps: int = 128):
        steps = int(maximum_steps)
        if steps < 1:
            raise ValueError("maximum_steps must be positive.")
        self.maximum_steps = steps
        self.definition_id = canonical_fingerprint(
            {"kind": "finance-gjr-garch-definition", "maximum_steps": steps}
        )


class EGARCHDefinition(StrictModule):
    maximum_steps: int = eqx.field(static=True)
    definition_id: str = eqx.field(static=True)

    def __init__(self, *, maximum_steps: int = 128):
        steps = int(maximum_steps)
        if steps < 1:
            raise ValueError("maximum_steps must be positive.")
        self.maximum_steps = steps
        self.definition_id = canonical_fingerprint(
            {"kind": "finance-egarch-definition", "maximum_steps": steps}
        )


class HARDefinition(StrictModule):
    windows: tuple[int, ...] = eqx.field(static=True)
    include_intercept: bool = eqx.field(static=True)
    ridge: float = eqx.field(static=True)
    definition_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        windows: tuple[int, ...] = (1, 5, 22),
        include_intercept: bool = True,
        ridge: float = 0.0,
    ):
        windows_ = tuple(int(window) for window in windows)
        ridge_ = float(ridge)
        if not windows_ or any(window < 1 for window in windows_):
            raise ValueError("windows must be nonempty and positive.")
        if len(set(windows_)) != len(windows_) or ridge_ < 0.0:
            raise ValueError("windows must be unique and ridge nonnegative.")
        self.windows = windows_
        self.include_intercept = bool(include_intercept)
        self.ridge = ridge_
        self.definition_id = canonical_fingerprint(
            {
                "kind": "finance-har-definition",
                "windows": windows_,
                "include_intercept": bool(include_intercept),
                "ridge": ridge_,
            }
        )


class StochasticVolatilityDefinition(StrictModule):
    model_id: str = eqx.field(static=True)
    covariance_regularization: float = eqx.field(static=True)
    definition_id: str = eqx.field(static=True)

    def __init__(self, model_id: str, /, *, covariance_regularization: float = 0.0):
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be nonempty.")
        regularization = float(covariance_regularization)
        if not jnp.isfinite(regularization) or regularization < 0.0:
            raise ValueError("covariance_regularization must be finite and nonnegative.")
        self.model_id = model_id.strip()
        self.covariance_regularization = regularization
        self.definition_id = canonical_fingerprint(
            {
                "kind": "finance-stochastic-volatility-definition",
                "model_id": self.model_id,
                "covariance_regularization": regularization,
                "approximation": "linear-gaussian-state-space",
            }
        )


class GARCHFit(StrictModule):
    fit: GenericGARCHFit
    evidence: FinanceEvidenceBinding = eqx.field(static=True)
    data_id: str = eqx.field(static=True)
    definition_id: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)
    series_index: int = eqx.field(static=True)


class HARFit(StrictModule):
    fit: GenericHARFit
    evidence: FinanceEvidenceBinding = eqx.field(static=True)
    data_id: str = eqx.field(static=True)
    definition_id: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)
    series_index: int = eqx.field(static=True)


class StochasticVolatilityFit(StrictModule):
    likelihood: Any
    log_likelihood: Any
    valid: Any
    evidence: FinanceEvidenceBinding = eqx.field(static=True)
    data_id: str = eqx.field(static=True)
    definition_id: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)
    approximation: str = eqx.field(static=True)


def _fit_conditional(
    returns: ReturnResult,
    definition: GARCHDefinition | GJRDefinition | EGARCHDefinition,
    law: PhysicalLaw,
    series_index: int,
    kind: str,
) -> GARCHFit:
    _physical(law)
    if not isinstance(returns, ReturnResult):
        raise TypeError("returns must be a ReturnResult.")
    series = int(series_index)
    if not 0 <= series < returns.values.shape[0]:
        raise IndexError("series_index is outside the return panel.")
    generic = fit_garch_arrays(
        returns.values[series],
        mask=returns.valid_mask[series],
        kind=kind,
        maximum_steps=definition.maximum_steps,
    )
    return GARCHFit(
        fit=generic,
        evidence=_binding(
            returns.result_id, definition.definition_id, f"{kind}-gaussian-qmle"
        ),
        data_id=returns.result_id,
        definition_id=definition.definition_id,
        law_id=law.law_id,
        series_index=series,
    )


def fit_garch(
    returns: ReturnResult,
    definition: GARCHDefinition,
    law: PhysicalLaw,
    /,
    *,
    series_index: int = 0,
) -> GARCHFit:
    if not isinstance(definition, GARCHDefinition):
        raise TypeError("definition must be a GARCHDefinition.")
    return _fit_conditional(returns, definition, law, series_index, "garch")


def fit_gjr_garch(
    returns: ReturnResult,
    definition: GJRDefinition,
    law: PhysicalLaw,
    /,
    *,
    series_index: int = 0,
) -> GARCHFit:
    if not isinstance(definition, GJRDefinition):
        raise TypeError("definition must be a GJRDefinition.")
    return _fit_conditional(returns, definition, law, series_index, "gjr-garch")


def fit_egarch(
    returns: ReturnResult,
    definition: EGARCHDefinition,
    law: PhysicalLaw,
    /,
    *,
    series_index: int = 0,
) -> GARCHFit:
    if not isinstance(definition, EGARCHDefinition):
        raise TypeError("definition must be an EGARCHDefinition.")
    return _fit_conditional(returns, definition, law, series_index, "egarch")


def fit_har(
    realized: RealizedMeasureResult,
    definition: HARDefinition,
    law: PhysicalLaw,
    /,
    *,
    series_index: int = 0,
) -> HARFit:
    _physical(law)
    if not isinstance(realized, RealizedMeasureResult) or not isinstance(
        definition, HARDefinition
    ):
        raise TypeError("realized and definition must have their declared finance types.")
    series = int(series_index)
    if not 0 <= series < realized.values.shape[0]:
        raise IndexError("series_index is outside the realized-measure panel.")
    generic = fit_har_arrays(
        realized.values[series],
        mask=realized.valid_mask[series],
        windows=definition.windows,
        include_intercept=definition.include_intercept,
        ridge=definition.ridge,
    )
    return HARFit(
        fit=generic,
        evidence=_binding(
            realized.result_id, definition.definition_id, "har-least-squares"
        ),
        data_id=realized.result_id,
        definition_id=definition.definition_id,
        law_id=law.law_id,
        series_index=series,
    )


def fit_stochastic_volatility(
    problem: StateSpaceProblem,
    data_id: str,
    definition: StochasticVolatilityDefinition,
    law: PhysicalLaw,
    /,
) -> StochasticVolatilityFit:
    """Evaluate an explicit linear-Gaussian latent-log-volatility approximation."""

    _physical(law)
    if not isinstance(problem, StateSpaceProblem):
        raise TypeError("problem must be a StateSpaceProblem.")
    if not isinstance(definition, StochasticVolatilityDefinition):
        raise TypeError("definition must be a StochasticVolatilityDefinition.")
    if problem.model.model_id != definition.model_id:
        raise ValueError("state-space problem model does not match the SV definition.")
    likelihood = exact_state_space_log_likelihood(
        problem,
        method="kalman",
        covariance_regularization=definition.covariance_regularization,
    )
    return StochasticVolatilityFit(
        likelihood=likelihood,
        log_likelihood=likelihood.total_log_likelihood,
        valid=likelihood.valid,
        evidence=_binding(str(data_id), definition.definition_id, "kalman-linearized-sv"),
        data_id=str(data_id),
        definition_id=definition.definition_id,
        law_id=law.law_id,
        approximation="linear-gaussian-state-space",
    )


__all__ = [
    "EGARCHDefinition",
    "GARCHDefinition",
    "GARCHFit",
    "GJRDefinition",
    "HARDefinition",
    "HARFit",
    "StochasticVolatilityDefinition",
    "StochasticVolatilityFit",
    "fit_egarch",
    "fit_garch",
    "fit_gjr_garch",
    "fit_har",
    "fit_stochastic_volatility",
]
