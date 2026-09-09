#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ..core import PhysicalLaw
from ..market import MarketState, RiskFactorLayout


class FactorRiskModel(StrictModule):
    """Measure-neutral factor covariance structure bound to one physical risk law."""

    asset_factor_loading: Array
    factor_covariance: Array
    specific_variance: Array
    layout: RiskFactorLayout = eqx.field(static=True)
    law: PhysicalLaw = eqx.field(static=True)
    asset_ids: tuple[str, ...] = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        asset_ids: tuple[str, ...],
        layout: RiskFactorLayout,
        asset_factor_loading: ArrayLike,
        factor_covariance: ArrayLike,
        specific_variance: ArrayLike,
        law: PhysicalLaw,
        /,
        *,
        model_id: str,
    ):
        assets = tuple(str(value) for value in asset_ids)
        if (
            not assets
            or any(not value for value in assets)
            or len(set(assets)) != len(assets)
        ):
            raise ValueError("asset_ids must be non-empty and unique.")
        if not isinstance(layout, RiskFactorLayout):
            raise TypeError("layout must be a RiskFactorLayout.")
        if not isinstance(law, PhysicalLaw):
            raise TypeError("Factor risk requires a PhysicalLaw.")
        if law.factor_layout_id != layout.layout_id:
            raise ValueError("Physical law and factor layout are incompatible.")
        loading = jnp.asarray(asset_factor_loading)
        covariance = jnp.asarray(factor_covariance)
        specific = jnp.asarray(specific_variance)
        expected_loading = (len(assets), layout.factor_count)
        if loading.shape != expected_loading:
            raise ValueError(f"asset_factor_loading must have shape {expected_loading}.")
        if covariance.shape != (layout.factor_count, layout.factor_count):
            raise ValueError("factor_covariance must be square on the factor layout.")
        if specific.shape != (len(assets),):
            raise ValueError("specific_variance must have one entry per asset.")
        dtype = jnp.result_type(loading, covariance, specific, jnp.float32)
        if not jnp.issubdtype(dtype, jnp.floating):
            raise TypeError("Factor risk arrays must be real-valued.")
        loading, covariance, specific = (
            loading.astype(dtype),
            covariance.astype(dtype),
            specific.astype(dtype),
        )
        host_covariance = np.asarray(covariance)
        if (
            not np.all(np.isfinite(np.asarray(loading)))
            or not np.all(np.isfinite(host_covariance))
            or not np.all(np.isfinite(np.asarray(specific)))
        ):
            raise ValueError("Factor risk arrays must be finite.")
        tolerance = (
            64.0
            * np.finfo(host_covariance.dtype).eps
            * max(float(np.max(np.abs(host_covariance))), 1.0)
        )
        if np.max(np.abs(host_covariance - host_covariance.T)) > tolerance:
            raise ValueError("factor_covariance must be symmetric.")
        symmetric = 0.5 * host_covariance + 0.5 * host_covariance.T
        if np.min(np.linalg.eigvalsh(symmetric)) < -tolerance or np.any(
            np.asarray(specific) < 0.0
        ):
            raise ValueError(
                "Factor and specific variances must be positive semidefinite."
            )
        identifier = str(model_id)
        if not identifier:
            raise ValueError("model_id must be non-empty.")
        self.asset_factor_loading = loading
        self.factor_covariance = jnp.asarray(symmetric, dtype=dtype)
        self.specific_variance = specific
        self.layout, self.law = layout, law
        self.asset_ids, self.model_id = assets, identifier

    @property
    def asset_covariance(self) -> Array:
        return (
            self.asset_factor_loading
            @ self.factor_covariance
            @ self.asset_factor_loading.T
            + jnp.diag(self.specific_variance)
        )


class MarketRiskReport(StrictModule):
    portfolio_factor_exposure: Array
    factor_variance: Array
    specific_variance: Array
    total_variance: Array
    volatility: Array
    marginal_volatility: Array
    component_volatility: Array
    factor_component_variance: Array
    law: PhysicalLaw = eqx.field(static=True)
    model_id: str = eqx.field(static=True)


class FactorPnL(StrictModule):
    factor_changes: Array
    factor_contributions: Array
    total_pnl: Array
    start_state_id: str = eqx.field(static=True)
    end_state_id: str = eqx.field(static=True)


def market_factor_risk(
    weights: ArrayLike,
    model: FactorRiskModel,
    /,
) -> MarketRiskReport:
    """Compute Euler-reconciled market and factor risk for one portfolio."""

    if not isinstance(model, FactorRiskModel):
        raise TypeError("model must be a FactorRiskModel.")
    weight = jnp.asarray(weights, dtype=model.asset_factor_loading.dtype)
    if weight.shape != (len(model.asset_ids),) or not np.all(
        np.isfinite(np.asarray(weight))
    ):
        raise ValueError("weights must be one finite value per model asset.")
    exposure = weight @ model.asset_factor_loading
    factor_marginal = model.factor_covariance @ exposure
    factor_variance = exposure @ factor_marginal
    specific_variance = jnp.sum(weight * weight * model.specific_variance)
    total = factor_variance + specific_variance
    volatility = jnp.sqrt(jnp.maximum(total, 0.0))
    covariance_weight = model.asset_covariance @ weight
    safe_volatility = jnp.where(volatility > 0.0, volatility, 1.0)
    marginal = jnp.where(volatility > 0.0, covariance_weight / safe_volatility, 0.0)
    component = weight * marginal
    factor_component = exposure * factor_marginal
    return MarketRiskReport(
        portfolio_factor_exposure=exposure,
        factor_variance=factor_variance,
        specific_variance=specific_variance,
        total_variance=total,
        volatility=volatility,
        marginal_volatility=marginal,
        component_volatility=component,
        factor_component_variance=factor_component,
        law=model.law,
        model_id=model.model_id,
    )


def explain_factor_pnl(
    start: MarketState,
    end: MarketState,
    factor_exposure: ArrayLike,
    /,
) -> FactorPnL:
    """Explain linear factor PnL only from accepted, identically ordered states."""

    if not isinstance(start, MarketState) or not isinstance(end, MarketState):
        raise TypeError("start and end must be MarketState values.")
    if start.layout.layout_id != end.layout.layout_id:
        raise ValueError("Market states must use the same factor layout.")
    if not bool(np.asarray(start.successful)) or not bool(np.asarray(end.successful)):
        raise ValueError("Factor PnL requires accepted market states.")
    exposure = jnp.asarray(factor_exposure, dtype=start.values.dtype)
    if exposure.shape != start.values.shape or not np.all(
        np.isfinite(np.asarray(exposure))
    ):
        raise ValueError("factor_exposure must match the market factor layout.")
    changes = end.values - start.values
    contributions = exposure * changes
    return FactorPnL(
        factor_changes=changes,
        factor_contributions=contributions,
        total_pnl=jnp.sum(contributions),
        start_state_id=start.state_id,
        end_state_id=end.state_id,
    )


__all__ = [
    "FactorPnL",
    "FactorRiskModel",
    "MarketRiskReport",
    "explain_factor_pnl",
    "market_factor_risk",
]
