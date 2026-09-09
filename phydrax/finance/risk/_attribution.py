#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ..core import StressLaw


class StressTestResult(StrictModule):
    scenario_pnl: Array
    worst_pnl: Array
    worst_scenario: Array
    breached: Array
    loss_limit: Array
    law: StressLaw = eqx.field(static=True)
    scenario_ids: tuple[str, ...] = eqx.field(static=True)


class PnLExplanation(StrictModule):
    market_pnl: Array
    fx_pnl: Array
    trading_pnl: Array
    cost_pnl: Array
    total_pnl: Array
    explained_pnl: Array
    residual: Array


class AttributionResult(StrictModule):
    allocation: Array
    selection: Array
    interaction: Array
    group_total: Array
    active_return: Array
    explained_return: Array
    residual: Array
    group_ids: tuple[str, ...] = eqx.field(static=True)


def stress_test(
    position_values: ArrayLike,
    relative_shocks: ArrayLike,
    scenario_ids: tuple[str, ...],
    law: StressLaw,
    /,
    *,
    loss_limit: float = np.inf,
) -> StressTestResult:
    """Apply a stress law without substituting statistical or pricing probabilities."""

    if not isinstance(law, StressLaw):
        raise TypeError("stress_test requires a StressLaw.")
    raw_positions = jnp.asarray(position_values)
    raw_shocks = jnp.asarray(relative_shocks)
    if jnp.issubdtype(raw_positions.dtype, jnp.complexfloating) or jnp.issubdtype(
        raw_shocks.dtype, jnp.complexfloating
    ):
        raise TypeError("Stress inputs must be real-valued.")
    dtype = jnp.result_type(raw_positions.dtype, raw_shocks.dtype, jnp.float32)
    positions = raw_positions.astype(dtype)
    shocks = raw_shocks.astype(dtype)
    if positions.ndim != 1 or positions.shape[0] == 0:
        raise ValueError("position_values must be a non-empty vector.")
    if shocks.ndim != 2 or shocks.shape[1] != positions.shape[0] or shocks.shape[0] == 0:
        raise ValueError("relative_shocks must have shape (scenario, position).")
    if not np.all(np.isfinite(np.asarray(positions))) or not np.all(
        np.isfinite(np.asarray(shocks))
    ):
        raise ValueError("Stress inputs must be finite.")
    identifiers = tuple(str(value) for value in scenario_ids)
    if (
        len(identifiers) != shocks.shape[0]
        or any(not value for value in identifiers)
        or len(set(identifiers)) != len(identifiers)
    ):
        raise ValueError("scenario_ids must uniquely identify every stress row.")
    limit = float(loss_limit)
    if np.isnan(limit) or limit <= 0.0:
        raise ValueError("loss_limit must be positive or positive infinity.")
    pnl = shocks @ positions
    worst_index = jnp.argmin(pnl)
    worst = pnl[worst_index]
    return StressTestResult(
        scenario_pnl=pnl,
        worst_pnl=worst,
        worst_scenario=worst_index.astype(jnp.int32),
        breached=-worst > limit,
        loss_limit=jnp.asarray(limit, dtype=pnl.dtype),
        law=law,
        scenario_ids=identifiers,
    )


def explain_pnl(
    start_quantities: ArrayLike,
    end_quantities: ArrayLike,
    start_prices: ArrayLike,
    end_prices: ArrayLike,
    asset_currency_index: ArrayLike,
    start_cash: ArrayLike,
    end_cash: ArrayLike,
    start_fx_to_base: ArrayLike,
    end_fx_to_base: ArrayLike,
    realized_costs_base: ArrayLike,
    /,
) -> PnLExplanation:
    """Reconcile market, FX, trading/cash-flow, and explicit cost PnL exactly."""

    start_q = np.asarray(start_quantities)
    end_q = np.asarray(end_quantities)
    start_p = np.asarray(start_prices)
    end_p = np.asarray(end_prices)
    currency = np.asarray(asset_currency_index)
    start_cash_ = np.asarray(start_cash)
    end_cash_ = np.asarray(end_cash)
    start_fx = np.asarray(start_fx_to_base)
    end_fx = np.asarray(end_fx_to_base)
    if (
        start_q.ndim != 1
        or start_q.size == 0
        or end_q.shape != start_q.shape
        or start_p.shape != start_q.shape
        or end_p.shape != start_q.shape
    ):
        raise ValueError(
            "Quantity and price inputs must be same-sized non-empty vectors."
        )
    if currency.shape != start_q.shape or not np.issubdtype(currency.dtype, np.integer):
        raise TypeError("asset_currency_index must be one integer per asset.")
    if (
        start_cash_.ndim != 1
        or end_cash_.shape != start_cash_.shape
        or start_fx.shape != start_cash_.shape
        or end_fx.shape != start_cash_.shape
    ):
        raise ValueError("Cash and FX inputs must share one currency-vector shape.")
    if np.any(currency < 0) or np.any(currency >= start_cash_.size):
        raise ValueError("asset_currency_index is out of range.")
    inputs = (start_q, end_q, start_p, end_p, start_cash_, end_cash_, start_fx, end_fx)
    if any(not np.all(np.isfinite(value)) for value in inputs):
        raise ValueError("PnL explain inputs must be finite.")
    costs = float(np.asarray(realized_costs_base))
    if not np.isfinite(costs) or costs < 0.0:
        raise ValueError("realized_costs_base must be finite and non-negative.")
    asset_start_fx = start_fx[currency]
    asset_end_fx = end_fx[currency]
    start_value = float(
        np.sum(start_q * start_p * asset_start_fx) + np.sum(start_cash_ * start_fx)
    )
    end_value = float(np.sum(end_q * end_p * asset_end_fx) + np.sum(end_cash_ * end_fx))
    market = float(np.sum(start_q * (end_p - start_p) * asset_start_fx))
    fx = float(
        np.sum(start_q * end_p * (asset_end_fx - asset_start_fx))
        + np.sum(start_cash_ * (end_fx - start_fx))
    )
    trading_net = float(
        np.sum((end_q - start_q) * end_p * asset_end_fx)
        + np.sum((end_cash_ - start_cash_) * end_fx)
    )
    trading_gross = trading_net + costs
    cost_pnl = -costs
    total = end_value - start_value
    explained = market + fx + trading_gross + cost_pnl
    return PnLExplanation(
        market_pnl=jnp.asarray(market),
        fx_pnl=jnp.asarray(fx),
        trading_pnl=jnp.asarray(trading_gross),
        cost_pnl=jnp.asarray(cost_pnl),
        total_pnl=jnp.asarray(total),
        explained_pnl=jnp.asarray(explained),
        residual=jnp.asarray(total - explained),
    )


def brinson_attribution(
    portfolio_weights: ArrayLike,
    benchmark_weights: ArrayLike,
    asset_returns: ArrayLike,
    group_labels: ArrayLike,
    group_ids: tuple[str, ...],
    /,
) -> AttributionResult:
    """Brinson--Fachler allocation/selection/interaction with explicit reconciliation."""

    portfolio = np.asarray(portfolio_weights)
    benchmark = np.asarray(benchmark_weights)
    returns = np.asarray(asset_returns)
    labels = np.asarray(group_labels)
    if (
        portfolio.ndim != 1
        or portfolio.size == 0
        or benchmark.shape != portfolio.shape
        or returns.shape != portfolio.shape
    ):
        raise ValueError("Weights and returns must be same-sized non-empty vectors.")
    if labels.shape != portfolio.shape or not np.issubdtype(labels.dtype, np.integer):
        raise TypeError("group_labels must be one integer per asset.")
    if any(not np.all(np.isfinite(value)) for value in (portfolio, benchmark, returns)):
        raise ValueError("Attribution inputs must be finite.")
    identifiers = tuple(str(value) for value in group_ids)
    groups = len(identifiers)
    if (
        groups == 0
        or any(not value for value in identifiers)
        or len(set(identifiers)) != groups
    ):
        raise ValueError("group_ids must be non-empty and unique.")
    if np.any(labels < 0) or np.any(labels >= groups):
        raise ValueError("group_labels are out of range.")
    benchmark_total = float(benchmark @ returns)
    allocation = np.zeros((groups,), dtype=np.result_type(portfolio, benchmark, returns))
    selection = np.zeros_like(allocation)
    interaction = np.zeros_like(allocation)
    for group in range(groups):
        members = labels == group
        wp, wb = float(np.sum(portfolio[members])), float(np.sum(benchmark[members]))
        rp = float(portfolio[members] @ returns[members] / wp) if wp != 0.0 else 0.0
        rb = float(benchmark[members] @ returns[members] / wb) if wb != 0.0 else 0.0
        allocation[group] = (wp - wb) * (rb - benchmark_total)
        selection[group] = wb * (rp - rb)
        interaction[group] = (wp - wb) * (rp - rb)
    group_total = allocation + selection + interaction
    active = float(portfolio @ returns - benchmark_total)
    explained = float(np.sum(group_total))
    return AttributionResult(
        allocation=jnp.asarray(allocation),
        selection=jnp.asarray(selection),
        interaction=jnp.asarray(interaction),
        group_total=jnp.asarray(group_total),
        active_return=jnp.asarray(active),
        explained_return=jnp.asarray(explained),
        residual=jnp.asarray(active - explained),
        group_ids=identifiers,
    )


__all__ = [
    "AttributionResult",
    "PnLExplanation",
    "StressTestResult",
    "brinson_attribution",
    "explain_pnl",
    "stress_test",
]
