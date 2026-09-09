#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ..core import AssetReference, Currency
from ._ledger import PortfolioLedger
from ._objectives import (
    BlackLittermanObjective,
    CVaRObjective,
    DrawdownRiskObjective,
    EVaRObjective,
    FiniteScenarioKellyObjective,
    KLDivergenceRobustObjective,
    MeanVarianceObjective,
    SpectralRiskObjective,
    TrackingErrorObjective,
)
from ._problem import PortfolioProblem


def _array(value: ArrayLike, name: str, /, *, ndim: int) -> Array:
    result = jnp.asarray(value)
    if result.ndim != ndim or 0 in result.shape:
        raise ValueError(f"{name} must be a non-empty rank-{ndim} array.")
    if jnp.issubdtype(result.dtype, jnp.complexfloating):
        raise TypeError(f"{name} must be real-valued.")
    result = result.astype(jnp.result_type(result.dtype, jnp.float32))
    if not np.all(np.isfinite(np.asarray(result))):
        raise ValueError(f"{name} must be finite.")
    return result


class PortfolioReplayMarket(StrictModule):
    """Fixed-shape realized market and financing inputs for independent replay."""

    prices: Array
    fx_to_base: Array
    funding_rates: Array
    cash_borrow_rates: Array
    asset_borrow_rates: Array
    transaction_cost_rates: Array
    benchmark_returns: Array | None
    assets: tuple[AssetReference, ...] = eqx.field(static=True)
    currencies: tuple[Currency, ...] = eqx.field(static=True)
    asset_currency_index: tuple[int, ...] = eqx.field(static=True)
    base_currency_index: int = eqx.field(static=True)

    def __init__(
        self,
        assets: tuple[AssetReference, ...],
        currencies: tuple[Currency, ...],
        prices: ArrayLike,
        fx_to_base: ArrayLike,
        funding_rates: ArrayLike,
        cash_borrow_rates: ArrayLike,
        asset_borrow_rates: ArrayLike,
        transaction_cost_rates: ArrayLike,
        /,
        *,
        base_currency: Currency,
        benchmark_returns: ArrayLike | None = None,
    ):
        assets_ = tuple(assets)
        currencies_ = tuple(currencies)
        if not assets_ or any(not isinstance(item, AssetReference) for item in assets_):
            raise TypeError("assets must be a non-empty tuple of AssetReference values.")
        if not currencies_ or any(not isinstance(item, Currency) for item in currencies_):
            raise TypeError("currencies must be a non-empty tuple of Currency values.")
        asset_ids = tuple(item.asset_id for item in assets_)
        currency_ids = tuple(item.currency_id for item in currencies_)
        if len(set(asset_ids)) != len(asset_ids) or len(set(currency_ids)) != len(
            currency_ids
        ):
            raise ValueError("Replay assets and currencies must have unique identities.")
        if (
            not isinstance(base_currency, Currency)
            or base_currency.currency_id not in currency_ids
        ):
            raise ValueError("base_currency must be present in currencies.")
        prices_ = _array(prices, "prices", ndim=2)
        fx = _array(fx_to_base, "fx_to_base", ndim=2).astype(prices_.dtype)
        times, asset_count = prices_.shape
        currency_count = len(currencies_)
        if asset_count != len(assets_) or fx.shape != (times, currency_count):
            raise ValueError("Price and FX axes must match assets, currencies, and time.")
        if np.any(np.asarray(prices_) <= 0.0) or np.any(np.asarray(fx) <= 0.0):
            raise ValueError("Prices and FX rates must be strictly positive.")
        rate_shape_cash = (max(times - 1, 0), currency_count)
        rate_shape_asset = (max(times - 1, 0), asset_count)
        funding = jnp.asarray(funding_rates, dtype=prices_.dtype)
        cash_borrow = jnp.asarray(cash_borrow_rates, dtype=prices_.dtype)
        asset_borrow = jnp.asarray(asset_borrow_rates, dtype=prices_.dtype)
        costs = _array(transaction_cost_rates, "transaction_cost_rates", ndim=2).astype(
            prices_.dtype
        )
        if funding.shape != rate_shape_cash or cash_borrow.shape != rate_shape_cash:
            raise ValueError(f"Cash-rate inputs must both have shape {rate_shape_cash}.")
        if asset_borrow.shape != rate_shape_asset:
            raise ValueError(f"asset_borrow_rates must have shape {rate_shape_asset}.")
        if costs.shape != prices_.shape:
            raise ValueError("transaction_cost_rates must match prices.")
        rate_arrays = (funding, cash_borrow, asset_borrow, costs)
        if any(not np.all(np.isfinite(np.asarray(value))) for value in rate_arrays):
            raise ValueError("Replay rates must be finite.")
        if any(np.any(np.asarray(value) < 0.0) for value in rate_arrays):
            raise ValueError("Replay rates must be non-negative.")
        base_index = currency_ids.index(base_currency.currency_id)
        tolerance = 64.0 * np.finfo(np.asarray(fx).dtype).eps
        if np.max(np.abs(np.asarray(fx[:, base_index]) - 1.0)) > tolerance:
            raise ValueError("The base-currency FX series must equal one.")
        asset_currency = tuple(
            currency_ids.index(asset.currency.currency_id)
            if asset.currency.currency_id in currency_ids
            else -1
            for asset in assets_
        )
        if any(index < 0 for index in asset_currency):
            raise ValueError("Every asset currency must be present in currencies.")
        benchmark = (
            None
            if benchmark_returns is None
            else _array(benchmark_returns, "benchmark_returns", ndim=1).astype(
                prices_.dtype
            )
        )
        if benchmark is not None and benchmark.shape != (max(times - 1, 0),):
            raise ValueError("benchmark_returns must have one entry per realized period.")
        self.prices, self.fx_to_base = prices_, fx
        self.funding_rates, self.cash_borrow_rates = funding, cash_borrow
        self.asset_borrow_rates, self.transaction_cost_rates = asset_borrow, costs
        self.benchmark_returns = benchmark
        self.assets, self.currencies = assets_, currencies_
        self.asset_currency_index = asset_currency
        self.base_currency_index = base_index


class ReplayCostInputs(StrictModule):
    """Caller-supplied realized commissions and tax costs, in trade currencies."""

    commissions: Array
    tax_costs: Array

    def __init__(self, commissions: ArrayLike, tax_costs: ArrayLike, /):
        commission = _array(commissions, "commissions", ndim=1)
        taxes = _array(tax_costs, "tax_costs", ndim=1).astype(commission.dtype)
        if taxes.shape != commission.shape:
            raise ValueError("commissions and tax_costs must have identical shapes.")
        if np.any(np.asarray(commission) < 0.0) or np.any(np.asarray(taxes) < 0.0):
            raise ValueError("Realized costs must be non-negative.")
        self.commissions, self.tax_costs = commission, taxes


class RealizedPortfolioResult(StrictModule):
    """Independent accounting replay; contains no forecast or optimizer certificate."""

    net_asset_value: Array
    period_returns: Array
    holdings: Array
    cash: Array
    unsettled_payables: Array
    weights: Array
    market_and_fx_pnl: Array
    funding_pnl: Array
    asset_borrow_cost: Array
    transaction_cost: Array
    tax_cost: Array
    constraint_violation: Array
    self_financing_residual: Array
    realized_objective: Array
    realized_var: Array
    realized_expected_shortfall: Array
    valid: Array
    ledger_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)


def _weighted_tail(
    losses: np.ndarray, probabilities: np.ndarray, confidence: float, /
) -> tuple[float, float]:
    order = np.argsort(losses, kind="stable")
    sorted_losses, sorted_probabilities = losses[order], probabilities[order]
    cumulative = np.cumsum(sorted_probabilities)
    index = min(
        int(np.searchsorted(cumulative, confidence, side="left")), losses.size - 1
    )
    value_at_risk = float(sorted_losses[index])
    previous = np.concatenate((np.asarray((0.0,)), cumulative[:-1]))
    mass = np.maximum(np.minimum(cumulative, 1.0) - np.maximum(previous, confidence), 0.0)
    expected_shortfall = float(np.sum(mass * sorted_losses) / (1.0 - confidence))
    return value_at_risk, expected_shortfall


def _entropic_risk(losses: np.ndarray, radius: float, /) -> float:
    spread = float(np.max(losses) - np.min(losses))
    if spread == 0.0:
        return float(losses[0])
    probabilities = np.full(losses.shape, 1.0 / losses.size)
    logarithms = np.log(probabilities)
    lower = max(np.finfo(losses.dtype).eps * spread, np.finfo(losses.dtype).tiny)
    upper = spread * 1e6
    for _ in range(80):
        tau = np.sqrt(lower * upper)
        shifted = losses / tau + logarithms
        maximum = float(np.max(shifted))
        log_partition = maximum + float(np.log(np.sum(np.exp(shifted - maximum))))
        tilted = np.exp(shifted - maximum)
        tilted /= np.sum(tilted)
        divergence = float(np.sum(tilted * (np.log(tilted) - logarithms)))
        if divergence > radius:
            lower = tau
        else:
            upper = tau
    tau = upper
    shifted = losses / tau + logarithms
    maximum = float(np.max(shifted))
    return tau * (radius + maximum + float(np.log(np.sum(np.exp(shifted - maximum)))))


def _constraint_violation(
    problem: PortfolioProblem, weights: np.ndarray, /
) -> np.ndarray:
    constraints = problem.constraints
    times = weights.shape[0]
    result = np.zeros((times,), dtype=weights.dtype)
    result = np.maximum(result, np.abs(np.sum(weights, axis=1) - constraints.budget))
    if constraints.lower_weights is not None:
        result = np.maximum(
            result,
            np.max(
                np.maximum(np.asarray(constraints.lower_weights)[None, :] - weights, 0.0),
                axis=1,
            ),
        )
    if constraints.upper_weights is not None:
        result = np.maximum(
            result,
            np.max(
                np.maximum(weights - np.asarray(constraints.upper_weights)[None, :], 0.0),
                axis=1,
            ),
        )
    if constraints.linear_matrix is not None:
        exposure = weights @ np.asarray(constraints.linear_matrix).T
        result = np.maximum(
            result,
            np.max(
                np.maximum(np.asarray(constraints.linear_lower)[None, :] - exposure, 0.0),
                axis=1,
            ),
        )
        result = np.maximum(
            result,
            np.max(
                np.maximum(exposure - np.asarray(constraints.linear_upper)[None, :], 0.0),
                axis=1,
            ),
        )
    if constraints.gross_limit is not None:
        result = np.maximum(
            result,
            np.maximum(np.sum(np.abs(weights), axis=1) - constraints.gross_limit, 0.0),
        )
    if constraints.turnover_limit is not None:
        initial = np.asarray(problem.current_weights)
        previous = np.concatenate((initial[None, :], weights[:-1]), axis=0)
        result = np.maximum(
            result,
            np.maximum(
                np.sum(np.abs(weights - previous), axis=1) - constraints.turnover_limit,
                0.0,
            ),
        )
    if constraints.lot_sizes is not None:
        ratios = weights / np.asarray(constraints.lot_sizes)[None, :]
        result = np.maximum(result, np.max(np.abs(ratios - np.rint(ratios)), axis=1))
    if constraints.maximum_cardinality is not None:
        count = np.sum(np.abs(weights) > 1e-10, axis=1)
        result = np.maximum(
            result, np.maximum(count - constraints.maximum_cardinality, 0.0)
        )
    for robust in constraints.robust:
        uncertain = np.sqrt(
            np.sum((weights @ np.asarray(robust.factor_loading)) ** 2, axis=1)
        )
        value = weights @ np.asarray(robust.nominal) + robust.radius * uncertain
        result = np.maximum(result, np.maximum(value - robust.bound, 0.0))
    return result


def _realized_objective(
    problem: PortfolioProblem,
    returns: np.ndarray,
    nav: np.ndarray,
    market: PortfolioReplayMarket,
    /,
) -> float:
    objective = problem.objective
    mean = float(np.mean(returns)) if returns.size else 0.0
    variance = float(np.var(returns)) if returns.size else 0.0
    if isinstance(objective, MeanVarianceObjective):
        return objective.risk_aversion * variance - objective.return_weight * mean
    if isinstance(objective, TrackingErrorObjective):
        if market.benchmark_returns is None:
            raise ValueError("Tracking replay requires benchmark_returns.")
        active = returns - np.asarray(market.benchmark_returns)
        return (
            objective.tracking_aversion * float(np.mean(active * active))
            - objective.return_weight * mean
        )
    if isinstance(objective, BlackLittermanObjective):
        return objective.risk_aversion * variance - mean
    if isinstance(objective, FiniteScenarioKellyObjective):
        if np.any(nav <= objective.bankruptcy_floor):
            return float("inf")
        return -float(np.mean(np.log1p(returns))) if returns.size else 0.0
    losses = -returns
    if isinstance(objective, CVaRObjective):
        probabilities = np.full(losses.shape, 1.0 / losses.size)
        _, tail = _weighted_tail(losses, probabilities, objective.confidence)
        return objective.risk_weight * tail - objective.return_weight * mean
    if isinstance(objective, SpectralRiskObjective):
        probabilities = np.full(losses.shape, 1.0 / losses.size)
        risk = 0.0
        for confidence, weight in zip(
            np.asarray(objective.confidences), np.asarray(objective.weights), strict=True
        ):
            _, tail = _weighted_tail(losses, probabilities, float(confidence))
            risk += float(weight) * tail
        return objective.risk_weight * risk - objective.return_weight * mean
    if isinstance(objective, (EVaRObjective, KLDivergenceRobustObjective)):
        radius = (
            objective.relative_entropy_radius
            if isinstance(objective, EVaRObjective)
            else objective.radius
        )
        return (
            objective.risk_weight * _entropic_risk(losses, radius)
            - objective.return_weight * mean
        )
    if isinstance(objective, DrawdownRiskObjective):
        peak = np.maximum.accumulate(nav)
        drawdown = np.max((peak - nav) / np.maximum(peak, np.finfo(nav.dtype).tiny))
        return objective.risk_weight * float(drawdown) - objective.return_weight * mean
    raise TypeError("Unsupported realized objective.")


def replay_self_financing(
    ledger: PortfolioLedger,
    market: PortfolioReplayMarket,
    problem: PortfolioProblem,
    costs: ReplayCostInputs,
    /,
    *,
    risk_confidence: float = 0.95,
) -> RealizedPortfolioResult:
    """Replay realized accounting without consulting compiled optimization matrices."""

    if not isinstance(ledger, PortfolioLedger) or not isinstance(
        market, PortfolioReplayMarket
    ):
        raise TypeError("ledger and market have incorrect types.")
    if not isinstance(problem, PortfolioProblem) or not isinstance(
        costs, ReplayCostInputs
    ):
        raise TypeError("problem and costs have incorrect types.")
    confidence = float(risk_confidence)
    if not isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("risk_confidence must lie in (0, 1).")
    trades = ledger.trades
    if costs.commissions.shape != (len(trades),) or costs.tax_costs.shape != (
        len(trades),
    ):
        raise ValueError("Replay costs must have one entry per ledger trade.")
    asset_ids = tuple(asset.asset_id for asset in market.assets)
    if problem.forecast.asset_ids != asset_ids:
        raise ValueError("Problem forecast and replay asset ordering must match exactly.")
    currency_ids = tuple(currency.currency_id for currency in market.currencies)
    asset_index = {identifier: index for index, identifier in enumerate(asset_ids)}
    currency_index = {identifier: index for index, identifier in enumerate(currency_ids)}
    times, asset_count = market.prices.shape
    if any(trade.asset.asset_id not in asset_index for trade in trades):
        raise ValueError("Every ledger trade asset must be present in the replay market.")
    if any(
        trade.execution_index >= times or trade.settlement_index >= times
        for trade in trades
    ):
        raise ValueError(
            "Every trade execution and settlement must lie in the replay horizon."
        )
    quantities = np.zeros((asset_count,), dtype=np.asarray(market.prices).dtype)
    for holding in ledger.initial_holdings:
        if holding.asset.asset_id not in asset_index:
            raise ValueError(
                "Every initial holding must be present in the replay market."
            )
        quantities[asset_index[holding.asset.asset_id]] += float(
            np.asarray(holding.quantity)
        )
    cash = np.zeros((len(market.currencies),), dtype=quantities.dtype)
    for balance in ledger.initial_cash:
        key = balance.currency.currency_id
        if key not in currency_index:
            raise ValueError(
                "Every initial cash currency must be present in the replay market."
            )
        cash[currency_index[key]] += (
            int(np.asarray(balance.amount.atoms)) / balance.currency.atoms_per_unit
        )
    pending = np.zeros_like(cash)
    liabilities = np.zeros((len(trades),), dtype=quantities.dtype)
    holdings_history = np.zeros((times, asset_count), dtype=quantities.dtype)
    cash_history = np.zeros((times, cash.size), dtype=quantities.dtype)
    pending_history = np.zeros_like(cash_history)
    weight_history = np.zeros((times, asset_count), dtype=quantities.dtype)
    nav = np.zeros((times,), dtype=quantities.dtype)
    revaluation = np.zeros_like(nav)
    funding_pnl = np.zeros_like(nav)
    borrow_cost = np.zeros_like(nav)
    transaction_cost = np.zeros_like(nav)
    tax_cost = np.zeros_like(nav)
    residual = np.zeros_like(nav)
    prices = np.asarray(market.prices)
    fx = np.asarray(market.fx_to_base)
    asset_currency = np.asarray(market.asset_currency_index, dtype=np.int64)
    trade_by_execution: list[list[int]] = [[] for _ in range(times)]
    trade_by_settlement: list[list[int]] = [[] for _ in range(times)]
    for index, trade in enumerate(trades):
        trade_by_execution[trade.execution_index].append(index)
        trade_by_settlement[trade.settlement_index].append(index)
    previous_nav = None
    for time in range(times):
        current_asset_base = prices[time] * fx[time, asset_currency]
        pre_event_nav = float(
            np.sum(quantities * current_asset_base) + np.sum((cash - pending) * fx[time])
        )
        if previous_nav is None:
            previous_nav = pre_event_nav
            revaluation[time] = 0.0
        else:
            revaluation[time] = pre_event_nav - previous_nav
        accrual_local = np.zeros_like(cash)
        if time > 0:
            positive = np.maximum(cash, 0.0) * np.asarray(market.funding_rates)[time - 1]
            negative = (
                np.minimum(cash, 0.0) * np.asarray(market.cash_borrow_rates)[time - 1]
            )
            accrual_local = positive + negative
            cash += accrual_local
            short_cost_local = np.zeros_like(cash)
            for asset in range(asset_count):
                cost = (
                    max(-quantities[asset], 0.0)
                    * prices[time, asset]
                    * float(np.asarray(market.asset_borrow_rates)[time - 1, asset])
                )
                short_cost_local[asset_currency[asset]] += cost
            cash -= short_cost_local
            funding_pnl[time] = float(np.sum(accrual_local * fx[time]))
            borrow_cost[time] = float(np.sum(short_cost_local * fx[time]))
        for trade_index in trade_by_execution[time]:
            trade = trades[trade_index]
            asset = asset_index[trade.asset.asset_id]
            currency = asset_currency[asset]
            quantity = float(np.asarray(trade.quantity))
            impact = (
                abs(quantity)
                * prices[time, asset]
                * float(np.asarray(market.transaction_cost_rates)[time, asset])
            )
            commission = float(np.asarray(costs.commissions)[trade_index])
            tax = float(np.asarray(costs.tax_costs)[trade_index])
            economic_cost = impact + commission
            liability = quantity * prices[time, asset] + economic_cost + tax
            liabilities[trade_index] = liability
            quantities[asset] += quantity
            pending[currency] += liability
            transaction_cost[time] += (economic_cost) * fx[time, currency]
            tax_cost[time] += tax * fx[time, currency]
        for trade_index in trade_by_settlement[time]:
            trade = trades[trade_index]
            asset = asset_index[trade.asset.asset_id]
            currency = asset_currency[asset]
            cash[currency] -= liabilities[trade_index]
            pending[currency] -= liabilities[trade_index]
        nav[time] = float(
            np.sum(quantities * current_asset_base) + np.sum((cash - pending) * fx[time])
        )
        expected = (
            previous_nav
            + revaluation[time]
            + funding_pnl[time]
            - borrow_cost[time]
            - transaction_cost[time]
            - tax_cost[time]
        )
        residual[time] = nav[time] - expected
        holdings_history[time], cash_history[time], pending_history[time] = (
            quantities,
            cash,
            pending,
        )
        asset_values = quantities * current_asset_base
        if nav[time] != 0.0:
            weight_history[time] = asset_values / nav[time]
        previous_nav = nav[time]
    if not np.isfinite(nav[0]) or nav[0] <= 0.0:
        raise ValueError(
            "Replay requires strictly positive initial post-event net asset value."
        )
    period_returns = nav[1:] / nav[:-1] - 1.0
    probabilities = (
        np.full(period_returns.shape, 1.0 / period_returns.size)
        if period_returns.size
        else np.empty((0,), dtype=nav.dtype)
    )
    if period_returns.size:
        value_at_risk, expected_shortfall = _weighted_tail(
            -period_returns, probabilities, confidence
        )
    else:
        value_at_risk, expected_shortfall = 0.0, 0.0
    violations = _constraint_violation(problem, weight_history)
    objective = _realized_objective(problem, period_returns, nav, market)
    tolerance = 256.0 * np.finfo(nav.dtype).eps * np.maximum(np.abs(nav), 1.0)
    valid = np.all(np.isfinite(nav)) and np.all(np.abs(residual) <= tolerance)
    return RealizedPortfolioResult(
        net_asset_value=jnp.asarray(nav),
        period_returns=jnp.asarray(period_returns),
        holdings=jnp.asarray(holdings_history),
        cash=jnp.asarray(cash_history),
        unsettled_payables=jnp.asarray(pending_history),
        weights=jnp.asarray(weight_history),
        market_and_fx_pnl=jnp.asarray(revaluation),
        funding_pnl=jnp.asarray(funding_pnl),
        asset_borrow_cost=jnp.asarray(borrow_cost),
        transaction_cost=jnp.asarray(transaction_cost),
        tax_cost=jnp.asarray(tax_cost),
        constraint_violation=jnp.asarray(violations),
        self_financing_residual=jnp.asarray(residual),
        realized_objective=jnp.asarray(objective),
        realized_var=jnp.asarray(value_at_risk),
        realized_expected_shortfall=jnp.asarray(expected_shortfall),
        valid=jnp.asarray(valid),
        ledger_id=ledger.ledger_id,
        problem_id=problem.problem_id,
    )


__all__ = [
    "PortfolioReplayMarket",
    "RealizedPortfolioResult",
    "ReplayCostInputs",
    "replay_self_financing",
]
