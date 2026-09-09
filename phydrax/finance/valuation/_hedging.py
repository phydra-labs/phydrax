#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Self-financing hedge replay with explicit financing and transaction costs."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ..core._currency import Currency
from ..core._evidence import FinanceEvidenceBinding
from ..core._laws import PricingLaw


class HedgeReplayEvidence(StrictModule):
    finite: Array
    self_financing: Array
    maximum_self_financing_residual: Array
    terminal_replication_error: Array
    maximum_absolute_replication_error: Array
    cumulative_transaction_cost: Array
    binding: FinanceEvidenceBinding | None
    law_id: str = eqx.field(static=True)


class HedgeReplay(StrictModule):
    times: Array
    underlying_values: Array
    derivative_values: Array
    hedge_units: Array
    cash_account: Array
    portfolio_values: Array
    transaction_costs: Array
    replication_errors: Array
    financing_rate: Array
    transaction_cost_rate: Array
    currency: Currency | None = eqx.field(static=True)
    evidence: HedgeReplayEvidence
    replay_id: str = eqx.field(static=True)

    @property
    def terminal_error(self) -> Array:
        return self.replication_errors[-1]


def evaluate_hedge_replay(
    times: ArrayLike,
    underlying_values: ArrayLike,
    derivative_values: ArrayLike,
    hedge_units: ArrayLike,
    financing_rate: ArrayLike,
    /,
    *,
    transaction_cost_rate: ArrayLike = 0.0,
    initial_cash: ArrayLike | None = None,
    currency: Currency | None = None,
    evidence_binding: FinanceEvidenceBinding | None = None,
    pricing_law: PricingLaw | None = None,
    replay_id: str,
    tolerance: float = 1.0e-10,
) -> HedgeReplay:
    """Replay a predictable hedge; row ``i`` units are installed at row ``i`` price."""

    times_, spots, derivatives, units = tuple(
        jnp.asarray(value, dtype=float)
        for value in (times, underlying_values, derivative_values, hedge_units)
    )
    if (
        times_.ndim != 1
        or times_.size < 2
        or spots.shape != times_.shape
        or derivatives.shape != times_.shape
        or units.shape != times_.shape
    ):
        raise ValueError(
            "hedge replay arrays must be aligned vectors with at least two nodes."
        )
    rate = jnp.asarray(financing_rate, dtype=float)
    costs_rate = jnp.asarray(transaction_cost_rate, dtype=float)
    if rate.shape not in ((), (times_.size - 1,)) or costs_rate.shape not in (
        (),
        times_.shape,
    ):
        raise ValueError(
            "financing_rate must be scalar/interval-aligned and transaction_cost_rate scalar/node-aligned."
        )
    rate = jnp.broadcast_to(rate, (times_.size - 1,))
    costs_rate = jnp.broadcast_to(costs_rate, times_.shape)
    invalid = (
        jnp.any(~jnp.isfinite(times_))
        | jnp.any(jnp.diff(times_) <= 0.0)
        | jnp.any(~jnp.isfinite(spots))
        | jnp.any(spots <= 0.0)
        | jnp.any(~jnp.isfinite(derivatives))
        | jnp.any(~jnp.isfinite(units))
        | jnp.any(~jnp.isfinite(rate))
        | jnp.any(~jnp.isfinite(costs_rate))
        | jnp.any(costs_rate < 0.0)
    )
    spots = eqx.error_if(spots, invalid, "hedge replay inputs are invalid.")
    if not isinstance(replay_id, str) or not replay_id:
        raise ValueError("replay_id must be a non-empty string.")
    if currency is not None and not isinstance(currency, Currency):
        raise TypeError("currency must be Currency or None.")
    if evidence_binding is not None and not isinstance(
        evidence_binding, FinanceEvidenceBinding
    ):
        raise TypeError("evidence_binding must be FinanceEvidenceBinding or None.")
    if pricing_law is not None and not isinstance(pricing_law, PricingLaw):
        raise TypeError("pricing_law must be PricingLaw or None.")
    tolerance_ = jnp.asarray(tolerance, dtype=float)
    tolerance_ = eqx.error_if(
        tolerance_,
        ~jnp.isfinite(tolerance_) | (tolerance_ < 0.0),
        "tolerance must be finite and non-negative.",
    )
    initial_trade_cost = costs_rate[0] * jnp.abs(units[0]) * spots[0]
    cash0 = (
        derivatives[0] - units[0] * spots[0] - initial_trade_cost
        if initial_cash is None
        else jnp.asarray(initial_cash, dtype=float)
    )
    cash0 = eqx.error_if(cash0, ~jnp.isfinite(cash0), "initial cash must be finite.")

    def step(carry, data):
        previous_cash, previous_units = carry
        dt, interval_rate, spot, derivative, next_units, cost_rate = data
        accrued = previous_cash * jnp.exp(interval_rate * dt)
        trade = next_units - previous_units
        cost = cost_rate * jnp.abs(trade) * spot
        cash = accrued - trade * spot - cost
        portfolio = cash + next_units * spot
        error = portfolio - derivative
        self_financing_residual = cash - (accrued - trade * spot - cost)
        return (cash, next_units), (cash, portfolio, cost, error, self_financing_residual)

    (_, _), history = jax.lax.scan(
        step,
        (cash0, units[0]),
        (jnp.diff(times_), rate, spots[1:], derivatives[1:], units[1:], costs_rate[1:]),
    )
    cash_tail, portfolio_tail, cost_tail, error_tail, residual_tail = history
    portfolio0 = cash0 + units[0] * spots[0]
    cash = jnp.concatenate((cash0[None], cash_tail))
    portfolio = jnp.concatenate((portfolio0[None], portfolio_tail))
    transaction_costs = jnp.concatenate((initial_trade_cost[None], cost_tail))
    errors = jnp.concatenate(((portfolio0 - derivatives[0])[None], error_tail))
    residuals = jnp.concatenate((jnp.asarray(0.0, dtype=cash.dtype)[None], residual_tail))
    maximum_residual = jnp.max(jnp.abs(residuals))
    finite = (
        jnp.all(jnp.isfinite(cash))
        & jnp.all(jnp.isfinite(portfolio))
        & jnp.all(jnp.isfinite(errors))
    )
    self_financing = finite & (maximum_residual <= tolerance_)
    evidence = HedgeReplayEvidence(
        finite,
        self_financing,
        maximum_residual,
        errors[-1],
        jnp.max(jnp.abs(errors)),
        jnp.sum(transaction_costs),
        evidence_binding,
        "" if pricing_law is None else pricing_law.law_id,
    )
    return HedgeReplay(
        times_,
        spots,
        derivatives,
        units,
        cash,
        portfolio,
        transaction_costs,
        errors,
        rate,
        costs_rate,
        currency,
        evidence,
        replay_id,
    )


__all__ = ["HedgeReplay", "HedgeReplayEvidence", "evaluate_hedge_replay"]
