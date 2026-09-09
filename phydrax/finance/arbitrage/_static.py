#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule


class OptionCallSlice(StrictModule):
    """One fixed-maturity call-price slice in an explicit numeraire convention."""

    strikes: Array
    call_prices: Array
    forward: Array
    discount_factor: Array
    maturity: float = eqx.field(static=True)
    asset_id: str = eqx.field(static=True)
    currency_code: str = eqx.field(static=True)
    numeraire_id: str = eqx.field(static=True)
    market_snapshot_id: str = eqx.field(static=True)

    def __init__(
        self,
        strikes: ArrayLike,
        call_prices: ArrayLike,
        forward: ArrayLike,
        discount_factor: ArrayLike,
        /,
        *,
        maturity: float,
        asset_id: str,
        currency_code: str,
        numeraire_id: str,
        market_snapshot_id: str,
    ):
        strikes_ = jnp.asarray(strikes, dtype=float)
        calls = jnp.asarray(call_prices, dtype=float)
        forward_ = jnp.asarray(forward, dtype=float)
        discount = jnp.asarray(discount_factor, dtype=float)
        if strikes_.ndim != 1 or strikes_.shape[0] < 2 or calls.shape != strikes_.shape:
            raise ValueError(
                "An option call slice needs at least two strike/price pairs."
            )
        if forward_.shape != () or discount.shape != ():
            raise ValueError("forward and discount_factor must be scalars.")
        if bool(
            jnp.any(~jnp.isfinite(strikes_))
            | jnp.any(~jnp.isfinite(calls))
            | ~jnp.isfinite(forward_)
            | ~jnp.isfinite(discount)
        ):
            raise ValueError("Option call-slice numerics must be finite.")
        if bool(jnp.any(jnp.diff(strikes_) <= 0.0)):
            raise ValueError("Option strikes must be strictly increasing.")
        if bool(discount <= 0.0):
            raise ValueError("discount_factor must be positive.")
        maturity_ = float(maturity)
        if not isfinite(maturity_) or maturity_ < 0.0:
            raise ValueError("maturity must be finite and nonnegative.")
        identifiers = tuple(
            str(value)
            for value in (
                asset_id,
                currency_code,
                numeraire_id,
                market_snapshot_id,
            )
        )
        if any(not value for value in identifiers):
            raise ValueError("Option call-slice identities must be nonempty.")
        self.strikes = strikes_
        self.call_prices = calls
        self.forward = forward_
        self.discount_factor = discount
        self.maturity = maturity_
        (
            self.asset_id,
            self.currency_code,
            self.numeraire_id,
            self.market_snapshot_id,
        ) = identifiers


class StaticOptionArbitrageEvidence(StrictModule):
    """Finite-grid call bounds, strike monotonicity and convexity evidence."""

    lower_bound_violation: Array
    upper_bound_violation: Array
    strike_monotonicity_violation: Array
    butterfly_convexity_violation: Array
    minimum_call_spread: Array
    minimum_butterfly_slope_change: Array
    finite: Array
    valid: Array
    support_limited: bool = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)


class CalendarArbitrageEvidence(StrictModule):
    """Finite common-strike normalized-call calendar evidence."""

    normalized_call_difference: Array
    minimum_normalized_call_difference: Array
    maximum_violation: Array
    valid: Array
    support_limited: bool = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)


def evaluate_static_option_arbitrage(
    option_slice: OptionCallSlice,
    /,
    *,
    tolerance: float = 1e-8,
) -> StaticOptionArbitrageEvidence:
    """Audit one observed strike grid without making a global surface claim."""
    if not isinstance(option_slice, OptionCallSlice):
        raise TypeError("option_slice must be an OptionCallSlice.")
    tolerance_ = float(tolerance)
    if not isfinite(tolerance_) or tolerance_ < 0.0:
        raise ValueError("tolerance must be finite and nonnegative.")
    strikes = option_slice.strikes
    calls = option_slice.call_prices
    discount = option_slice.discount_factor
    lower = discount * jnp.maximum(option_slice.forward - strikes, 0.0)
    upper = discount * jnp.maximum(option_slice.forward, 0.0)
    lower_violation = jnp.max(jnp.maximum(lower - calls, 0.0))
    upper_violation = jnp.max(jnp.maximum(calls - upper, 0.0))
    spreads = jnp.diff(calls)
    monotonicity = jnp.max(jnp.maximum(spreads, 0.0))
    slopes = spreads / jnp.diff(strikes)
    slope_changes = jnp.diff(slopes)
    convexity = (
        jnp.max(jnp.maximum(-slope_changes, 0.0))
        if slope_changes.size
        else jnp.asarray(0.0, dtype=calls.dtype)
    )
    finite = jnp.all(jnp.isfinite(calls))
    valid = (
        finite
        & (lower_violation <= tolerance_)
        & (upper_violation <= tolerance_)
        & (monotonicity <= tolerance_)
        & (convexity <= tolerance_)
    )
    return StaticOptionArbitrageEvidence(
        lower_violation,
        upper_violation,
        monotonicity,
        convexity,
        jnp.min(-spreads),
        (
            jnp.min(slope_changes)
            if slope_changes.size
            else jnp.asarray(jnp.inf, dtype=calls.dtype)
        ),
        finite,
        valid,
        True,
        tolerance_,
    )


def evaluate_calendar_arbitrage(
    earlier: OptionCallSlice,
    later: OptionCallSlice,
    /,
    *,
    tolerance: float = 1e-8,
) -> CalendarArbitrageEvidence:
    """Compare numeraire-normalized calls on exactly one shared strike grid."""
    if not isinstance(earlier, OptionCallSlice) or not isinstance(later, OptionCallSlice):
        raise TypeError("Calendar diagnostics require two OptionCallSlice objects.")
    if earlier.maturity >= later.maturity:
        raise ValueError("Calendar diagnostics require strictly increasing maturities.")
    if (
        earlier.asset_id != later.asset_id
        or earlier.currency_code != later.currency_code
        or earlier.numeraire_id != later.numeraire_id
    ):
        raise ValueError("Calendar slices must share asset, currency, and numeraire.")
    if earlier.strikes.shape != later.strikes.shape or not bool(
        jnp.allclose(earlier.strikes, later.strikes, rtol=0.0, atol=0.0)
    ):
        raise ValueError("Calendar diagnostics require an identical strike grid.")
    tolerance_ = float(tolerance)
    if not isfinite(tolerance_) or tolerance_ < 0.0:
        raise ValueError("tolerance must be finite and nonnegative.")
    earlier_normalized = earlier.call_prices / earlier.discount_factor
    later_normalized = later.call_prices / later.discount_factor
    difference = later_normalized - earlier_normalized
    minimum = jnp.min(difference)
    violation = jnp.maximum(-minimum, 0.0)
    return CalendarArbitrageEvidence(
        difference,
        minimum,
        violation,
        jnp.all(jnp.isfinite(difference)) & (violation <= tolerance_),
        True,
        tolerance_,
    )


__all__ = [
    "CalendarArbitrageEvidence",
    "OptionCallSlice",
    "StaticOptionArbitrageEvidence",
    "evaluate_calendar_arbitrage",
    "evaluate_static_option_arbitrage",
]
