#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Avellaneda--Stoikov and GLFT market-making reference policies."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ...dynamics._grid import TimeGrid


def _identifier(value: str, owner: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} must be a non-empty string.")
    return value


def _positive(value: float, owner: str, /) -> float:
    resolved = float(value)
    if not isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{owner} must be finite and positive.")
    return resolved


def _nonnegative(value: float, owner: str, /) -> float:
    resolved = float(value)
    if not isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{owner} must be finite and nonnegative.")
    return resolved


def _inventory_grid(value: ArrayLike, bound: float, /) -> Array:
    grid = jnp.asarray(value)
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError("inventory_grid must be a nonempty rank-one vector.")
    if jnp.issubdtype(grid.dtype, jnp.complexfloating):
        raise TypeError("inventory_grid must be real-valued.")
    grid = grid.astype(jnp.result_type(grid, float))
    if not bool(jnp.all(jnp.isfinite(grid))):
        raise ValueError("inventory_grid must be finite.")
    if bool(jnp.any(jnp.diff(grid) <= 0.0)):
        raise ValueError("inventory_grid must be strictly increasing.")
    if bool(jnp.any(jnp.abs(grid) > bound)):
        raise ValueError("inventory_grid exceeds the declared inventory bound.")
    return grid


class AvellanedaStoikovPlan(StrictModule):
    """Finite-horizon exponential-arrival AS reference assumptions."""

    risk_aversion: float = eqx.field(static=True)
    volatility: float = eqx.field(static=True)
    arrival_scale: float = eqx.field(static=True)
    arrival_decay: float = eqx.field(static=True)
    inventory_bound: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        risk_aversion: float,
        volatility: float,
        arrival_scale: float,
        arrival_decay: float,
        inventory_bound: float,
        plan_id: str,
    ):
        self.risk_aversion = _nonnegative(risk_aversion, "risk_aversion")
        self.volatility = _nonnegative(volatility, "volatility")
        self.arrival_scale = _positive(arrival_scale, "arrival_scale")
        self.arrival_decay = _positive(arrival_decay, "arrival_decay")
        self.inventory_bound = _positive(inventory_bound, "inventory_bound")
        self.plan_id = _identifier(plan_id, "plan_id")


class GLFTPlan(StrictModule):
    """Small-tick GLFT asymptotic reference assumptions."""

    risk_aversion: float = eqx.field(static=True)
    volatility: float = eqx.field(static=True)
    arrival_scale: float = eqx.field(static=True)
    arrival_decay: float = eqx.field(static=True)
    inventory_bound: float = eqx.field(static=True)
    benchmark_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        risk_aversion: float,
        volatility: float,
        arrival_scale: float,
        arrival_decay: float,
        inventory_bound: float,
        benchmark_tolerance: float,
        plan_id: str,
    ):
        self.risk_aversion = _positive(risk_aversion, "risk_aversion")
        self.volatility = _nonnegative(volatility, "volatility")
        self.arrival_scale = _positive(arrival_scale, "arrival_scale")
        self.arrival_decay = _positive(arrival_decay, "arrival_decay")
        self.inventory_bound = _positive(inventory_bound, "inventory_bound")
        self.benchmark_tolerance = _nonnegative(
            benchmark_tolerance, "benchmark_tolerance"
        )
        self.plan_id = _identifier(plan_id, "plan_id")


class MarketMakingReference(StrictModule):
    """Time/inventory quote table and finite-arrival evidence."""

    time_grid: TimeGrid
    inventory_grid: Array
    reservation_price_adjustment: Array
    bid_offsets: Array
    ask_offsets: Array
    bid_intensities: Array
    ask_intensities: Array
    finite: Array
    admissible: Array
    inventory_skew_passed: Array
    plan_id: str = eqx.field(static=True)
    method: str = eqx.field(static=True)
    scope: str = eqx.field(static=True)


def _as_half_spread(plan: AvellanedaStoikovPlan, remaining: Array, /) -> Array:
    gamma = plan.risk_aversion
    if gamma <= 1.0e-12:
        return jnp.full_like(remaining, 1.0 / plan.arrival_decay)
    return (
        jnp.log1p(gamma / plan.arrival_decay) / gamma
        + 0.5 * gamma * plan.volatility * plan.volatility * remaining
    )


def solve_avellaneda_stoikov_reference(
    plan: AvellanedaStoikovPlan,
    time_grid: TimeGrid,
    inventory_grid: ArrayLike,
    /,
) -> MarketMakingReference:
    """Evaluate the AS closed-form reservation-price/quote reference."""

    if not isinstance(plan, AvellanedaStoikovPlan):
        raise TypeError("plan must be an AvellanedaStoikovPlan.")
    if not isinstance(time_grid, TimeGrid):
        raise TypeError("time_grid must be a TimeGrid.")
    inventory = _inventory_grid(inventory_grid, plan.inventory_bound)
    remaining = time_grid.times[-1] - time_grid.times
    skew = (
        -inventory[None, :]
        * plan.risk_aversion
        * plan.volatility
        * plan.volatility
        * remaining[:, None]
    )
    half_spread = _as_half_spread(plan, remaining)[:, None]
    bid_offsets = half_spread - skew
    ask_offsets = half_spread + skew
    bid_intensities = plan.arrival_scale * jnp.exp(-plan.arrival_decay * bid_offsets)
    ask_intensities = plan.arrival_scale * jnp.exp(-plan.arrival_decay * ask_offsets)
    finite = jnp.all(
        jnp.isfinite(
            jnp.stack((skew, bid_offsets, ask_offsets, bid_intensities, ask_intensities))
        )
    )
    admissible = jnp.all(bid_offsets >= 0.0) & jnp.all(ask_offsets >= 0.0)
    skew_passed = jnp.all(jnp.diff(skew, axis=1) <= 0.0)
    return MarketMakingReference(
        time_grid=time_grid,
        inventory_grid=inventory,
        reservation_price_adjustment=skew,
        bid_offsets=bid_offsets,
        ask_offsets=ask_offsets,
        bid_intensities=bid_intensities,
        ask_intensities=ask_intensities,
        finite=finite,
        admissible=admissible,
        inventory_skew_passed=skew_passed,
        plan_id=plan.plan_id,
        method="avellaneda-stoikov-exponential-arrival-closed-form-reference",
        scope="declared-bounded-inventory-grid-only",
    )


def solve_glft_reference(
    plan: GLFTPlan,
    time_grid: TimeGrid,
    inventory_grid: ArrayLike,
    /,
) -> MarketMakingReference:
    """Evaluate the stationary small-tick GLFT asymptotic quote reference."""

    if not isinstance(plan, GLFTPlan):
        raise TypeError("plan must be a GLFTPlan.")
    if not isinstance(time_grid, TimeGrid):
        raise TypeError("time_grid must be a TimeGrid.")
    inventory = _inventory_grid(inventory_grid, plan.inventory_bound)
    gamma = plan.risk_aversion
    decay = plan.arrival_decay
    base = jnp.log1p(gamma / decay) / gamma
    inventory_increment = jnp.sqrt(
        plan.volatility
        * plan.volatility
        * gamma
        / (2.0 * decay * plan.arrival_scale)
        * (1.0 + gamma / decay) ** (1.0 + decay / gamma)
    )
    bid_row = base + (inventory + 0.5) * inventory_increment
    ask_row = base - (inventory - 0.5) * inventory_increment
    bid_offsets = jnp.broadcast_to(
        bid_row[None, :], (time_grid.num_times, inventory.size)
    )
    ask_offsets = jnp.broadcast_to(
        ask_row[None, :], (time_grid.num_times, inventory.size)
    )
    reservation_adjustment = 0.5 * (ask_offsets - bid_offsets)
    bid_intensities = plan.arrival_scale * jnp.exp(-decay * bid_offsets)
    ask_intensities = plan.arrival_scale * jnp.exp(-decay * ask_offsets)
    finite = jnp.all(
        jnp.isfinite(
            jnp.stack(
                (
                    reservation_adjustment,
                    bid_offsets,
                    ask_offsets,
                    bid_intensities,
                    ask_intensities,
                )
            )
        )
    )
    admissible = jnp.all(bid_offsets >= 0.0) & jnp.all(ask_offsets >= 0.0)
    skew_passed = jnp.all(jnp.diff(reservation_adjustment, axis=1) <= 0.0)
    return MarketMakingReference(
        time_grid=time_grid,
        inventory_grid=inventory,
        reservation_price_adjustment=reservation_adjustment,
        bid_offsets=bid_offsets,
        ask_offsets=ask_offsets,
        bid_intensities=bid_intensities,
        ask_intensities=ask_intensities,
        finite=finite,
        admissible=admissible,
        inventory_skew_passed=skew_passed,
        plan_id=plan.plan_id,
        method="glft-small-tick-stationary-asymptotic-reference",
        scope="declared-bounded-inventory-grid-candidate-only",
    )


class GLFTBenchmarkEvidence(StrictModule):
    """Error of a GLFT candidate against independently supplied finite-grid quotes."""

    maximum_bid_error: Array
    maximum_ask_error: Array
    threshold: Array
    passed: Array
    plan_id: str = eqx.field(static=True)
    scope: str = eqx.field(static=True)


def compare_glft_to_finite_grid(
    candidate: MarketMakingReference,
    benchmark_bid_offsets: ArrayLike,
    benchmark_ask_offsets: ArrayLike,
    /,
    *,
    tolerance: float,
) -> GLFTBenchmarkEvidence:
    """Gate an asymptotic table against an independently computed quote table."""

    if not isinstance(candidate, MarketMakingReference):
        raise TypeError("candidate must be a MarketMakingReference.")
    if candidate.method != "glft-small-tick-stationary-asymptotic-reference":
        raise ValueError("candidate is not a GLFT asymptotic reference.")
    threshold = _nonnegative(tolerance, "tolerance")
    bids = jnp.asarray(benchmark_bid_offsets)
    asks = jnp.asarray(benchmark_ask_offsets)
    if (
        bids.shape != candidate.bid_offsets.shape
        or asks.shape != candidate.ask_offsets.shape
    ):
        raise ValueError("Benchmark quote tables must match the candidate table shape.")
    if not bool(jnp.all(jnp.isfinite(bids))) or not bool(jnp.all(jnp.isfinite(asks))):
        raise ValueError("Benchmark quote tables must be finite.")
    bid_error = jnp.max(jnp.abs(candidate.bid_offsets - bids))
    ask_error = jnp.max(jnp.abs(candidate.ask_offsets - asks))
    return GLFTBenchmarkEvidence(
        maximum_bid_error=bid_error,
        maximum_ask_error=ask_error,
        threshold=jnp.asarray(threshold),
        passed=(bid_error <= threshold) & (ask_error <= threshold),
        plan_id=candidate.plan_id,
        scope="independently-supplied-finite-grid-quote-table",
    )


__all__ = [
    "AvellanedaStoikovPlan",
    "GLFTBenchmarkEvidence",
    "GLFTPlan",
    "MarketMakingReference",
    "compare_glft_to_finite_grid",
    "solve_avellaneda_stoikov_reference",
    "solve_glft_reference",
]
