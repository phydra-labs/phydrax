#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Reference execution-impact models with explicit manipulation diagnostics."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ...dynamics._grid import TimeGrid


def _identifier(value: str, owner: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} must be a non-empty string.")
    return value


def _finite(value: float, owner: str, /) -> float:
    resolved = float(value)
    if not isfinite(resolved):
        raise ValueError(f"{owner} must be finite.")
    return resolved


def _nonnegative(value: float, owner: str, /) -> float:
    resolved = _finite(value, owner)
    if resolved < 0.0:
        raise ValueError(f"{owner} must be nonnegative.")
    return resolved


def _positive(value: float, owner: str, /) -> float:
    resolved = _finite(value, owner)
    if resolved <= 0.0:
        raise ValueError(f"{owner} must be positive.")
    return resolved


def _finite_vector(value: ArrayLike, owner: str, /) -> Array:
    array = jnp.asarray(value)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{owner} must be a nonempty rank-one vector.")
    if jnp.issubdtype(array.dtype, jnp.complexfloating):
        raise TypeError(f"{owner} must be real-valued.")
    array = array.astype(jnp.result_type(array, float))
    if not bool(jnp.all(jnp.isfinite(array))):
        raise ValueError(f"{owner} must be finite.")
    return array


class AlmgrenChrissModel(StrictModule):
    """Linear permanent and quadratic temporary impact liquidation reference."""

    volatility: float = eqx.field(static=True)
    risk_aversion: float = eqx.field(static=True)
    temporary_impact: float = eqx.field(static=True)
    permanent_impact: float = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        volatility: float,
        risk_aversion: float,
        temporary_impact: float,
        permanent_impact: float = 0.0,
        model_id: str,
    ):
        self.volatility = _nonnegative(volatility, "volatility")
        self.risk_aversion = _nonnegative(risk_aversion, "risk_aversion")
        self.temporary_impact = _positive(temporary_impact, "temporary_impact")
        self.permanent_impact = _nonnegative(permanent_impact, "permanent_impact")
        self.model_id = _identifier(model_id, "model_id")


class AlmgrenChrissSchedule(StrictModule):
    """Analytic finite-grid inventory and rate schedule with cost decomposition."""

    time_grid: TimeGrid
    inventory: Array
    trading_rates: Array
    temporary_cost: Array
    permanent_cost: Array
    inventory_risk: Array
    objective: Array
    conservation_residual: Array
    model_id: str = eqx.field(static=True)
    method: str = eqx.field(static=True)


def solve_almgren_chriss_schedule(
    model: AlmgrenChrissModel,
    time_grid: TimeGrid,
    parent_quantity: ArrayLike,
    /,
) -> AlmgrenChrissSchedule:
    """Return the continuous-time AC optimum sampled on a supplied time grid."""

    if not isinstance(model, AlmgrenChrissModel):
        raise TypeError("model must be an AlmgrenChrissModel.")
    if not isinstance(time_grid, TimeGrid):
        raise TypeError("time_grid must be a TimeGrid.")
    quantity = jnp.asarray(parent_quantity)
    if quantity.shape != () or jnp.issubdtype(quantity.dtype, jnp.complexfloating):
        raise ValueError("parent_quantity must be a real scalar.")
    quantity = quantity.astype(jnp.result_type(quantity, float))
    if not bool(jnp.isfinite(quantity)):
        raise ValueError("parent_quantity must be finite.")
    elapsed = time_grid.times - time_grid.times[0]
    horizon = time_grid.times[-1] - time_grid.times[0]
    kappa_squared = (
        model.risk_aversion * model.volatility * model.volatility / model.temporary_impact
    )
    linear_inventory = quantity * (1.0 - elapsed / horizon)
    if kappa_squared <= 1.0e-14:
        inventory = linear_inventory
    else:
        kappa = jnp.sqrt(kappa_squared)
        scaled = kappa * horizon
        inventory = quantity * jnp.sinh(kappa * (horizon - elapsed)) / jnp.sinh(scaled)
    rates = (inventory[:-1] - inventory[1:]) / time_grid.durations
    temporary_cost = model.temporary_impact * jnp.sum(rates * rates * time_grid.durations)
    permanent_cost = 0.5 * model.permanent_impact * quantity * quantity
    inventory_risk = (
        model.risk_aversion
        * model.volatility
        * model.volatility
        * jnp.sum(inventory[:-1] * inventory[:-1] * time_grid.durations)
    )
    conservation = jnp.abs(
        quantity - jnp.sum(rates * time_grid.durations) - inventory[-1]
    )
    return AlmgrenChrissSchedule(
        time_grid=time_grid,
        inventory=inventory,
        trading_rates=rates,
        temporary_cost=temporary_cost,
        permanent_cost=permanent_cost,
        inventory_risk=inventory_risk,
        objective=temporary_cost + permanent_cost + inventory_risk,
        conservation_residual=conservation,
        model_id=model.model_id,
        method="continuous-time-linear-impact-analytic-reference-on-supplied-grid",
    )


class TransientPropagatorModel(StrictModule):
    """Causal sum-of-exponentials transient impact kernel."""

    weights: Array
    decay_rates: Array
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        weights: ArrayLike,
        decay_rates: ArrayLike,
        /,
        *,
        model_id: str,
    ):
        weights_array = _finite_vector(weights, "weights")
        rates = _finite_vector(decay_rates, "decay_rates")
        if weights_array.shape != rates.shape:
            raise ValueError("weights and decay_rates must have equal shape.")
        if bool(jnp.any(rates <= 0.0)):
            raise ValueError("decay_rates must be strictly positive.")
        self.weights = weights_array
        self.decay_rates = rates
        self.model_id = _identifier(model_id, "model_id")

    @property
    def num_components(self) -> int:
        return int(self.weights.size)

    def kernel(self, lag: ArrayLike, /) -> Array:
        lag_array = jnp.asarray(lag)
        if bool(jnp.any(lag_array < 0.0)):
            raise ValueError("Transient kernel lags must be nonnegative.")
        return jnp.sum(
            self.weights * jnp.exp(-lag_array[..., None] * self.decay_rates),
            axis=-1,
        )


class ImpactManipulationEvidence(StrictModule):
    """Finite-grid positive-semidefinite kernel evidence only."""

    minimum_kernel_eigenvalue: Array
    symmetry_residual: Array
    threshold: Array
    finite: Array
    passed: Array
    model_id: str = eqx.field(static=True)
    scope: str = eqx.field(static=True)


def diagnose_transient_manipulation(
    model: TransientPropagatorModel,
    time_grid: TimeGrid,
    /,
    *,
    tolerance: float = 1.0e-10,
) -> ImpactManipulationEvidence:
    """Check the sampled symmetric cost kernel; no continuous-time claim is made."""

    if not isinstance(model, TransientPropagatorModel):
        raise TypeError("model must be a TransientPropagatorModel.")
    if not isinstance(time_grid, TimeGrid):
        raise TypeError("time_grid must be a TimeGrid.")
    threshold = _nonnegative(tolerance, "tolerance")
    times = np.asarray(time_grid.times[:-1], dtype=float)
    lags = np.abs(times[:, None] - times[None, :])
    kernel = np.sum(
        np.asarray(model.weights)[None, None, :]
        * np.exp(-lags[:, :, None] * np.asarray(model.decay_rates)[None, None, :]),
        axis=-1,
    )
    duration = np.asarray(time_grid.durations, dtype=float)
    weighted = duration[:, None] * kernel * duration[None, :]
    symmetry = float(np.max(np.abs(weighted - weighted.T)))
    finite = bool(np.all(np.isfinite(weighted)))
    minimum = float(np.min(np.linalg.eigvalsh(0.5 * (weighted + weighted.T))))
    passed = finite and symmetry <= threshold and minimum >= -threshold
    return ImpactManipulationEvidence(
        minimum_kernel_eigenvalue=jnp.asarray(minimum),
        symmetry_residual=jnp.asarray(symmetry),
        threshold=jnp.asarray(threshold),
        finite=jnp.asarray(finite),
        passed=jnp.asarray(passed),
        model_id=model.model_id,
        scope="supplied-grid-symmetric-quadratic-cost-kernel-only",
    )


class TransientImpactPath(StrictModule):
    """Causal component states and aggregate impact for supplied signed rates."""

    time_grid: TimeGrid
    component_states: Array
    impact: Array
    trading_rates: Array
    model_id: str = eqx.field(static=True)


def transient_impact_path(
    model: TransientPropagatorModel,
    time_grid: TimeGrid,
    trading_rates: ArrayLike,
    /,
) -> TransientImpactPath:
    """Propagate exponential memory using only past interval flow."""

    if not isinstance(model, TransientPropagatorModel):
        raise TypeError("model must be a TransientPropagatorModel.")
    if not isinstance(time_grid, TimeGrid):
        raise TypeError("time_grid must be a TimeGrid.")
    rates = _finite_vector(trading_rates, "trading_rates")
    if rates.shape != (time_grid.num_steps,):
        raise ValueError(f"trading_rates must have shape ({time_grid.num_steps},).")
    states = [jnp.zeros_like(model.weights)]
    for step in range(time_grid.num_steps):
        decay = jnp.exp(-model.decay_rates * time_grid.durations[step])
        signed_quantity = rates[step] * time_grid.durations[step]
        states.append(decay * (states[-1] + model.weights * signed_quantity))
    components = jnp.stack(states)
    return TransientImpactPath(
        time_grid=time_grid,
        component_states=components,
        impact=jnp.sum(components, axis=-1),
        trading_rates=rates,
        model_id=model.model_id,
    )


class ObizhaevaWangModel(StrictModule):
    """One-level block-book displacement reference without venue simulation."""

    depth: float = eqx.field(static=True)
    resilience: float = eqx.field(static=True)
    maximum_trade_quantity: float = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        depth: float,
        resilience: float,
        maximum_trade_quantity: float,
        model_id: str,
    ):
        self.depth = _positive(depth, "depth")
        self.resilience = _positive(resilience, "resilience")
        self.maximum_trade_quantity = _positive(
            maximum_trade_quantity, "maximum_trade_quantity"
        )
        self.model_id = _identifier(model_id, "model_id")


class ObizhaevaWangPath(StrictModule):
    """Pre/post-trade displacement and resilience recovery decomposition."""

    time_grid: TimeGrid
    signed_quantities: Array
    pre_trade_displacement: Array
    post_trade_displacement: Array
    recovered_displacement: Array
    execution_cost: Array
    model_id: str = eqx.field(static=True)


def obizhaeva_wang_path(
    model: ObizhaevaWangModel,
    time_grid: TimeGrid,
    signed_quantities: ArrayLike,
    /,
) -> ObizhaevaWangPath:
    """Evaluate causal block-book displacement for supplied interval quantities."""

    if not isinstance(model, ObizhaevaWangModel):
        raise TypeError("model must be an ObizhaevaWangModel.")
    if not isinstance(time_grid, TimeGrid):
        raise TypeError("time_grid must be a TimeGrid.")
    quantities = _finite_vector(signed_quantities, "signed_quantities")
    if quantities.shape != (time_grid.num_steps,):
        raise ValueError(f"signed_quantities must have shape ({time_grid.num_steps},).")
    if bool(jnp.any(jnp.abs(quantities) > model.maximum_trade_quantity)):
        raise ValueError(
            "signed_quantities exceed the declared block-book consumption bound."
        )
    before: list[Array] = []
    after: list[Array] = []
    recovered: list[Array] = []
    displacement = jnp.asarray(0.0, dtype=quantities.dtype)
    total_cost = jnp.asarray(0.0, dtype=quantities.dtype)
    for step in range(time_grid.num_steps):
        before.append(displacement)
        post = displacement + quantities[step] / model.depth
        after.append(post)
        total_cost = total_cost + quantities[step] * (
            displacement + 0.5 * quantities[step] / model.depth
        )
        next_displacement = post * jnp.exp(-model.resilience * time_grid.durations[step])
        recovered.append(post - next_displacement)
        displacement = next_displacement
    return ObizhaevaWangPath(
        time_grid=time_grid,
        signed_quantities=quantities,
        pre_trade_displacement=jnp.stack(before),
        post_trade_displacement=jnp.stack(after),
        recovered_displacement=jnp.stack(recovered),
        execution_cost=total_cost,
        model_id=model.model_id,
    )


__all__ = [
    "AlmgrenChrissModel",
    "AlmgrenChrissSchedule",
    "ImpactManipulationEvidence",
    "ObizhaevaWangModel",
    "ObizhaevaWangPath",
    "TransientImpactPath",
    "TransientPropagatorModel",
    "diagnose_transient_manipulation",
    "obizhaeva_wang_path",
    "solve_almgren_chriss_schedule",
    "transient_impact_path",
]
