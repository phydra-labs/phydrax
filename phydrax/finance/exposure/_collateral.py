# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Caller-supplied netting, collateral, MPOR, and closeout mechanics.

These records encode already-resolved economic terms.  They neither parse an
agreement nor determine whether netting or collateral is legally enforceable.
"""

from __future__ import annotations

from math import isfinite
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..contracts._credit import DefaultEventState
from ..core import Currency


CloseoutValueSource: TypeAlias = Literal["risk_free", "replacement"]
CloseoutObservation: TypeAlias = Literal["default_time", "mpor_end"]
SimultaneousDefaultRule: TypeAlias = Literal["counterparty", "own", "zero"]


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a canonical non-empty string.")
    return value


def _scalar(
    value: ArrayLike,
    name: str,
    /,
    *,
    nonnegative: bool = False,
    positive: bool = False,
) -> Array:
    result = jnp.asarray(value, dtype=float)
    if result.shape != ():
        raise ValueError(f"{name} must be scalar.")
    host = float(np.asarray(jax.device_get(result)))
    if not isfinite(host):
        raise ValueError(f"{name} must be finite.")
    if nonnegative and host < 0.0:
        raise ValueError(f"{name} must be nonnegative.")
    if positive and host <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return result


def _time_grid(value: ArrayLike, /) -> Array:
    result = jnp.asarray(value, dtype=float)
    host = np.asarray(jax.device_get(result))
    if result.ndim != 1 or result.shape[0] < 2:
        raise ValueError("times must be a vector with at least two nodes.")
    if not np.all(np.isfinite(host)) or host[0] < 0.0 or np.any(np.diff(host) <= 0.0):
        raise ValueError("times must be nonnegative, finite, and strictly increasing.")
    return result


class NettingSet(StrictModule):
    """Resolved caller-owned trade membership for one contractual netting scope."""

    trade_indices: Array
    trade_ids: tuple[str, ...] = eqx.field(static=True)
    base_currency: Currency = eqx.field(static=True)
    agreement_scope_id: str = eqx.field(static=True)
    netting_set_id: str = eqx.field(static=True)

    def __init__(
        self,
        trade_ids: tuple[str, ...],
        trade_indices: ArrayLike,
        base_currency: Currency,
        /,
        *,
        agreement_scope_id: str,
        netting_set_id: str,
    ):
        ids = tuple(_identifier(value, "trade_id") for value in trade_ids)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("trade_ids must be non-empty and unique.")
        indices = jnp.asarray(trade_indices)
        host = np.asarray(jax.device_get(indices))
        if indices.shape != (len(ids),) or not jnp.issubdtype(indices.dtype, jnp.integer):
            raise TypeError(
                "trade_indices must be an integer vector aligned with trade_ids."
            )
        if np.any(host < 0) or len(set(int(value) for value in host)) != len(ids):
            raise ValueError("trade_indices must be nonnegative and unique.")
        if not isinstance(base_currency, Currency):
            raise TypeError("base_currency must be a Currency.")
        self.trade_indices = indices.astype(jnp.int32)
        self.trade_ids = ids
        self.base_currency = base_currency
        self.agreement_scope_id = _identifier(agreement_scope_id, "agreement_scope_id")
        self.netting_set_id = _identifier(netting_set_id, "netting_set_id")


class NettingSetCollection(StrictModule):
    """Disjoint resolved netting sets; membership is caller supplied, never inferred."""

    netting_sets: tuple[NettingSet, ...]
    netting_set_ids: tuple[str, ...] = eqx.field(static=True)
    collection_id: str = eqx.field(static=True)

    def __init__(
        self,
        netting_sets: tuple[NettingSet, ...],
        /,
        *,
        collection_id: str,
    ):
        sets = tuple(netting_sets)
        if not sets or any(not isinstance(value, NettingSet) for value in sets):
            raise TypeError(
                "netting_sets must be a non-empty tuple of NettingSet values."
            )
        identifiers = tuple(value.netting_set_id for value in sets)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("netting set identities must be unique.")
        memberships = tuple(trade_id for value in sets for trade_id in value.trade_ids)
        if len(set(memberships)) != len(memberships):
            raise ValueError("A trade cannot belong to more than one netting set.")
        indices = tuple(
            int(index)
            for value in sets
            for index in np.asarray(jax.device_get(value.trade_indices))
        )
        if len(set(indices)) != len(indices):
            raise ValueError(
                "A trade-axis index cannot belong to more than one netting set."
            )
        self.netting_sets = sets
        self.netting_set_ids = identifiers
        self.collection_id = _identifier(collection_id, "collection_id")


class CollateralAgreement(StrictModule):
    """Deterministic single-currency collateral terms already resolved by the caller."""

    threshold_receivable: Array
    threshold_postable: Array
    minimum_transfer_amount: Array
    call_lag: Array
    margin_period_of_risk: Array
    closeout_lag: Array
    independent_amount_receivable: Array
    independent_amount_postable: Array
    remuneration_rate: Array
    collateral_currency: Currency = eqx.field(static=True)
    netting_set_id: str = eqx.field(static=True)
    agreement_id: str = eqx.field(static=True)

    def __init__(
        self,
        collateral_currency: Currency,
        /,
        *,
        netting_set_id: str,
        threshold_receivable: ArrayLike,
        threshold_postable: ArrayLike,
        minimum_transfer_amount: ArrayLike,
        call_lag: ArrayLike,
        margin_period_of_risk: ArrayLike,
        closeout_lag: ArrayLike,
        independent_amount_receivable: ArrayLike = 0.0,
        independent_amount_postable: ArrayLike = 0.0,
        remuneration_rate: ArrayLike = 0.0,
        agreement_id: str,
    ):
        if not isinstance(collateral_currency, Currency):
            raise TypeError("collateral_currency must be a Currency.")
        self.threshold_receivable = _scalar(
            threshold_receivable, "threshold_receivable", nonnegative=True
        )
        self.threshold_postable = _scalar(
            threshold_postable, "threshold_postable", nonnegative=True
        )
        self.minimum_transfer_amount = _scalar(
            minimum_transfer_amount, "minimum_transfer_amount", nonnegative=True
        )
        self.call_lag = _scalar(call_lag, "call_lag", nonnegative=True)
        self.margin_period_of_risk = _scalar(
            margin_period_of_risk, "margin_period_of_risk", positive=True
        )
        self.closeout_lag = _scalar(closeout_lag, "closeout_lag", nonnegative=True)
        self.independent_amount_receivable = _scalar(
            independent_amount_receivable,
            "independent_amount_receivable",
            nonnegative=True,
        )
        self.independent_amount_postable = _scalar(
            independent_amount_postable,
            "independent_amount_postable",
            nonnegative=True,
        )
        self.remuneration_rate = _scalar(remuneration_rate, "remuneration_rate")
        self.collateral_currency = collateral_currency
        self.netting_set_id = _identifier(netting_set_id, "netting_set_id")
        self.agreement_id = _identifier(agreement_id, "agreement_id")


class PreparedCollateralAgreement(StrictModule):
    """Host-resolved call-settlement indices on one fixed exposure grid."""

    agreement: CollateralAgreement
    times: Array
    settlement_indices: Array
    settles_on_grid: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        agreement: CollateralAgreement,
        times: ArrayLike,
        settlement_indices: ArrayLike,
        settles_on_grid: ArrayLike,
        prepared_id: str,
        /,
    ):
        if not isinstance(agreement, CollateralAgreement):
            raise TypeError("agreement must be a CollateralAgreement.")
        nodes = _time_grid(times)
        indices = jnp.asarray(settlement_indices)
        settles = jnp.asarray(settles_on_grid)
        if settles.dtype != jnp.dtype(bool):
            raise TypeError("settles_on_grid must have a boolean dtype.")
        host_indices = np.asarray(jax.device_get(indices))
        host_settles = np.asarray(jax.device_get(settles))
        expected = (nodes.shape[0],)
        if indices.shape != expected or not jnp.issubdtype(indices.dtype, jnp.integer):
            raise TypeError("settlement_indices must be an integer time-axis vector.")
        if settles.shape != expected:
            raise ValueError("settles_on_grid must match the time axis.")
        if (
            np.any(host_indices < np.arange(nodes.shape[0]))
            or np.any(host_indices >= nodes.shape[0])
            or np.any(~host_settles & (host_indices != nodes.shape[0] - 1))
        ):
            raise ValueError("Resolved settlement indices are inconsistent.")
        self.agreement = agreement
        self.times = nodes
        self.settlement_indices = indices.astype(jnp.int32)
        self.settles_on_grid = settles
        self.prepared_id = _identifier(prepared_id, "prepared_id")


class CollateralState(StrictModule):
    """Terminal settled balance and any calls settling beyond the simulated grid."""

    settled_balance: Array
    unsettled_transfer: Array
    valid: Array
    agreement_id: str = eqx.field(static=True)


class CollateralPath(StrictModule):
    """Pathwise collateral calls and settled balances on one exposure grid."""

    times: Array
    balances: Array
    calls: Array
    settlements: Array
    valid: Array
    state: CollateralState
    agreement_id: str = eqx.field(static=True)
    netting_set_id: str = eqx.field(static=True)


class CloseoutConvention(StrictModule):
    """Explicit closeout calculation choice, not a legal interpretation."""

    value_source: CloseoutValueSource = eqx.field(static=True)
    observation: CloseoutObservation = eqx.field(static=True)
    simultaneous_default: SimultaneousDefaultRule = eqx.field(static=True)
    convention_id: str = eqx.field(static=True)

    def __init__(
        self,
        value_source: CloseoutValueSource,
        observation: CloseoutObservation,
        simultaneous_default: SimultaneousDefaultRule,
        /,
        *,
        convention_id: str,
    ):
        if value_source not in ("risk_free", "replacement"):
            raise ValueError("Unsupported closeout value source.")
        if observation not in ("default_time", "mpor_end"):
            raise ValueError("Unsupported closeout observation.")
        if simultaneous_default not in ("counterparty", "own", "zero"):
            raise ValueError("Unsupported simultaneous-default rule.")
        self.value_source = value_source
        self.observation = observation
        self.simultaneous_default = simultaneous_default
        self.convention_id = _identifier(convention_id, "convention_id")


class CloseoutPath(StrictModule):
    """Raw closeout residual before positive/negative exposure is taken."""

    default_times: Array
    closeout_times: Array
    default_indices: Array
    closeout_indices: Array
    closeout_values: Array
    collateral_at_default: Array
    residual_values: Array
    occurred: Array
    counterparty_first: Array
    own_first: Array
    valid: Array
    convention_id: str = eqx.field(static=True)
    agreement_id: str = eqx.field(static=True)


def net_trade_values(trade_values: ArrayLike, netting_set: NettingSet, /) -> Array:
    """Net base-currency trade values before collateral or positive-part operations."""

    if not isinstance(netting_set, NettingSet):
        raise TypeError("netting_set must be a NettingSet.")
    values = jnp.asarray(trade_values, dtype=float)
    if values.ndim != 3:
        raise ValueError("trade_values must have shape (path, time, trade).")
    if (
        int(np.max(np.asarray(jax.device_get(netting_set.trade_indices))))
        >= values.shape[-1]
    ):
        raise ValueError("A netting-set trade index lies outside the trade value axis.")
    selected = jnp.take(values, netting_set.trade_indices, axis=-1)
    return jnp.sum(selected, axis=-1)


def collateral_target(
    agreement: CollateralAgreement,
    netted_value: ArrayLike,
    /,
) -> Array:
    """Target balance; positive means collateral held, negative means collateral posted."""

    if not isinstance(agreement, CollateralAgreement):
        raise TypeError("agreement must be a CollateralAgreement.")
    value = jnp.asarray(netted_value, dtype=float)
    variation = jnp.where(
        value > agreement.threshold_receivable,
        value - agreement.threshold_receivable,
        jnp.where(
            value < -agreement.threshold_postable,
            value + agreement.threshold_postable,
            0.0,
        ),
    )
    return (
        variation
        + agreement.independent_amount_receivable
        - agreement.independent_amount_postable
    )


def prepare_collateral_agreement(
    agreement: CollateralAgreement,
    times: ArrayLike,
    /,
) -> PreparedCollateralAgreement:
    """Resolve deterministic call lag to fixed device indices on the host."""

    if not isinstance(agreement, CollateralAgreement):
        raise TypeError("agreement must be a CollateralAgreement.")
    nodes = _time_grid(times)
    host_nodes = np.asarray(jax.device_get(nodes))
    lag = float(np.asarray(jax.device_get(agreement.call_lag)))
    indices = np.searchsorted(host_nodes, host_nodes + lag, side="left")
    settles = indices < host_nodes.shape[0]
    safe = np.minimum(indices, host_nodes.shape[0] - 1).astype(np.int32)
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-collateral-agreement",
            "agreement_id": agreement.agreement_id,
            "times": host_nodes.tolist(),
            "settlement_indices": safe.tolist(),
            "settles_on_grid": settles.tolist(),
        }
    )
    return PreparedCollateralAgreement(
        agreement,
        nodes,
        jnp.asarray(safe),
        jnp.asarray(settles),
        prepared_id,
    )


def evolve_collateral(
    prepared: PreparedCollateralAgreement,
    netted_values: ArrayLike,
    /,
    *,
    path_valid: ArrayLike | None = None,
    initial_balance: ArrayLike = 0.0,
) -> CollateralPath:
    """Advance collateral on device after host resolution of settlement lag."""

    if not isinstance(prepared, PreparedCollateralAgreement):
        raise TypeError("prepared must be a PreparedCollateralAgreement.")
    agreement = prepared.agreement
    nodes = prepared.times
    values = jnp.asarray(netted_values, dtype=float)
    if values.ndim != 2 or values.shape[1] != nodes.shape[0]:
        raise ValueError("netted_values must have shape (path, time).")
    path_count, time_count = values.shape
    if path_valid is None:
        valid = jnp.ones((path_count,), dtype=bool)
    else:
        valid = jnp.asarray(path_valid)
        if valid.dtype != jnp.dtype(bool):
            raise TypeError("path_valid must have a boolean dtype.")
    if valid.shape != (path_count,):
        raise ValueError("path_valid must have shape (path,).")
    valid = valid & jnp.all(jnp.isfinite(values), axis=-1)
    initial = jnp.asarray(initial_balance, dtype=values.dtype)
    if initial.shape not in ((), (path_count,)):
        raise ValueError("initial_balance must be scalar or have shape (path,).")
    initial = eqx.error_if(
        initial, jnp.any(~jnp.isfinite(initial)), "initial_balance must be finite."
    )
    current = jnp.broadcast_to(initial, (path_count,))
    pending = jnp.zeros((path_count, time_count), dtype=values.dtype)
    unsettled = jnp.zeros((path_count,), dtype=values.dtype)

    def collateral_step(carry, index):
        current_, pending_, outstanding_, unsettled_ = carry
        previous = jnp.maximum(index - 1, 0)
        duration = jnp.where(index > 0, nodes[index] - nodes[previous], 0.0)
        current_ = current_ * jnp.exp(agreement.remuneration_rate * duration)
        settled = pending_[:, index]
        current_ = current_ + settled
        outstanding_ = outstanding_ - settled
        target = collateral_target(agreement, values[:, index])
        requested = target - current_ - outstanding_ - unsettled_
        requested = jnp.where(
            jnp.abs(requested) >= agreement.minimum_transfer_amount,
            requested,
            0.0,
        )
        requested = jnp.where(valid, requested, 0.0)
        settlement_index = prepared.settlement_indices[index]
        settles_on_grid = prepared.settles_on_grid[index]
        settles_immediately = settles_on_grid & (settlement_index == index)
        immediate = jnp.where(settles_immediately, requested, 0.0)
        deferred = jnp.where(
            settles_on_grid & ~settles_immediately,
            requested,
            0.0,
        )
        current_ = current_ + immediate
        settlement = jnp.where(valid, settled + immediate, 0.0)
        pending_ = pending_.at[:, settlement_index].add(deferred)
        outstanding_ = outstanding_ + deferred
        unsettled_ = unsettled_ + jnp.where(settles_on_grid, 0.0, requested)
        balance = jnp.where(valid, current_, 0.0)
        return (
            current_,
            pending_,
            outstanding_,
            unsettled_,
        ), (balance, requested, settlement)

    (current, pending, _, unsettled), outputs = jax.lax.scan(
        collateral_step,
        (
            current,
            pending,
            jnp.zeros((path_count,), dtype=values.dtype),
            unsettled,
        ),
        jnp.arange(time_count, dtype=jnp.int32),
    )
    del pending
    balances, calls, settlements = (jnp.swapaxes(output, 0, 1) for output in outputs)
    state = CollateralState(
        jnp.where(valid, current, 0.0),
        jnp.where(valid, unsettled, 0.0),
        valid,
        agreement.agreement_id,
    )
    return CollateralPath(
        nodes,
        balances,
        calls,
        settlements,
        valid,
        state,
        agreement.agreement_id,
        agreement.netting_set_id,
    )


def resolve_closeout(
    times: ArrayLike,
    netted_values: ArrayLike,
    collateral: CollateralPath,
    agreement: CollateralAgreement,
    convention: CloseoutConvention,
    counterparty_default: DefaultEventState,
    own_default: DefaultEventState,
    /,
    *,
    replacement_values: ArrayLike | None = None,
) -> CloseoutPath:
    """Apply default ordering, MPOR/lag, collateral freeze, then calculate residual."""

    nodes = _time_grid(times)
    values = jnp.asarray(netted_values, dtype=float)
    if values.ndim != 2 or values.shape[1] != nodes.shape[0]:
        raise ValueError("netted_values must have shape (path, time).")
    if not isinstance(collateral, CollateralPath):
        raise TypeError("collateral must be a CollateralPath.")
    if not isinstance(agreement, CollateralAgreement):
        raise TypeError("agreement must be a CollateralAgreement.")
    if not isinstance(convention, CloseoutConvention):
        raise TypeError("convention must be a CloseoutConvention.")
    if collateral.agreement_id != agreement.agreement_id:
        raise ValueError("Collateral path and agreement identities differ.")
    if collateral.times.shape != nodes.shape or not np.array_equal(
        np.asarray(jax.device_get(collateral.times)), np.asarray(jax.device_get(nodes))
    ):
        raise ValueError("Collateral and closeout grids differ.")
    if collateral.balances.shape != values.shape:
        raise ValueError("Collateral balances must match netted_values.")
    path_count = values.shape[0]
    for event, name in (
        (counterparty_default, "counterparty_default"),
        (own_default, "own_default"),
    ):
        if not isinstance(event, DefaultEventState):
            raise TypeError(f"{name} must be a DefaultEventState.")
        if event.default_times.shape != (path_count,):
            raise ValueError(f"{name} path shape is incompatible with values.")

    cp_time = jnp.where(
        counterparty_default.occurred, counterparty_default.default_times, jnp.inf
    )
    own_time = jnp.where(own_default.occurred, own_default.default_times, jnp.inf)
    simultaneous = (cp_time == own_time) & jnp.isfinite(cp_time)
    cp_first = cp_time < own_time
    own_first = own_time < cp_time
    if convention.simultaneous_default == "counterparty":
        cp_first = cp_first | simultaneous
    elif convention.simultaneous_default == "own":
        own_first = own_first | simultaneous
    occurred = cp_first | own_first | simultaneous
    default_times = jnp.minimum(cp_time, own_time)
    default_times = jnp.where(occurred, default_times, 0.0)
    default_indices = jnp.searchsorted(nodes, default_times, side="left")
    default_indices = jnp.minimum(default_indices, nodes.shape[0] - 1)
    horizon_time = (
        default_times + agreement.margin_period_of_risk + agreement.closeout_lag
    )
    observation_times = jnp.where(
        convention.observation == "default_time", default_times, horizon_time
    )
    beyond_grid = occurred & (observation_times > nodes[-1])
    observation_times = eqx.error_if(
        observation_times,
        jnp.any(beyond_grid),
        "Exposure grid does not cover the resolved MPOR/closeout horizon.",
    )
    closeout_indices = jnp.searchsorted(nodes, observation_times, side="left")
    closeout_indices = jnp.minimum(closeout_indices, nodes.shape[0] - 1)
    if convention.value_source == "replacement":
        if replacement_values is None:
            raise ValueError("Replacement closeout requires explicit replacement_values.")
        source = jnp.asarray(replacement_values, dtype=values.dtype)
        if source.shape != values.shape:
            raise ValueError("replacement_values must match netted_values.")
    else:
        if replacement_values is not None:
            raise ValueError("risk_free closeout does not accept replacement_values.")
        source = values
    closeout_value = jnp.take_along_axis(source, closeout_indices[:, None], axis=1)[:, 0]
    collateral_default = jnp.take_along_axis(
        collateral.balances, default_indices[:, None], axis=1
    )[:, 0]
    valid = (
        collateral.valid
        & counterparty_default.valid
        & own_default.valid
        & jnp.all(jnp.isfinite(values), axis=-1)
        & (~occurred | (jnp.isfinite(closeout_value) & jnp.isfinite(collateral_default)))
    )
    closeout_value = jnp.where(occurred & valid, closeout_value, 0.0)
    collateral_default = jnp.where(occurred & valid, collateral_default, 0.0)
    residual = jnp.where(occurred & valid, closeout_value - collateral_default, 0.0)
    if convention.simultaneous_default == "zero":
        residual = jnp.where(simultaneous, 0.0, residual)
    return CloseoutPath(
        default_times,
        jnp.where(occurred, observation_times, 0.0),
        default_indices,
        closeout_indices,
        closeout_value,
        collateral_default,
        residual,
        occurred,
        cp_first,
        own_first,
        valid,
        convention.convention_id,
        agreement.agreement_id,
    )


def collateral_agreement_identity(agreement: CollateralAgreement, /) -> str:
    if not isinstance(agreement, CollateralAgreement):
        raise TypeError("agreement must be a CollateralAgreement.")
    return canonical_fingerprint(
        {
            "kind": "collateral-agreement",
            "agreement_id": agreement.agreement_id,
            "netting_set_id": agreement.netting_set_id,
            "currency_id": agreement.collateral_currency.currency_id,
        }
    )


__all__ = [
    "CloseoutConvention",
    "CloseoutObservation",
    "CloseoutPath",
    "CloseoutValueSource",
    "CollateralAgreement",
    "CollateralPath",
    "CollateralState",
    "PreparedCollateralAgreement",
    "NettingSet",
    "NettingSetCollection",
    "SimultaneousDefaultRule",
    "collateral_agreement_identity",
    "collateral_target",
    "evolve_collateral",
    "net_trade_values",
    "prepare_collateral_agreement",
    "resolve_closeout",
]
