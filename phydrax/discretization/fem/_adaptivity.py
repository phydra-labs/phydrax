#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite-element error estimators and deterministic cell markers.

Mesh refinement, coarsening, and lineage live in ``phydrax.meshing``
(`prepare_mesh_adaptation`); FE state transfer uses the adaptation result's
`FiniteElementTopologyTransfer`.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._strict import StrictModule


class FiniteElementErrorEstimate(StrictModule):
    """Cell-local residual/jump or goal-oriented error evidence."""

    cell_indicators: Array
    global_estimate: Array
    estimator_id: str = eqx.field(static=True)

    def __init__(self, cell_indicators: ArrayLike, estimator_id: str, /) -> None:
        indicators = jnp.asarray(cell_indicators)
        if indicators.ndim != 1:
            raise ValueError("cell_indicators must be rank-1.")
        identifier = str(estimator_id)
        if not identifier:
            raise ValueError("estimator_id must be non-empty.")
        self.cell_indicators = indicators
        self.global_estimate = jnp.sqrt(jnp.sum(indicators**2))
        self.estimator_id = identifier


def residual_jump_estimate(
    cell_residual: ArrayLike,
    cell_measure: ArrayLike,
    facet_jump: ArrayLike,
    facet_measure: ArrayLike,
    facet_owner: ArrayLike,
    facet_neighbor: ArrayLike,
    /,
) -> FiniteElementErrorEstimate:
    residual = jnp.asarray(cell_residual)
    cells = jnp.asarray(cell_measure)
    jumps = jnp.asarray(facet_jump)
    facets = jnp.asarray(facet_measure)
    owner = jnp.asarray(facet_owner, dtype=jnp.int32)
    neighbor = jnp.asarray(facet_neighbor, dtype=jnp.int32)
    if residual.shape != cells.shape or jumps.shape != facets.shape:
        raise ValueError("Residual/jump values must match their measures.")
    indicators = cells * residual**2
    contributions = facets * jumps**2
    indicators = indicators.at[owner].add(0.5 * contributions)
    active_neighbor = neighbor >= 0
    safe_neighbor = jnp.where(active_neighbor, neighbor, 0)
    indicators = indicators.at[safe_neighbor].add(
        jnp.where(active_neighbor, 0.5 * contributions, 0.0)
    )
    return FiniteElementErrorEstimate(
        jnp.sqrt(jnp.maximum(indicators, 0.0)),
        "residual-jump",
    )


def dual_weighted_residual_estimate(
    residual: ArrayLike,
    dual_correction: ArrayLike,
    /,
) -> Array:
    residual_ = jnp.asarray(residual)
    correction = jnp.asarray(dual_correction)
    if residual_.shape != correction.shape:
        raise ValueError("Residual and dual correction shapes must match.")
    return jnp.real(jnp.vdot(correction.reshape((-1,)), residual_.reshape((-1,))))


def dorfler_mark(
    indicators: ArrayLike,
    theta: float,
    /,
    *,
    cell_global_ids: ArrayLike | None = None,
) -> Array:
    values = np.asarray(indicators, dtype=np.float64)
    ids = (
        np.arange(values.size, dtype=np.int64)
        if cell_global_ids is None
        else np.asarray(cell_global_ids, dtype=np.int64)
    )
    fraction = float(theta)
    if (
        values.ndim != 1
        or ids.shape != values.shape
        or np.any(~np.isfinite(values))
        or np.any(values < 0.0)
        or not 0.0 < fraction <= 1.0
    ):
        raise ValueError("Dörfler indicators, IDs, or theta are invalid.")
    squared = values**2
    total = float(np.sum(squared))
    if total == 0.0:
        return jnp.asarray(np.empty((0,), dtype=np.int64))
    order = np.lexsort((ids, -squared))
    count = int(
        np.searchsorted(
            np.cumsum(squared[order]),
            fraction * total,
            side="left",
        )
        + 1
    )
    return jnp.asarray(np.sort(ids[order[:count]]))


def maximum_mark(
    indicators: ArrayLike,
    fraction: float,
    /,
    *,
    cell_global_ids: ArrayLike | None = None,
) -> Array:
    values = np.asarray(indicators, dtype=np.float64)
    ids = (
        np.arange(values.size, dtype=np.int64)
        if cell_global_ids is None
        else np.asarray(cell_global_ids, dtype=np.int64)
    )
    fraction_ = float(fraction)
    if (
        values.ndim != 1
        or ids.shape != values.shape
        or np.any(~np.isfinite(values))
        or np.any(values < 0.0)
        or not 0.0 < fraction_ <= 1.0
    ):
        raise ValueError("Maximum indicators, IDs, or fraction are invalid.")
    count = max(1, int(np.ceil(fraction_ * values.size)))
    order = np.lexsort((ids, -values))
    return jnp.asarray(np.sort(ids[order[:count]]))


class FiniteElementDWRIndicators(StrictModule):
    signed: Array
    absolute: Array
    global_estimate: Array

    def __init__(self, signed: ArrayLike, /) -> None:
        signed_ = jnp.asarray(signed)
        if signed_.ndim != 1:
            raise ValueError("DWR signed indicators must be rank-1.")
        self.signed = signed_
        self.absolute = jnp.abs(signed_)
        self.global_estimate = jnp.sum(signed_)


def local_dual_weighted_residual(
    cell_residual: ArrayLike,
    dual_correction: ArrayLike,
    /,
) -> FiniteElementDWRIndicators:
    residual = jnp.asarray(cell_residual)
    correction = jnp.asarray(dual_correction)
    if residual.shape != correction.shape or residual.ndim < 2:
        raise ValueError(
            "Local DWR residual/correction arrays must share cell-leading shape."
        )
    axes = tuple(range(1, residual.ndim))
    signed = jnp.sum(jnp.real(jnp.conj(correction) * residual), axis=axes)
    return FiniteElementDWRIndicators(signed)


__all__ = [
    "FiniteElementErrorEstimate",
    "FiniteElementDWRIndicators",
    "dorfler_mark",
    "dual_weighted_residual_estimate",
    "local_dual_weighted_residual",
    "maximum_mark",
    "residual_jump_estimate",
]
