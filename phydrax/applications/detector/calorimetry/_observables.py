#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax import ein

from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ._geometry import CalorimeterGeometry


class CalorimeterObservables(StrictModule, NonTrainableState):
    total_energy: Array
    layer_energies: Array
    layer_fractions: Array
    occupancy: Array
    zero_fraction: Array
    depth: Array
    transverse_width: Array
    hottest_cell_fraction: Array
    valid: Array
    geometry_id: str = eqx.field(static=True)


def calorimeter_observables(
    geometry: CalorimeterGeometry,
    cell_energies: ArrayLike,
    /,
    *,
    threshold: float = 0.0,
) -> CalorimeterObservables:
    if not isinstance(geometry, CalorimeterGeometry):
        raise TypeError("geometry must be CalorimeterGeometry.")
    energies = jnp.asarray(cell_energies)
    if energies.ndim != 2 or energies.shape[1] != geometry.cell_count:
        raise ValueError("cell_energies must have shape (event, geometry.cell_count).")
    layer_count = int(jnp.max(geometry.layer_ids)) + 1
    layer_membership = jax.nn.one_hot(
        geometry.layer_ids, layer_count, dtype=energies.dtype
    )
    live = geometry.active & ~geometry.dead
    sanitized = jnp.where(live[None, :], energies, 0.0)
    total = jnp.sum(sanitized, axis=1)
    layers = ein.contract("ec,cl->el", sanitized, layer_membership)
    fractions = layers / jnp.maximum(total[:, None], jnp.finfo(energies.dtype).tiny)
    occupied = sanitized > float(threshold)
    occupancy = jnp.sum(occupied, axis=1, dtype=jnp.int32)
    active_count = jnp.maximum(jnp.sum(live, dtype=jnp.int32), 1)
    zero_fraction = jnp.sum((sanitized == 0.0) & live[None, :], axis=1) / active_count
    depth_coordinate = geometry.geometry_features[:, 3]
    depth = ein.contract("ec,c->e", sanitized, depth_coordinate) / jnp.maximum(
        total, jnp.finfo(energies.dtype).tiny
    )
    transverse = geometry.centroids[:, :2]
    centroid = ein.contract("ec,cd->ed", sanitized, transverse) / jnp.maximum(
        total[:, None], jnp.finfo(energies.dtype).tiny
    )
    displacement = transverse[None, :, :] - centroid[:, None, :]
    width = jnp.sqrt(
        jnp.sum(sanitized * jnp.sum(displacement * displacement, axis=-1), axis=1)
        / jnp.maximum(total, jnp.finfo(energies.dtype).tiny)
    )
    hottest = jnp.max(sanitized, axis=1) / jnp.maximum(
        total, jnp.finfo(energies.dtype).tiny
    )
    valid = (
        jnp.all(jnp.isfinite(energies), axis=1)
        & jnp.all(energies >= 0.0, axis=1)
        & (total > 0.0)
    )
    return CalorimeterObservables(
        total,
        layers,
        fractions,
        occupancy,
        zero_fraction,
        depth,
        width,
        hottest,
        valid,
        geometry.geometry_id,
    )


__all__ = ["CalorimeterObservables", "calorimeter_observables"]
