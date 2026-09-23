#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax import ein

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ._geometry import CalorimeterGeometry
from ._response import CalorimeterResponse


class CalorimeterClusteringPlan(StrictModule, NonTrainableState):
    geometry: CalorimeterGeometry
    cell_to_cluster: Array
    cluster_capacity: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        geometry: CalorimeterGeometry,
        cell_to_cluster: ArrayLike,
        /,
        *,
        cluster_capacity: int,
    ):
        if not isinstance(geometry, CalorimeterGeometry):
            raise TypeError("geometry must be CalorimeterGeometry.")
        mapping = np.asarray(cell_to_cluster)
        capacity = int(cluster_capacity)
        if mapping.shape != (geometry.cell_count,) or not np.issubdtype(
            mapping.dtype, np.integer
        ):
            raise ValueError("cell_to_cluster must contain one integer per cell.")
        if capacity < 1 or np.any(mapping < -1) or np.any(mapping >= capacity):
            raise ValueError("cell_to_cluster exceeds cluster support.")
        self.geometry = geometry
        self.cell_to_cluster = jnp.asarray(mapping, dtype=jnp.int32)
        self.cluster_capacity = capacity
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fixed-calorimeter-clustering",
                "geometry": geometry.geometry_id,
                "mapping": array_tree_fingerprint(mapping),
                "cluster_capacity": capacity,
            }
        )


class CalorimeterClusterBank(StrictModule, NonTrainableState):
    event_ids: Array
    cluster_ids: Array
    energies: Array
    positions: Array
    cell_counts: Array
    active: Array
    valid: Array
    plan_id: str = eqx.field(static=True)


def reconstruct_calorimeter_clusters(
    plan: CalorimeterClusteringPlan,
    response: CalorimeterResponse,
    /,
) -> CalorimeterClusterBank:
    """Aggregate a predeclared fixed cell partition into calibrated clusters."""
    if not isinstance(plan, CalorimeterClusteringPlan) or not isinstance(
        response, CalorimeterResponse
    ):
        raise TypeError("plan and response must use calorimeter types.")
    if response.geometry_id != plan.geometry.geometry_id:
        raise ValueError("Calorimeter response and clustering geometry differ.")
    mapping = plan.cell_to_cluster
    valid_cell = (mapping >= 0) & plan.geometry.active & ~plan.geometry.dead
    membership = (
        jax.nn.one_hot(
            jnp.clip(mapping, 0, plan.cluster_capacity - 1),
            plan.cluster_capacity,
            dtype=response.reconstructed_cell_energy.dtype,
        )
        * valid_cell[:, None]
    )
    energies = ein.contract("ec,ck->ek", response.reconstructed_cell_energy, membership)
    weighted_positions = ein.contract(
        "ec,ck,cd->ekd",
        response.reconstructed_cell_energy,
        membership,
        plan.geometry.centroids,
    )
    positions = weighted_positions / jnp.maximum(
        energies[..., None], jnp.finfo(energies.dtype).tiny
    )
    counts = jnp.sum(membership > 0.0, axis=0, dtype=jnp.int32)
    counts = jnp.broadcast_to(counts, energies.shape)
    active = energies > 0.0
    valid = jnp.isfinite(energies) & jnp.all(jnp.isfinite(positions), axis=-1)
    cluster_ids = jnp.broadcast_to(
        jnp.arange(plan.cluster_capacity, dtype=jnp.int32), energies.shape
    )
    return CalorimeterClusterBank(
        response.digits.event_ids,
        cluster_ids,
        energies,
        jnp.where(active[..., None], positions, 0.0),
        counts,
        active,
        jnp.where(active, valid, True),
        plan.plan_id,
    )


__all__ = [
    "CalorimeterClusterBank",
    "CalorimeterClusteringPlan",
    "reconstruct_calorimeter_clusters",
]
