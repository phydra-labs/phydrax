#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CochainDiscretization


class DegreeAwareEntityOwnership(StrictModule, NonTrainableState):
    owners_by_degree: tuple[Array, ...]
    shard_count: int = eqx.field(static=True)
    ownership_id: str = eqx.field(static=True)

    def __init__(
        self,
        cochain: CochainDiscretization,
        shard_count: int,
        /,
    ):
        shards = int(shard_count)
        if not isinstance(cochain, CochainDiscretization) or shards <= 0:
            raise ValueError("Distributed cochain ownership is invalid.")
        owners = tuple(
            jnp.arange(count, dtype=jnp.int32) % shards for count in cochain.cell_counts
        )
        self.owners_by_degree = owners
        self.shard_count = shards
        self.ownership_id = canonical_fingerprint(
            {
                "kind": "degree-aware-entity-ownership",
                "cochain": cochain.prepared_id,
                "shard_count": shards,
                "owners": array_tree_fingerprint(
                    tuple(np.asarray(value) for value in owners)
                ),
            }
        )

    def owned_mask(self, degree: int, shard: int, /) -> Array:
        degree_ = int(degree)
        shard_ = int(shard)
        if (
            degree_ < 0
            or degree_ >= len(self.owners_by_degree)
            or not 0 <= shard_ < self.shard_count
        ):
            raise ValueError("Distributed ownership query is invalid.")
        return self.owners_by_degree[degree_] == shard_

    def reconcile(self, degree: int, replicas: ArrayLike, /) -> Array:
        values = jnp.asarray(replicas)
        owners = self.owners_by_degree[int(degree)]
        if values.shape[:2] != (self.shard_count, owners.size):
            raise ValueError(
                "Distributed replicas must have shape (shards, entities, ...)."
            )
        indices = owners.reshape((1, owners.size) + (1,) * (values.ndim - 2))
        return jnp.take_along_axis(values, indices, axis=0)[0]


class DistributedGravitySolveResult(StrictModule):
    potential_shards: Array
    acceleration_shards: Array
    compatibility_residual: Array
    successful: Array


class DistributedGravitySolvePlan(StrictModule, NonTrainableState):
    global_solve: object = eqx.field(static=True)
    shard_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, global_solve, shard_count: int, /, *, solve_id: str):
        count = int(shard_count)
        if not callable(global_solve) or count <= 0 or not solve_id:
            raise ValueError("Distributed gravity solve plan is invalid.")
        self.global_solve = global_solve
        self.shard_count = count
        self.plan_id = canonical_fingerprint(
            {
                "kind": "distributed-gravity-solve",
                "solve_id": solve_id,
                "shard_count": count,
            }
        )

    def solve(self, density_shards: ArrayLike, /) -> DistributedGravitySolveResult:
        shards = jnp.asarray(density_shards)
        if shards.shape[0] != self.shard_count:
            raise ValueError("Distributed gravity density does not align with shards.")
        density = jnp.concatenate(
            tuple(shards[index] for index in range(self.shard_count)), axis=0
        )
        potential, acceleration, diagnostics = self.global_solve(density)
        potential_shards = jnp.stack(jnp.array_split(potential, self.shard_count, axis=0))
        acceleration_shards = jnp.stack(
            jnp.array_split(acceleration, self.shard_count, axis=0)
        )
        residual = jnp.sum(density)
        successful = diagnostics.finite & jnp.all(jnp.isfinite(potential_shards))
        return DistributedGravitySolveResult(
            potential_shards=potential_shards,
            acceleration_shards=acceleration_shards,
            compatibility_residual=residual,
            successful=successful,
        )


class DistributedMultiphysicsSynchronizationResult(StrictModule):
    gravity_compatibility_residual: Array
    synchronized_forcing_coefficients: Array
    synchronized_amr_register: Array
    successful: Array


class DistributedMultiphysicsSynchronizationPlan(StrictModule, NonTrainableState):
    shard_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, shard_count: int, /):
        count = int(shard_count)
        if count <= 0:
            raise ValueError("Distributed synchronization requires positive shard count.")
        self.shard_count = count
        self.plan_id = canonical_fingerprint(
            {"kind": "distributed-multiphysics-synchronization", "shard_count": count}
        )

    def synchronize(
        self,
        gravity_rhs_replicas: ArrayLike,
        forcing_coefficient_replicas: ArrayLike,
        amr_register_replicas: ArrayLike,
        /,
    ) -> DistributedMultiphysicsSynchronizationResult:
        gravity = jnp.asarray(gravity_rhs_replicas)
        forcing = jnp.asarray(forcing_coefficient_replicas)
        register = jnp.asarray(amr_register_replicas)
        if (
            gravity.shape[0] != self.shard_count
            or forcing.shape[0] != self.shard_count
            or register.shape[0] != self.shard_count
        ):
            raise ValueError(
                "Distributed multiphysics replicas do not align with shards."
            )
        gravity_global = jnp.sum(gravity, axis=0)
        forcing_global = jnp.mean(forcing, axis=0)
        register_global = jnp.sum(register, axis=0)
        residual = jnp.sum(gravity_global)
        successful = (
            jnp.all(jnp.isfinite(gravity_global))
            & jnp.all(jnp.isfinite(forcing_global))
            & jnp.all(jnp.isfinite(register_global))
        )
        return DistributedMultiphysicsSynchronizationResult(
            gravity_compatibility_residual=residual,
            synchronized_forcing_coefficients=forcing_global,
            synchronized_amr_register=register_global,
            successful=successful,
        )


class DistributedMHDReconciliationDiagnostics(StrictModule):
    face_replica_defect: Array
    edge_replica_defect: Array
    successful: Array


def reconcile_distributed_mhd_entities(
    ownership: DegreeAwareEntityOwnership,
    magnetic_degree: int,
    face_replicas: ArrayLike,
    edge_replicas: ArrayLike,
    /,
) -> tuple[Array, Array, DistributedMHDReconciliationDiagnostics]:
    faces = jnp.asarray(face_replicas)
    edges = jnp.asarray(edge_replicas)
    face = ownership.reconcile(magnetic_degree, faces)
    edge = ownership.reconcile(magnetic_degree - 1, edges)
    face_defect = jnp.max(jnp.abs(faces - face[None, ...]), initial=0.0)
    edge_defect = jnp.max(jnp.abs(edges - edge[None, ...]), initial=0.0)
    successful = jnp.all(jnp.isfinite(face)) & jnp.all(jnp.isfinite(edge))
    return (
        face,
        edge,
        DistributedMHDReconciliationDiagnostics(
            face_replica_defect=face_defect,
            edge_replica_defect=edge_defect,
            successful=successful,
        ),
    )


__all__ = [
    "DegreeAwareEntityOwnership",
    "DistributedGravitySolvePlan",
    "DistributedGravitySolveResult",
    "DistributedMultiphysicsSynchronizationPlan",
    "DistributedMultiphysicsSynchronizationResult",
    "DistributedMHDReconciliationDiagnostics",
    "reconcile_distributed_mhd_entities",
]
