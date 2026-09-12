#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from phydrax._fingerprint import canonical_fingerprint
from phydrax._strict import StrictModule
from phydrax._trainable import NonTrainableState
from phydrax.backends.distributed import JaxCollectiveProvider

from ._morton import MortonAddressPlan
from ._neighbor_query import _minimum_image, MortonNeighborQueryPlan


class DistributedMortonNeighborEvidence(NonTrainableState, StrictModule):
    """Global exactness and identity evidence for a distributed query."""

    successful: jax.Array
    complete: jax.Array
    finite: jax.Array
    stable_ids_unique: jax.Array
    shard_count: jax.Array
    local_source_capacity: jax.Array
    local_target_capacity: jax.Array
    maximum_local_candidates: jax.Array


class DistributedMortonNeighborResult(NonTrainableState, StrictModule):
    """Global-sharded exact neighbors using global logical source indices."""

    source_indices: jax.Array
    source_stable_ids: jax.Array
    distance_squared: jax.Array
    valid: jax.Array
    counts: jax.Array
    evidence: DistributedMortonNeighborEvidence


class DistributedMortonNeighborQueryPlan(StrictModule):
    """Shard sources, merge exact local top-k sets, and shard target rows."""

    address_plan: MortonAddressPlan
    local_plan: MortonNeighborQueryPlan
    source_capacity: int = eqx.field(static=True)
    target_capacity: int = eqx.field(static=True)
    maximum_neighbors: int = eqx.field(static=True)
    shard_count: int = eqx.field(static=True)
    local_source_capacity: int = eqx.field(static=True)
    local_target_capacity: int = eqx.field(static=True)
    axis_name: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        address_plan: MortonAddressPlan,
        source_capacity: int,
        target_capacity: int,
        maximum_neighbors: int,
        shard_count: int,
        *,
        axis_name: str = "spatial",
        maximum_leaf_occupancy: int = 32,
        coarsening_factor: int = 8,
        target_top_nodes: int = 1024,
    ) -> None:
        sources = int(source_capacity)
        targets = int(target_capacity)
        neighbors = int(maximum_neighbors)
        shards = int(shard_count)
        axis = str(axis_name).strip()
        if sources < 1 or targets < 1 or shards < 1:
            raise ValueError(
                "Distributed Morton capacities and shard_count must be positive."
            )
        if sources % shards or targets % shards:
            raise ValueError(
                "source_capacity and target_capacity must be divisible by shard_count."
            )
        if neighbors < 1 or neighbors > sources:
            raise ValueError("maximum_neighbors must lie in [1, source_capacity].")
        if not axis:
            raise ValueError("axis_name must be non-empty.")
        local_sources = sources // shards
        local_targets = targets // shards
        local_neighbors = min(neighbors, local_sources)
        local_plan = MortonNeighborQueryPlan(
            address_plan,
            local_sources,
            targets,
            local_neighbors,
            maximum_candidates=local_sources,
            maximum_leaf_occupancy=maximum_leaf_occupancy,
            coarsening_factor=coarsening_factor,
            target_top_nodes=target_top_nodes,
        )
        object.__setattr__(self, "address_plan", address_plan)
        object.__setattr__(self, "local_plan", local_plan)
        object.__setattr__(self, "source_capacity", sources)
        object.__setattr__(self, "target_capacity", targets)
        object.__setattr__(self, "maximum_neighbors", neighbors)
        object.__setattr__(self, "shard_count", shards)
        object.__setattr__(self, "local_source_capacity", local_sources)
        object.__setattr__(self, "local_target_capacity", local_targets)
        object.__setattr__(self, "axis_name", axis)
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "distributed-morton-neighbor-query-plan",
                    "address_plan_id": address_plan.plan_id,
                    "source_capacity": sources,
                    "target_capacity": targets,
                    "maximum_neighbors": neighbors,
                    "shard_count": shards,
                    "axis_name": axis,
                    "local_plan_id": local_plan.plan_id,
                }
            ),
        )

    def query(
        self,
        source_points: jax.Array,
        target_points: jax.Array,
        *,
        source_mask: jax.Array | None = None,
        target_mask: jax.Array | None = None,
        source_stable_ids: jax.Array | None = None,
        target_stable_ids: jax.Array | None = None,
        exclude_self: bool = False,
        radius: float | None = None,
        devices: Sequence[jax.Device] | None = None,
    ) -> DistributedMortonNeighborResult:
        sources = jnp.asarray(source_points)
        targets = jnp.asarray(target_points)
        source_shape = (self.source_capacity, self.address_plan.dimension)
        target_shape = (self.target_capacity, self.address_plan.dimension)
        if sources.shape != source_shape:
            raise ValueError(f"source_points must have shape {source_shape}.")
        if targets.shape != target_shape:
            raise ValueError(f"target_points must have shape {target_shape}.")
        source_valid = (
            jnp.ones((self.source_capacity,), dtype=bool)
            if source_mask is None
            else jnp.asarray(source_mask, dtype=bool)
        )
        target_valid = (
            jnp.ones((self.target_capacity,), dtype=bool)
            if target_mask is None
            else jnp.asarray(target_mask, dtype=bool)
        )
        if source_valid.shape != (self.source_capacity,):
            raise ValueError("source_mask must match source_capacity.")
        if target_valid.shape != (self.target_capacity,):
            raise ValueError("target_mask must match target_capacity.")
        source_ids = (
            jnp.arange(self.source_capacity, dtype=jnp.int64)
            if source_stable_ids is None
            else jnp.asarray(source_stable_ids)
        )
        target_ids = (
            jnp.arange(self.target_capacity, dtype=jnp.int64)
            if target_stable_ids is None
            else jnp.asarray(target_stable_ids)
        )
        if source_ids.shape != (self.source_capacity,) or not jnp.issubdtype(
            source_ids.dtype, jnp.integer
        ):
            raise ValueError("source_stable_ids must be one integer per source.")
        if target_ids.shape != (self.target_capacity,) or not jnp.issubdtype(
            target_ids.dtype, jnp.integer
        ):
            raise ValueError("target_stable_ids must be one integer per target.")
        if (
            exclude_self
            and target_stable_ids is None
            and (self.source_capacity != self.target_capacity)
        ):
            raise ValueError(
                "exclude_self with unequal capacities requires target_stable_ids."
            )

        selected_devices = (
            tuple(jax.devices()[: self.shard_count])
            if devices is None
            else tuple(devices)
        )
        if len(selected_devices) != self.shard_count:
            raise ValueError(
                "Distributed Morton execution requires one device per shard."
            )
        mesh = Mesh(np.asarray(selected_devices, dtype=object), (self.axis_name,))
        point_spec = PartitionSpec(self.axis_name, None)
        vector_spec = PartitionSpec(self.axis_name)
        row_spec = PartitionSpec(self.axis_name, None)
        replicated = PartitionSpec()
        provider = JaxCollectiveProvider(self.axis_name)

        sources = jax.device_put(sources, NamedSharding(mesh, point_spec))
        targets = jax.device_put(targets, NamedSharding(mesh, point_spec))
        source_valid = jax.device_put(source_valid, NamedSharding(mesh, vector_spec))
        target_valid = jax.device_put(target_valid, NamedSharding(mesh, vector_spec))
        source_ids = jax.device_put(source_ids, NamedSharding(mesh, vector_spec))
        target_ids = jax.device_put(target_ids, NamedSharding(mesh, vector_spec))

        def execute_local(
            local_sources,
            local_targets,
            local_source_valid,
            local_target_valid,
            local_source_ids,
            local_target_ids,
        ):
            rank = jax.lax.axis_index(self.axis_name).astype(jnp.int64)
            global_targets = provider.all_gather(local_targets, axis=0, tiled=True)
            global_target_valid = provider.all_gather(
                local_target_valid, axis=0, tiled=True
            )
            global_target_ids = provider.all_gather(local_target_ids, axis=0, tiled=True)
            local = self.local_plan.query(
                local_sources,
                global_targets,
                source_mask=local_source_valid,
                target_mask=global_target_valid,
                source_stable_ids=local_source_ids,
                target_stable_ids=global_target_ids,
                exclude_self=exclude_self,
                radius=radius,
            )
            local_indices = local.source_indices
            local_coordinates = local_sources[local_indices]
            relative = _minimum_image(
                global_targets[:, None, :] - local_coordinates,
                self.address_plan,
            )
            local_distance_squared = jnp.sum(relative * relative, axis=-1)
            local_global_indices = (
                rank * self.local_source_capacity + local_indices.astype(jnp.int64)
            )
            local_candidate_ids = local_source_ids[local_indices]

            gathered_indices = provider.all_gather(
                local_global_indices, axis=0, tiled=False
            )
            gathered_ids = provider.all_gather(local_candidate_ids, axis=0, tiled=False)
            gathered_distance = provider.all_gather(
                local_distance_squared, axis=0, tiled=False
            )
            gathered_valid = provider.all_gather(local.valid, axis=0, tiled=False)
            candidate_indices = jnp.moveaxis(gathered_indices, 0, 1).reshape(
                (self.target_capacity, -1)
            )
            candidate_ids = jnp.moveaxis(gathered_ids, 0, 1).reshape(
                (self.target_capacity, -1)
            )
            candidate_distance = jnp.moveaxis(gathered_distance, 0, 1).reshape(
                (self.target_capacity, -1)
            )
            candidate_valid = jnp.moveaxis(gathered_valid, 0, 1).reshape(
                (self.target_capacity, -1)
            )

            all_source_ids = provider.all_gather(local_source_ids, axis=0, tiled=True)
            all_source_valid = provider.all_gather(local_source_valid, axis=0, tiled=True)
            identifier_order = jnp.lexsort(
                (all_source_ids, (~all_source_valid).astype(jnp.int32))
            )
            ordered_ids = all_source_ids[identifier_order]
            ordered_valid = all_source_valid[identifier_order]
            stable_ids_unique = ~jnp.any(
                ordered_valid[1:]
                & ordered_valid[:-1]
                & (ordered_ids[1:] == ordered_ids[:-1])
            )
            local_complete = local.evidence.successful.astype(jnp.int32)
            complete = provider.minimum(local_complete).astype(bool) & stable_ids_unique
            maximum_local_candidates = provider.maximum(
                local.evidence.required_candidates
            )

            maximum_id = jnp.asarray(
                jnp.iinfo(candidate_ids.dtype).max, dtype=candidate_ids.dtype
            )
            sortable_distance = jnp.where(candidate_valid, candidate_distance, jnp.inf)
            sortable_id = jnp.where(candidate_valid, candidate_ids, maximum_id)
            candidate_order = jnp.lexsort((sortable_id, sortable_distance), axis=1)
            selected = candidate_order[:, : self.maximum_neighbors]
            output_indices = jnp.take_along_axis(candidate_indices, selected, axis=1)
            output_ids = jnp.take_along_axis(candidate_ids, selected, axis=1)
            output_distance = jnp.take_along_axis(candidate_distance, selected, axis=1)
            output_valid = jnp.take_along_axis(candidate_valid, selected, axis=1)
            output_valid = output_valid & complete
            output_indices = jnp.where(output_valid, output_indices, 0)
            output_ids = jnp.where(output_valid, output_ids, 0)
            output_distance = jnp.where(output_valid, output_distance, 0)

            start = jax.lax.axis_index(self.axis_name) * self.local_target_capacity
            local_output_indices = jax.lax.dynamic_slice_in_dim(
                output_indices,
                start,
                self.local_target_capacity,
                axis=0,
            )
            local_output_ids = jax.lax.dynamic_slice_in_dim(
                output_ids,
                start,
                self.local_target_capacity,
                axis=0,
            )
            local_output_distance = jax.lax.dynamic_slice_in_dim(
                output_distance,
                start,
                self.local_target_capacity,
                axis=0,
            )
            local_output_valid = jax.lax.dynamic_slice_in_dim(
                output_valid,
                start,
                self.local_target_capacity,
                axis=0,
            )
            local_counts = jnp.sum(local_output_valid, axis=1, dtype=jnp.int32)
            finite = provider.minimum(
                (local.evidence.finite & local.evidence.topology_successful).astype(
                    jnp.int32
                )
            ).astype(bool)
            successful = complete & finite
            return (
                local_output_indices,
                local_output_ids,
                local_output_distance,
                local_output_valid,
                local_counts,
                successful,
                finite,
                stable_ids_unique,
                maximum_local_candidates,
            )

        mapped = jax.shard_map(
            execute_local,
            mesh=mesh,
            in_specs=(
                point_spec,
                point_spec,
                vector_spec,
                vector_spec,
                vector_spec,
                vector_spec,
            ),
            out_specs=(
                row_spec,
                row_spec,
                row_spec,
                row_spec,
                vector_spec,
                replicated,
                replicated,
                replicated,
                replicated,
            ),
            check_vma=False,
        )
        (
            source_indices,
            output_ids,
            distance_squared,
            valid,
            counts,
            successful,
            finite,
            stable_ids_unique,
            maximum_local_candidates,
        ) = mapped(
            sources,
            targets,
            source_valid,
            target_valid,
            source_ids,
            target_ids,
        )
        evidence = DistributedMortonNeighborEvidence(
            successful=successful,
            complete=successful,
            finite=finite,
            stable_ids_unique=stable_ids_unique,
            shard_count=jnp.asarray(self.shard_count, dtype=jnp.int32),
            local_source_capacity=jnp.asarray(
                self.local_source_capacity, dtype=jnp.int32
            ),
            local_target_capacity=jnp.asarray(
                self.local_target_capacity, dtype=jnp.int32
            ),
            maximum_local_candidates=maximum_local_candidates,
        )
        return DistributedMortonNeighborResult(
            source_indices=source_indices,
            source_stable_ids=output_ids,
            distance_squared=distance_squared,
            valid=valid,
            counts=counts,
            evidence=evidence,
        )


__all__ = [
    "DistributedMortonNeighborEvidence",
    "DistributedMortonNeighborQueryPlan",
    "DistributedMortonNeighborResult",
]
