#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from math import prod

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._interpolation import GatherStencil
from ..._strict import AbstractAttribute, StrictModule
from ..._trainable import NonTrainableState
from ..spatial import SparseBlockTopologyPlan, SparseBlockTopologyState
from ..splatting import (
    ParticleGridSplatState,
    PreparedParticleGridSplat,
    SplatDepositResult,
    SplatRouteScatterResult,
)


class AbstractMPMNodalStoragePlan(StrictModule, NonTrainableState):
    """Static MPM nodal placement policy."""

    storage_id: AbstractAttribute[str]

    @property
    @abc.abstractmethod
    def storage_capacity(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def pack(self, dense: Array, topology: SparseBlockTopologyState | None, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def unpack(
        self, compact: Array, topology: SparseBlockTopologyState | None, /
    ) -> Array:
        raise NotImplementedError


class DenseMPMNodalStoragePlan(AbstractMPMNodalStoragePlan):
    """Identity storage over the complete logical nodal grid."""

    grid_shape: tuple[int, ...] = eqx.field(static=True)
    storage_id: str = eqx.field(static=True)

    def __init__(self, grid_shape, /):
        shape = tuple(int(value) for value in grid_shape)
        if not shape or any(size <= 0 for size in shape):
            raise ValueError("grid_shape must contain positive dimensions.")
        self.grid_shape = shape
        self.storage_id = canonical_fingerprint(
            {"kind": "dense-mpm-nodal-storage", "grid_shape": shape}
        )

    @property
    def storage_capacity(self) -> int:
        return prod(self.grid_shape)

    def pack(self, dense, topology, /):
        del topology
        return jnp.asarray(dense)

    def unpack(self, compact, topology, /):
        del topology
        return jnp.asarray(compact)


class BlockSparseMPMNodalStoragePlan(AbstractMPMNodalStoragePlan):
    """Operational MPM storage on canonical sparse tensor blocks."""

    topology_plan: SparseBlockTopologyPlan
    storage_id: str = eqx.field(static=True)

    def __init__(self, topology_plan: SparseBlockTopologyPlan, /):
        if not isinstance(topology_plan, SparseBlockTopologyPlan):
            raise TypeError("topology_plan must be SparseBlockTopologyPlan.")
        self.topology_plan = topology_plan
        self.storage_id = canonical_fingerprint(
            {
                "kind": "block-sparse-mpm-storage",
                "topology": topology_plan.plan_id,
            }
        )

    @property
    def storage_capacity(self) -> int:
        return self.topology_plan.storage_capacity

    @property
    def grid_shape(self) -> tuple[int, ...]:
        return self.topology_plan.layout.shape

    def build(
        self,
        routes: ParticleGridSplatState,
        previous: SparseBlockTopologyState | None = None,
        /,
    ) -> SparseBlockTopologyState:
        logical = routes.stencil.indices.reshape((-1,))
        valid = routes.stencil.valid.reshape((-1,))
        if previous is None:
            return self.topology_plan.build(logical, valid)
        return self.topology_plan.refresh(previous, logical, valid).candidate

    def mapped_stencil(
        self,
        routes: ParticleGridSplatState,
        topology: SparseBlockTopologyState,
        /,
    ) -> GatherStencil:
        lookup = topology.lookup(
            routes.stencil.indices,
            routes.stencil.valid,
        )
        return GatherStencil(
            indices=lookup.storage_slots,
            weights=routes.stencil.weights,
            source_size=self.storage_capacity,
            valid=routes.stencil.valid & lookup.supported,
            support=routes.stencil.support & jnp.all(lookup.supported, axis=-1),
            case_shape=routes.stencil.case_shape,
        )

    def mapped_routes(
        self,
        routes: ParticleGridSplatState,
        topology: SparseBlockTopologyState,
        /,
    ) -> ParticleGridSplatState:
        stencil = self.mapped_stencil(routes, topology)
        successful = (
            routes.successful
            & topology.evidence.successful
            & jnp.all(~routes.stencil.valid | stencil.valid)
        )
        mapped = eqx.tree_at(lambda value: value.stencil, routes, stencil)
        return eqx.tree_at(lambda value: value.successful, mapped, successful)

    def target_measure(self, topology: SparseBlockTopologyState, /) -> Array:
        logical = topology.logical_node_ids.reshape((-1,))
        valid = topology.node_valid.reshape((-1,))
        measure, supported = self.topology_plan.layout.measure_at(logical)
        return jnp.where(valid & supported, measure, 1.0)

    def coordinates(self, topology: SparseBlockTopologyState, /) -> Array:
        logical = topology.logical_node_ids.reshape((-1,))
        valid = topology.node_valid.reshape((-1,))
        coordinates, supported = self.topology_plan.layout.coordinates_at(logical)
        return jnp.where((valid & supported)[:, None], coordinates, 0.0)

    def deposit_content(
        self,
        splat: PreparedParticleGridSplat,
        routes: ParticleGridSplatState,
        topology: SparseBlockTopologyState,
        content: ArrayLike,
        /,
    ) -> SplatDepositResult:
        mapped = self.mapped_stencil(routes, topology)
        return splat.deposit_content_mapped(
            routes,
            content,
            mapped,
            self.target_measure(topology),
        )

    def scatter_route_payload(
        self,
        splat: PreparedParticleGridSplat,
        routes: ParticleGridSplatState,
        topology: SparseBlockTopologyState,
        payload: ArrayLike,
        /,
    ) -> SplatRouteScatterResult:
        return splat.scatter_route_payload_mapped(
            routes,
            payload,
            self.mapped_stencil(routes, topology),
        )

    def pack(
        self,
        dense: ArrayLike,
        topology: SparseBlockTopologyState | None,
        /,
    ) -> Array:
        if topology is None:
            raise ValueError("Block-sparse packing requires a topology state.")
        value = jnp.asarray(dense)
        if value.shape[: len(self.grid_shape)] != self.grid_shape:
            raise ValueError("Dense MPM storage has the wrong grid prefix.")
        trailing = value.shape[len(self.grid_shape) :]
        flat = value.reshape((prod(self.grid_shape),) + trailing)
        logical = topology.logical_node_ids.reshape((-1,))
        valid = topology.node_valid.reshape((-1,))
        packed = flat[logical]
        return jnp.where(
            valid.reshape(valid.shape + (1,) * len(trailing)),
            packed,
            jnp.zeros((), dtype=packed.dtype),
        )

    def unpack(
        self,
        compact: ArrayLike,
        topology: SparseBlockTopologyState | None,
        /,
    ) -> Array:
        if topology is None:
            raise ValueError("Block-sparse unpacking requires a topology state.")
        value = jnp.asarray(compact)
        if value.ndim < 1 or value.shape[0] != self.storage_capacity:
            raise ValueError("Compact MPM storage has the wrong storage capacity.")
        logical = topology.logical_node_ids.reshape((-1,))
        valid = topology.node_valid.reshape((-1,))
        trailing = value.shape[1:]
        flat = jnp.zeros((prod(self.grid_shape),) + trailing, dtype=value.dtype)
        payload = jnp.where(
            valid.reshape(valid.shape + (1,) * len(trailing)),
            value,
            jnp.zeros((), dtype=value.dtype),
        )
        flat = flat.at[jnp.where(valid, logical, 0)].add(payload)
        return flat.reshape(self.grid_shape + trailing)


__all__ = [
    "AbstractMPMNodalStoragePlan",
    "BlockSparseMPMNodalStoragePlan",
    "DenseMPMNodalStoragePlan",
]
