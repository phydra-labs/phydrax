#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Named-sharding and deterministic AMR ownership for numerical relativity."""

from __future__ import annotations

from collections.abc import Sequence
from math import prod
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec
from jaxtyping import Array, ArrayLike

from ..._execution_runtime import ExecutionGroup
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import StructuredCochainBridge
from ...discretization.amr import (
    BlockAMRPartitionPlan,
    BlockHierarchyPlan,
    BlockHierarchyState,
    BlockHierarchyTopology,
    BlockTopologyCompileResult,
    PreparedDistributedBlockAMRHierarchy,
    PreparedFDAMRHierarchy,
)
from ...discretization.finite_volume import (
    FiniteVolumeDecompositionPlan,
    PreparedFiniteVolumeDecomposition,
)


NumericalRelativityFormulation: TypeAlias = Literal[
    "z4c",
    "grhd",
    "grmhd",
    "grrmhd",
    "z4c-grhd",
    "z4c-grmhd",
    "z4c-grrmhd",
]
_FORMULATIONS = frozenset(
    ("z4c", "grhd", "grmhd", "grrmhd", "z4c-grhd", "z4c-grmhd", "z4c-grrmhd")
)


def _formulation(value: str, /) -> NumericalRelativityFormulation:
    normalized = str(value)
    if normalized not in _FORMULATIONS:
        raise ValueError(f"Unknown numerical-relativity formulation {normalized!r}.")
    return normalized  # type: ignore[return-value]


def formulation_field_names(
    formulation: NumericalRelativityFormulation, /
) -> tuple[str, ...]:
    """Return the canonical independent restart fields for one formulation."""

    normalized = _formulation(formulation)
    if normalized == "z4c":
        return ("z4c",)
    if normalized == "grhd":
        return ("material",)
    if normalized == "grmhd":
        return ("material", "magnetic_flux")
    if normalized == "grrmhd":
        return ("material", "radiation", "magnetic_flux")
    if normalized == "z4c-grhd":
        return ("z4c", "material")
    if normalized == "z4c-grmhd":
        return ("z4c", "material", "magnetic_flux")
    return ("z4c", "material", "radiation", "magnetic_flux")


class NumericalRelativityOwnership(StrictModule, NonTrainableState):
    """Fixed-capacity stable-block ownership and owner-local slot evidence."""

    owner_indices: Array
    local_indices: Array
    active: Array
    device_count: int = eqx.field(static=True)
    per_device_capacity: int = eqx.field(static=True)
    single_device_authority: bool = eqx.field(static=True)
    ownership_id: str = eqx.field(static=True)

    def __init__(
        self,
        owner_indices: ArrayLike,
        local_indices: ArrayLike,
        active: ArrayLike,
        /,
        *,
        device_count: int,
        per_device_capacity: int,
        plan_id: str,
    ):
        owners = np.asarray(owner_indices, dtype=np.int32)
        local = np.asarray(local_indices, dtype=np.int32)
        mask = np.asarray(active, dtype=np.bool_)
        devices = int(device_count)
        capacity = int(per_device_capacity)
        if (
            owners.ndim != 1
            or local.shape != owners.shape
            or mask.shape != owners.shape
            or devices <= 0
            or capacity <= 0
            or np.any(owners[mask] < 0)
            or np.any(owners[mask] >= devices)
            or np.any(local[mask] < 0)
            or np.any(owners[~mask] != -1)
            or np.any(local[~mask] != -1)
            or not plan_id
        ):
            raise ValueError("Numerical-relativity ownership metadata is invalid.")
        self.owner_indices = jnp.asarray(owners)
        self.local_indices = jnp.asarray(local)
        self.active = jnp.asarray(mask)
        self.device_count = devices
        self.per_device_capacity = capacity
        self.single_device_authority = devices == 1
        self.ownership_id = canonical_fingerprint(
            {
                "kind": "numerical-relativity-amr-ownership",
                "plan": plan_id,
                "owners": array_tree_fingerprint(owners),
                "local_indices": array_tree_fingerprint(local),
                "active": array_tree_fingerprint(mask),
                "per_device_capacity": capacity,
            }
        )


class DistributedCochainState(StrictModule):
    """An oriented cochain whose fixed components retain named shardings."""

    components: tuple[Array, ...]
    degree: int = eqx.field(static=True)
    bridge_id: str = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)


class NumericalRelativityDistributedPlan(StrictModule, NonTrainableState):
    """Cartesian named-sharding policy shared by Z4c, GRHD, and GRMHD."""

    formulation: NumericalRelativityFormulation = eqx.field(static=True)
    grid_id: str = eqx.field(static=True)
    decomposition: FiniteVolumeDecompositionPlan
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        formulation: NumericalRelativityFormulation,
        global_shape: Sequence[int],
        split_factors: Sequence[int],
        /,
        *,
        axis_names: Sequence[str] = ("x", "y", "z"),
        halo_width: int = 3,
        periodic: Sequence[bool] | None = None,
        grid_id: str,
    ):
        formulation_ = _formulation(formulation)
        identifier = str(grid_id).strip()
        if not identifier:
            raise ValueError("Grid identity must be non-empty.")
        decomposition = FiniteVolumeDecompositionPlan(
            global_shape,
            split_factors,
            axis_names,
            halo_width=halo_width,
            periodic=periodic,
        )
        if len(decomposition.global_shape) != 3:
            raise ValueError(
                "Numerical-relativity production decomposition is three-dimensional."
            )
        self.formulation = formulation_
        self.grid_id = identifier
        self.decomposition = decomposition
        self.plan_id = canonical_fingerprint(
            {
                "kind": "numerical-relativity-distributed-plan",
                "formulation": formulation_,
                "grid": identifier,
                "decomposition": decomposition.plan_id,
            }
        )

    def prepare(
        self,
        devices: Sequence[jax.Device] | None = None,
        /,
        *,
        execution_group: ExecutionGroup | None = None,
    ) -> PreparedNumericalRelativityDistributed:
        if devices is not None and execution_group is not None:
            raise ValueError("devices and execution_group are mutually exclusive.")
        prepared = self.decomposition.prepare(
            devices,
            execution_group=execution_group,
        )
        return PreparedNumericalRelativityDistributed(self, prepared)


class PreparedNumericalRelativityDistributed(StrictModule, NonTrainableState):
    """Prepared named mesh with explicit field layouts and stable ownership."""

    plan: NumericalRelativityDistributedPlan
    decomposition: PreparedFiniteVolumeDecomposition
    z4c_sharding: NamedSharding
    material_sharding: NamedSharding
    radiation_sharding: NamedSharding
    scalar_sharding: NamedSharding
    replicated_sharding: NamedSharding
    device_count: int = eqx.field(static=True)
    single_device_authority: bool = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: NumericalRelativityDistributedPlan,
        decomposition: PreparedFiniteVolumeDecomposition,
        /,
    ):
        if not isinstance(plan, NumericalRelativityDistributedPlan) or not isinstance(
            decomposition, PreparedFiniteVolumeDecomposition
        ):
            raise TypeError(
                "Prepared NR distribution requires its plan and FV decomposition."
            )
        if decomposition.plan.plan_id != plan.decomposition.plan_id:
            raise ValueError("Prepared FV decomposition does not belong to the NR plan.")
        spatial = tuple(
            mesh_axis if split > 1 else None
            for mesh_axis, split in zip(
                decomposition.mesh_axis_names,
                plan.decomposition.split_factors,
                strict=True,
            )
        )
        devices = prod(plan.decomposition.split_factors)
        self.plan = plan
        self.decomposition = decomposition
        self.z4c_sharding = NamedSharding(
            decomposition.mesh, PartitionSpec(None, *spatial)
        )
        self.material_sharding = decomposition.cell_sharding
        self.radiation_sharding = decomposition.cell_sharding
        self.scalar_sharding = NamedSharding(decomposition.mesh, PartitionSpec(*spatial))
        self.replicated_sharding = NamedSharding(decomposition.mesh, PartitionSpec())
        self.device_count = devices
        self.single_device_authority = devices == 1
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-numerical-relativity-distribution",
                "plan": plan.plan_id,
                "decomposition": decomposition.prepared_id,
            }
        )

    def shard_z4c(self, values: ArrayLike, /) -> Array:
        field = jnp.asarray(values)
        expected = (25,) + self.plan.decomposition.global_shape
        if field.shape != expected:
            raise ValueError(f"Packed Z4c field must have shape {expected}.")
        return jax.device_put(field, self.z4c_sharding)

    def shard_material(self, values: ArrayLike, /) -> Array:
        field = jnp.asarray(values)
        expected = self.plan.decomposition.global_shape + (5,)
        if field.shape != expected:
            raise ValueError(f"Relativistic material state must have shape {expected}.")
        return jax.device_put(field, self.material_sharding)

    def shard_radiation(self, values: ArrayLike, /) -> Array:
        field = jnp.asarray(values)
        expected = self.plan.decomposition.global_shape + (4,)
        if field.shape != expected:
            raise ValueError(f"Relativistic radiation state must have shape {expected}.")
        return jax.device_put(field, self.radiation_sharding)

    def shard_scalar(self, values: ArrayLike, /) -> Array:
        field = jnp.asarray(values)
        if field.shape != self.plan.decomposition.global_shape:
            raise ValueError("NR scalar field does not match the distributed grid.")
        return jax.device_put(field, self.scalar_sharding)

    def periodic_z4c_halo(self, values: ArrayLike, axis: int, /) -> Array:
        """Materialize one periodic Z4c halo through the prepared FV route."""

        field = jnp.asarray(values)
        expected = (25,) + self.plan.decomposition.global_shape
        if field.shape != expected:
            raise ValueError(f"Packed Z4c field must have shape {expected}.")
        component_last = jnp.moveaxis(field, 0, -1)
        halo = self.decomposition.periodic_halo(component_last, axis)
        return jnp.moveaxis(halo, -1, 0)

    def periodic_material_halo(self, values: ArrayLike, axis: int, /) -> Array:
        """Materialize one periodic material halo through the named decomposition."""

        field = jnp.asarray(values)
        expected = self.plan.decomposition.global_shape + (5,)
        if field.shape != expected:
            raise ValueError(f"Relativistic material state must have shape {expected}.")
        return self.decomposition.periodic_halo(field, axis)

    def periodic_radiation_halo(self, values: ArrayLike, axis: int, /) -> Array:
        field = jnp.asarray(values)
        expected = self.plan.decomposition.global_shape + (4,)
        if field.shape != expected:
            raise ValueError(f"Relativistic radiation state must have shape {expected}.")
        return self.decomposition.periodic_halo(field, axis)

    def periodic_scalar_halo(self, values: ArrayLike, axis: int, /) -> Array:
        field = jnp.asarray(values)
        if field.shape != self.plan.decomposition.global_shape:
            raise ValueError("NR scalar field does not match the distributed grid.")
        return self.decomposition.periodic_halo(field, axis)

    def shard_cochain(
        self,
        bridge: StructuredCochainBridge,
        degree: int,
        packed_values: ArrayLike,
        /,
    ) -> DistributedCochainState:
        """Shard each oriented cochain component without flattening its topology."""

        if not isinstance(bridge, StructuredCochainBridge):
            raise TypeError("bridge must be a StructuredCochainBridge.")
        if tuple(bridge.grid.shape) != self.plan.decomposition.global_shape:
            raise ValueError("Cochain bridge and distributed NR grid shapes differ.")
        degree_ = int(degree)
        components = bridge.unpack(degree_, packed_values)
        shardings = self.cochain_component_shardings(bridge, degree_)
        distributed = tuple(
            jax.device_put(component, sharding)
            for component, sharding in zip(components, shardings, strict=True)
        )
        return DistributedCochainState(
            distributed,
            degree_,
            bridge.bridge_id,
            canonical_fingerprint(
                {
                    "kind": "distributed-numerical-relativity-cochain",
                    "distribution": self.prepared_id,
                    "bridge": bridge.bridge_id,
                    "degree": degree_,
                    "shapes": [list(component.shape) for component in components],
                    "specifications": [str(value.spec) for value in shardings],
                }
            ),
        )

    def cochain_component_shardings(
        self,
        bridge: StructuredCochainBridge,
        degree: int,
        /,
    ) -> tuple[NamedSharding, ...]:
        degree_ = int(degree)
        if not isinstance(bridge, StructuredCochainBridge):
            raise TypeError("bridge must be a StructuredCochainBridge.")
        if degree_ < 0 or degree_ > bridge.dimension:
            raise ValueError("Cochain degree is outside the bridge dimension.")
        split_factors = self.plan.decomposition.split_factors
        mesh_axes = self.decomposition.mesh_axis_names
        shardings = []
        for shape in bridge.orientation_shapes[degree_]:
            specification = tuple(
                mesh_axis if split > 1 and extent % split == 0 else None
                for extent, split, mesh_axis in zip(
                    shape, split_factors, mesh_axes, strict=True
                )
            )
            shardings.append(
                NamedSharding(self.decomposition.mesh, PartitionSpec(*specification))
            )
        return tuple(shardings)


class NumericalRelativityAMRDistributionPlan(StrictModule, NonTrainableState):
    """NR identity over the authoritative Morton-contiguous block partition."""

    formulation: NumericalRelativityFormulation = eqx.field(static=True)
    partition: BlockAMRPartitionPlan
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        formulation: NumericalRelativityFormulation,
        hierarchy: BlockHierarchyPlan,
        part_count: int,
        /,
        *,
        axis_name: str = "nr_blocks",
    ):
        formulation_ = _formulation(formulation)
        partition = BlockAMRPartitionPlan(
            hierarchy,
            part_count,
            axis_name=axis_name,
        )
        self.formulation = formulation_
        self.partition = partition
        self.plan_id = canonical_fingerprint(
            {
                "kind": "numerical-relativity-amr-distribution",
                "formulation": formulation_,
                "partition": partition.plan_id,
            }
        )

    def prepare(
        self,
        topology: BlockTopologyCompileResult | BlockHierarchyTopology,
        hierarchy: PreparedFDAMRHierarchy,
        /,
        *,
        costs: Sequence[ArrayLike | None] | None = None,
        execution_group: ExecutionGroup | None = None,
    ) -> PreparedNumericalRelativityAMRDistribution:
        prepared = self.partition.prepare(
            topology,
            hierarchy,
            costs=costs,
            execution_group=execution_group,
        )
        return PreparedNumericalRelativityAMRDistribution(self, prepared)


class PreparedNumericalRelativityAMRDistribution(StrictModule, NonTrainableState):
    """Owner-computes block hierarchy with canonical FillPatch route transposes."""

    plan: NumericalRelativityAMRDistributionPlan
    distribution: PreparedDistributedBlockAMRHierarchy
    ownership: tuple[NumericalRelativityOwnership, ...]
    single_device_authority: bool = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: NumericalRelativityAMRDistributionPlan,
        distribution: PreparedDistributedBlockAMRHierarchy,
        /,
    ):
        if not isinstance(plan, NumericalRelativityAMRDistributionPlan) or not isinstance(
            distribution, PreparedDistributedBlockAMRHierarchy
        ):
            raise TypeError(
                "Prepared NR AMR distribution requires its plan and block distribution."
            )
        if distribution.partition.plan_id != plan.partition.plan_id:
            raise ValueError(
                "Prepared block distribution does not belong to the NR plan."
            )
        owners = tuple(
            NumericalRelativityOwnership(
                layout.block_owner,
                layout.canonical_to_local,
                distribution.topology.levels[level].active,
                device_count=layout.part_count,
                per_device_capacity=layout.local_block_capacity,
                plan_id=layout.layout_id,
            )
            for level, layout in enumerate(distribution.layouts)
        )
        self.plan = plan
        self.distribution = distribution
        self.ownership = owners
        self.single_device_authority = plan.partition.part_count == 1
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-numerical-relativity-amr-distribution",
                "plan": plan.plan_id,
                "distribution": distribution.prepared_id,
                "ownership": [value.ownership_id for value in owners],
            }
        )

    def pack(self, state: BlockHierarchyState, /) -> tuple[Array, ...]:
        return self.distribution.pack(state)

    def unpack(self, packed_values: Sequence[ArrayLike], /) -> BlockHierarchyState:
        return self.distribution.unpack(packed_values)


__all__ = [
    "DistributedCochainState",
    "NumericalRelativityAMRDistributionPlan",
    "NumericalRelativityDistributedPlan",
    "NumericalRelativityFormulation",
    "NumericalRelativityOwnership",
    "PreparedNumericalRelativityAMRDistribution",
    "PreparedNumericalRelativityDistributed",
    "formulation_field_names",
]
