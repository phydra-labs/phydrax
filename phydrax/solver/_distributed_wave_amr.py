#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Owner-computes distributed complex-wave evolution on block AMR."""

from __future__ import annotations

from collections.abc import Sequence
from math import pi, prod
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec
from jaxtyping import Array, ArrayLike

from .._execution_plan import ExecutionPlan
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._distributed_field import DistributedHaloPlan
from ..discretization.amr._distributed import (
    BlockAMRStableIDMigrationPlan,
    PreparedDistributedBlockAMRHierarchy,
)
from ..discretization.amr._topology_transfer import BlockFieldTopologyTransition
from ..lifecycle._chunk_repository import ArtifactRepository
from ..lifecycle._distributed_checkpoint import (
    ProcessCheckpointPublication,
    publish_process_checkpoint,
    restore_global_array_from_checkpoint,
)
from ..lifecycle._models import CheckpointManifest


def _part_spec(axis_name: str, rank: int, /) -> PartitionSpec:
    return PartitionSpec(axis_name, *((None,) * (rank - 1)))


def _global_all(value: Array, axis_name: str, /) -> Array:
    return jax.lax.pmin(jnp.asarray(value, dtype=jnp.int32), axis_name) == 1


def _global_agreement(value: Array, axis_name: str, /) -> Array:
    encoded = jnp.asarray(value, dtype=jnp.int32)
    return jax.lax.pmin(encoded, axis_name) == jax.lax.pmax(encoded, axis_name)


def _safe_halo_exchange(
    halo: DistributedHaloPlan,
    local: Array,
    part: Array,
    axis_name: str,
    /,
) -> Array:
    """Exchange fixed packets without allowing padded receives to overwrite row zero."""
    values = local
    for phase, permutation in enumerate(halo.permutations):
        send_indices = halo.phase_send_indices[phase, part]
        send_valid = halo.phase_send_valid[phase, part]
        payload = values[send_indices]
        payload = jnp.where(
            send_valid.reshape(send_valid.shape + (1,) * (payload.ndim - 1)),
            payload,
            0,
        )
        received = jax.lax.ppermute(
            payload,
            axis_name=axis_name,
            perm=permutation,
        )
        receive_indices = halo.phase_receive_indices[phase, part]
        receive_valid = halo.phase_receive_valid[phase, part]
        destination = jnp.any(
            jnp.asarray(
                [target == part for _, target in permutation],
                dtype=jnp.bool_,
            )
        )
        values = jax.lax.cond(
            destination,
            lambda current: current.at[receive_indices].set(
                jnp.where(
                    receive_valid.reshape(
                        receive_valid.shape + (1,) * (received.ndim - 1)
                    ),
                    received,
                    current[receive_indices],
                )
            ),
            lambda current: current,
            values,
        )
    return values


class DistributedWaveAMRState(StrictModule):
    """Packed complex block shards at one globally accepted scale-factor level."""

    psi: tuple[Array, ...]
    scale_factor: Array
    accepted_boundary: Array
    execution_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    partition_id: str = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    physics_id: str = eqx.field(static=True)


class DistributedWaveAMRLinearEvidence(StrictModule):
    iterations: Array
    residual_norm: Array
    rhs_norm: Array
    relative_residual: Array
    finite: Array
    converged: Array
    method: str = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)


class DistributedWaveAMRGravityEvidence(StrictModule):
    potential: tuple[Array, ...]
    source: tuple[Array, ...]
    mean_density: Array
    source_integral: Array
    gauge_defect: Array
    interface_flux_conservation_defect: Array
    solve: DistributedWaveAMRLinearEvidence
    finite: Array
    successful: Array
    operator_id: str = eqx.field(static=True)


class DistributedWaveAMRDiagnostics(StrictModule):
    initial_probability: Array
    final_probability: Array
    probability_relative_error: Array
    initial_current: Array
    final_current: Array
    current_relative_defect: Array
    current_absolute_defect: Array
    cayley_relative_residual: Array
    self_adjoint_residual: Array
    maximum_kinetic_phase: Array
    maximum_potential_phase: Array
    kinetic_energy: Array
    potential_energy: Array
    finite: Array
    initial_poisson_closed: Array
    final_poisson_closed: Array
    poisson_closed: Array
    kinetic_closed: Array
    halo_complete: Array
    rank_agreement: Array
    accepted: Array
    status: Array
    status_meanings: tuple[str, ...] = eqx.field(
        static=True,
        default=(
            "accepted",
            "non_finite",
            "poisson_failed",
            "kinetic_solve_failed",
            "cayley_residual",
            "self_adjoint_residual",
            "probability_drift",
            "phase_unresolved",
            "collective_disagreement",
            "halo_incomplete",
        ),
    )


class DistributedWaveAMRResult(StrictModule):
    state: DistributedWaveAMRState
    candidate_state: DistributedWaveAMRState
    diagnostics: DistributedWaveAMRDiagnostics
    gravity: DistributedWaveAMRGravityEvidence
    second_gravity: DistributedWaveAMRGravityEvidence
    kinetic_solve: DistributedWaveAMRLinearEvidence
    successful: Array
    execution_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)


class DistributedWaveAMRCheckpointEvidence(StrictModule, NonTrainableState):
    checkpoint_id: str = eqx.field(static=True)
    execution_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    partition_id: str = eqx.field(static=True)
    array_paths: tuple[str, ...] = eqx.field(static=True)
    exact_coverage_required: bool = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class DistributedWaveAMRRestoreEvidence(StrictModule, NonTrainableState):
    source_execution_id: str = eqx.field(static=True)
    target_execution_id: str = eqx.field(static=True)
    manifest_id: str = eqx.field(static=True)
    migration_id: str | None = eqx.field(static=True)
    changed_partition: bool = eqx.field(static=True)
    exact_coverage: bool = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class DistributedWaveAMRObservables(StrictModule):
    probability: Array
    current: Array
    winding: Array
    node_count: Array
    vortex_count: Array
    finite: Array


class DistributedWaveAMRTopologyTransferResult(StrictModule):
    candidate: DistributedWaveAMRState
    probability_density_conservation_defect: Array
    phase_defect: Array
    negative_density: Array
    route_complete: Array
    finite: Array
    successful: Array
    transition_id: str = eqx.field(static=True)


class _PackedCompositeRoutes(StrictModule, NonTrainableState):
    """Compact leaf ownership, exact face routes, and block-packing relations."""

    halo: DistributedHaloPlan
    owned_packed_indices: Array
    entity_packed_indices: Array
    local_cell_measures: Array
    entity_storage_indices: Array
    entity_measures: Array
    local_entity_ids: Array
    edge_left_local: Array
    edge_right_local: Array
    edge_weights: Array
    edge_axis: Array
    edge_area: Array
    edge_level_jump: Array
    edge_valid: Array
    global_edge_left: Array
    global_edge_right: Array
    global_edge_axis: Array
    global_edge_area: Array
    level_leaf_masks: tuple[Array, ...]
    level_local_shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    level_local_sizes: tuple[int, ...] = eqx.field(static=True)
    local_packed_size: int = eqx.field(static=True)
    edge_capacity: int = eqx.field(static=True)
    crossing_edge_count: int = eqx.field(static=True)
    level_jump_edge_count: int = eqx.field(static=True)
    dynamic_array_bytes: int = eqx.field(static=True)
    route_id: str = eqx.field(static=True)

    def __init__(
        self,
        distribution: PreparedDistributedBlockAMRHierarchy,
        layout: Any,
        diffusion: Any,
        /,
    ):
        flat_leaf = np.asarray(layout.flat_leaf_mask, dtype=np.bool_)
        leaf_cells = np.flatnonzero(flat_leaf).astype(np.int32)
        compact = np.full((flat_leaf.size,), -1, dtype=np.int32)
        compact[leaf_cells] = np.arange(leaf_cells.size, dtype=np.int32)
        part_count = distribution.partition.part_count
        level_local_shapes = tuple(
            (level_layout.local_block_capacity, *level_plan.block_shape)
            for level_layout, level_plan in zip(
                distribution.layouts, distribution.topology.plan.levels, strict=True
            )
        )
        level_local_sizes = tuple(prod(shape) for shape in level_local_shapes)
        level_offsets = np.cumsum((0, *level_local_sizes[:-1]), dtype=np.int64)
        entity_owner = np.empty((leaf_cells.size,), dtype=np.int32)
        entity_packed = np.empty((leaf_cells.size,), dtype=np.int32)
        for entity, canonical_cell in enumerate(leaf_cells):
            level = max(
                index
                for index, offset in enumerate(layout.real_layout.cell_offsets)
                if int(offset) <= int(canonical_cell)
            )
            level_plan = distribution.topology.plan.levels[level]
            block_cells = prod(level_plan.block_shape)
            level_cell = int(canonical_cell) - int(layout.real_layout.cell_offsets[level])
            block_slot, local_cell = divmod(level_cell, block_cells)
            owner = int(np.asarray(distribution.layouts[level].block_owner)[block_slot])
            local_block = int(
                np.asarray(distribution.layouts[level].canonical_to_local)[block_slot]
            )
            if owner < 0 or local_block < 0:
                raise ValueError("A composite leaf cell has no distributed block owner.")
            entity_owner[entity] = owner
            entity_packed[entity] = (
                int(level_offsets[level]) + local_block * block_cells + local_cell
            )
        if set(entity_owner.tolist()) != set(range(part_count)):
            raise ValueError(
                "Distributed Wave AMR needs at least one composite leaf cell per part."
            )

        face_routes = diffusion.plan.routes
        canonical_left = np.asarray(face_routes.edge_left, dtype=np.int32)
        canonical_right = np.asarray(face_routes.edge_right, dtype=np.int32)
        left = compact[canonical_left]
        right = compact[canonical_right]
        if np.any(left < 0) or np.any(right < 0):
            raise ValueError("Composite face routes reference non-leaf storage.")
        adjacency = np.stack((left, right), axis=1)
        halo = DistributedHaloPlan(entity_owner, adjacency, part_count)
        halo_ids = np.asarray(halo.local_global_ids, dtype=np.int32)
        halo_valid = np.asarray(halo.local_valid, dtype=np.bool_)
        halo_owned = np.asarray(halo.local_owned, dtype=np.bool_)
        owned_packed = np.zeros_like(halo_ids)
        local_entity_ids = np.zeros_like(halo_ids)
        measures = np.ones(halo_ids.shape, dtype=np.dtype(layout.real_dtype))
        compact_measures = np.asarray(layout.cell_measures)[leaf_cells]
        local_maps: list[dict[int, int]] = []
        for part in range(part_count):
            local_maps.append(
                {
                    int(entity): local
                    for local, entity in enumerate(halo_ids[part])
                    if halo_valid[part, local]
                }
            )
            for local, entity in enumerate(halo_ids[part]):
                if not halo_valid[part, local]:
                    continue
                local_entity_ids[part, local] = int(entity)
                measures[part, local] = compact_measures[entity]
                if halo_owned[part, local]:
                    owned_packed[part, local] = entity_packed[entity]

        records: list[list[tuple[int, int, float, int, float, bool]]] = [
            [] for _ in range(part_count)
        ]
        edge_weight = np.asarray(diffusion.edge_weights)
        edge_axis = np.asarray(face_routes.edge_axis, dtype=np.int32)
        edge_area = np.asarray(face_routes.edge_area)
        level_jump = np.asarray(face_routes.edge_level_jump, dtype=np.bool_)
        crossing = 0
        for index, (left_entity, right_entity) in enumerate(
            zip(left, right, strict=True)
        ):
            owner = int(entity_owner[left_entity])
            if entity_owner[left_entity] != entity_owner[right_entity]:
                crossing += 1
            records[owner].append(
                (
                    local_maps[owner][int(left_entity)],
                    local_maps[owner][int(right_entity)],
                    float(edge_weight[index]),
                    int(edge_axis[index]),
                    float(edge_area[index]),
                    bool(level_jump[index]),
                )
            )
        edge_capacity = max(1, *(len(value) for value in records))
        edge_left = np.zeros((part_count, edge_capacity), dtype=np.int32)
        edge_right = np.zeros_like(edge_left)
        weights = np.zeros((part_count, edge_capacity), dtype=edge_weight.dtype)
        axes = np.zeros((part_count, edge_capacity), dtype=np.int32)
        areas = np.zeros((part_count, edge_capacity), dtype=edge_area.dtype)
        jumps = np.zeros((part_count, edge_capacity), dtype=np.bool_)
        valid = np.zeros((part_count, edge_capacity), dtype=np.bool_)
        for part, part_records in enumerate(records):
            count = len(part_records)
            if count == 0:
                continue
            edge_left[part, :count] = [value[0] for value in part_records]
            edge_right[part, :count] = [value[1] for value in part_records]
            weights[part, :count] = [value[2] for value in part_records]
            axes[part, :count] = [value[3] for value in part_records]
            areas[part, :count] = [value[4] for value in part_records]
            jumps[part, :count] = [value[5] for value in part_records]
            valid[part, :count] = True
        packed_leaf_masks = tuple(
            level_layout.pack(mask)
            for level_layout, mask in zip(
                distribution.layouts, layout.leaf_mask, strict=True
            )
        )

        dynamic = (
            owned_packed,
            measures,
            local_entity_ids,
            edge_left,
            edge_right,
            weights,
            axes,
            areas,
            jumps,
            valid,
            entity_packed,
            left,
            leaf_cells,
            compact_measures,
            right,
            edge_axis,
            edge_area,
            np.asarray(halo.local_global_ids),
            np.asarray(halo.local_valid),
            np.asarray(halo.local_owned),
            np.asarray(halo.phase_send_indices),
            np.asarray(halo.phase_receive_indices),
            np.asarray(halo.phase_send_valid),
            np.asarray(halo.phase_receive_valid),
        )
        dynamic_bytes = sum(value.size * value.dtype.itemsize for value in dynamic) + sum(
            value.size * value.dtype.itemsize for value in packed_leaf_masks
        )
        route_id = canonical_fingerprint(
            {
                "kind": "distributed-wave-amr-composite-routes",
                "distribution": distribution.prepared_id,
                "layout": layout.layout_id,
                "operator": diffusion.operator_id,
                "halo": halo.plan_id,
                "owned_packed": array_tree_fingerprint(owned_packed),
                "edge_left": array_tree_fingerprint(edge_left),
                "edge_right": array_tree_fingerprint(edge_right),
                "edge_weights": array_tree_fingerprint(weights),
                "edge_valid": array_tree_fingerprint(valid),
            }
        )
        self.halo = halo
        self.owned_packed_indices = jnp.asarray(owned_packed)
        self.entity_packed_indices = jnp.asarray(entity_packed)
        self.local_cell_measures = jnp.asarray(measures, dtype=layout.real_dtype)
        self.entity_storage_indices = jnp.asarray(leaf_cells)
        self.entity_measures = jnp.asarray(compact_measures, dtype=layout.real_dtype)
        self.local_entity_ids = jnp.asarray(local_entity_ids)
        self.edge_left_local = jnp.asarray(edge_left)
        self.edge_right_local = jnp.asarray(edge_right)
        self.edge_weights = jnp.asarray(weights, dtype=layout.real_dtype)
        self.edge_axis = jnp.asarray(axes)
        self.edge_area = jnp.asarray(areas, dtype=layout.real_dtype)
        self.edge_level_jump = jnp.asarray(jumps)
        self.edge_valid = jnp.asarray(valid)
        self.global_edge_left = jnp.asarray(left)
        self.global_edge_right = jnp.asarray(right)
        self.global_edge_axis = jnp.asarray(edge_axis)
        self.global_edge_area = jnp.asarray(edge_area, dtype=layout.real_dtype)
        self.level_leaf_masks = packed_leaf_masks
        self.level_local_shapes = level_local_shapes
        self.level_local_sizes = level_local_sizes
        self.local_packed_size = sum(level_local_sizes)
        self.edge_capacity = edge_capacity
        self.crossing_edge_count = crossing
        self.level_jump_edge_count = int(np.count_nonzero(level_jump))
        self.dynamic_array_bytes = int(dynamic_bytes)
        self.route_id = route_id


class PreparedDistributedWaveAMR(StrictModule, NonTrainableState):
    """Prepared global composite operator and bounded collective solve owner."""

    hierarchy: PreparedDistributedBlockAMRHierarchy
    routes: _PackedCompositeRoutes
    execution_plan: ExecutionPlan | None = eqx.field(static=True)
    boson_mass: float = eqx.field(static=True)
    gravitational_constant: float = eqx.field(static=True)
    reduced_planck_constant: float = eqx.field(static=True)
    solve_relative_tolerance: float = eqx.field(static=True)
    solve_absolute_tolerance: float = eqx.field(static=True)
    maximum_solve_steps: int = eqx.field(static=True)
    norm_relative_tolerance: float = eqx.field(static=True)
    self_adjoint_tolerance: float = eqx.field(static=True)
    maximum_phase_radians: float = eqx.field(static=True)
    kinetic_spectral_upper_bound: float = eqx.field(static=True)
    real_dtype: np.dtype = eqx.field(static=True)
    complex_dtype: np.dtype = eqx.field(static=True)
    source_prepared_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    partition_id: str = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    authority_operator_id: str = eqx.field(static=True)
    physics_id: str = eqx.field(static=True)
    mesh_id: str | None = eqx.field(static=True)
    required_bytes: int = eqx.field(static=True)
    execution_id: str = eqx.field(static=True)

    def __init__(
        self,
        hierarchy: PreparedDistributedBlockAMRHierarchy,
        layout: Any,
        diffusion: Any,
        /,
        *,
        execution_plan: ExecutionPlan | None,
        boson_mass: float,
        gravitational_constant: float,
        reduced_planck_constant: float,
        solve_relative_tolerance: float,
        solve_absolute_tolerance: float,
        maximum_solve_steps: int,
        norm_relative_tolerance: float,
        self_adjoint_tolerance: float,
        maximum_phase_radians: float,
        kinetic_spectral_upper_bound: float,
        source_prepared_id: str,
        physics_id: str,
    ):
        if not isinstance(hierarchy, PreparedDistributedBlockAMRHierarchy):
            raise TypeError("hierarchy must be PreparedDistributedBlockAMRHierarchy.")
        if execution_plan is not None and not isinstance(execution_plan, ExecutionPlan):
            raise TypeError("execution_plan must be ExecutionPlan or None.")
        routes = _PackedCompositeRoutes(hierarchy, layout, diffusion)
        complex_dtype = np.dtype(layout.dtype)
        real_dtype = np.dtype(layout.real_dtype)
        local_cells = routes.local_packed_size
        parts = hierarchy.partition.part_count
        # Two gravity solves, CGNE Cayley vectors, candidate/rollback, and diagnostics.
        workspace = (
            parts * local_cells * (12 * complex_dtype.itemsize + 10 * real_dtype.itemsize)
        )
        required = (
            hierarchy.resources.dynamic_route_array_bytes
            + routes.dynamic_array_bytes
            + workspace
        )
        mesh_id = None
        if hierarchy.mesh is not None:
            mesh_id = canonical_fingerprint(
                {
                    "axis_names": list(hierarchy.mesh.axis_names),
                    "device_keys": [
                        [device.process_index, device.id]
                        for device in hierarchy.mesh.devices.flat
                    ],
                }
            )
        operator_id = canonical_fingerprint(
            {
                "kind": "distributed-wave-amr-composite-operator",
                "local_operator": diffusion.operator_id,
                "routes": routes.route_id,
                "pairing": layout.space.space_id,
                "convention": "owner-computes-volume-paired-positive-laplacian",
            }
        )
        execution_id = canonical_fingerprint(
            {
                "kind": "prepared-distributed-wave-amr",
                "source_prepared": source_prepared_id,
                "hierarchy": hierarchy.prepared_id,
                "topology": hierarchy.topology.topology_id,
                "partition": hierarchy.partition.plan_id,
                "operator": operator_id,
                "physics": physics_id,
                "mesh": mesh_id,
                "execution_plan": (
                    None if execution_plan is None else execution_plan.plan_fingerprint
                ),
                "maximum_solve_steps": int(maximum_solve_steps),
                "required_bytes": required,
            }
        )
        self.hierarchy = hierarchy
        self.routes = routes
        self.execution_plan = execution_plan
        self.boson_mass = float(boson_mass)
        self.gravitational_constant = float(gravitational_constant)
        self.reduced_planck_constant = float(reduced_planck_constant)
        self.solve_relative_tolerance = float(solve_relative_tolerance)
        self.solve_absolute_tolerance = float(solve_absolute_tolerance)
        self.maximum_solve_steps = int(maximum_solve_steps)
        self.norm_relative_tolerance = float(norm_relative_tolerance)
        self.self_adjoint_tolerance = float(self_adjoint_tolerance)
        self.maximum_phase_radians = float(maximum_phase_radians)
        self.kinetic_spectral_upper_bound = float(kinetic_spectral_upper_bound)
        self.real_dtype = real_dtype
        self.complex_dtype = complex_dtype
        self.source_prepared_id = str(source_prepared_id)
        self.topology_id = hierarchy.topology.topology_id
        self.partition_id = hierarchy.partition.plan_id
        self.authority_operator_id = diffusion.operator_id
        self.operator_id = operator_id
        self.physics_id = str(physics_id)
        self.mesh_id = mesh_id
        self.required_bytes = required
        self.execution_id = execution_id

    @property
    def executable(self) -> bool:
        return self.hierarchy.mesh is not None

    @property
    def route_crossing_count(self) -> int:
        return self.routes.crossing_edge_count

    def _place(self, values: Sequence[ArrayLike], /) -> tuple[Array, ...]:
        arrays = tuple(jnp.asarray(value, dtype=self.complex_dtype) for value in values)
        expected = tuple(
            (self.hierarchy.partition.part_count, *shape)
            for shape in self.routes.level_local_shapes
        )
        if tuple(value.shape for value in arrays) != expected:
            raise ValueError(
                "Packed wave shards do not match distributed block capacities."
            )
        if self.hierarchy.mesh is None:
            return arrays
        axis = self.hierarchy.partition.axis_name
        return tuple(
            jax.device_put(
                value,
                NamedSharding(self.hierarchy.mesh, _part_spec(axis, value.ndim)),
            )
            for value in arrays
        )

    def pack_canonical_values(self, values: Sequence[ArrayLike], /) -> tuple[Array, ...]:
        arrays = tuple(jnp.asarray(value, dtype=self.complex_dtype) for value in values)
        expected = tuple(
            (
                level_plan.maximum_blocks,
                *level_plan.block_shape,
            )
            for level_plan in self.hierarchy.topology.plan.levels
        )
        if tuple(value.shape for value in arrays) != expected:
            raise ValueError("Canonical wave values do not match the AMR hierarchy.")
        packed = tuple(
            level_layout.pack(value)
            for level_layout, value in zip(self.hierarchy.layouts, arrays, strict=True)
        )
        return self._place(packed)

    def _replicate_scalar(self, value: ArrayLike, dtype: Any, /) -> Array:
        scalar = jnp.asarray(value, dtype=dtype)
        if scalar.shape != ():
            raise ValueError("Distributed Wave AMR scalar state must be rank zero.")
        if self.hierarchy.mesh is None:
            return scalar
        return jax.device_put(
            scalar,
            NamedSharding(self.hierarchy.mesh, PartitionSpec()),
        )

    def _bind_unchecked(
        self,
        arrays: tuple[Array, ...],
        scale: Array,
        accepted: Array,
        /,
    ) -> DistributedWaveAMRState:
        return DistributedWaveAMRState(
            arrays,
            scale,
            accepted,
            self.execution_id,
            self.topology_id,
            self.partition_id,
            self.operator_id,
            self.physics_id,
        )

    def _validate_payload(
        self,
        arrays: tuple[Array, ...],
        scale: Array,
        accepted: Array,
        /,
    ) -> None:
        finite = jnp.asarray(True)
        zero_masked = jnp.asarray(True)
        probability = jnp.asarray(0.0, dtype=self.real_dtype)
        for value, mask, spacing in zip(
            arrays,
            self.routes.level_leaf_masks,
            self.hierarchy.topology.plan.level_spacings,
            strict=True,
        ):
            finite = finite & jnp.all(jnp.isfinite(value))
            zero_masked = zero_masked & jnp.all(
                jnp.where(mask, jnp.zeros((), dtype=value.dtype), value) == 0
            )
            probability = probability + prod(spacing) * jnp.sum(
                jnp.where(mask, jnp.abs(value) ** 2, 0.0)
            )
        valid = (
            finite
            & zero_masked
            & jnp.isfinite(scale)
            & (scale > 0.0)
            & jnp.isfinite(probability)
            & (probability > 0.0)
        )
        if not bool(np.asarray(valid)):
            raise ValueError(
                "Distributed Wave AMR state must be finite, positive in scale/global "
                "probability, and zero on inactive or covered storage."
            )
        if accepted.shape != () or accepted.dtype != jnp.dtype(jnp.bool_):
            raise ValueError(
                "Distributed Wave AMR accepted-boundary state must be Boolean scalar."
            )

    def bind_packed_state(
        self,
        values: Sequence[ArrayLike],
        scale_factor: ArrayLike,
        /,
        *,
        accepted_boundary: ArrayLike = True,
    ) -> DistributedWaveAMRState:
        arrays = self._place(values)
        scale = self._replicate_scalar(scale_factor, self.real_dtype)
        accepted = self._replicate_scalar(accepted_boundary, bool)
        self._validate_payload(arrays, scale, accepted)
        return self._bind_unchecked(arrays, scale, accepted)

    def validate_state(
        self, state: DistributedWaveAMRState, /
    ) -> DistributedWaveAMRState:
        if not isinstance(state, DistributedWaveAMRState):
            raise TypeError("state must be DistributedWaveAMRState.")
        if (
            state.execution_id != self.execution_id
            or state.topology_id != self.topology_id
            or state.partition_id != self.partition_id
            or state.operator_id != self.operator_id
            or state.physics_id != self.physics_id
        ):
            raise ValueError(
                "Distributed wave state belongs to another prepared execution."
            )
        return self.bind_packed_state(
            state.psi,
            state.scale_factor,
            accepted_boundary=state.accepted_boundary,
        )

    def migrate_accepted_state(
        self,
        state: DistributedWaveAMRState,
        target: PreparedDistributedWaveAMR,
        /,
    ) -> DistributedWaveAMRState:
        checked = self.validate_state(state)
        if not bool(np.asarray(checked.accepted_boundary)):
            raise ValueError(
                "Distributed Wave AMR migration requires an accepted boundary."
            )
        if not isinstance(target, PreparedDistributedWaveAMR):
            raise TypeError("target must be PreparedDistributedWaveAMR.")
        if (
            self.source_prepared_id != target.source_prepared_id
            or self.topology_id != target.topology_id
            or self.physics_id != target.physics_id
            or self.authority_operator_id != target.authority_operator_id
        ):
            raise ValueError("Distributed migration may change only partition ownership.")
        migration = self.hierarchy.migration_to(target.hierarchy)
        values = migration.migrate(checked.psi)
        return target.bind_packed_state(
            values,
            checked.scale_factor,
            accepted_boundary=checked.accepted_boundary,
        )

    def checkpoint_evidence(
        self, checkpoint_id: str, /
    ) -> DistributedWaveAMRCheckpointEvidence:
        identifier = str(checkpoint_id).strip()
        if not identifier:
            raise ValueError("checkpoint_id must be nonempty.")
        paths = tuple(
            [f"['psi'][{level}]" for level in range(len(self.routes.level_local_shapes))]
            + ["['scale_factor']", "['accepted_boundary']"]
        )
        evidence_id = canonical_fingerprint(
            {
                "kind": "distributed-wave-amr-checkpoint-evidence",
                "checkpoint": identifier,
                "execution": self.execution_id,
                "topology": self.topology_id,
                "partition": self.partition_id,
                "paths": list(paths),
                "coverage": "exact-addressable-shards",
            }
        )
        return DistributedWaveAMRCheckpointEvidence(
            identifier,
            self.execution_id,
            self.topology_id,
            self.partition_id,
            paths,
            True,
            evidence_id,
        )

    def publish_checkpoint(
        self,
        repository: ArtifactRepository,
        checkpoint_id: str,
        state: DistributedWaveAMRState,
        /,
        *,
        writer_id: str,
        attempt_id: str | None = None,
        parent_manifest: CheckpointManifest | None = None,
    ) -> tuple[ProcessCheckpointPublication, DistributedWaveAMRCheckpointEvidence]:
        checked = self.validate_state(state)
        evidence = self.checkpoint_evidence(checkpoint_id)
        tree = {
            "psi": checked.psi,
            "scale_factor": checked.scale_factor,
            "accepted_boundary": checked.accepted_boundary,
        }
        publication = publish_process_checkpoint(
            repository,
            evidence.checkpoint_id,
            self.execution_id,
            tree,
            analysis_plan_id=self.source_prepared_id,
            numeric_revision_id=self.physics_id,
            writer_id=writer_id,
            attempt_id=attempt_id,
            topology_epoch=self.hierarchy.topology.epoch.index,
            parent_manifest=parent_manifest,
        )
        observed_paths = {
            dict(shard.metadata)["array_path"] for shard in publication.shards
        }
        expected_paths = set(evidence.array_paths)
        if not observed_paths.issubset(expected_paths):
            raise RuntimeError(
                "Distributed Wave AMR checkpoint published an unknown local array."
            )
        return publication, evidence

    def _validate_checkpoint_manifest(
        self,
        manifest: CheckpointManifest,
        source: PreparedDistributedWaveAMR,
        /,
    ) -> None:
        if not isinstance(manifest, CheckpointManifest) or not manifest.complete:
            raise ValueError("Distributed Wave AMR restore requires a complete manifest.")
        if not isinstance(source, PreparedDistributedWaveAMR):
            raise TypeError("source must be PreparedDistributedWaveAMR.")
        if manifest.execution_plan_id != source.execution_id:
            raise ValueError("Checkpoint execution identity does not match its source.")
        if (
            self.source_prepared_id != source.source_prepared_id
            or self.topology_id != source.topology_id
            or self.physics_id != source.physics_id
            or self.authority_operator_id != source.authority_operator_id
            or self.hierarchy.topology.plan.plan_id
            != source.hierarchy.topology.plan.plan_id
        ):
            raise ValueError(
                "Checkpoint restore may change only distributed partition/sharding."
            )
        expected_paths = set(
            source.checkpoint_evidence(manifest.checkpoint_id).array_paths
        )
        observed_paths = {
            dict(shard.metadata).get("array_path") for shard in manifest.shards
        }
        if observed_paths != expected_paths:
            raise ValueError(
                "Distributed Wave AMR checkpoint manifest state inventory is incomplete."
            )

    def restore_checkpoint(
        self,
        repository: ArtifactRepository,
        manifest: CheckpointManifest,
        source: PreparedDistributedWaveAMR,
        /,
        *,
        parent_manifest: CheckpointManifest | None = None,
    ) -> tuple[DistributedWaveAMRState, DistributedWaveAMRRestoreEvidence]:
        self._validate_checkpoint_manifest(manifest, source)
        if source.hierarchy.mesh is None:
            raise ValueError(
                "Checkpoint source sharding is unavailable for direct restore."
            )
        source_axis = source.hierarchy.partition.axis_name
        restored = []
        for level, shape in enumerate(source.routes.level_local_shapes):
            global_shape = (source.hierarchy.partition.part_count, *shape)
            sharding = NamedSharding(
                source.hierarchy.mesh,
                _part_spec(source_axis, len(global_shape)),
            )
            restored.append(
                restore_global_array_from_checkpoint(
                    repository,
                    manifest,
                    f"['psi'][{level}]",
                    sharding,
                    parent_manifest=parent_manifest,
                )
            )
        replicated = NamedSharding(source.hierarchy.mesh, PartitionSpec())
        scale = restore_global_array_from_checkpoint(
            repository,
            manifest,
            "['scale_factor']",
            replicated,
            parent_manifest=parent_manifest,
        )
        accepted = restore_global_array_from_checkpoint(
            repository,
            manifest,
            "['accepted_boundary']",
            replicated,
            parent_manifest=parent_manifest,
        )
        source_state = source.bind_packed_state(
            tuple(restored), scale, accepted_boundary=accepted
        )
        migration_id = None
        changed = source.execution_id != self.execution_id
        if changed:
            migration: BlockAMRStableIDMigrationPlan = source.hierarchy.migration_to(
                self.hierarchy
            )
            migration_id = migration.migration_id
            result = self.bind_packed_state(
                migration.migrate(source_state.psi),
                source_state.scale_factor,
                accepted_boundary=source_state.accepted_boundary,
            )
        else:
            result = self.validate_state(source_state)
        evidence_id = canonical_fingerprint(
            {
                "kind": "distributed-wave-amr-restore-evidence",
                "source": source.execution_id,
                "target": self.execution_id,
                "manifest": manifest.manifest_id,
                "migration": migration_id,
                "changed_partition": changed,
                "coverage": "exact",
            }
        )
        evidence = DistributedWaveAMRRestoreEvidence(
            source.execution_id,
            self.execution_id,
            manifest.manifest_id,
            migration_id,
            changed,
            True,
            evidence_id,
        )
        return result, evidence

    def observables(
        self,
        state: DistributedWaveAMRState,
        /,
        *,
        relative_node_floor: float,
    ) -> DistributedWaveAMRObservables:
        checked = self.validate_state(state)
        if self.hierarchy.mesh is None:
            raise RuntimeError("Distributed observables require an execution mesh.")
        floor = float(relative_node_floor)
        if not np.isfinite(floor) or not 0.0 < floor < 1.0:
            raise ValueError(
                "relative_node_floor must lie strictly between zero and one."
            )
        axis_name = self.hierarchy.partition.axis_name
        routes = self.routes
        halo = routes.halo
        dimension = len(self.hierarchy.topology.plan.grid.shape)
        lengths = tuple(
            float(np.asarray(axis.bounds[1] - axis.bounds[0]))
            for axis in self.hierarchy.topology.plan.grid.structured_axes
        )
        volume = prod(lengths)
        hbar = self.reduced_planck_constant
        mass = self.boson_mass

        def local_observables(level_values):
            part = jax.lax.axis_index(axis_name)
            packed = jnp.concatenate(
                tuple(value[0].reshape((-1,)) for value in level_values)
            )
            owned = halo.local_owned[part]
            local = jnp.where(
                owned,
                packed[routes.owned_packed_indices[part]],
                0.0,
            )
            full = _safe_halo_exchange(halo, local, part, axis_name)
            measures = routes.local_cell_measures[part]
            left = routes.edge_left_local[part]
            right = routes.edge_right_local[part]
            valid = routes.edge_valid[part]
            axes = routes.edge_axis[part]
            areas = routes.edge_area[part]
            probability = jax.lax.psum(
                jnp.sum(jnp.where(owned, measures * jnp.abs(local) ** 2, 0.0)),
                axis_name,
            )
            current_face = (
                hbar / mass * jnp.imag(jnp.conj(full[left]) * full[right]) * areas
            )
            current = jnp.stack(
                tuple(
                    jax.lax.psum(
                        jnp.sum(
                            jnp.where(
                                valid & (axes == component),
                                current_face,
                                0.0,
                            )
                        ),
                        axis_name,
                    )
                    for component in range(dimension)
                )
            )
            phase_jump = jnp.angle(jnp.conj(full[left]) * full[right])
            winding = jnp.stack(
                tuple(
                    jax.lax.psum(
                        jnp.sum(
                            jnp.where(
                                valid & (axes == component),
                                areas * phase_jump,
                                0.0,
                            )
                        ),
                        axis_name,
                    )
                    / (2.0 * pi * volume / lengths[component])
                    for component in range(dimension)
                )
            )
            power = jnp.where(owned, jnp.abs(local) ** 2, 0.0)
            maximum_power = jax.lax.pmax(jnp.max(power, initial=0.0), axis_name)
            node_count = jax.lax.psum(
                jnp.sum(
                    (owned & (jnp.abs(local) ** 2 <= floor * maximum_power)).astype(
                        jnp.int32
                    )
                ),
                axis_name,
            )
            finite = _global_all(
                jnp.all(jnp.isfinite(jnp.where(owned, local, 0.0)))
                & jnp.isfinite(probability)
                & jnp.all(jnp.isfinite(current))
                & jnp.all(jnp.isfinite(winding)),
                axis_name,
            )
            return probability, current, winding, node_count, finite

        level_specs = tuple(_part_spec(axis_name, value.ndim) for value in checked.psi)
        scalar_spec = PartitionSpec()
        probability, current, winding, node_count, finite = jax.shard_map(
            local_observables,
            mesh=self.hierarchy.mesh,
            in_specs=(level_specs,),
            out_specs=(
                scalar_spec,
                scalar_spec,
                scalar_spec,
                scalar_spec,
                scalar_spec,
            ),
            check_vma=False,
        )(checked.psi)
        if dimension < 2:
            vortex_count = jnp.asarray(0, dtype=jnp.int32)
        else:
            real = tuple(jnp.real(value) for value in checked.psi)
            imaginary = tuple(jnp.imag(value) for value in checked.psi)

            def fill(values):
                boundaries, supplied = self.hierarchy._boundary_inputs(values, None)
                return self.hierarchy._packed_fill_values(
                    values,
                    values,
                    values,
                    boundaries,
                    supplied,
                    0.0,
                    0.0,
                    0.0,
                    distributed=True,
                )

            real_filled, real_valid = fill(real)
            imaginary_filled, imaginary_valid = fill(imaginary)
            vortex_count = jnp.asarray(0, dtype=jnp.int32)
            fill_complete = jnp.asarray(True)
            for level, (
                real_workspace,
                imaginary_workspace,
                valid_real,
                valid_imaginary,
                leaf_mask,
                level_plan,
                fill_route,
            ) in enumerate(
                zip(
                    real_filled,
                    imaginary_filled,
                    real_valid,
                    imaginary_valid,
                    routes.level_leaf_masks,
                    self.hierarchy.topology.plan.levels,
                    self.hierarchy.fill_routes,
                    strict=True,
                )
            ):
                del level
                workspace = jax.lax.complex(real_workspace, imaginary_workspace)
                target = fill_route.target_valid.reshape(
                    fill_route.target_valid.shape + (1,) * (valid_real.ndim - 2)
                )
                fill_complete = fill_complete & jnp.all(
                    (valid_real & valid_imaginary) | ~target
                )
                halo_width = level_plan.halo_width

                def point(first: int, second: int, /):
                    slices = [slice(None), slice(None)]
                    for component, (width, size) in enumerate(
                        zip(
                            halo_width,
                            level_plan.block_shape,
                            strict=True,
                        )
                    ):
                        offset = int(component == first) + int(component == second)
                        slices.append(slice(width + offset, width + offset + size))
                    return workspace[tuple(slices)]

                center = point(-1, -1)
                for first_axis in range(dimension):
                    for second_axis in range(first_axis + 1, dimension):
                        first = point(first_axis, -1)
                        second = point(second_axis, -1)
                        diagonal = point(first_axis, second_axis)
                        circulation = (
                            jnp.angle(jnp.conj(center) * first)
                            + jnp.angle(jnp.conj(first) * diagonal)
                            + jnp.angle(jnp.conj(diagonal) * second)
                            + jnp.angle(jnp.conj(second) * center)
                        )
                        vortex_count = vortex_count + jnp.sum(
                            leaf_mask & (jnp.abs(circulation) >= pi)
                        ).astype(jnp.int32)
            finite = finite & fill_complete & jnp.isfinite(vortex_count)
        return DistributedWaveAMRObservables(
            probability,
            current,
            winding,
            node_count,
            vortex_count,
            finite,
        )

    def step(
        self,
        state: DistributedWaveAMRState,
        end_scale_factor: ArrayLike,
        kick_factor: ArrayLike,
        drift_factor: ArrayLike,
        /,
    ) -> DistributedWaveAMRResult:
        """Run one all-rank transaction without materializing canonical hierarchy data."""

        checked = self.validate_state(state)
        if self.hierarchy.mesh is None:
            raise RuntimeError(
                "Distributed Wave AMR execution requires a prepared execution mesh."
            )
        end_host = np.asarray(end_scale_factor)
        start_host = np.asarray(checked.scale_factor)
        kick_host = np.asarray(kick_factor)
        drift_host = np.asarray(drift_factor)
        if (
            end_host.shape != ()
            or start_host.shape != ()
            or kick_host.shape != ()
            or drift_host.shape != ()
            or not np.isfinite(end_host).item()
            or not np.isfinite(start_host).item()
            or not np.isfinite(kick_host).item()
            or not np.isfinite(drift_host).item()
            or float(end_host) <= float(start_host)
        ):
            raise ValueError(
                "Distributed Wave AMR end scale must be finite and strictly increasing before collective execution."
            )
        end = self._replicate_scalar(end_scale_factor, self.real_dtype)
        kick = self._replicate_scalar(kick_factor, self.real_dtype)
        drift = self._replicate_scalar(drift_factor, self.real_dtype)
        axis = self.hierarchy.partition.axis_name
        routes = self.routes
        halo = routes.halo
        dimension = len(self.hierarchy.topology.plan.grid.shape)
        relative_tolerance = self.solve_relative_tolerance
        absolute_tolerance = self.solve_absolute_tolerance
        maximum_steps = self.maximum_solve_steps
        mass = self.boson_mass
        gravity_constant = self.gravitational_constant
        hbar = self.reduced_planck_constant
        norm_tolerance = self.norm_relative_tolerance
        adjoint_tolerance = self.self_adjoint_tolerance
        maximum_phase = self.maximum_phase_radians
        spectral_upper = self.kinetic_spectral_upper_bound
        local_shapes = routes.level_local_shapes
        local_sizes = routes.level_local_sizes
        real_dtype = self.real_dtype
        fillpatch_routes_ok = jnp.asarray(True)
        for level in range(len(checked.psi)):
            routed_values = (
                self.hierarchy.route_exchange(
                    "same_level",
                    level,
                    checked.psi[level],
                    distributed=True,
                ),
                self.hierarchy.route_exchange(
                    "coarse_fine",
                    level,
                    checked.psi[max(0, level - 1)],
                    distributed=True,
                ),
                self.hierarchy.route_exchange(
                    "interface",
                    level,
                    checked.psi[level],
                    distributed=True,
                ),
            )
            route_plans = (
                self.hierarchy.same_level_routes[level],
                self.hierarchy.coarse_fine_routes[level],
                self.hierarchy.interface_routes[level],
            )
            for routed, route_plan in zip(routed_values, route_plans, strict=True):
                route_valid = route_plan.received_block_valid.reshape(
                    route_plan.received_block_valid.shape + (1,) * (routed.ndim - 2)
                )
                fillpatch_routes_ok = fillpatch_routes_ok & jnp.all(
                    jnp.isfinite(jnp.where(route_valid, routed, 0.0))
                )

        def local_step(
            level_values,
            initial_scale,
            end_scale,
            kick_value,
            drift_value,
            fillpatch_complete,
        ):
            part = jax.lax.axis_index(axis)
            local_flat = jnp.concatenate(
                tuple(value[0].reshape((-1,)) for value in level_values)
            )
            packed_indices = routes.owned_packed_indices[part]
            owned = halo.local_owned[part]
            valid_halo = halo.local_valid[part]
            measures = routes.local_cell_measures[part]
            entity_ids = routes.local_entity_ids[part]
            edge_left = routes.edge_left_local[part]
            edge_right = routes.edge_right_local[part]
            edge_weights = routes.edge_weights[part]
            edge_axes = routes.edge_axis[part]
            edge_areas = routes.edge_area[part]
            edge_jump = routes.edge_level_jump[part]
            edge_valid = routes.edge_valid[part]

            def to_owned(packed):
                return jnp.where(owned, packed[packed_indices], 0.0)

            def to_packed(local):
                packed = jnp.zeros((routes.local_packed_size,), dtype=local.dtype)
                return packed.at[packed_indices].add(jnp.where(owned, local, 0.0))

            def split_packed(flat):
                output = []
                start = 0
                for shape, size in zip(local_shapes, local_sizes, strict=True):
                    output.append(flat[start : start + size].reshape((1, *shape)))
                    start += size
                return tuple(output)

            def exchange(value):
                return _safe_halo_exchange(halo, value, part, axis)

            def operator(value):
                full = exchange(jnp.where(owned, value, 0.0))
                difference = full[edge_left] - full[edge_right]
                flux = jnp.where(edge_valid, edge_weights * difference, 0.0)
                integrated = jnp.zeros_like(full)
                integrated = integrated.at[edge_left].add(flux)
                integrated = integrated.at[edge_right].add(-flux)
                integrated = halo.accumulate_halo(integrated, part, axis_name=axis)
                return jnp.where(owned, integrated / measures, 0.0)

            def inner(left, right):
                local = jnp.sum(jnp.where(owned, measures * jnp.conj(left) * right, 0.0))
                return jax.lax.psum(local, axis)

            def weighted_mean(value):
                numerator = jax.lax.psum(
                    jnp.sum(jnp.where(owned, measures * value, 0.0)), axis
                )
                denominator = jax.lax.psum(jnp.sum(jnp.where(owned, measures, 0.0)), axis)
                return numerator / denominator

            def project(value):
                return jnp.where(owned, value - weighted_mean(value), 0.0)

            def global_norm(value):
                return jnp.sqrt(jnp.maximum(jnp.real(inner(value, value)), 0.0))

            def finite_local(value):
                return jnp.all(jnp.isfinite(jnp.where(owned, value, 0.0)))

            def pcg(action, rhs, *, gauge):
                initial = jnp.zeros_like(rhs)
                residual = project(rhs) if gauge else rhs
                direction = residual
                squared = jnp.real(inner(residual, residual))
                rhs_norm = global_norm(rhs)
                tolerance = jnp.maximum(absolute_tolerance, relative_tolerance * rhs_norm)
                valid = _global_all(
                    finite_local(rhs)
                    & jnp.isfinite(squared)
                    & (squared >= 0.0)
                    & jnp.isfinite(rhs_norm),
                    axis,
                )

                def condition(carry):
                    iteration, _, _, _, residual_squared, okay = carry
                    return (
                        (iteration < maximum_steps)
                        & okay
                        & (residual_squared > tolerance * tolerance)
                    )

                def body(carry):
                    iteration, solution, residual_, direction_, old_squared, okay = carry
                    image = action(direction_)
                    denominator = jnp.real(inner(direction_, image))
                    step_valid = _global_all(
                        jnp.isfinite(denominator)
                        & (denominator > 0.0)
                        & jnp.isfinite(old_squared)
                        & (old_squared >= 0.0),
                        axis,
                    )
                    alpha = jnp.where(step_valid, old_squared / denominator, 0.0)
                    candidate_solution = solution + alpha * direction_
                    candidate_residual = residual_ - alpha * image
                    if gauge:
                        candidate_solution = project(candidate_solution)
                        candidate_residual = project(candidate_residual)
                    new_squared = jnp.real(inner(candidate_residual, candidate_residual))
                    candidate_valid = (
                        okay
                        & step_valid
                        & _global_all(
                            finite_local(candidate_solution)
                            & finite_local(candidate_residual)
                            & jnp.isfinite(new_squared)
                            & (new_squared >= 0.0),
                            axis,
                        )
                    )
                    beta = jnp.where(
                        candidate_valid & (old_squared > 0.0),
                        new_squared / old_squared,
                        0.0,
                    )
                    candidate_direction = candidate_residual + beta * direction_
                    return (
                        iteration + 1,
                        jnp.where(candidate_valid, candidate_solution, solution),
                        jnp.where(candidate_valid, candidate_residual, residual_),
                        jnp.where(candidate_valid, candidate_direction, direction_),
                        jnp.where(candidate_valid, new_squared, old_squared),
                        candidate_valid,
                    )

                iteration, solution, _, _, _, valid = jax.lax.while_loop(
                    condition,
                    body,
                    (
                        jnp.asarray(0, dtype=jnp.int32),
                        initial,
                        residual,
                        direction,
                        squared,
                        valid,
                    ),
                )
                if gauge:
                    solution = project(solution)
                residual_value = action(solution) - rhs
                if gauge:
                    residual_value = project(residual_value)
                residual_norm = global_norm(residual_value)
                finite = valid & _global_all(
                    finite_local(solution)
                    & finite_local(residual_value)
                    & jnp.isfinite(residual_norm),
                    axis,
                )
                converged = finite & (residual_norm <= tolerance)
                relative = residual_norm / jnp.where(rhs_norm > 0.0, rhs_norm, 1.0)
                return (
                    solution,
                    iteration,
                    residual_norm,
                    rhs_norm,
                    relative,
                    finite,
                    converged,
                )

            def solve_poisson(wave):
                density = jnp.where(owned, mass * jnp.abs(wave) ** 2, 0.0)
                mean_density = weighted_mean(density)
                source = jnp.where(
                    owned,
                    -4.0 * jnp.pi * gravity_constant * (density - mean_density),
                    0.0,
                )
                source = project(source)
                solution = pcg(
                    lambda value: project(operator(project(value))), source, gauge=True
                )
                potential = solution[0]
                source_integral = jax.lax.psum(
                    jnp.sum(jnp.where(owned, measures * source, 0.0)), axis
                )
                gauge_defect = jnp.abs(weighted_mean(potential))
                interface_flux = jnp.where(
                    edge_valid & edge_jump,
                    edge_weights
                    * (exchange(potential)[edge_left] - exchange(potential)[edge_right]),
                    0.0,
                )
                interface_defect = jax.lax.pmax(
                    jnp.max(jnp.abs(interface_flux + (-interface_flux)), initial=0.0),
                    axis,
                )
                return (
                    potential,
                    source,
                    mean_density,
                    source_integral,
                    gauge_defect,
                    interface_defect,
                    *solution[1:],
                )

            def potential_kick(wave, potential, fraction):
                coefficient = fraction * mass * kick_value / hbar
                phase = coefficient * potential
                candidate = jnp.where(owned, wave * jnp.exp(-1j * phase), 0.0)
                local_maximum = jnp.max(
                    jnp.where(owned, jnp.abs(phase), 0.0), initial=0.0
                )
                return candidate, jax.lax.pmax(local_maximum, axis)

            def kinetic_drift(wave):
                action_coefficient = hbar * drift_value / (2.0 * mass)
                alpha = 0.5 * action_coefficient

                def left(value):
                    return jnp.where(owned, value + 1j * alpha * operator(value), 0.0)

                def adjoint_left(value):
                    return jnp.where(owned, value - 1j * alpha * operator(value), 0.0)

                right = jnp.where(owned, wave - 1j * alpha * operator(wave), 0.0)
                normal_rhs = adjoint_left(right)
                solved = pcg(
                    lambda value: adjoint_left(left(value)), normal_rhs, gauge=False
                )
                candidate = jnp.where(owned, solved[0], 0.0)
                residual = left(candidate) - right
                residual_norm = global_norm(residual)
                rhs_norm = global_norm(right)
                relative = residual_norm / jnp.where(rhs_norm > 0.0, rhs_norm, 1.0)
                finite = solved[5] & _global_all(
                    finite_local(candidate)
                    & finite_local(residual)
                    & jnp.isfinite(relative),
                    axis,
                )
                converged = finite & (
                    residual_norm
                    <= jnp.maximum(absolute_tolerance, relative_tolerance * rhs_norm)
                )
                return (
                    candidate,
                    solved[1],
                    residual_norm,
                    rhs_norm,
                    relative,
                    finite,
                    converged,
                )

            def probability(value):
                return jnp.real(inner(value, value))

            def integrated_current(value):
                full = exchange(value)
                face = (
                    hbar
                    / mass
                    * jnp.imag(jnp.conj(full[edge_left]) * full[edge_right])
                    * edge_areas
                )
                return jnp.stack(
                    tuple(
                        jax.lax.psum(
                            jnp.sum(
                                jnp.where(
                                    edge_valid & (edge_axes == component), face, 0.0
                                )
                            ),
                            axis,
                        )
                        for component in range(dimension)
                    )
                )

            wave = to_owned(local_flat)
            initial_probability = probability(wave)
            initial_current = integrated_current(wave)
            first_gravity = solve_poisson(wave)
            first, first_phase = potential_kick(wave, first_gravity[0], 0.5)
            kinetic = kinetic_drift(first)
            second_gravity = solve_poisson(kinetic[0])
            candidate, second_phase = potential_kick(kinetic[0], second_gravity[0], 0.5)
            final_probability = probability(candidate)
            probability_error = jnp.abs(
                final_probability - initial_probability
            ) / jnp.where(initial_probability > 0.0, initial_probability, 1.0)
            final_current = integrated_current(candidate)
            current_delta = final_current - initial_current
            current_absolute = jnp.sqrt(jnp.sum(current_delta * current_delta))
            initial_current_norm = jnp.sqrt(jnp.sum(initial_current * initial_current))
            current_relative = jnp.where(
                initial_current_norm > 32.0 * jnp.finfo(real_dtype).eps,
                current_absolute / initial_current_norm,
                0.0,
            )
            probe = jnp.where(
                owned,
                wave
                * (
                    1.0 + entity_ids.astype(real_dtype) / max(1, routes.halo.entity_count)
                ),
                0.0,
            )
            left_pairing = inner(wave, operator(probe))
            right_pairing = inner(operator(wave), probe)
            adjoint_scale = jnp.maximum(
                jnp.maximum(jnp.abs(left_pairing), jnp.abs(right_pairing)), 1.0
            )
            self_adjoint = jnp.abs(left_pairing - right_pairing) / adjoint_scale
            action_coefficient = hbar * drift_value / (2.0 * mass)
            kinetic_phase = 2.0 * jnp.arctan(
                0.5 * jnp.abs(action_coefficient) * spectral_upper
            )
            potential_phase = jnp.maximum(first_phase, second_phase)
            resolved_phase = jnp.maximum(kinetic_phase, potential_phase)
            kinetic_energy = (
                hbar**2
                / (2.0 * mass * end_scale**2)
                * jnp.real(inner(candidate, operator(candidate)))
            )
            potential_energy = (
                0.5
                / end_scale
                * jax.lax.psum(
                    jnp.sum(
                        jnp.where(
                            owned,
                            measures * second_gravity[0] * mass * jnp.abs(candidate) ** 2,
                            0.0,
                        )
                    ),
                    axis,
                )
            )
            halo_complete = _global_all(
                jnp.all(jnp.isfinite(jnp.where(valid_halo, exchange(wave), 0.0)))
                & fillpatch_complete,
                axis,
            )
            finite = _global_all(
                finite_local(candidate)
                & jnp.isfinite(final_probability)
                & jnp.all(jnp.isfinite(final_current))
                & jnp.isfinite(current_absolute)
                & jnp.isfinite(current_relative)
                & jnp.isfinite(kinetic[4])
                & jnp.isfinite(self_adjoint)
                & jnp.isfinite(kinetic_phase)
                & jnp.isfinite(kinetic_energy)
                & jnp.isfinite(potential_energy),
                axis,
            )
            initial_poisson = first_gravity[-1]
            final_poisson = second_gravity[-1]
            poisson_closed = initial_poisson & final_poisson
            kinetic_closed = kinetic[5]
            cayley_closed = kinetic[6]
            adjoint_closed = self_adjoint <= adjoint_tolerance
            norm_closed = probability_error <= norm_tolerance
            phase_closed = resolved_phase <= maximum_phase
            local_accept = (
                finite
                & poisson_closed
                & kinetic_closed
                & cayley_closed
                & adjoint_closed
                & norm_closed
                & phase_closed
                & halo_complete
                & (end_scale > initial_scale)
                & jnp.isfinite(end_scale)
                & jnp.isfinite(kick_value)
                & jnp.isfinite(drift_value)
            )
            agreement = _global_agreement(local_accept, axis)
            accepted = _global_all(local_accept, axis) & agreement
            committed = jnp.where(accepted, candidate, wave)
            status = jnp.where(
                ~finite,
                1,
                jnp.where(
                    ~poisson_closed,
                    2,
                    jnp.where(
                        ~kinetic_closed,
                        3,
                        jnp.where(
                            ~cayley_closed,
                            4,
                            jnp.where(
                                ~adjoint_closed,
                                5,
                                jnp.where(
                                    ~norm_closed,
                                    6,
                                    jnp.where(
                                        ~phase_closed,
                                        7,
                                        jnp.where(
                                            ~agreement,
                                            8,
                                            jnp.where(~halo_complete, 9, 0),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ).astype(jnp.int32)
            candidate_packed = split_packed(to_packed(candidate))
            committed_packed = split_packed(to_packed(committed))
            first_potential = split_packed(to_packed(first_gravity[0]))
            first_source = split_packed(to_packed(first_gravity[1]))
            second_potential = split_packed(to_packed(second_gravity[0]))
            second_source = split_packed(to_packed(second_gravity[1]))
            scalar_evidence = (
                initial_probability,
                final_probability,
                probability_error,
                initial_current,
                final_current,
                current_relative,
                current_absolute,
                kinetic[4],
                self_adjoint,
                kinetic_phase,
                potential_phase,
                kinetic_energy,
                potential_energy,
                finite,
                initial_poisson,
                final_poisson,
                poisson_closed,
                kinetic_closed,
                halo_complete,
                agreement,
                accepted,
                status,
                first_gravity[2],
                first_gravity[3],
                first_gravity[4],
                first_gravity[5],
                first_gravity[6],
                first_gravity[7],
                first_gravity[8],
                first_gravity[9],
                first_gravity[10],
                first_gravity[11],
                second_gravity[2],
                second_gravity[3],
                second_gravity[4],
                second_gravity[5],
                second_gravity[6],
                second_gravity[7],
                second_gravity[8],
                second_gravity[9],
                second_gravity[10],
                second_gravity[11],
                kinetic[1],
                kinetic[2],
                kinetic[3],
                kinetic[4],
                kinetic[5],
                kinetic[6],
            )
            return (
                candidate_packed,
                committed_packed,
                first_potential,
                first_source,
                second_potential,
                second_source,
                scalar_evidence,
            )

        level_specs = tuple(_part_spec(axis, value.ndim) for value in checked.psi)
        scalar_spec = PartitionSpec()
        evidence_specs = (
            scalar_spec,
            scalar_spec,
            scalar_spec,
            scalar_spec,
            scalar_spec,
            scalar_spec,
            scalar_spec,
            scalar_spec,
            scalar_spec,
            scalar_spec,
            scalar_spec,
            scalar_spec,
            scalar_spec,
            scalar_spec,
            scalar_spec,
            scalar_spec,
            scalar_spec,
            scalar_spec,
            scalar_spec,
            scalar_spec,
            scalar_spec,
            scalar_spec,
            scalar_spec,
            scalar_spec,
            scalar_spec,
            scalar_spec,
            scalar_spec,
            scalar_spec,
            scalar_spec,
            scalar_spec,
            scalar_spec,
            scalar_spec,
            scalar_spec,
            scalar_spec,
            scalar_spec,
            scalar_spec,
            scalar_spec,
            scalar_spec,
            scalar_spec,
            scalar_spec,
            scalar_spec,
            scalar_spec,
            scalar_spec,
            scalar_spec,
            scalar_spec,
            scalar_spec,
            scalar_spec,
            scalar_spec,
        )
        mapped = jax.shard_map(
            local_step,
            mesh=self.hierarchy.mesh,
            in_specs=(
                level_specs,
                scalar_spec,
                scalar_spec,
                scalar_spec,
                scalar_spec,
                scalar_spec,
            ),
            out_specs=(
                level_specs,
                level_specs,
                level_specs,
                level_specs,
                level_specs,
                level_specs,
                evidence_specs,
            ),
            check_vma=False,
        )
        (
            candidate_values,
            committed_values,
            first_potential,
            first_source,
            second_potential,
            second_source,
            evidence,
        ) = mapped(
            checked.psi,
            checked.scale_factor,
            end,
            kick,
            drift,
            fillpatch_routes_ok,
        )
        (
            initial_probability,
            final_probability,
            probability_error,
            initial_current,
            final_current,
            current_relative,
            current_absolute,
            cayley_relative,
            self_adjoint,
            kinetic_phase,
            potential_phase,
            kinetic_energy,
            potential_energy,
            finite,
            initial_poisson,
            final_poisson,
            poisson_closed,
            kinetic_closed,
            halo_complete,
            agreement,
            accepted,
            status,
            first_mean,
            first_integral,
            first_gauge,
            first_interface,
            first_iterations,
            first_residual,
            first_rhs_norm,
            first_relative,
            first_finite,
            first_converged,
            second_mean,
            second_integral,
            second_gauge,
            second_interface,
            second_iterations,
            second_residual,
            second_rhs_norm,
            second_relative,
            second_finite,
            second_converged,
            kinetic_iterations,
            kinetic_residual,
            kinetic_rhs_norm,
            kinetic_relative,
            kinetic_finite,
            kinetic_converged,
        ) = evidence
        candidate_state = self._bind_unchecked(
            candidate_values,
            end,
            self._replicate_scalar(True, bool),
        )
        committed_state = self._bind_unchecked(
            committed_values,
            jnp.where(accepted, end, checked.scale_factor),
            jnp.where(
                accepted,
                self._replicate_scalar(True, bool),
                checked.accepted_boundary,
            ),
        )
        first_linear = DistributedWaveAMRLinearEvidence(
            first_iterations,
            first_residual,
            first_rhs_norm,
            first_relative,
            first_finite,
            first_converged,
            "collective-projected-pcg",
            self.maximum_solve_steps,
        )
        second_linear = DistributedWaveAMRLinearEvidence(
            second_iterations,
            second_residual,
            second_rhs_norm,
            second_relative,
            second_finite,
            second_converged,
            "collective-projected-pcg",
            self.maximum_solve_steps,
        )
        kinetic_linear = DistributedWaveAMRLinearEvidence(
            kinetic_iterations,
            kinetic_residual,
            kinetic_rhs_norm,
            kinetic_relative,
            kinetic_finite,
            kinetic_converged,
            "collective-cgne-cayley",
            self.maximum_solve_steps,
        )
        first_gravity = DistributedWaveAMRGravityEvidence(
            first_potential,
            first_source,
            first_mean,
            first_integral,
            first_gauge,
            first_interface,
            first_linear,
            first_finite,
            first_converged,
            self.operator_id,
        )
        second_gravity = DistributedWaveAMRGravityEvidence(
            second_potential,
            second_source,
            second_mean,
            second_integral,
            second_gauge,
            second_interface,
            second_linear,
            second_finite,
            second_converged,
            self.operator_id,
        )
        diagnostics = DistributedWaveAMRDiagnostics(
            initial_probability,
            final_probability,
            probability_error,
            initial_current,
            final_current,
            current_relative,
            current_absolute,
            cayley_relative,
            self_adjoint,
            kinetic_phase,
            potential_phase,
            kinetic_energy,
            potential_energy,
            finite,
            initial_poisson,
            final_poisson,
            poisson_closed,
            kinetic_closed,
            halo_complete,
            agreement,
            accepted,
            status,
        )
        result_id = canonical_fingerprint(
            {
                "kind": "distributed-wave-amr-step-result",
                "execution": self.execution_id,
                "contract": "atomic-global-commit-or-rollback",
                "operator": self.operator_id,
                "physics": self.physics_id,
            }
        )
        return DistributedWaveAMRResult(
            committed_state,
            candidate_state,
            diagnostics,
            first_gravity,
            second_gravity,
            kinetic_linear,
            accepted,
            self.execution_id,
            result_id,
        )


class PreparedDistributedWaveAMRTopologyTransition(StrictModule, NonTrainableState):
    """Owner-local phase reconstruction over exact source/target overlap routes."""

    source: PreparedDistributedWaveAMR
    target: PreparedDistributedWaveAMR
    transition: BlockFieldTopologyTransition
    halo: DistributedHaloPlan
    source_load_indices: Array
    source_load_valid: Array
    target_packed_indices: Array
    target_valid: Array
    target_measures: Array
    overlap_source_indices: Array
    overlap_weights: Array
    overlap_valid: Array
    nearest_source_indices: Array
    plus_source_indices: Array
    minus_source_indices: Array
    target_displacements: Array
    coordinate_differences: Array
    local_target_capacity: int = eqx.field(static=True)
    overlap_capacity: int = eqx.field(static=True)
    common_stable_block_ids: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    required_bytes: int = eqx.field(static=True)
    transition_id: str = eqx.field(static=True)

    @staticmethod
    def estimate_required_bytes(
        source: PreparedDistributedWaveAMR,
        target: PreparedDistributedWaveAMR,
        transition: BlockFieldTopologyTransition,
        /,
    ) -> int:
        """Exact padded-capacity envelope before packet arrays are allocated."""
        if (
            not isinstance(source, PreparedDistributedWaveAMR)
            or not isinstance(target, PreparedDistributedWaveAMR)
            or not isinstance(transition, BlockFieldTopologyTransition)
        ):
            raise TypeError(
                "Topology transition preflight requires prepared source/target plans."
            )
        parts = source.hierarchy.partition.part_count
        source_cells = source.routes.halo.entity_count
        target_cells = target.routes.halo.entity_count
        dimension = len(source.hierarchy.topology.plan.grid.shape)
        source_owner = np.asarray(source.routes.halo.entity_owner, dtype=np.int32)
        target_owner = np.asarray(target.routes.halo.entity_owner, dtype=np.int32)
        combined_owner = np.concatenate((source_owner, target_owner))
        relation_source = np.asarray(
            transition.leaf_routes.relation.source_indices, dtype=np.int32
        )
        relation_target = np.asarray(
            transition.leaf_routes.relation.target_indices, dtype=np.int32
        )
        relation_weights = np.asarray(transition.leaf_routes.weights)
        relation_valid = np.asarray(transition.leaf_routes.relation.valid, dtype=np.bool_)
        overlap_by_target: list[set[int]] = [set() for _ in range(target_cells)]
        nearest = np.zeros((target_cells,), dtype=np.int32)
        best = np.full((target_cells,), -np.inf)
        for source_index, target_index, weight, valid in zip(
            relation_source,
            relation_target,
            relation_weights,
            relation_valid,
            strict=True,
        ):
            if not valid:
                continue
            source_index_ = int(source_index)
            target_index_ = int(target_index)
            weight_ = float(weight)
            overlap_by_target[target_index_].add(source_index_)
            if weight_ > best[target_index_]:
                nearest[target_index_] = source_index_
                best[target_index_] = weight_
        if np.any(~np.isfinite(best)):
            raise ValueError(
                "Topology transition preflight found incomplete target overlap."
            )
        plus = np.broadcast_to(
            np.arange(source_cells, dtype=np.int32)[:, None],
            (source_cells, dimension),
        ).copy()
        minus = plus.copy()
        plus_area = np.full(plus.shape, -np.inf)
        minus_area = np.full(minus.shape, -np.inf)
        for left, right, edge_axis, area in zip(
            np.asarray(source.routes.global_edge_left),
            np.asarray(source.routes.global_edge_right),
            np.asarray(source.routes.global_edge_axis),
            np.asarray(source.routes.global_edge_area),
            strict=True,
        ):
            left_ = int(left)
            right_ = int(right)
            axis_ = int(edge_axis)
            area_ = float(area)
            if area_ > plus_area[left_, axis_]:
                plus[left_, axis_] = right_
                plus_area[left_, axis_] = area_
            if area_ > minus_area[right_, axis_]:
                minus[right_, axis_] = left_
                minus_area[right_, axis_] = area_
        if np.any(~np.isfinite(plus_area)) or np.any(~np.isfinite(minus_area)):
            raise ValueError(
                "Topology transition preflight found incomplete phase neighbors."
            )
        adjacency: set[tuple[int, int]] = set()
        for target_entity in range(target_cells):
            selected = int(nearest[target_entity])
            requested = set(overlap_by_target[target_entity])
            requested.add(selected)
            requested.update(int(value) for value in plus[selected])
            requested.update(int(value) for value in minus[selected])
            proxy = source_cells + target_entity
            adjacency.update((source_entity, proxy) for source_entity in requested)
        local_counts = []
        pair_messages: dict[tuple[int, int], set[int]] = {}
        for part in range(parts):
            owned = set(np.flatnonzero(combined_owner == part).tolist())
            halo_entities: set[int] = set()
            for left, right in adjacency:
                if left in owned and combined_owner[right] != part:
                    halo_entities.add(right)
                if right in owned and combined_owner[left] != part:
                    halo_entities.add(left)
            local_counts.append(len(owned) + len(halo_entities))
            for entity in halo_entities:
                pair_messages.setdefault((int(combined_owner[entity]), part), set()).add(
                    entity
                )
        remaining = set(pair_messages)
        phase_count = 0
        while remaining:
            used_source: set[int] = set()
            used_target: set[int] = set()
            phase = []
            for source_part, target_part in sorted(remaining):
                if source_part not in used_source and target_part not in used_target:
                    phase.append((source_part, target_part))
                    used_source.add(source_part)
                    used_target.add(target_part)
            remaining.difference_update(phase)
            phase_count += 1
        local_capacity = max(1, *local_counts)
        message_capacity = max(1, *(len(values) for values in pair_messages.values()))
        target_counts = np.bincount(target_owner, minlength=parts)
        local_target_capacity = max(1, int(np.max(target_counts)))
        overlap_counts = np.asarray(
            [len(values) for values in overlap_by_target], dtype=np.int32
        )
        overlap_capacity = max(
            1,
            int(transition.leaf_routes.maximum_target_row_width),
            int(np.max(overlap_counts, initial=0)),
        )
        integer_bytes = np.dtype(np.int32).itemsize
        boolean_bytes = np.dtype(np.bool_).itemsize
        real_bytes = source.real_dtype.itemsize
        complex_bytes = source.complex_dtype.itemsize
        target_rows = parts * local_target_capacity
        overlap_rows = target_rows * overlap_capacity
        halo_rows = parts * local_capacity
        phase_rows = phase_count * parts * message_capacity
        plan_bytes = (
            halo_rows * (integer_bytes + boolean_bytes)
            + target_rows * (integer_bytes + boolean_bytes + real_bytes)
            + overlap_rows * (integer_bytes + boolean_bytes + real_bytes)
            + target_rows * integer_bytes
            + target_rows * dimension * 2 * integer_bytes
            + target_rows * dimension * 2 * real_bytes
            + (source_cells + target_cells) * integer_bytes
            + halo_rows * (integer_bytes + 2 * boolean_bytes)
            + phase_rows * (2 * integer_bytes + 2 * boolean_bytes)
        )
        workspace_bytes = (
            2 * halo_rows * complex_bytes
            + overlap_rows * complex_bytes
            + 2 * target_rows * dimension * complex_bytes
            + target_rows * (3 * complex_bytes + 8 * real_bytes)
            + parts * target.routes.local_packed_size * complex_bytes
        )
        return int(plan_bytes + workspace_bytes)

    def __init__(
        self,
        source: PreparedDistributedWaveAMR,
        target: PreparedDistributedWaveAMR,
        transition: BlockFieldTopologyTransition,
        /,
    ):
        if (
            not isinstance(source, PreparedDistributedWaveAMR)
            or not isinstance(target, PreparedDistributedWaveAMR)
            or not isinstance(transition, BlockFieldTopologyTransition)
        ):
            raise TypeError(
                "Distributed topology transfer requires source, target, and overlap plan."
            )
        if (
            source.hierarchy.partition.part_count != target.hierarchy.partition.part_count
            or source.hierarchy.partition.axis_name
            != target.hierarchy.partition.axis_name
            or source.mesh_id != target.mesh_id
            or transition.source_topology.epoch.epoch_id
            != source.hierarchy.topology.epoch.epoch_id
            or transition.target_topology.epoch.epoch_id
            != target.hierarchy.topology.epoch.epoch_id
            or source.physics_id != target.physics_id
        ):
            raise ValueError(
                "Distributed topology transfer requires consecutive epochs on one mesh with one physics identity."
            )
        source_storage = []
        source_offsets = []
        next_offset = 0
        for level_plan in transition.source_topology.plan.levels:
            source_offsets.append(next_offset)
            next_offset += level_plan.maximum_blocks * prod(level_plan.block_shape)
        for level, slot, local in zip(
            np.asarray(transition.source_levels),
            np.asarray(transition.source_slots),
            np.asarray(transition.source_local),
            strict=True,
        ):
            source_storage.append(
                source_offsets[int(level)]
                + int(slot)
                * prod(transition.source_topology.plan.levels[int(level)].block_shape)
                + int(local)
            )
        if not np.array_equal(
            np.asarray(source_storage, dtype=np.int32),
            np.asarray(source.routes.entity_storage_indices),
        ):
            raise ValueError(
                "Distributed topology transfer source leaf ordering is inconsistent."
            )
        source_count = source.routes.halo.entity_count
        target_count = target.routes.halo.entity_count
        leaf_routes = transition.leaf_routes
        relation_source = np.asarray(leaf_routes.relation.source_indices, dtype=np.int32)
        relation_target = np.asarray(leaf_routes.relation.target_indices, dtype=np.int32)
        relation_weights = np.asarray(leaf_routes.weights)
        relation_valid = np.asarray(leaf_routes.relation.valid, dtype=np.bool_)
        if (
            leaf_routes.relation.source_size != source_count
            or leaf_routes.relation.target_size != target_count
        ):
            raise ValueError(
                "Distributed topology overlap relation does not cover leaf entities."
            )

        source_owner = np.asarray(source.routes.halo.entity_owner, dtype=np.int32)
        target_owner = np.asarray(target.routes.halo.entity_owner, dtype=np.int32)
        source_centers = _entity_centers(
            transition.source_topology,
            np.asarray(source.routes.entity_storage_indices, dtype=np.int32),
        )
        target_centers = _entity_centers(
            transition.target_topology,
            np.asarray(target.routes.entity_storage_indices, dtype=np.int32),
        )
        dimension = source_centers.shape[1]
        bounds = np.asarray(
            [
                np.asarray(axis.bounds)
                for axis in transition.source_topology.plan.grid.structured_axes
            ],
            dtype=np.float64,
        )
        lengths = bounds[:, 1] - bounds[:, 0]

        nearest = np.zeros((target_count,), dtype=np.int32)
        best_weight = np.full((target_count,), -np.inf)
        overlap_by_target: list[list[tuple[int, float]]] = [
            [] for _ in range(target_count)
        ]
        for source_index, target_index, weight, valid in zip(
            relation_source,
            relation_target,
            relation_weights,
            relation_valid,
            strict=True,
        ):
            if not valid:
                continue
            source_index_ = int(source_index)
            target_index_ = int(target_index)
            weight_ = float(weight)
            overlap_by_target[target_index_].append((source_index_, weight_))
            if weight_ > best_weight[target_index_]:
                nearest[target_index_] = source_index_
                best_weight[target_index_] = weight_
        if np.any(~np.isfinite(best_weight)):
            raise ValueError(
                "Distributed phase transfer lacks source coverage for a target leaf."
            )

        plus = np.broadcast_to(
            np.arange(source_count, dtype=np.int32)[:, None],
            (source_count, dimension),
        ).copy()
        minus = plus.copy()
        plus_area = np.full(plus.shape, -np.inf)
        minus_area = np.full(minus.shape, -np.inf)
        for left, right, edge_axis, area in zip(
            np.asarray(source.routes.global_edge_left),
            np.asarray(source.routes.global_edge_right),
            np.asarray(source.routes.global_edge_axis),
            np.asarray(source.routes.global_edge_area),
            strict=True,
        ):
            left_ = int(left)
            right_ = int(right)
            axis_ = int(edge_axis)
            area_ = float(area)
            if area_ > plus_area[left_, axis_]:
                plus[left_, axis_] = right_
                plus_area[left_, axis_] = area_
            if area_ > minus_area[right_, axis_]:
                minus[right_, axis_] = left_
                minus_area[right_, axis_] = area_
        if np.any(~np.isfinite(plus_area)) or np.any(~np.isfinite(minus_area)):
            raise ValueError(
                "Distributed phase transfer source face neighbors are incomplete."
            )
        coordinate_difference = np.stack(
            tuple(
                (
                    source_centers[plus[:, axis], axis]
                    - source_centers[:, axis]
                    - np.round(
                        (source_centers[plus[:, axis], axis] - source_centers[:, axis])
                        / lengths[axis]
                    )
                    * lengths[axis]
                )
                - (
                    source_centers[minus[:, axis], axis]
                    - source_centers[:, axis]
                    - np.round(
                        (source_centers[minus[:, axis], axis] - source_centers[:, axis])
                        / lengths[axis]
                    )
                    * lengths[axis]
                )
                for axis in range(dimension)
            ),
            axis=1,
        )
        target_displacement = target_centers - source_centers[nearest]
        target_displacement -= np.round(target_displacement / lengths) * lengths

        adjacency_records: set[tuple[int, int]] = set()
        for target_entity in range(target_count):
            proxy = source_count + target_entity
            requested = {
                source_index for source_index, _ in overlap_by_target[target_entity]
            }
            selected = int(nearest[target_entity])
            requested.add(selected)
            requested.update(int(value) for value in plus[selected])
            requested.update(int(value) for value in minus[selected])
            adjacency_records.update(
                (source_entity, proxy) for source_entity in requested
            )
        adjacency = np.asarray(sorted(adjacency_records), dtype=np.int32)
        combined_owner = np.concatenate((source_owner, target_owner))
        halo = DistributedHaloPlan(
            combined_owner,
            adjacency,
            source.hierarchy.partition.part_count,
        )
        halo_ids = np.asarray(halo.local_global_ids, dtype=np.int32)
        halo_valid = np.asarray(halo.local_valid, dtype=np.bool_)
        halo_owned = np.asarray(halo.local_owned, dtype=np.bool_)
        local_maps = tuple(
            {
                int(entity): local
                for local, entity in enumerate(halo_ids[part])
                if halo_valid[part, local]
            }
            for part in range(source.hierarchy.partition.part_count)
        )
        source_load_indices = np.zeros_like(halo_ids)
        source_load_valid = np.zeros_like(halo_valid)
        source_packed = np.asarray(source.routes.entity_packed_indices)
        for part in range(source.hierarchy.partition.part_count):
            for local, entity in enumerate(halo_ids[part]):
                if (
                    halo_valid[part, local]
                    and halo_owned[part, local]
                    and int(entity) < source_count
                ):
                    source_load_indices[part, local] = source_packed[int(entity)]
                    source_load_valid[part, local] = True

        targets_by_part = tuple(
            np.flatnonzero(target_owner == part)
            for part in range(source.hierarchy.partition.part_count)
        )
        local_target_capacity = max(1, *(value.size for value in targets_by_part))
        overlap_capacity = max(1, *(len(value) for value in overlap_by_target))
        target_packed_indices = np.zeros(
            (source.hierarchy.partition.part_count, local_target_capacity),
            dtype=np.int32,
        )
        target_valid = np.zeros_like(target_packed_indices, dtype=np.bool_)
        target_measures = np.ones(target_packed_indices.shape, dtype=source.real_dtype)
        overlap_source_indices = np.zeros(
            (
                source.hierarchy.partition.part_count,
                local_target_capacity,
                overlap_capacity,
            ),
            dtype=np.int32,
        )
        overlap_weights = np.zeros(overlap_source_indices.shape, dtype=source.real_dtype)
        overlap_valid = np.zeros(overlap_source_indices.shape, dtype=np.bool_)
        nearest_source_indices = np.zeros_like(target_packed_indices)
        plus_source_indices = np.zeros(
            target_packed_indices.shape + (dimension,), dtype=np.int32
        )
        minus_source_indices = np.zeros_like(plus_source_indices)
        target_displacements = np.zeros(
            target_packed_indices.shape + (dimension,), dtype=source.real_dtype
        )
        coordinate_differences = np.ones_like(target_displacements)
        target_packed = np.asarray(target.routes.entity_packed_indices)
        target_measure_values = np.asarray(target.routes.entity_measures)
        for part, entities in enumerate(targets_by_part):
            for local_target, target_entity_value in enumerate(entities):
                target_entity = int(target_entity_value)
                target_valid[part, local_target] = True
                target_packed_indices[part, local_target] = target_packed[target_entity]
                target_measures[part, local_target] = target_measure_values[target_entity]
                records = overlap_by_target[target_entity]
                for route, (source_entity, weight) in enumerate(records):
                    overlap_source_indices[part, local_target, route] = local_maps[part][
                        source_entity
                    ]
                    overlap_weights[part, local_target, route] = weight
                    overlap_valid[part, local_target, route] = True
                selected = int(nearest[target_entity])
                nearest_source_indices[part, local_target] = local_maps[part][selected]
                plus_source_indices[part, local_target] = [
                    local_maps[part][int(value)] for value in plus[selected]
                ]
                minus_source_indices[part, local_target] = [
                    local_maps[part][int(value)] for value in minus[selected]
                ]
                target_displacements[part, local_target] = target_displacement[
                    target_entity
                ]
                coordinate_differences[part, local_target] = coordinate_difference[
                    selected
                ]
        required_bytes = self.estimate_required_bytes(
            source,
            target,
            transition,
        )
        common_stable_ids = tuple(
            sorted(
                set(
                    int(value)
                    for value in np.asarray(source_layout.stable_block_ids)[
                        : source_layout.active_count
                    ]
                ).intersection(
                    int(value)
                    for value in np.asarray(target_layout.stable_block_ids)[
                        : target_layout.active_count
                    ]
                )
            )
            for source_layout, target_layout in zip(
                source.hierarchy.layouts,
                target.hierarchy.layouts,
                strict=True,
            )
        )
        transition_id = canonical_fingerprint(
            {
                "kind": "prepared-distributed-wave-amr-topology-transition",
                "source": source.execution_id,
                "target": target.execution_id,
                "overlap": transition.transition_id,
                "halo": halo.plan_id,
                "common_stable_block_ids": common_stable_ids,
                "required_bytes": required_bytes,
                "contract": "accepted-boundary-owner-computes",
            }
        )
        self.source = source
        self.target = target
        self.transition = transition
        self.halo = halo
        self.source_load_indices = jnp.asarray(source_load_indices)
        self.source_load_valid = jnp.asarray(source_load_valid)
        self.target_packed_indices = jnp.asarray(target_packed_indices)
        self.target_valid = jnp.asarray(target_valid)
        self.target_measures = jnp.asarray(target_measures, dtype=source.real_dtype)
        self.overlap_source_indices = jnp.asarray(overlap_source_indices)
        self.overlap_weights = jnp.asarray(overlap_weights, dtype=source.real_dtype)
        self.overlap_valid = jnp.asarray(overlap_valid)
        self.nearest_source_indices = jnp.asarray(nearest_source_indices)
        self.plus_source_indices = jnp.asarray(plus_source_indices)
        self.minus_source_indices = jnp.asarray(minus_source_indices)
        self.target_displacements = jnp.asarray(
            target_displacements, dtype=source.real_dtype
        )
        self.coordinate_differences = jnp.asarray(
            coordinate_differences, dtype=source.real_dtype
        )
        self.local_target_capacity = local_target_capacity
        self.overlap_capacity = overlap_capacity
        self.common_stable_block_ids = common_stable_ids
        self.required_bytes = int(required_bytes)
        self.transition_id = transition_id

    def execute(
        self,
        state: DistributedWaveAMRState,
        /,
        *,
        relative_node_floor: float,
    ) -> DistributedWaveAMRTopologyTransferResult:
        if self.source.hierarchy.mesh is None:
            raise RuntimeError(
                "Distributed topology transfer execution requires a JAX mesh."
            )
        checked = self.source.validate_state(state)
        if not bool(np.asarray(checked.accepted_boundary)):
            raise ValueError(
                "Distributed topology transfer requires an accepted boundary."
            )
        floor = float(relative_node_floor)
        if not np.isfinite(floor) or not 0.0 < floor < 1.0:
            raise ValueError(
                "relative_node_floor must lie strictly between zero and one."
            )
        axis_name = self.source.hierarchy.partition.axis_name
        self.target_displacements.shape[-1]
        source_shapes = self.source.routes.level_local_shapes
        target_shapes = self.target.routes.level_local_shapes
        target_sizes = self.target.routes.level_local_sizes
        source_count = self.source.routes.halo.entity_count
        real_dtype = self.source.real_dtype

        def transfer_local(level_values):
            part = jax.lax.axis_index(axis_name)
            source_packed = jnp.concatenate(
                tuple(value[0].reshape((-1,)) for value in level_values)
            )
            load_valid = self.source_load_valid[part]
            local = jnp.zeros((self.halo.local_capacity,), dtype=source_packed.dtype)
            local = local.at[jnp.arange(self.halo.local_capacity)].set(
                jnp.where(
                    load_valid,
                    source_packed[self.source_load_indices[part]],
                    0.0,
                )
            )
            full = local
            for phase, permutation in enumerate(self.halo.permutations):
                send_indices = self.halo.phase_send_indices[phase, part]
                send_valid = self.halo.phase_send_valid[phase, part]
                payload = jnp.where(
                    send_valid,
                    full[send_indices],
                    jnp.zeros((), dtype=full.dtype),
                )
                received = jax.lax.ppermute(
                    payload,
                    axis_name=axis_name,
                    perm=permutation,
                )
                receive_indices = self.halo.phase_receive_indices[phase, part]
                receive_valid = self.halo.phase_receive_valid[phase, part]
                destination = jnp.any(
                    jnp.asarray(
                        [target_part == part for _, target_part in permutation],
                        dtype=jnp.bool_,
                    )
                )
                full = jax.lax.cond(
                    destination,
                    lambda current: current.at[receive_indices].set(
                        jnp.where(
                            receive_valid,
                            received,
                            current[receive_indices],
                        )
                    ),
                    lambda current: current,
                    full,
                )
            target_valid = self.target_valid[part]
            source_indices = self.overlap_source_indices[part]
            route_valid = self.overlap_valid[part]
            weights = self.overlap_weights[part]
            source_values = full[source_indices]
            density = jnp.sum(
                jnp.where(
                    route_valid,
                    weights * jnp.abs(source_values) ** 2,
                    0.0,
                ),
                axis=1,
            )
            reference_real = jnp.sum(
                jnp.where(route_valid, weights * jnp.real(source_values), 0.0),
                axis=1,
            )
            reference_imaginary = jnp.sum(
                jnp.where(route_valid, weights * jnp.imag(source_values), 0.0),
                axis=1,
            )
            reference = jax.lax.complex(reference_real, reference_imaginary)
            nearest = self.nearest_source_indices[part]
            plus = full[self.plus_source_indices[part]]
            minus = full[self.minus_source_indices[part]]
            phase_difference = jnp.angle(
                jnp.exp(1j * (jnp.angle(plus) - jnp.angle(minus)))
            )
            coordinate = self.coordinate_differences[part]
            safe_coordinate = jnp.where(coordinate != 0.0, coordinate, 1.0)
            phase_gradient = phase_difference / safe_coordinate
            phase = jnp.angle(full[nearest]) + jnp.sum(
                phase_gradient * self.target_displacements[part],
                axis=1,
            )
            negative_local = jnp.any(target_valid & (density < 0.0))
            maximum_density = jax.lax.pmax(
                jnp.max(jnp.where(target_valid, density, 0.0), initial=0.0),
                axis_name,
            )
            occupied = target_valid & (density > floor * maximum_density)
            density_phase_candidate = jnp.sqrt(
                jnp.where(target_valid & (density >= 0.0), density, 0.0)
            ).astype(self.source.complex_dtype) * jnp.exp(1j * phase)
            reference_phase = jnp.angle(reference)
            phase_difference_from_overlap = jnp.abs(
                jnp.angle(jnp.exp(1j * (phase - reference_phase)))
            )
            complex_overlap_branch = (~occupied) | (
                phase_difference_from_overlap >= 0.5 * pi
            )
            candidate = jnp.where(
                target_valid,
                jnp.where(
                    complex_overlap_branch,
                    reference,
                    density_phase_candidate,
                ),
                0.0,
            )
            phase_defect = jax.lax.pmax(
                jnp.max(
                    jnp.where(
                        occupied,
                        phase_difference_from_overlap / pi,
                        0.0,
                    ),
                    initial=0.0,
                ),
                axis_name,
            )
            source_ids = self.halo.local_global_ids[part]
            source_owned = self.halo.local_owned[part] & (source_ids < source_count)
            source_measure = self.source.routes.entity_measures[
                jnp.minimum(source_ids, source_count - 1)
            ]
            source_probability = jax.lax.psum(
                jnp.sum(
                    jnp.where(
                        source_owned,
                        source_measure * jnp.abs(full) ** 2,
                        0.0,
                    )
                ),
                axis_name,
            )
            target_probability = jax.lax.psum(
                jnp.sum(
                    jnp.where(
                        target_valid,
                        self.target_measures[part] * density,
                        0.0,
                    )
                ),
                axis_name,
            )
            conservation_defect = jnp.abs(
                target_probability - source_probability
            ) / jnp.where(source_probability > 0.0, source_probability, 1.0)
            route_complete = _global_all(
                jnp.all(jnp.isfinite(jnp.where(self.halo.local_valid[part], full, 0.0)))
                & jnp.all(jnp.any(route_valid, axis=1) | ~target_valid),
                axis_name,
            )
            negative = ~_global_all(~negative_local, axis_name)
            finite = _global_all(
                jnp.all(jnp.isfinite(jnp.where(target_valid, candidate, 0.0)))
                & jnp.isfinite(conservation_defect)
                & jnp.isfinite(phase_defect),
                axis_name,
            )
            tolerance = 4096.0 * jnp.finfo(real_dtype).eps
            successful = (
                finite & route_complete & ~negative & (conservation_defect <= tolerance)
            )
            target_flat = jnp.zeros(
                (self.target.routes.local_packed_size,),
                dtype=candidate.dtype,
            )
            target_flat = target_flat.at[self.target_packed_indices[part]].add(
                jnp.where(target_valid & successful, candidate, 0.0)
            )
            output = []
            start = 0
            for shape, size in zip(target_shapes, target_sizes, strict=True):
                output.append(target_flat[start : start + size].reshape((1, *shape)))
                start += size
            return (
                tuple(output),
                conservation_defect,
                phase_defect,
                negative,
                route_complete,
                finite,
                successful,
            )

        source_specs = tuple(
            _part_spec(axis_name, 1 + len(shape)) for shape in source_shapes
        )
        target_specs = tuple(
            _part_spec(axis_name, 1 + len(shape)) for shape in target_shapes
        )
        scalar_spec = PartitionSpec()
        (
            candidate_values,
            conservation_defect,
            phase_defect,
            negative_density,
            route_complete,
            finite,
            successful,
        ) = jax.shard_map(
            transfer_local,
            mesh=self.source.hierarchy.mesh,
            in_specs=(source_specs,),
            out_specs=(
                target_specs,
                scalar_spec,
                scalar_spec,
                scalar_spec,
                scalar_spec,
                scalar_spec,
                scalar_spec,
            ),
            check_vma=False,
        )(checked.psi)
        candidate = self.target.bind_packed_state(
            candidate_values,
            checked.scale_factor,
            accepted_boundary=True,
        )
        return DistributedWaveAMRTopologyTransferResult(
            candidate,
            conservation_defect,
            phase_defect,
            negative_density,
            route_complete,
            finite,
            successful,
            self.transition_id,
        )


def _entity_centers(topology: Any, storage_indices: np.ndarray, /) -> np.ndarray:
    offsets = []
    next_offset = 0
    for level_plan in topology.plan.levels:
        offsets.append(next_offset)
        next_offset += level_plan.maximum_blocks * prod(level_plan.block_shape)
    bounds = np.asarray(
        [np.asarray(axis.bounds) for axis in topology.plan.grid.structured_axes],
        dtype=np.float64,
    )
    centers = []
    for storage in storage_indices:
        level = max(
            index for index, offset in enumerate(offsets) if offset <= int(storage)
        )
        level_plan = topology.plan.levels[level]
        block_cells = prod(level_plan.block_shape)
        block, local = divmod(int(storage) - offsets[level], block_cells)
        local_index = np.asarray(
            np.unravel_index(local, level_plan.block_shape), dtype=np.int32
        )
        logical = np.asarray(topology.levels[level].logical_indices)[block]
        global_index = logical * np.asarray(level_plan.block_shape) + local_index
        spacing = np.asarray(topology.plan.level_spacings[level])
        centers.append(bounds[:, 0] + (global_index + 0.5) * spacing)
    return np.asarray(centers)


__all__ = [
    "DistributedWaveAMRCheckpointEvidence",
    "DistributedWaveAMRDiagnostics",
    "DistributedWaveAMRGravityEvidence",
    "DistributedWaveAMRLinearEvidence",
    "DistributedWaveAMRRestoreEvidence",
    "DistributedWaveAMRObservables",
    "DistributedWaveAMRResult",
    "DistributedWaveAMRState",
    "DistributedWaveAMRTopologyTransferResult",
    "PreparedDistributedWaveAMR",
    "PreparedDistributedWaveAMRTopologyTransition",
]
