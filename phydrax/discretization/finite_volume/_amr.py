#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import pairwise
from math import prod

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import ArraySpace
from ...sparse import EdgeRelation, SparseCoordinateOperator
from .._conservation_ledger import (
    AcceptedConservationFluxIntegralBlock,
    AcceptedConservationIntegralLedger,
)
from ..amr import BlockHierarchyTopology, FluxRegister, PreparedFDAMRHierarchy
from ._precision import FiniteVolumePrecisionPolicy


def _time_tolerance(first: Array, second: Array, /) -> Array:
    dtype = jnp.result_type(first, second, jnp.asarray(1.0))
    first_ = first.astype(dtype)
    second_ = second.astype(dtype)
    scale = jnp.maximum(
        jnp.asarray(1.0, dtype=dtype),
        jnp.maximum(jnp.abs(first_), jnp.abs(second_)),
    )
    return jnp.asarray(16.0, dtype=dtype) * jnp.finfo(dtype).eps * scale


def _route_block(
    ledger: AcceptedConservationIntegralLedger,
    route_id: str,
    /,
) -> AcceptedConservationFluxIntegralBlock:
    matches = tuple(block for block in ledger.blocks if block.route_id == route_id)
    if len(matches) != 1:
        raise ValueError(
            f"Accepted ledger must contain exactly one block for route {route_id!r}."
        )
    return matches[0]


class BlockAMRConservationPlan(StrictModule, NonTrainableState):
    """Accepted-ledger synchronization bound to one immutable block topology."""

    hierarchy: PreparedFDAMRHierarchy
    topology: BlockHierarchyTopology
    precision: FiniteVolumePrecisionPolicy
    cell_volumes: Array
    active_cell_mask: Array
    covered_cell_masks: tuple[Array, ...]
    restriction_operators: tuple[SparseCoordinateOperator, ...]
    level_cell_offsets: tuple[int, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        hierarchy: PreparedFDAMRHierarchy,
        topology: BlockHierarchyTopology,
        /,
        *,
        precision: FiniteVolumePrecisionPolicy | None = None,
    ):
        if not isinstance(hierarchy, PreparedFDAMRHierarchy):
            raise TypeError("hierarchy must be PreparedFDAMRHierarchy.")
        if not isinstance(topology, BlockHierarchyTopology) or (
            topology.plan.plan_id != hierarchy.plan.hierarchy.plan_id
        ):
            raise ValueError("topology must belong to the prepared hierarchy.")
        precision_ = (
            FiniteVolumePrecisionPolicy(hierarchy.plan.precision.field_dtype)
            if precision is None
            else precision
        )
        if not isinstance(precision_, FiniteVolumePrecisionPolicy):
            raise TypeError("precision must be FiniteVolumePrecisionPolicy.")
        if precision_.storage_dtype != hierarchy.plan.precision.field_dtype:
            raise ValueError(
                "Finite-volume storage precision must match hierarchy field precision."
            )
        offsets: list[int] = []
        volumes: list[np.ndarray] = []
        active: list[np.ndarray] = []
        next_offset = 0
        for level_plan, metadata, spacing in zip(
            topology.plan.levels,
            topology.levels,
            topology.plan.level_spacings,
            strict=True,
        ):
            offsets.append(next_offset)
            cell_count = level_plan.maximum_blocks * prod(level_plan.block_shape)
            next_offset += cell_count
            volumes.append(
                np.full((cell_count,), prod(spacing), dtype=precision_.reduction_dtype)
            )
            active.append(
                np.repeat(
                    np.asarray(metadata.active, dtype=np.bool_),
                    prod(level_plan.block_shape),
                )
            )
        operators: list[SparseCoordinateOperator] = []
        for level, coarse_plan in enumerate(topology.plan.levels[:-1]):
            fine_plan = topology.plan.levels[level + 1]
            ratio = coarse_plan.refinement_ratio
            coarse_size = coarse_plan.maximum_blocks * prod(coarse_plan.block_shape)
            fine_size = fine_plan.maximum_blocks * prod(fine_plan.block_shape)
            source_indices: list[int] = []
            target_indices: list[int] = []
            coarse_active = np.asarray(topology.levels[level].active, dtype=np.bool_)
            coarse_logical = np.asarray(
                topology.levels[level].logical_indices, dtype=np.int32
            )
            covered = np.asarray(topology.covered_cells[level], dtype=np.bool_)
            for coarse_slot in np.flatnonzero(coarse_active):
                origin = tuple(
                    int(index) * size
                    for index, size in zip(
                        coarse_logical[coarse_slot],
                        coarse_plan.block_shape,
                        strict=True,
                    )
                )
                for local in np.ndindex(coarse_plan.block_shape):
                    if not covered[(int(coarse_slot),) + local]:
                        continue
                    coarse_index = coarse_slot * prod(coarse_plan.block_shape) + int(
                        np.ravel_multi_index(local, coarse_plan.block_shape)
                    )
                    lower = tuple(
                        (start + offset) * ratio
                        for start, offset in zip(origin, local, strict=True)
                    )
                    for child in np.ndindex((ratio,) * len(coarse_plan.block_shape)):
                        fine_coordinate = tuple(
                            start + offset
                            for start, offset in zip(lower, child, strict=True)
                        )
                        fine_slot = topology.cell_slot(level + 1, fine_coordinate)
                        if fine_slot is None:
                            raise RuntimeError(
                                "Covered coarse AMR cell lacks a fine restriction donor."
                            )
                        fine_box = topology.patch_boxes(level + 1)[fine_slot]
                        fine_local = tuple(
                            value - start
                            for value, start in zip(
                                fine_coordinate, fine_box.lower, strict=True
                            )
                        )
                        source_indices.append(
                            fine_slot * prod(fine_plan.block_shape)
                            + int(
                                np.ravel_multi_index(
                                    fine_local,
                                    fine_plan.block_shape,
                                )
                            )
                        )
                        target_indices.append(coarse_index)
            relation = EdgeRelation(
                np.asarray(source_indices, dtype=np.int32),
                np.asarray(target_indices, dtype=np.int32),
                source_size=fine_size,
                target_size=coarse_size,
            )
            operators.append(
                SparseCoordinateOperator(
                    relation,
                    jnp.full(
                        relation.route_shape,
                        1.0 / ratio ** len(coarse_plan.block_shape),
                        dtype=precision_.reduction_dtype,
                    ),
                    source=ArraySpace(
                        (fine_size,),
                        dtype=precision_.reduction_dtype,
                    ),
                    target=ArraySpace(
                        (coarse_size,),
                        dtype=precision_.reduction_dtype,
                    ),
                    operator_id=canonical_fingerprint(
                        {
                            "kind": "block-amr-covered-cell-restriction",
                            "epoch": topology.epoch.epoch_id,
                            "level": level,
                            "relation": {
                                "sources": source_indices,
                                "targets": target_indices,
                            },
                        }
                    ),
                    accumulation_dtype=precision_.reduction_dtype,
                )
            )
        self.hierarchy = hierarchy
        self.topology = topology
        self.precision = precision_
        self.cell_volumes = jnp.asarray(np.concatenate(volumes))
        self.active_cell_mask = jnp.asarray(np.concatenate(active))
        self.covered_cell_masks = tuple(topology.covered_cells[:-1])
        self.restriction_operators = tuple(operators)
        self.level_cell_offsets = tuple(offsets)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "block-amr-conservation-plan",
                "hierarchy": hierarchy.prepared_id,
                "epoch": topology.epoch.epoch_id,
                "precision": precision_.policy_id,
            }
        )

    def _validate_ledgers(
        self,
        ledgers: Sequence[AcceptedConservationIntegralLedger],
        /,
    ) -> tuple[tuple[AcceptedConservationIntegralLedger, ...], Array]:
        values = tuple(ledgers)
        if not values or any(
            not isinstance(value, AcceptedConservationIntegralLedger) for value in values
        ):
            raise TypeError(
                "Accepted AMR aggregation requires nonempty accepted-ledger sequences."
            )
        epoch = self.topology.epoch.epoch_id
        if any(
            value.units != "content"
            or value.start_topology_epoch_id != epoch
            or value.end_topology_epoch_id != epoch
            or value.geometry_family_id != self.topology.plan.geometry_id
            or value.geometry_layout_id != self.topology.partition_id
            or value.evidence_policy_id != self.precision.policy_id
            or value.cell_count != self.active_cell_mask.size
            or value.active_cell_mask.shape != self.active_cell_mask.shape
            for value in values
        ):
            raise ValueError(
                "Accepted ledgers must share this hierarchy epoch, geometry, "
                "precision identity, units, and cell layout."
            )
        token = values[0].source_integral
        for value in values:
            token = eqx.error_if(
                token,
                jnp.any(value.active_cell_mask != self.active_cell_mask),
                "Accepted ledger active cells do not match the hierarchy topology.",
            )
        for index, (previous, current) in enumerate(pairwise(values), start=1):
            difference = current.start_time - previous.end_time
            tolerance = _time_tolerance(current.start_time, previous.end_time)
            token = eqx.error_if(
                token,
                difference > tolerance,
                f"Accepted ledger intervals contain a gap before ledger {index}.",
            )
            token = eqx.error_if(
                token,
                difference < -tolerance,
                f"Accepted ledger intervals overlap before ledger {index}.",
            )
            token = eqx.error_if(
                token,
                current.accepted_step <= previous.accepted_step,
                "Accepted-step IDs must be strictly monotone.",
            )
        return values, token

    def aggregate_accepted_route(
        self,
        ledgers: Sequence[AcceptedConservationIntegralLedger],
        route_id: str,
        /,
    ) -> Array:
        """Sum one immutable route across a contiguous accepted interval union."""
        values, token = self._validate_ledgers(ledgers)
        blocks = tuple(_route_block(ledger, str(route_id)) for ledger in values)
        identity = (
            blocks[0].route_id,
            blocks[0].block_kind,
            blocks[0].component_shape,
        )
        if any(
            (block.route_id, block.block_kind, block.component_shape) != identity
            for block in blocks[1:]
        ):
            raise ValueError("Accepted route aggregation requires one canonical route.")
        total = jnp.zeros_like(self.precision.reduction(blocks[0].flux_integral))
        total = total + jnp.asarray(0.0, dtype=total.dtype) * jnp.sum(token)
        for block in blocks:
            total = self.precision.reduction(
                total + self.precision.reduction(block.flux_integral)
            )
        return total

    def flux_register(
        self,
        coarse_ledger: AcceptedConservationIntegralLedger,
        fine_ledgers: Sequence[AcceptedConservationIntegralLedger],
        coarse_route_id: str,
        fine_route_id: str,
        restrict_flux: Callable[[Array], Array],
        interface_mask: ArrayLike,
        /,
    ) -> FluxRegister:
        """Build a cell-routed register from exact accepted face-route integrals."""
        if not callable(restrict_flux):
            raise TypeError("restrict_flux must be callable.")
        coarse_values, coarse_token = self._validate_ledgers((coarse_ledger,))
        fine_values, _ = self._validate_ledgers(fine_ledgers)
        coarse = coarse_values[0]
        token = coarse_token
        token = eqx.error_if(
            token,
            jnp.abs(fine_values[0].start_time - coarse.start_time)
            > _time_tolerance(fine_values[0].start_time, coarse.start_time),
            "Fine accepted intervals must start at the coarse interval start.",
        )
        token = eqx.error_if(
            token,
            jnp.abs(fine_values[-1].end_time - coarse.end_time)
            > _time_tolerance(fine_values[-1].end_time, coarse.end_time),
            "Fine accepted intervals must end at the coarse interval end.",
        )
        coarse_block = _route_block(coarse, str(coarse_route_id))
        fine_flux = self.aggregate_accepted_route(fine_values, str(fine_route_id))
        restricted = self.precision.reduction(restrict_flux(fine_flux))
        coarse_flux = self.precision.reduction(coarse_block.flux_integral)
        coarse_flux = coarse_flux + jnp.asarray(0.0, dtype=coarse_flux.dtype) * jnp.sum(
            token
        )
        if restricted.shape != coarse_flux.shape:
            raise ValueError("Restricted fine flux must match the coarse route shape.")
        mask = jnp.asarray(interface_mask)
        if (
            mask.dtype != jnp.dtype(jnp.bool_)
            or mask.shape != coarse_block.active_mask.shape
        ):
            raise ValueError("interface_mask must select the coarse route faces.")
        mask = mask & coarse_block.active_mask
        component_rank = coarse_flux.ndim - mask.ndim
        masked_coarse = jnp.where(
            mask.reshape(mask.shape + (1,) * component_rank), coarse_flux, 0.0
        )
        masked_fine = jnp.where(
            mask.reshape(mask.shape + (1,) * component_rank), restricted, 0.0
        )

        def scatter(values: Array) -> Array:
            result = jnp.zeros(
                (coarse.cell_count,) + coarse.component_shape,
                dtype=self.precision.reduction_dtype,
            )
            result = result.at[coarse_block.owner_cells].add(-values)
            neighbors = coarse_block.neighbor_cells
            safe = jnp.maximum(neighbors, 0)
            neighbor_mask = (neighbors >= 0).reshape(
                neighbors.shape + (1,) * component_rank
            )
            return result.at[safe].add(jnp.where(neighbor_mask, values, 0.0))

        cell_coarse = scatter(masked_coarse)
        cell_fine = scatter(masked_fine)
        cell_mask = jnp.zeros((coarse.cell_count,), dtype=jnp.bool_)
        cell_mask = cell_mask.at[coarse_block.owner_cells].max(mask)
        valid_neighbor = mask & (coarse_block.neighbor_cells >= 0)
        cell_mask = cell_mask.at[jnp.maximum(coarse_block.neighbor_cells, 0)].max(
            valid_neighbor
        )
        return FluxRegister(
            cell_coarse,
            cell_fine,
            cell_mask,
            accumulated_time=coarse.end_time - coarse.start_time,
            refinement_ratio=len(fine_values),
            register_id=canonical_fingerprint(
                {
                    "kind": "block-amr-accepted-flux-register",
                    "plan": self.plan_id,
                    "coarse_route": coarse_block.route_id,
                    "fine_route": str(fine_route_id),
                    "precision": self.precision.policy_id,
                }
            ),
        )

    def reflux(
        self,
        level_values: Sequence[ArrayLike],
        register: FluxRegister,
        /,
    ) -> tuple[Array, ...]:
        """Apply one route-bound cell register to the matching hierarchy payload."""
        values = tuple(jnp.asarray(value) for value in level_values)
        if len(values) != len(self.topology.plan.levels):
            raise ValueError("Reflux requires one value array per hierarchy level.")
        components = register.coarse_flux.shape[1:]
        flat_chunks = []
        for level_plan, value in zip(self.topology.plan.levels, values, strict=True):
            expected_prefix = (level_plan.maximum_blocks,) + level_plan.block_shape
            if value.shape[: len(expected_prefix)] != expected_prefix or (
                value.shape[len(expected_prefix) :] != components
            ):
                raise ValueError("Reflux values do not match hierarchy cells/components.")
            flat_chunks.append(value.reshape((-1,) + components))
        flat = jnp.concatenate(tuple(flat_chunks), axis=0)
        if flat.shape != register.coarse_flux.shape:
            raise ValueError("Flux register does not match this hierarchy cell layout.")
        refluxed = register.apply(
            flat,
            self.cell_volumes,
            accumulation_dtype=self.precision.reduction_dtype,
            output_dtype=self.precision.storage_dtype,
        )
        refluxed = jnp.where(
            self.active_cell_mask.reshape(
                self.active_cell_mask.shape + (1,) * len(components)
            ),
            refluxed,
            0.0,
        )
        result = []
        for level, (level_plan, value) in enumerate(
            zip(self.topology.plan.levels, values, strict=True)
        ):
            begin = self.level_cell_offsets[level]
            count = level_plan.maximum_blocks * prod(level_plan.block_shape)
            result.append(refluxed[begin : begin + count].reshape(value.shape))
        return tuple(result)

    def covered_cell_mask(self, coarse_level: int, /) -> Array:
        """Return block-shaped coarse cells wholly covered by the next fine level."""
        level = int(coarse_level)
        if level < 0 or level >= len(self.topology.plan.levels) - 1:
            raise ValueError("coarse_level must have an adjacent fine level.")
        return self.covered_cell_masks[level]

    def restrict_covered(
        self,
        coarse_values: ArrayLike,
        fine_values: ArrayLike,
        coarse_level: int,
        /,
    ) -> Array:
        """Restrict covered cells through canonical sparse fine-to-coarse routes."""
        level = int(coarse_level)
        covered = self.covered_cell_mask(level)
        coarse_plan = self.topology.plan.levels[level]
        fine_plan = self.topology.plan.levels[level + 1]
        coarse = jnp.asarray(coarse_values)
        fine = jnp.asarray(fine_values)
        coarse_prefix = (coarse_plan.maximum_blocks,) + coarse_plan.block_shape
        fine_prefix = (fine_plan.maximum_blocks,) + fine_plan.block_shape
        if (
            coarse.shape[: len(coarse_prefix)] != coarse_prefix
            or fine.shape[: len(fine_prefix)] != fine_prefix
            or coarse.shape[len(coarse_prefix) :] != fine.shape[len(fine_prefix) :]
        ):
            raise ValueError("Covered restriction values do not match adjacent levels.")
        component_shape = coarse.shape[len(coarse_prefix) :]
        component_count = prod(component_shape) if component_shape else 1
        operator = self.restriction_operators[level]
        fine_flat = self.precision.reduction(fine).reshape((-1, component_count))
        restricted_flat = jnp.stack(
            tuple(
                operator.mv(fine_flat[:, component])
                for component in range(component_count)
            ),
            axis=1,
        )
        restricted = restricted_flat.reshape(coarse_prefix + component_shape)
        mask = covered.reshape(covered.shape + (1,) * len(component_shape))
        result = jnp.where(mask, restricted, self.precision.reduction(coarse))
        active = self.topology.levels[level].active.reshape(
            (coarse_plan.maximum_blocks,) + (1,) * (result.ndim - 1)
        )
        return self.precision.storage(jnp.where(active, result, 0.0))


__all__ = ["BlockAMRConservationPlan"]
