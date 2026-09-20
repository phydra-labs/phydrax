#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from math import prod

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._core import BlockHierarchyState, BlockHierarchyTopology
from ._fd_transfer import AMREntityTransferPlan


class FillPatchSource(IntEnum):
    """Per-cell FillPatch source classification in precedence order."""

    INACTIVE = -1
    INTERIOR = 0
    SAME_LEVEL = 1
    PERIODIC = 2
    COARSE_TIME_INTERPOLATED = 3
    PHYSICAL_BOUNDARY = 4
    UNRESOLVED = 5


class FDAMRPhysicalBoundaryRequest(StrictModule, NonTrainableState):
    """Caller-owned physical boundary values requested by one prepared level route."""

    level: int = eqx.field(static=True)
    mask: Array
    request_id: str = eqx.field(static=True)

    def __init__(self, level: int, mask: ArrayLike, plan_id: str, /):
        mask_ = jnp.asarray(mask, dtype=jnp.bool_)
        self.level = int(level)
        self.mask = mask_
        self.request_id = canonical_fingerprint(
            {
                "kind": "fd-amr-physical-boundary-request",
                "plan": plan_id,
                "level": int(level),
            }
        )

    @property
    def required(self) -> bool:
        return bool(np.any(np.asarray(self.mask)))


class FDAMRFillPatchWorkspace(StrictModule):
    """Padded cell-centered blocks plus validity and per-cell source evidence."""

    values: Array
    valid: Array
    source_class: Array
    workspace_id: str = eqx.field(static=True)

    def __init__(
        self,
        values: ArrayLike,
        valid: ArrayLike,
        source_class: ArrayLike,
        plan_id: str,
        /,
    ):
        values_ = jnp.asarray(values)
        valid_ = jnp.asarray(valid, dtype=jnp.bool_)
        sources = jnp.asarray(source_class, dtype=jnp.int8)
        if valid_.shape != sources.shape or values_.shape[: valid_.ndim] != valid_.shape:
            raise ValueError("FillPatch values, validity, and source classes must align.")
        self.values = values_
        self.valid = valid_
        self.source_class = sources
        self.workspace_id = canonical_fingerprint(
            {
                "kind": "fd-amr-fill-patch-workspace",
                "plan": plan_id,
                "shape": list(values_.shape),
            }
        )


class FDAMRFillPatchResult(StrictModule):
    """All prepared level workspaces and outstanding physical boundary requests."""

    workspaces: tuple[FDAMRFillPatchWorkspace, ...]
    physical_boundary_requests: tuple[FDAMRPhysicalBoundaryRequest, ...]
    complete: Array
    result_id: str = eqx.field(static=True)

    def require_complete(self, /) -> tuple[FDAMRFillPatchWorkspace, ...]:
        if not bool(self.complete):
            raise ValueError(
                "FillPatch has unresolved cells; provide every requested physical boundary value."
            )
        return self.workspaces


class FDAMRFillPatchPlan(StrictModule, NonTrainableState):
    """Prepared, source-classified cell-centered FillPatch routes for one AMR level."""

    topology: BlockHierarchyTopology
    level: int = eqx.field(static=True)
    transfer: AMREntityTransferPlan | None
    source_class: Array
    source_slots: Array
    source_local_indices: Array
    mapped_global_indices: Array
    coarse_donor_slots: Array
    coarse_donor_local_indices: Array
    coarse_donor_valid: Array
    coarse_child_indices: Array
    physical_boundary_mask: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        topology: BlockHierarchyTopology,
        level: int,
        transfer: AMREntityTransferPlan | None = None,
        /,
    ):
        if not isinstance(topology, BlockHierarchyTopology):
            raise TypeError("FillPatch preparation requires BlockHierarchyTopology.")
        level_ = int(level)
        if level_ < 0 or level_ >= len(topology.plan.levels):
            raise ValueError("FillPatch level is out of range.")
        dimension = len(topology.plan.grid.shape)
        if level_ == 0:
            if transfer is not None:
                raise ValueError("Level-zero FillPatch has no coarse transfer.")
        else:
            if not isinstance(transfer, AMREntityTransferPlan):
                raise TypeError("Fine FillPatch requires an AMREntityTransferPlan.")
            if transfer.axis_entities != ("interval",) * dimension:
                raise NotImplementedError(
                    "Prepared FillPatch is cell-centered; AMREntityTransferPlan "
                    "is the explicit non-cell extension seam."
                )
            if (
                transfer.refinement_ratio
                != topology.plan.levels[level_ - 1].refinement_ratio
            ):
                raise ValueError(
                    "FillPatch transfer ratio must match adjacent AMR levels."
                )
        level_plan = topology.plan.levels[level_]
        padded_shape = tuple(
            size + 2 * width
            for size, width in zip(
                level_plan.block_shape, level_plan.halo_width, strict=True
            )
        )
        route_shape = (level_plan.maximum_blocks,) + padded_shape
        sources = np.full(route_shape, int(FillPatchSource.INACTIVE), dtype=np.int8)
        source_slots = np.full(route_shape, -1, dtype=np.int32)
        source_local = np.full(route_shape + (dimension,), -1, dtype=np.int32)
        mapped_global = np.full(route_shape + (dimension,), -1, dtype=np.int32)
        metadata = topology.levels[level_]
        count = int(np.count_nonzero(np.asarray(metadata.active)))
        logical = np.asarray(metadata.logical_indices, dtype=np.int32)
        global_shape = topology.plan.global_cell_shapes[level_]
        block_shape = level_plan.block_shape
        logical_to_slot = {tuple(logical[slot]): slot for slot in range(count)}
        ratio = None if level_ == 0 else topology.plan.levels[level_ - 1].refinement_ratio
        for slot in range(count):
            block_origin = tuple(
                int(index) * size
                for index, size in zip(logical[slot], block_shape, strict=True)
            )
            for padded_index in np.ndindex(padded_shape):
                local = tuple(
                    index - width
                    for index, width in zip(
                        padded_index, level_plan.halo_width, strict=True
                    )
                )
                route_index = (slot,) + padded_index
                if all(
                    0 <= index < size
                    for index, size in zip(local, block_shape, strict=True)
                ):
                    sources[route_index] = int(FillPatchSource.INTERIOR)
                    source_slots[route_index] = slot
                    source_local[route_index] = local
                    mapped_global[route_index] = tuple(
                        origin + index
                        for origin, index in zip(block_origin, local, strict=True)
                    )
                    continue
                raw_global = [
                    origin + index
                    for origin, index in zip(block_origin, local, strict=True)
                ]
                mapped = list(raw_global)
                wrapped = False
                physical = False
                for axis, extent in enumerate(global_shape):
                    if 0 <= mapped[axis] < extent:
                        continue
                    if topology.plan.periodic_axes[axis]:
                        mapped[axis] %= extent
                        wrapped = True
                    else:
                        physical = True
                if physical:
                    sources[route_index] = int(FillPatchSource.PHYSICAL_BOUNDARY)
                    continue
                mapped_tuple = tuple(mapped)
                mapped_global[route_index] = mapped_tuple
                source_logical = tuple(
                    index // size
                    for index, size in zip(mapped_tuple, block_shape, strict=True)
                )
                same_slot = logical_to_slot.get(source_logical)
                if same_slot is not None:
                    sources[route_index] = int(
                        FillPatchSource.PERIODIC
                        if wrapped
                        else FillPatchSource.SAME_LEVEL
                    )
                    source_slots[route_index] = same_slot
                    source_local[route_index] = tuple(
                        index % size
                        for index, size in zip(mapped_tuple, block_shape, strict=True)
                    )
                    continue
                if level_ > 0 and ratio is not None:
                    coarse_index = tuple(index // ratio for index in mapped_tuple)
                    if topology.cell_is_active(level_ - 1, coarse_index):
                        sources[route_index] = int(
                            FillPatchSource.COARSE_TIME_INTERPOLATED
                        )
                        continue
                sources[route_index] = int(FillPatchSource.UNRESOLVED)
        if np.any(sources == int(FillPatchSource.UNRESOLVED)):
            raise ValueError(
                "Prepared FillPatch rejected an unresolved interior/coarse route; increase proper nesting or coverage."
            )
        donor_count = 3**dimension
        coarse_donor_slots = np.full(route_shape + (donor_count,), -1, dtype=np.int32)
        coarse_donor_local = np.full(
            route_shape + (donor_count, dimension), -1, dtype=np.int32
        )
        coarse_donor_valid = np.zeros(route_shape + (donor_count,), dtype=np.bool_)
        coarse_child_indices = np.full(route_shape + (dimension,), -1, dtype=np.int32)
        if level_ > 0 and ratio is not None:
            coarse_metadata = topology.levels[level_ - 1]
            coarse_count = int(np.count_nonzero(np.asarray(coarse_metadata.active)))
            coarse_logical = np.asarray(coarse_metadata.logical_indices, dtype=np.int32)
            coarse_logical_to_slot = {
                tuple(coarse_logical[slot]): slot for slot in range(coarse_count)
            }
            coarse_block_shape = topology.plan.levels[level_ - 1].block_shape
            coarse_global_shape = topology.plan.global_cell_shapes[level_ - 1]
            stencil_offsets = tuple(
                tuple(index - 1 for index in stencil_index)
                for stencil_index in np.ndindex((3,) * dimension)
            )
            coarse_routes = np.argwhere(
                sources == int(FillPatchSource.COARSE_TIME_INTERPOLATED)
            )
            for route_row in coarse_routes:
                route_index = tuple(route_row)
                fine_global = tuple(mapped_global[route_index])
                coarse_center = tuple(value // ratio for value in fine_global)
                coarse_child_indices[route_index] = tuple(
                    value % ratio for value in fine_global
                )
                for donor, offset in enumerate(stencil_offsets):
                    donor_global = [
                        center + delta
                        for center, delta in zip(coarse_center, offset, strict=True)
                    ]
                    for axis, extent in enumerate(coarse_global_shape):
                        if topology.plan.periodic_axes[axis]:
                            donor_global[axis] %= extent
                        else:
                            donor_global[axis] = min(
                                max(donor_global[axis], 0), extent - 1
                            )
                    donor_logical = tuple(
                        index // size
                        for index, size in zip(
                            donor_global, coarse_block_shape, strict=True
                        )
                    )
                    donor_slot = coarse_logical_to_slot.get(donor_logical)
                    if donor_slot is None:
                        raise ValueError(
                            "Prepared FillPatch rejected a missing active interior "
                            "coarse limiter donor; increase proper nesting."
                        )
                    donor_route = route_index + (donor,)
                    coarse_donor_slots[donor_route] = donor_slot
                    coarse_donor_local[donor_route] = tuple(
                        index % size
                        for index, size in zip(
                            donor_global, coarse_block_shape, strict=True
                        )
                    )
                    coarse_donor_valid[donor_route] = True
        physical_mask = sources == int(FillPatchSource.PHYSICAL_BOUNDARY)
        self.topology = topology
        self.level = level_
        self.transfer = transfer
        self.source_class = jnp.asarray(sources)
        self.source_slots = jnp.asarray(source_slots)
        self.source_local_indices = jnp.asarray(source_local)
        self.mapped_global_indices = jnp.asarray(mapped_global)
        self.coarse_donor_slots = jnp.asarray(coarse_donor_slots)
        self.coarse_donor_local_indices = jnp.asarray(coarse_donor_local)
        self.coarse_donor_valid = jnp.asarray(coarse_donor_valid)
        self.coarse_child_indices = jnp.asarray(coarse_child_indices)
        self.physical_boundary_mask = jnp.asarray(physical_mask)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fd-amr-fill-patch-plan",
                "epoch": topology.epoch.epoch_id,
                "level": level_,
                "transfer": None if transfer is None else transfer.transfer_id,
                "sources": array_tree_fingerprint(sources),
                "source_slots": array_tree_fingerprint(source_slots),
                "source_local": array_tree_fingerprint(source_local),
                "mapped_global": array_tree_fingerprint(mapped_global),
                "coarse_donor_slots": array_tree_fingerprint(coarse_donor_slots),
                "coarse_donor_local": array_tree_fingerprint(coarse_donor_local),
                "coarse_donor_valid": array_tree_fingerprint(coarse_donor_valid),
                "coarse_child_indices": array_tree_fingerprint(coarse_child_indices),
            }
        )

    def _same_level_values(self, state: BlockHierarchyState, /) -> Array:
        level_state = state.levels[self.level]
        safe_slots = jnp.maximum(self.source_slots, 0)
        safe_local = jnp.maximum(self.source_local_indices, 0)
        index = (safe_slots,) + tuple(
            safe_local[..., axis] for axis in range(safe_local.shape[-1])
        )
        return level_state.safe_values()[index]

    def _coarse_donor_values(
        self,
        state: BlockHierarchyState,
        /,
    ) -> Array:
        if self.level == 0:
            raise RuntimeError("Level-zero FillPatch has no coarse donor routes.")
        coarse_values = state.levels[self.level - 1].safe_values()
        safe_slots = jnp.maximum(self.coarse_donor_slots, 0)
        safe_local = jnp.maximum(self.coarse_donor_local_indices, 0)
        index = (safe_slots,) + tuple(
            safe_local[..., axis] for axis in range(safe_local.shape[-1])
        )
        gathered = coarse_values[index]
        component_rank = gathered.ndim - self.coarse_donor_valid.ndim
        return jnp.where(
            self.coarse_donor_valid.reshape(
                self.coarse_donor_valid.shape + (1,) * component_rank
            ),
            gathered,
            jnp.zeros((), dtype=gathered.dtype),
        )

    def execute(
        self,
        state: BlockHierarchyState,
        coarse_old: BlockHierarchyState,
        coarse_new: BlockHierarchyState,
        coarse_old_time: ArrayLike,
        coarse_new_time: ArrayLike,
        fill_time: ArrayLike,
        physical_boundary_values: ArrayLike | None = None,
        /,
    ) -> tuple[FDAMRFillPatchWorkspace, FDAMRPhysicalBoundaryRequest]:
        epoch_id = self.topology.epoch.epoch_id
        if any(
            not isinstance(value, BlockHierarchyState)
            or value.topology.epoch.epoch_id != epoch_id
            for value in (state, coarse_old, coarse_new)
        ):
            raise ValueError(
                "FillPatch states must share the prepared fixed topology epoch."
            )
        same_values = self._same_level_values(state)
        component_rank = same_values.ndim - self.source_class.ndim
        same_mask = (
            (self.source_class == int(FillPatchSource.INTERIOR))
            | (self.source_class == int(FillPatchSource.SAME_LEVEL))
            | (self.source_class == int(FillPatchSource.PERIODIC))
        )
        values = jnp.where(
            same_mask.reshape(same_mask.shape + (1,) * component_rank),
            same_values,
            jnp.zeros((), dtype=same_values.dtype),
        )
        valid = same_mask
        if self.level > 0:
            dtype = jnp.result_type(coarse_old_time, coarse_new_time, fill_time, 1.0)
            old_time = jnp.asarray(coarse_old_time, dtype=dtype)
            new_time = jnp.asarray(coarse_new_time, dtype=dtype)
            target_time = jnp.asarray(fill_time, dtype=dtype)
            scale = jnp.maximum(
                1.0,
                jnp.maximum(
                    jnp.abs(old_time),
                    jnp.maximum(jnp.abs(new_time), jnp.abs(target_time)),
                ),
            )
            tolerance = 16.0 * jnp.finfo(dtype).eps * scale
            span = new_time - old_time
            same_time = jnp.abs(span) <= tolerance
            denominator = jnp.where(same_time, 1.0, span)
            alpha = jnp.where(same_time, 0.0, (target_time - old_time) / denominator)
            invalid_time = (
                ~jnp.isfinite(old_time)
                | ~jnp.isfinite(new_time)
                | ~jnp.isfinite(target_time)
                | (span < -tolerance)
                | (target_time < old_time - tolerance)
                | (target_time > new_time + tolerance)
                | (same_time & (jnp.abs(target_time - old_time) > tolerance))
            )
            alpha = eqx.error_if(
                alpha,
                invalid_time,
                "FillPatch time must lie within the coarse old/new interval.",
            )
            alpha = jnp.clip(alpha, 0.0, 1.0)
            old_donors = self._coarse_donor_values(coarse_old)
            new_donors = self._coarse_donor_values(coarse_new)
            donors = old_donors + alpha * (new_donors - old_donors)
            if self.transfer is None:
                raise RuntimeError("Fine FillPatch lost its prepared cell transfer.")
            dimension = len(self.topology.plan.grid.shape)
            component_shape = donors.shape[self.coarse_donor_valid.ndim :]
            donor_patches = donors.reshape(
                self.source_class.shape + (3,) * dimension + component_shape
            )
            route_count = prod(self.source_class.shape)
            flat_patches = donor_patches.reshape(
                (route_count,) + (3,) * dimension + component_shape
            )
            prolonged = jax.vmap(self.transfer.prolong)(flat_patches)
            child = jnp.maximum(
                self.coarse_child_indices.reshape((route_count, dimension)), 0
            )
            ratio = self.transfer.refinement_ratio
            fine_index = (jnp.arange(route_count, dtype=jnp.int32),) + tuple(
                ratio + child[:, axis] for axis in range(dimension)
            )
            coarse_values = prolonged[fine_index].reshape(
                self.source_class.shape + component_shape
            )
            coarse_mask = self.source_class == int(
                FillPatchSource.COARSE_TIME_INTERPOLATED
            )
            values = jnp.where(
                coarse_mask.reshape(coarse_mask.shape + (1,) * component_rank),
                coarse_values,
                values,
            )
            valid = valid | coarse_mask
        if physical_boundary_values is not None:
            boundary = jnp.asarray(physical_boundary_values, dtype=values.dtype)
            if boundary.shape != values.shape:
                raise ValueError(
                    "Physical boundary values must match the padded level workspace."
                )
            physical_mask = self.physical_boundary_mask
            values = jnp.where(
                physical_mask.reshape(physical_mask.shape + (1,) * component_rank),
                boundary,
                values,
            )
            valid = valid | physical_mask
        active = self.topology.levels[self.level].active.reshape(
            (self.topology.plan.levels[self.level].maximum_blocks,)
            + (1,) * (valid.ndim - 1)
        )
        valid = valid & active
        values = jnp.where(
            valid.reshape(valid.shape + (1,) * component_rank),
            values,
            jnp.zeros((), dtype=values.dtype),
        )
        return (
            FDAMRFillPatchWorkspace(values, valid, self.source_class, self.plan_id),
            FDAMRPhysicalBoundaryRequest(
                self.level, self.physical_boundary_mask, self.plan_id
            ),
        )


__all__ = [
    "FDAMRFillPatchPlan",
    "FDAMRFillPatchResult",
    "FDAMRFillPatchWorkspace",
    "FDAMRPhysicalBoundaryRequest",
    "FillPatchSource",
]
