#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._core import BlockLevelPlan, BlockLevelState
from ._fd_transfer import AMREntityTransferPlan


class FDAMRHaloWorkspace(StrictModule):
    values: Array
    valid: Array
    source_kind: str = eqx.field(static=True)
    workspace_id: str = eqx.field(static=True)

    def __init__(
        self,
        values: ArrayLike,
        valid: ArrayLike,
        source_kind: str,
        plan_id: str,
        /,
    ):
        values_ = jnp.asarray(values)
        valid_ = jnp.asarray(valid, dtype=bool)
        if valid_.shape != values_.shape[: valid_.ndim]:
            raise ValueError("AMR halo validity mask must prefix the workspace shape.")
        source = str(source_kind)
        if source not in ("same_level", "coarse_fine", "combined"):
            raise ValueError("Unknown AMR halo source kind.")
        self.values = values_
        self.valid = valid_
        self.source_kind = source
        self.workspace_id = canonical_fingerprint(
            {
                "kind": "fd-amr-halo-workspace",
                "plan": plan_id,
                "source_kind": source,
                "shape": list(values_.shape),
            }
        )


class FDAMRHaloPlan(StrictModule, NonTrainableState):
    """Multidimensional same-level and coarse/fine block halo realization."""

    level_plan: BlockLevelPlan
    plan_id: str = eqx.field(static=True)

    def __init__(self, level_plan: BlockLevelPlan, /):
        if not isinstance(level_plan, BlockLevelPlan):
            raise TypeError("FD AMR halo plan requires BlockLevelPlan.")
        self.level_plan = level_plan
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fd-amr-halo-plan",
                "level": level_plan.plan_id,
            }
        )

    def fill_same_level(self, state: BlockLevelState, /) -> FDAMRHaloWorkspace:
        if (
            not isinstance(state, BlockLevelState)
            or state.plan.plan_id != self.level_plan.plan_id
        ):
            raise ValueError("AMR same-level halo state does not match the plan.")
        values = state.safe_values()
        padded = values
        dimension = len(self.level_plan.block_shape)
        for axis in range(dimension):
            width = self.level_plan.halo_width[axis]
            if width == 0:
                continue
            array_axis = axis + 1
            padding = [(0, 0)] * padded.ndim
            padding[array_axis] = (width, width)
            next_values = jnp.pad(padded, tuple(padding))
            lower_slots = state.metadata.neighbor_slots[:, axis, 0]
            upper_slots = state.metadata.neighbor_slots[:, axis, 1]
            safe_lower = jnp.where(lower_slots >= 0, lower_slots, 0)
            safe_upper = jnp.where(upper_slots >= 0, upper_slots, 0)
            lower_source = [slice(None)] * padded.ndim
            upper_source = [slice(None)] * padded.ndim
            lower_source[array_axis] = slice(padded.shape[array_axis] - width, None)
            upper_source[array_axis] = slice(0, width)
            lower_data = padded[safe_lower][tuple(lower_source)]
            upper_data = padded[safe_upper][tuple(upper_source)]
            lower_mask = (lower_slots >= 0).reshape((-1,) + (1,) * (lower_data.ndim - 1))
            upper_mask = (upper_slots >= 0).reshape((-1,) + (1,) * (upper_data.ndim - 1))
            lower_target = [slice(None)] * next_values.ndim
            upper_target = [slice(None)] * next_values.ndim
            lower_target[array_axis] = slice(0, width)
            upper_target[array_axis] = slice(next_values.shape[array_axis] - width, None)
            next_values = next_values.at[tuple(lower_target)].set(
                jnp.where(lower_mask, lower_data, 0.0)
            )
            next_values = next_values.at[tuple(upper_target)].set(
                jnp.where(upper_mask, upper_data, 0.0)
            )
            padded = next_values
        valid = state.metadata.active.reshape(
            (self.level_plan.maximum_blocks,) + (1,) * dimension
        )
        valid = jnp.broadcast_to(
            valid,
            (self.level_plan.maximum_blocks,)
            + tuple(
                size + 2 * width
                for size, width in zip(
                    self.level_plan.block_shape,
                    self.level_plan.halo_width,
                    strict=True,
                )
            ),
        )
        return FDAMRHaloWorkspace(
            padded,
            valid,
            "same_level",
            self.plan_id,
        )

    def fill_coarse_fine(
        self,
        fine_state: BlockLevelState,
        coarse_state: BlockLevelState,
        parent_slots: ArrayLike,
        child_offsets: ArrayLike,
        transfer: AMREntityTransferPlan,
        /,
    ) -> FDAMRHaloWorkspace:
        if fine_state.plan.plan_id != self.level_plan.plan_id:
            raise ValueError("Fine state must match this halo plan level.")
        slots = jnp.asarray(parent_slots, dtype=jnp.int32)
        offsets = jnp.asarray(child_offsets, dtype=jnp.int32)
        if slots.shape != (self.level_plan.maximum_blocks,):
            raise ValueError(
                "parent_slots must contain one coarse slot per fine capacity."
            )
        if offsets.shape != (
            self.level_plan.maximum_blocks,
            len(self.level_plan.block_shape),
        ):
            raise ValueError(
                "child_offsets must align with fine block capacity and rank."
            )
        if not isinstance(transfer, AMREntityTransferPlan):
            raise TypeError("coarse/fine halo requires AMREntityTransferPlan.")
        offsets = eqx.error_if(
            offsets,
            jnp.any((offsets < 0) | (offsets >= self.level_plan.refinement_ratio)),
            "Active child offsets must lie within the refinement ratio.",
        )
        safe_slots = jnp.where(slots >= 0, slots, 0)
        parent_values = coarse_state.safe_values()[safe_slots]
        prolonged_parent = jax.vmap(transfer.prolong)(parent_values)
        expected_parent_shape = tuple(
            size * self.level_plan.refinement_ratio
            for size in self.level_plan.block_shape
        )
        if (
            prolonged_parent.shape[1 : 1 + len(self.level_plan.block_shape)]
            != expected_parent_shape
        ):
            raise ValueError("Prolonged parent block has incompatible refined shape.")

        def select_child(parent: Array, offset: Array) -> Array:
            spatial_rank = len(self.level_plan.block_shape)
            starts = tuple(
                offset[axis] * self.level_plan.block_shape[axis]
                for axis in range(spatial_rank)
            ) + (0,) * (parent.ndim - spatial_rank)
            sizes = self.level_plan.block_shape + parent.shape[spatial_rank:]
            return jax.lax.dynamic_slice(parent, starts, sizes)

        prolonged = jax.vmap(select_child)(prolonged_parent, offsets)
        fine = fine_state.safe_values()
        padding = [(0, 0)]
        for width in self.level_plan.halo_width:
            padding.append((width, width))
        padding.extend((0, 0) for _ in fine.shape[1 + len(self.level_plan.block_shape) :])
        parent_padded = jnp.pad(prolonged, tuple(padding), mode="edge")
        interior = [slice(None)]
        for size, width in zip(
            self.level_plan.block_shape,
            self.level_plan.halo_width,
            strict=True,
        ):
            interior.append(slice(width, width + size))
        interior.extend(
            slice(None) for _ in fine.shape[1 + len(self.level_plan.block_shape) :]
        )
        parent_padded = parent_padded.at[tuple(interior)].set(fine)
        valid_parent = (slots >= 0) & fine_state.metadata.active
        valid = valid_parent.reshape(
            (self.level_plan.maximum_blocks,) + (1,) * len(self.level_plan.block_shape)
        )
        valid = jnp.broadcast_to(valid, parent_padded.shape[: valid.ndim])
        return FDAMRHaloWorkspace(
            jnp.where(
                valid.reshape(valid.shape + (1,) * (parent_padded.ndim - valid.ndim)),
                parent_padded,
                0.0,
            ),
            valid,
            "coarse_fine",
            self.plan_id,
        )


__all__ = ["FDAMRHaloPlan", "FDAMRHaloWorkspace"]
