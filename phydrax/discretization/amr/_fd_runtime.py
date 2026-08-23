#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._core import BlockHierarchyPlan, BlockHierarchyState, BlockLevelState
from ._fd_halo import FDAMRHaloPlan, FDAMRHaloWorkspace
from ._fd_transfer import AMREntityTransferPlan
from ._refinement import FixedCapacityRefinementPlan, RefinementDecision
from ._reflux import FluxRegister


class AMRSubcycleResult(StrictModule):
    coarse_state: Array
    fine_state: Array
    flux_register: FluxRegister
    substeps: int = eqx.field(static=True)
    temporal_method_id: str = eqx.field(static=True)


class ConservativeAMRSubcyclingPlan(StrictModule, NonTrainableState):
    """Two-level temporal subcycling with time-integrated conservative reflux."""

    refinement_ratio: int = eqx.field(static=True)
    temporal_method_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        refinement_ratio: int = 2,
        /,
        *,
        temporal_method_id: str = "temporal:caller-supplied",
    ):
        ratio = int(refinement_ratio)
        method_id = str(temporal_method_id)
        if ratio <= 1:
            raise ValueError("AMR subcycling refinement ratio must exceed one.")
        if not method_id:
            raise ValueError("temporal_method_id must be non-empty.")
        self.refinement_ratio = ratio
        self.temporal_method_id = method_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "conservative-amr-subcycling",
                "refinement_ratio": ratio,
                "temporal_method_id": method_id,
            }
        )

    def advance(
        self,
        time: ArrayLike,
        coarse_state: ArrayLike,
        fine_state: ArrayLike,
        step_size: ArrayLike,
        coarse_step: Callable[[Array, Array, Array, Any], Array],
        fine_step: Callable[[Array, Array, Array, Any], Array],
        coarse_flux: Callable[[Array, Any], Array],
        fine_flux: Callable[[Array, Any], Array],
        restrict_flux: Callable[[Array], Array],
        interface_mask: ArrayLike,
        coarse_volume: ArrayLike,
        args: Any = None,
        /,
    ) -> AMRSubcycleResult:
        if not all(
            callable(value)
            for value in (coarse_step, fine_step, coarse_flux, fine_flux, restrict_flux)
        ):
            raise TypeError(
                "AMR subcycling steps, fluxes, and restriction must be callable."
            )
        time_ = jnp.asarray(time)
        dt = jnp.asarray(step_size)
        coarse = jnp.asarray(coarse_state)
        fine = jnp.asarray(fine_state)
        fine_dt = dt / self.refinement_ratio
        coarse_new = coarse_step(time_, coarse, dt, args)
        fine_flux_integral = jnp.zeros_like(jnp.asarray(coarse_flux(coarse, args)))
        fine_time = time_
        fine_new = fine
        for _ in range(self.refinement_ratio):
            fine_new = fine_step(fine_time, fine_new, fine_dt, args)
            fine_flux_integral = fine_flux_integral + fine_dt * restrict_flux(
                fine_flux(fine_new, args)
            )
            fine_time = fine_time + fine_dt
        coarse_flux_integral = dt * coarse_flux(coarse_new, args)
        register = FluxRegister(
            coarse_flux_integral,
            fine_flux_integral,
            interface_mask,
            accumulated_time=dt,
            orientation=1,
            refinement_ratio=self.refinement_ratio,
            register_id=canonical_fingerprint(
                {
                    "kind": "conservative-amr-subcycle-register",
                    "plan": self.plan_id,
                    "coarse_flux_shape": list(coarse_flux_integral.shape),
                }
            ),
        )
        coarse_refluxed = register.apply(coarse_new, coarse_volume)
        return AMRSubcycleResult(
            coarse_state=coarse_refluxed,
            fine_state=fine_new,
            flux_register=register,
            substeps=self.refinement_ratio,
            temporal_method_id=self.temporal_method_id,
        )


class FDRegridResult(StrictModule):
    decision: RefinementDecision
    child_values: Array
    regrid_trace_id: str = eqx.field(static=True)


class FDRegridPlan(StrictModule, NonTrainableState):
    """Deterministic fixed-capacity parent refinement and child-state population."""

    refinement: FixedCapacityRefinementPlan
    transfer: AMREntityTransferPlan
    child_offsets: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        refinement: FixedCapacityRefinementPlan,
        transfer: AMREntityTransferPlan,
        child_offsets: ArrayLike,
        /,
    ):
        if not isinstance(refinement, FixedCapacityRefinementPlan) or not isinstance(
            transfer, AMREntityTransferPlan
        ):
            raise TypeError("FD regrid requires refinement and entity-transfer plans.")
        offsets = np.asarray(child_offsets, dtype=np.int32)
        if offsets.shape != refinement.parent_to_children.shape + (
            len(transfer.axis_entities),
        ):
            raise ValueError("child_offsets must align with every parent-to-child route.")
        if np.any(offsets < 0) or np.any(offsets >= transfer.refinement_ratio):
            raise ValueError("child_offsets must lie inside the refinement ratio.")
        self.refinement = refinement
        self.transfer = transfer
        self.child_offsets = jnp.asarray(offsets)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fd-regrid-plan",
                "refinement": refinement.plan_id,
                "transfer": transfer.transfer_id,
                "child_offsets": array_tree_fingerprint(offsets),
            }
        )

    def apply(
        self,
        parent_state: BlockLevelState,
        child_state: BlockLevelState,
        indicators: ArrayLike,
        threshold: ArrayLike,
        /,
    ) -> FDRegridResult:
        decision = self.refinement.decide(
            parent_state.metadata,
            child_state.metadata,
            indicators,
            threshold,
        )
        parent_values = parent_state.safe_values()
        child_values = child_state.safe_values()
        mapping = self.refinement.parent_to_children
        for parent_slot in range(mapping.shape[0]):
            prolonged = self.transfer.prolong(parent_values[parent_slot])
            for child_route in range(mapping.shape[1]):
                route = mapping[parent_slot, child_route]
                valid_route = (route >= 0) & (route < child_state.plan.maximum_blocks)
                child_slot = jnp.clip(
                    route,
                    0,
                    child_state.plan.maximum_blocks - 1,
                )
                offset = self.child_offsets[parent_slot, child_route]
                spatial_rank = len(self.transfer.axis_entities)
                starts = tuple(
                    offset[axis] * child_state.plan.block_shape[axis]
                    for axis in range(spatial_rank)
                ) + (0,) * (prolonged.ndim - spatial_rank)
                sizes = child_state.plan.block_shape + prolonged.shape[spatial_rank:]
                child = jax.lax.dynamic_slice(prolonged, starts, sizes)
                requested = decision.selected_parents[parent_slot] & valid_route
                existing = child_state.metadata.active[child_slot]
                child_values = child_values.at[child_slot].set(
                    jnp.where(
                        requested & ~existing,
                        child,
                        child_values[child_slot],
                    )
                )
        active_shape = (child_state.plan.maximum_blocks,) + (1,) * (child_values.ndim - 1)
        child_values = jnp.where(
            decision.child_active.reshape(active_shape),
            child_values,
            0.0,
        )
        trace_id = canonical_fingerprint(
            {
                "kind": "fd-regrid-trace",
                "plan": self.plan_id,
                "parent_metadata": parent_state.metadata.metadata_id,
                "child_metadata": child_state.metadata.metadata_id,
            }
        )
        return FDRegridResult(
            decision=decision,
            child_values=child_values,
            regrid_trace_id=trace_id,
        )


class AMRMigrationResult(StrictModule):
    active: Array
    block_ids: Array
    parent_ids: Array
    logical_indices: Array
    values: Array
    migration_id: str = eqx.field(static=True)


class AMRMigrationPlan(StrictModule, NonTrainableState):
    """Deterministic capacity-slot migration for repartitioned AMR levels."""

    source_to_target: Array
    target_capacity: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, source_to_target: ArrayLike, target_capacity: int, /):
        routes = np.asarray(source_to_target, dtype=np.int32).reshape((-1,))
        capacity = int(target_capacity)
        active = routes >= 0
        if capacity <= 0 or np.any(routes[active] >= capacity):
            raise ValueError("AMR migration routes exceed target capacity.")
        if np.unique(routes[active]).size != np.count_nonzero(active):
            raise ValueError("AMR migration target slots must be unique.")
        self.source_to_target = jnp.asarray(routes)
        self.target_capacity = capacity
        self.plan_id = canonical_fingerprint(
            {
                "kind": "amr-migration-plan",
                "routes": array_tree_fingerprint(routes),
                "target_capacity": capacity,
            }
        )

    def migrate(self, state: BlockLevelState, /) -> AMRMigrationResult:
        if self.source_to_target.shape != (state.plan.maximum_blocks,):
            raise ValueError("AMR migration route count must match source capacity.")
        active_source = state.metadata.active & (self.source_to_target >= 0)
        block_ids = jnp.full(
            (self.target_capacity,),
            -1,
            dtype=state.metadata.block_ids.dtype,
        )
        parent_ids = jnp.full(
            (self.target_capacity,),
            -1,
            dtype=state.metadata.parent_ids.dtype,
        )
        logical = jnp.zeros(
            (self.target_capacity, state.metadata.logical_indices.shape[1]),
            dtype=state.metadata.logical_indices.dtype,
        )
        values = jnp.zeros(
            (self.target_capacity,) + state.values.shape[1:],
            dtype=state.values.dtype,
        )
        active = jnp.zeros((self.target_capacity,), dtype=bool)
        safe_values = state.safe_values()
        for source_slot in range(state.plan.maximum_blocks):
            valid = active_source[source_slot]
            target = jnp.clip(
                self.source_to_target[source_slot],
                0,
                self.target_capacity - 1,
            )
            active = active.at[target].set(active[target] | valid)
            block_ids = block_ids.at[target].set(
                jnp.where(valid, state.metadata.block_ids[source_slot], block_ids[target])
            )
            parent_ids = parent_ids.at[target].set(
                jnp.where(
                    valid, state.metadata.parent_ids[source_slot], parent_ids[target]
                )
            )
            logical = logical.at[target].set(
                jnp.where(
                    valid,
                    state.metadata.logical_indices[source_slot],
                    logical[target],
                )
            )
            values = values.at[target].set(
                jnp.where(valid, safe_values[source_slot], values[target])
            )
        return AMRMigrationResult(
            active=active,
            block_ids=block_ids,
            parent_ids=parent_ids,
            logical_indices=logical,
            values=values,
            migration_id=canonical_fingerprint(
                {
                    "kind": "amr-migration-result",
                    "plan": self.plan_id,
                    "metadata": state.metadata.metadata_id,
                }
            ),
        )


class FDAMRHierarchyPlan(StrictModule, NonTrainableState):
    """FD halo, transfer, and subcycling policies aligned to a block hierarchy."""

    hierarchy: BlockHierarchyPlan
    transfers: tuple[AMREntityTransferPlan, ...]
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        hierarchy: BlockHierarchyPlan,
        transfers: Sequence[AMREntityTransferPlan],
        /,
    ):
        transfers_ = tuple(transfers)
        if (
            not isinstance(hierarchy, BlockHierarchyPlan)
            or len(transfers_) != len(hierarchy.levels) - 1
        ):
            raise ValueError(
                "FD AMR hierarchy requires one transfer per level transition."
            )
        if any(
            transfer.refinement_ratio != hierarchy.levels[index].refinement_ratio
            for index, transfer in enumerate(transfers_)
        ):
            raise ValueError("FD AMR transfer ratios must match level plans.")
        self.hierarchy = hierarchy
        self.transfers = transfers_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fd-amr-hierarchy-plan",
                "hierarchy": hierarchy.plan_id,
                "transfers": [value.transfer_id for value in transfers_],
            }
        )

    def prepare(self, /) -> "PreparedFDAMRHierarchy":
        return PreparedFDAMRHierarchy(self)


class PreparedFDAMRHierarchy(StrictModule, NonTrainableState):
    plan: FDAMRHierarchyPlan
    halo_plans: tuple[FDAMRHaloPlan, ...]
    subcycling: tuple[ConservativeAMRSubcyclingPlan, ...]
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: FDAMRHierarchyPlan, /):
        if not isinstance(plan, FDAMRHierarchyPlan):
            raise TypeError("plan must be FDAMRHierarchyPlan.")
        halos = tuple(FDAMRHaloPlan(level) for level in plan.hierarchy.levels)
        subcycling = tuple(
            ConservativeAMRSubcyclingPlan(level.refinement_ratio)
            for level in plan.hierarchy.levels[:-1]
        )
        self.plan = plan
        self.halo_plans = halos
        self.subcycling = subcycling
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-fd-amr-hierarchy",
                "plan": plan.plan_id,
                "halos": [value.plan_id for value in halos],
                "subcycling": [value.plan_id for value in subcycling],
            }
        )

    def fill_same_level(
        self,
        state: BlockHierarchyState,
        /,
    ) -> tuple[FDAMRHaloWorkspace, ...]:
        if (
            not isinstance(state, BlockHierarchyState)
            or state.plan.plan_id != self.plan.hierarchy.plan_id
        ):
            raise ValueError("FD AMR hierarchy state does not match its plan.")
        return tuple(
            halo.fill_same_level(level)
            for halo, level in zip(self.halo_plans, state.levels, strict=True)
        )


__all__ = [
    "AMRMigrationPlan",
    "AMRMigrationResult",
    "AMRSubcycleResult",
    "FDAMRHierarchyPlan",
    "ConservativeAMRSubcyclingPlan",
    "FDRegridPlan",
    "FDRegridResult",
    "PreparedFDAMRHierarchy",
]
