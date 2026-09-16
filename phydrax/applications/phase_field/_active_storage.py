#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...sparse import align_key_groups, KeyGroupPlan, KeyGroupState, KeyGroupTransition


class ActivePhaseStorageEvidence(StrictModule):
    required_per_dof: Array
    required_per_cell: Array
    dof_overflow: Array
    cell_overflow: Array
    simplex_defect: Array
    finite: Array
    successful: Array


class ActivePhaseFieldState(StrictModule):
    phase_ids: Array
    values: Array
    active: Array
    dwell: Array
    evidence: ActivePhaseStorageEvidence
    storage_id: str = eqx.field(static=True)


class ActivePhaseTransition(StrictModule):
    previous: ActivePhaseFieldState
    candidate: ActivePhaseFieldState
    cell_groups: KeyGroupState
    group_transition: KeyGroupTransition
    activated: Array
    pruned: Array
    successful: Array


class ActivePhaseStoragePlan(StrictModule, NonTrainableState):
    """Canonical fixed-capacity phase IDs and values on FE coordinates."""

    cell_dofs: Array
    dof_count: int = eqx.field(static=True)
    phase_count: int = eqx.field(static=True)
    local_capacity: int = eqx.field(static=True)
    cell_phase_capacity: int = eqx.field(static=True)
    activation_tolerance: float = eqx.field(static=True)
    pruning_tolerance: float = eqx.field(static=True)
    minimum_dwell: int = eqx.field(static=True)
    group_plan: KeyGroupPlan
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        cell_dofs: ArrayLike,
        phase_count: int,
        local_capacity: int,
        /,
        *,
        cell_phase_capacity: int | None = None,
        activation_tolerance: float = 1.0e-12,
        pruning_tolerance: float = 1.0e-14,
        minimum_dwell: int = 2,
    ):
        cells = np.asarray(cell_dofs, dtype=np.int32)
        phases = int(phase_count)
        capacity = int(local_capacity)
        cell_capacity = (
            capacity if cell_phase_capacity is None else int(cell_phase_capacity)
        )
        activation = float(activation_tolerance)
        pruning = float(pruning_tolerance)
        dwell = int(minimum_dwell)
        if (
            cells.ndim != 2
            or cells.size == 0
            or np.any(cells < 0)
            or phases < 2
            or capacity < 1
            or capacity > phases
            or cell_capacity < capacity
            or cell_capacity > cells.shape[1] * capacity
            or not np.isfinite(activation)
            or not np.isfinite(pruning)
            or activation < 0.0
            or pruning < 0.0
            or pruning > activation
            or dwell < 0
        ):
            raise ValueError("Active phase-storage configuration is invalid.")
        dof_count = int(np.max(cells)) + 1
        self.cell_dofs = jnp.asarray(cells)
        self.dof_count = dof_count
        self.phase_count = phases
        self.local_capacity = capacity
        self.cell_phase_capacity = cell_capacity
        self.activation_tolerance = activation
        self.pruning_tolerance = pruning
        self.minimum_dwell = dwell
        self.group_plan = KeyGroupPlan(
            cells.shape[1] * capacity,
            cell_capacity,
            phases - 1,
            maximum_group_size=cells.shape[1] * capacity,
            case_shape=(cells.shape[0],),
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "active-phase-storage-plan",
                "cell_dofs": array_tree_fingerprint(cells),
                "phase_count": phases,
                "local_capacity": capacity,
                "cell_phase_capacity": cell_capacity,
                "activation_tolerance": activation,
                "pruning_tolerance": pruning,
                "minimum_dwell": dwell,
            }
        )

    @staticmethod
    def _project_simplex(values: Array) -> Array:
        ordered = jnp.sort(values)[::-1]
        cumulative = jnp.cumsum(ordered) - 1.0
        indices = jnp.arange(1, values.size + 1, dtype=values.dtype)
        active = ordered - cumulative / indices > 0.0
        rho = jnp.maximum(jnp.sum(active, dtype=jnp.int32) - 1, 0)
        threshold = cumulative[rho] / (rho.astype(values.dtype) + 1.0)
        return jnp.maximum(values - threshold, 0.0)

    def from_dense(
        self,
        values: ArrayLike,
        /,
        *,
        project_simplex: bool = False,
    ) -> ActivePhaseFieldState:
        dense = jnp.asarray(values)
        if dense.shape != (self.dof_count, self.phase_count):
            raise ValueError(
                "Dense phase values must have shape (dof_count, phase_count)."
            )
        physical = jax.vmap(self._project_simplex)(dense) if project_simplex else dense
        finite = jnp.all(jnp.isfinite(physical))
        simplex_defect = jnp.max(jnp.abs(jnp.sum(physical, axis=1) - 1.0))
        valid = physical > self.activation_tolerance
        required = jnp.sum(valid, axis=1, dtype=jnp.int32)
        dof_overflow = jnp.any(required > self.local_capacity)
        phase_ids = jnp.broadcast_to(
            jnp.arange(self.phase_count, dtype=jnp.int32), physical.shape
        )
        sentinel = jnp.asarray(self.phase_count, dtype=jnp.int32)
        sort_key = jnp.where(valid, phase_ids, sentinel)
        order = jnp.argsort(sort_key, axis=1, stable=True)
        selected = order[:, : self.local_capacity]
        selected_ids = jnp.take_along_axis(phase_ids, selected, axis=1)
        selected_values = jnp.take_along_axis(physical, selected, axis=1)
        selected_active = jnp.take_along_axis(valid, selected, axis=1)
        selected_ids = jnp.where(selected_active, selected_ids, -1)
        selected_values = jnp.where(selected_active, selected_values, 0.0)
        dwell = jnp.zeros_like(selected_ids, dtype=jnp.int32)
        cell_groups = self._cell_groups(selected_ids, selected_active)
        required_cell = cell_groups.evidence.required_groups
        cell_overflow = jnp.any(required_cell > self.cell_phase_capacity)
        successful = (
            finite
            & (simplex_defect <= 1.0e-10)
            & ~dof_overflow
            & ~cell_overflow
            & jnp.all(required >= 1)
            & jnp.all(cell_groups.evidence.successful)
        )
        evidence = ActivePhaseStorageEvidence(
            required,
            required_cell,
            dof_overflow,
            cell_overflow,
            simplex_defect,
            finite,
            successful,
        )
        return ActivePhaseFieldState(
            selected_ids,
            selected_values,
            selected_active,
            dwell,
            evidence,
            self.plan_id,
        )

    def dense(self, state: ActivePhaseFieldState, /) -> Array:
        self._validate(state)
        result = jnp.zeros((self.dof_count, self.phase_count), dtype=state.values.dtype)
        safe = jnp.where(state.active, state.phase_ids, 0)
        dofs = jnp.broadcast_to(
            jnp.arange(self.dof_count, dtype=jnp.int32)[:, None], safe.shape
        )
        return result.at[dofs, safe].add(jnp.where(state.active, state.values, 0.0))

    def _cell_groups(self, phase_ids: Array, active: Array) -> KeyGroupState:
        gathered_ids = phase_ids[self.cell_dofs]
        gathered_active = active[self.cell_dofs]
        stable = self.cell_dofs[..., None] * self.local_capacity + jnp.arange(
            self.local_capacity, dtype=jnp.int32
        )
        stable = jnp.broadcast_to(stable, gathered_ids.shape)
        cell_count = self.cell_dofs.shape[0]
        return self.group_plan.build(
            gathered_ids.reshape((cell_count, -1)),
            gathered_active.reshape((cell_count, -1)),
            stable_ids=stable.reshape((cell_count, -1)),
        )

    def cell_groups(self, state: ActivePhaseFieldState, /) -> KeyGroupState:
        self._validate(state)
        return self._cell_groups(state.phase_ids, state.active)

    def transition(
        self,
        previous: ActivePhaseFieldState,
        dense_candidate: ArrayLike,
        /,
    ) -> ActivePhaseTransition:
        self._validate(previous)
        candidate = self.from_dense(dense_candidate)
        previous_groups = self.cell_groups(previous)
        candidate_groups = self.cell_groups(candidate)
        aligned = align_key_groups(previous_groups, candidate_groups)
        previous_dense = self.dense(previous)
        candidate_dense = self.dense(candidate)
        activated = (candidate_dense > self.activation_tolerance) & ~(
            previous_dense > self.activation_tolerance
        )
        pruned = (previous_dense > self.pruning_tolerance) & ~(
            candidate_dense > self.pruning_tolerance
        )
        successful = (
            previous.evidence.successful
            & candidate.evidence.successful
            & jnp.all(aligned.successful)
        )
        return ActivePhaseTransition(
            previous,
            candidate,
            candidate_groups,
            aligned,
            activated,
            pruned,
            successful,
        )

    def _validate(self, state: ActivePhaseFieldState, /) -> None:
        if not isinstance(state, ActivePhaseFieldState):
            raise TypeError("state must be ActivePhaseFieldState.")
        shape = (self.dof_count, self.local_capacity)
        if (
            state.phase_ids.shape != shape
            or state.values.shape != shape
            or state.active.shape != shape
            or state.dwell.shape != shape
            or state.storage_id != self.plan_id
        ):
            raise ValueError("Active phase state is incompatible with its plan.")


__all__ = [
    "ActivePhaseFieldState",
    "ActivePhaseStorageEvidence",
    "ActivePhaseStoragePlan",
    "ActivePhaseTransition",
]
