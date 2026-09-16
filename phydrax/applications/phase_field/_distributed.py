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
from ...discretization import FiniteElementDiscretization
from ...discretization.fem import (
    CostAwareFiniteElementPartition,
    DistributedFiniteElementOperator,
    FiniteElementDistributedPhasePlan,
    JaxCollectiveBackend,
    lower_distributed_finite_element_phases,
    partition_cells_cost_aware,
)
from ...linalg import AbstractLinearOperator


class DistributedPhaseFieldEvidence(StrictModule):
    local_value: Array
    global_value: Array
    ownership_complete: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class DistributedPhaseFieldCheckpointManifest(StrictModule, NonTrainableState):
    topology_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)
    global_cell_ids: Array
    global_dof_ids: tuple[Array, ...]
    method_id: str = eqx.field(static=True)
    stochastic_id: str | None = eqx.field(static=True)
    active_storage_id: str | None = eqx.field(static=True)
    manifest_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: FiniteElementDiscretization,
        method_id: str,
        /,
        *,
        stochastic_id: str | None = None,
        active_storage_id: str | None = None,
    ):
        if not isinstance(discretization, FiniteElementDiscretization):
            raise TypeError("discretization must be FiniteElementDiscretization.")
        method = str(method_id)
        if not method:
            raise ValueError("Distributed phase-field manifest needs a method_id.")
        cells = np.concatenate(
            tuple(np.asarray(block.global_ids) for block in discretization.mesh.blocks)
        )
        dofs = tuple(
            np.arange(dof_map.global_dof_count, dtype=np.int64)
            for dof_map in discretization.dof_maps
        )
        self.topology_id = discretization.default_runtime.topology_id
        self.geometry_layout_id = discretization.default_runtime.geometry_layout_id
        self.global_cell_ids = jnp.asarray(cells)
        self.global_dof_ids = tuple(jnp.asarray(value) for value in dofs)
        self.method_id = method
        self.stochastic_id = stochastic_id
        self.active_storage_id = active_storage_id
        self.manifest_id = canonical_fingerprint(
            {
                "kind": "distributed-phase-field-checkpoint-manifest",
                "topology": self.topology_id,
                "geometry_layout": self.geometry_layout_id,
                "cells": array_tree_fingerprint(cells),
                "dofs": [array_tree_fingerprint(value) for value in dofs],
                "method": method,
                "stochastic": stochastic_id,
                "active_storage": active_storage_id,
            }
        )


class DistributedPhaseFieldPlan(StrictModule, NonTrainableState):
    """Exactly-once FE partition phases and real named-axis collectives."""

    partition: CostAwareFiniteElementPartition
    phases: FiniteElementDistributedPhasePlan
    collective: JaxCollectiveBackend
    axis_name: str = eqx.field(static=True)
    part_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: FiniteElementDiscretization,
        part_count: int,
        /,
        *,
        axis_name: str = "phase_field_parts",
        physics_weight: float = 1.0,
        cut_penalty: float = 0.25,
    ):
        if not isinstance(discretization, FiniteElementDiscretization):
            raise TypeError("discretization must be FiniteElementDiscretization.")
        parts = int(part_count)
        partition = partition_cells_cost_aware(
            discretization,
            parts,
            physics_weight=physics_weight,
            cut_penalty=cut_penalty,
        )
        phases = lower_distributed_finite_element_phases(discretization, partition)
        collective = JaxCollectiveBackend(axis_name)
        self.partition = partition
        self.phases = phases
        self.collective = collective
        self.axis_name = collective.axis_name
        self.part_count = parts
        self.plan_id = canonical_fingerprint(
            {
                "kind": "distributed-phase-field-plan",
                "discretization": discretization.prepared_id,
                "partition": partition.plan_id,
                "phases": phases.plan_id,
                "axis_name": collective.axis_name,
            }
        )

    def distributed_operator(
        self, local_operator: AbstractLinearOperator, /
    ) -> DistributedFiniteElementOperator:
        if not isinstance(local_operator, AbstractLinearOperator):
            raise TypeError("local_operator must be AbstractLinearOperator.")
        return DistributedFiniteElementOperator(local_operator, self.collective)

    def reference_owned_sum(
        self,
        cell_values: ArrayLike,
        /,
    ) -> Array:
        values = jnp.asarray(cell_values)
        if values.shape[0] != self.phases.partition.cell_owner.shape[0]:
            raise ValueError("Distributed cell values do not match partition cells.")
        result = jnp.zeros(values.shape[1:], dtype=values.dtype)
        for part in range(self.part_count):
            result = result + self.phases.local_contribution(part, values)
        return result

    def collective_owned_sum(
        self,
        cell_values: ArrayLike,
        /,
    ) -> DistributedPhaseFieldEvidence:
        values = jnp.asarray(cell_values)
        if values.shape[0] != self.phases.partition.cell_owner.shape[0]:
            raise ValueError("Distributed cell values do not match partition cells.")
        part = jax.lax.axis_index(self.axis_name)
        owned = self.phases.partition.cell_owner == part
        mask = owned.reshape(owned.shape + (1,) * (values.ndim - 1))
        local = jnp.sum(jnp.where(mask, values, 0.0), axis=0)
        global_value = self.collective.sum(local)
        expected = self.reference_owned_sum(values)
        scale = jnp.maximum(jnp.max(jnp.abs(expected)), 1.0)
        tolerance = 256.0 * jnp.finfo(values.dtype).eps * scale
        complete = jnp.max(jnp.abs(global_value - expected)) <= tolerance
        finite = jnp.all(jnp.isfinite(global_value))
        return DistributedPhaseFieldEvidence(
            local,
            global_value,
            complete,
            finite,
            complete & finite,
            self.plan_id,
        )


__all__ = [
    "DistributedPhaseFieldCheckpointManifest",
    "DistributedPhaseFieldEvidence",
    "DistributedPhaseFieldPlan",
]
