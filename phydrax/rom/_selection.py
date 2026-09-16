#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._production import ROMCostEstimate, ROMResourcePolicy


class SelectedEntitySet(StrictModule, NonTrainableState):
    entity_indices: Array
    closure_dof_indices: Array
    entity_kind: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    ownership_id: str = eqx.field(static=True)
    entity_set_id: str = eqx.field(static=True)

    def __init__(
        self,
        entity_indices: ArrayLike,
        closure_dof_indices: ArrayLike,
        /,
        *,
        entity_kind: str,
        topology_id: str,
        support_id: str,
        geometry_id: str,
        ownership_id: str,
    ):
        entities = np.asarray(entity_indices, dtype=np.int64)
        closure = np.asarray(closure_dof_indices, dtype=np.int64)
        identifiers = tuple(
            str(value)
            for value in (entity_kind, topology_id, support_id, geometry_id, ownership_id)
        )
        if (
            entities.ndim != 1
            or closure.ndim != 1
            or entities.size == 0
            or closure.size == 0
        ):
            raise ValueError(
                "Selected entities and closure DOFs must be non-empty vectors."
            )
        if np.any(entities < 0) or np.any(closure < 0):
            raise ValueError("Selected indices must be nonnegative.")
        if (
            np.unique(entities).size != entities.size
            or np.unique(closure).size != closure.size
        ):
            raise ValueError("Selected entities and closure DOFs must be unique.")
        if any(not value for value in identifiers):
            raise ValueError("Selected-entity identities must be non-empty.")
        self.entity_indices = jnp.asarray(entities, dtype=jnp.int32)
        self.closure_dof_indices = jnp.asarray(closure, dtype=jnp.int32)
        self.entity_kind = identifiers[0]
        self.topology_id = identifiers[1]
        self.support_id = identifiers[2]
        self.geometry_id = identifiers[3]
        self.ownership_id = identifiers[4]
        self.entity_set_id = canonical_fingerprint(
            {
                "kind": "selected-entity-set",
                "entity_kind": identifiers[0],
                "topology": identifiers[1],
                "support": identifiers[2],
                "geometry": identifiers[3],
                "ownership": identifiers[4],
                "content": array_tree_fingerprint(
                    {"entities": entities, "closure": closure}
                )["sha256"],
            }
        )


class SelectedEvaluationPlan(StrictModule, NonTrainableState):
    entities: SelectedEntitySet
    residual_output_size: int = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    cost: ROMCostEstimate

    def __init__(
        self,
        entities: SelectedEntitySet,
        /,
        *,
        residual_output_size: int,
        provider_id: str,
        resource_policy: ROMResourcePolicy,
    ):
        if not isinstance(entities, SelectedEntitySet):
            raise TypeError("entities must be a SelectedEntitySet.")
        if not isinstance(resource_policy, ROMResourcePolicy):
            raise TypeError("resource_policy must be a ROMResourcePolicy.")
        output = int(residual_output_size)
        provider = str(provider_id)
        workspace = 8 * (
            int(entities.closure_dof_indices.size)
            + int(entities.entity_indices.size)
            + output
        )
        if output <= 0 or not provider:
            raise ValueError(
                "Selected residual output size and provider ID must be valid."
            )
        if not resource_policy.admit(
            full_dimension=max(int(entities.closure_dof_indices.size), 1),
            reduced_dimension=max(output, 1),
            samples=int(entities.entity_indices.size),
            workspace_bytes=max(workspace, 1),
        ):
            raise ValueError("Selected evaluation exceeds the ROM resource policy.")
        self.entities = entities
        self.residual_output_size = output
        self.provider_id = provider
        self.cost = ROMCostEstimate(
            online_operations=workspace // 8,
            local_memory_bytes=workspace,
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "selected-evaluation-plan",
                "entities": entities.entity_set_id,
                "provider": provider,
                "output_size": output,
                "resource_policy": resource_policy.policy_id,
            }
        )

    def gather_state(self, full_state: ArrayLike, /) -> Array:
        state = jnp.asarray(full_state)
        if state.ndim != 1:
            raise ValueError("Selected evaluation requires one flattened full state.")
        if int(jnp.max(self.entities.closure_dof_indices)) >= state.size:
            raise ValueError("Closure DOF index exceeds the full-state size.")
        return state[self.entities.closure_dof_indices]

    def evaluate(
        self,
        provider: Callable[[Array, Array, object], Array],
        full_state: ArrayLike,
        inputs: object = None,
        /,
    ) -> Array:
        compact = self.gather_state(full_state)
        value = jnp.asarray(provider(compact, self.entities.entity_indices, inputs))
        if value.shape != (self.residual_output_size,):
            raise ValueError("Selected provider output shape does not match its plan.")
        return value


__all__ = ["SelectedEntitySet", "SelectedEvaluationPlan"]
