#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.contact._kinematics import ContactKinematicsEpoch
from ...linalg import SmallLinearSolvePlan, solve_small_linear


class ContactGraphPlan(StrictModule, NonTrainableState):
    component_ids: Array
    component_count: int = eqx.field(static=True)
    route_count: int = eqx.field(static=True)
    graph_id: str = eqx.field(static=True)

    @classmethod
    def from_kinematics(cls, kinematics: ContactKinematicsEpoch, /) -> ContactGraphPlan:
        if not isinstance(kinematics, ContactKinematicsEpoch):
            raise TypeError("kinematics must be ContactKinematicsEpoch.")
        endpoints = []
        valid_values = []
        for batch in kinematics.batches:
            endpoints.extend(np.asarray(batch.vertex_indices).tolist())
            valid_values.extend(np.asarray(batch.valid, dtype=bool).tolist())
        route_count = len(endpoints)
        parent = list(range(route_count))

        def root(index):
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left, right):
            left_root = root(left)
            right_root = root(right)
            if left_root != right_root:
                parent[max(left_root, right_root)] = min(left_root, right_root)

        vertex_routes: dict[int, list[int]] = {}
        for route, (indices, valid) in enumerate(
            zip(endpoints, valid_values, strict=True)
        ):
            if not valid:
                continue
            for vertex in indices:
                if vertex >= 0:
                    vertex_routes.setdefault(int(vertex), []).append(route)
        for routes in vertex_routes.values():
            for route in routes[1:]:
                union(routes[0], route)
        roots = [root(index) for index in range(route_count)]
        unique = {value: index for index, value in enumerate(sorted(set(roots)))}
        components = np.asarray([unique[value] for value in roots], dtype=np.int32)
        return cls(
            jnp.asarray(components),
            len(unique),
            route_count,
            canonical_fingerprint(
                {
                    "kind": "contact-graph-plan",
                    "components": array_tree_fingerprint(components),
                    "kinematics": kinematics.epoch_id,
                }
            ),
        )


class ContactBlockPreconditionerPlan(StrictModule, NonTrainableState):
    solve_plan: SmallLinearSolvePlan
    regularization: float = eqx.field(static=True)
    coarse_weight: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        local_dimension: int,
        /,
        *,
        regularization: float = 1.0e-10,
        coarse_weight: float = 0.5,
    ):
        regularization_ = float(regularization)
        coarse = float(coarse_weight)
        if regularization_ <= 0.0 or not np.isfinite(regularization_):
            raise ValueError("Preconditioner regularization must be positive.")
        if not 0.0 <= coarse <= 1.0:
            raise ValueError("coarse_weight must lie in [0, 1].")
        solve_plan = SmallLinearSolvePlan(local_dimension)
        self.solve_plan = solve_plan
        self.regularization = regularization_
        self.coarse_weight = coarse
        self.plan_id = canonical_fingerprint(
            {
                "kind": "contact-block-preconditioner-plan",
                "solve": solve_plan.plan_id,
                "regularization": regularization_.hex(),
                "coarse_weight": coarse.hex(),
            }
        )


class ContactPreconditionerResult(StrictModule):
    value: Array
    local_success: Array
    component_count: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


def apply_contact_block_preconditioner(
    plan: ContactBlockPreconditionerPlan,
    local_blocks: ArrayLike,
    right_hand_side: ArrayLike,
    /,
    *,
    graph: ContactGraphPlan | None = None,
) -> ContactPreconditionerResult:
    if not isinstance(plan, ContactBlockPreconditionerPlan):
        raise TypeError("plan must be ContactBlockPreconditionerPlan.")
    blocks = jnp.asarray(local_blocks)
    right = jnp.asarray(right_hand_side, dtype=blocks.dtype)
    dimension = plan.solve_plan.dimension
    if blocks.ndim != 3 or blocks.shape[-2:] != (dimension, dimension):
        raise ValueError("Contact preconditioner blocks have invalid shape.")
    if right.shape != blocks.shape[:-1]:
        raise ValueError("Contact preconditioner right-hand side has invalid shape.")
    identity = jnp.eye(dimension, dtype=blocks.dtype)
    regularized = blocks + plan.regularization * identity[None, :, :]
    fine = solve_small_linear(plan.solve_plan, regularized, right)
    value = fine.value
    component_count = 0
    if graph is not None:
        if graph.route_count != blocks.shape[0]:
            raise ValueError("Contact graph and preconditioner route counts differ.")
        component_count = graph.component_count
        diagonal = jnp.maximum(
            jnp.mean(jnp.diagonal(regularized, axis1=-2, axis2=-1), axis=-1),
            jnp.finfo(blocks.dtype).eps,
        )
        for component in range(graph.component_count):
            mask = graph.component_ids == component
            count = jnp.maximum(jnp.sum(mask), 1)
            coarse_rhs = jnp.sum(jnp.where(mask[:, None], right, 0.0), axis=0) / count
            coarse_diagonal = jnp.sum(jnp.where(mask, diagonal, 0.0)) / count
            correction = coarse_rhs / coarse_diagonal
            value = value + plan.coarse_weight * jnp.where(
                mask[:, None], correction[None, :], 0.0
            )
    finite = jnp.all(jnp.isfinite(value))
    successful = finite & jnp.all(fine.successful)
    return ContactPreconditionerResult(
        value,
        fine.successful,
        jnp.asarray(component_count, dtype=jnp.int32),
        finite,
        successful,
        plan.plan_id,
    )


__all__ = [
    "ContactBlockPreconditionerPlan",
    "ContactGraphPlan",
    "ContactPreconditionerResult",
    "apply_contact_block_preconditioner",
]
