#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike, PyTree

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import AbstractPreconditioner, PreconditionerProperties
from ._generic import FiniteElementDiscretization


class FiniteElementPatchPlan(StrictModule, NonTrainableState):
    gathers: Array
    valid: Array
    partition_weights: Array
    global_size: int = eqx.field(static=True)
    patch_id: str = eqx.field(static=True)

    def __init__(
        self,
        gathers: ArrayLike,
        valid: ArrayLike,
        partition_weights: ArrayLike,
        global_size: int,
        /,
    ):
        routes = jnp.asarray(gathers, dtype=jnp.int32)
        valid_ = jnp.asarray(valid, dtype=bool)
        weights = jnp.asarray(partition_weights)
        size = int(global_size)
        if (
            routes.ndim != 2
            or valid_.shape != routes.shape
            or weights.shape != routes.shape
        ):
            raise ValueError(
                "Patch gather, validity, and weight layouts are incompatible."
            )
        if size <= 0 or bool(jnp.any(valid_ & ((routes < 0) | (routes >= size)))):
            raise ValueError("Patch routes/global size are invalid.")
        if bool(jnp.any(jnp.where(valid_, weights <= 0.0, weights != 0.0))):
            raise ValueError(
                "Patch partition weights must be positive only on valid routes."
            )
        self.gathers = routes
        self.valid = valid_
        self.partition_weights = weights
        self.global_size = size
        self.patch_id = canonical_fingerprint(
            {
                "kind": "finite-element-one-ring-patch-plan",
                "patches": int(routes.shape[0]),
                "width": int(routes.shape[1]),
                "global_size": size,
            }
        )


def one_ring_patch_plan(
    discretization: FiniteElementDiscretization,
    field_name: str,
    /,
) -> FiniteElementPatchPlan:
    if not isinstance(discretization, FiniteElementDiscretization):
        raise TypeError("discretization must be FiniteElementDiscretization.")
    field_index = discretization._field_index(field_name)
    dof_map = discretization.dof_maps[field_index]
    cell_count = sum(block.cell_count for block in discretization.mesh.blocks)
    block_by_cell = []
    local_by_cell = []
    for block_index, block in enumerate(discretization.mesh.blocks):
        block_by_cell.extend((block_index,) * block.cell_count)
        local_by_cell.extend(range(block.cell_count))
    neighbours = [set((cell,)) for cell in range(cell_count)]
    for owner, neighbour in zip(
        np.asarray(discretization.interior_facet_domain.owner_cells),
        np.asarray(discretization.interior_facet_domain.neighbour_cells),
        strict=True,
    ):
        neighbours[int(owner)].add(int(neighbour))
        neighbours[int(neighbour)].add(int(owner))
    patches = []
    for cells in neighbours:
        dofs = set()
        for cell in sorted(cells):
            block = block_by_cell[cell]
            local = local_by_cell[cell]
            dofs.update(np.asarray(dof_map.cell_dofs[block][local]).tolist())
        patches.append(tuple(sorted(dofs)))
    width = max(len(patch) for patch in patches)
    routes = np.zeros((cell_count, width), dtype=np.int32)
    valid = np.zeros((cell_count, width), dtype=bool)
    for patch, values in enumerate(patches):
        routes[patch, : len(values)] = values
        valid[patch, : len(values)] = True
    coverage = np.zeros((dof_map.global_dof_count,), dtype=np.int32)
    np.add.at(coverage, routes[valid], 1)
    weights = np.zeros_like(
        routes, dtype=np.asarray(discretization.mesh.coordinates).dtype
    )
    weights[valid] = 1.0 / coverage[routes[valid]]
    return FiniteElementPatchPlan(
        routes,
        valid,
        weights,
        dof_map.global_dof_count,
    )


class FiniteElementPatchPreconditioner(AbstractPreconditioner):
    plan: FiniteElementPatchPlan
    local_inverse: Array

    def __init__(
        self,
        plan: FiniteElementPatchPlan,
        local_inverse: ArrayLike,
        space,
        /,
    ):
        if not isinstance(plan, FiniteElementPatchPlan):
            raise TypeError("plan must be FiniteElementPatchPlan.")
        inverse = jnp.asarray(local_inverse)
        if inverse.shape != plan.gathers.shape + (plan.gathers.shape[1],):
            raise ValueError("Patch inverse matrices have an incompatible shape.")
        if space.size != plan.global_size:
            raise ValueError("Patch preconditioner space does not match patch plan.")
        self.plan = plan
        self.local_inverse = inverse
        self.space = space
        self.properties = PreconditionerProperties(
            linear=True,
            stationary=True,
            evidence={"linear": "construction", "stationary": "construction"},
        )
        self.preconditioner_id = canonical_fingerprint(
            {
                "kind": "finite-element-one-ring-preconditioner",
                "patch": plan.patch_id,
                "space": space.space_id,
                "inverse_shape": list(inverse.shape),
            }
        )

    def apply(
        self,
        residual: PyTree,
        /,
        *,
        iteration: ArrayLike | None = None,
    ):
        coordinates = self.space.flatten(self.space.validate(residual))
        safe_routes = jnp.maximum(self.plan.gathers, 0)
        local = coordinates[safe_routes]
        local = jnp.where(self.plan.valid, local, 0.0)
        correction = oe.contract("pij,pj->pi", self.local_inverse, local)
        correction = jnp.where(
            self.plan.valid,
            correction * self.plan.partition_weights,
            0.0,
        )
        assembled = (
            jnp.zeros((self.plan.global_size,), dtype=correction.dtype)
            .at[safe_routes]
            .add(correction)
        )
        return self.space.unflatten(assembled)


__all__ = [
    "FiniteElementPatchPlan",
    "FiniteElementPatchPreconditioner",
    "one_ring_patch_plan",
]
