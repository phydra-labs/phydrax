#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.mpm import (
    BlockSparseMPMNodalStoragePlan,
    KWayMPMContactPlan,
    MPMActiveBlockState,
    MPMContactGraph,
)
from ..discretization.splatting import ParticleGridSplatState, PreparedParticleGridSplat


class MPMImplicitUnknownLayout(StrictModule, NonTrainableState):
    field_count: int = eqx.field(static=True)
    node_capacity: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    free_mask: Array
    essential_mask: Array
    contact_multiplier_capacity: int = eqx.field(static=True)
    rigid_dof_capacity: int = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        free_mask: ArrayLike,
        essential_mask: ArrayLike,
        /,
        *,
        contact_multiplier_capacity: int = 0,
        rigid_dof_capacity: int = 0,
    ):
        free = np.asarray(free_mask, dtype=bool)
        essential = np.asarray(essential_mask, dtype=bool)
        if free.shape != essential.shape or free.ndim < 3:
            raise ValueError("Implicit MPM free/essential masks must share K,G,d shape.")
        if np.any(free & essential):
            raise ValueError("Implicit MPM DOFs cannot be free and essential together.")
        contact = int(contact_multiplier_capacity)
        rigid = int(rigid_dof_capacity)
        if contact < 0 or rigid < 0:
            raise ValueError("Implicit multiplier/rigid capacities must be nonnegative.")
        self.field_count = int(free.shape[0])
        self.node_capacity = int(np.prod(free.shape[1:-1]))
        self.dimension = int(free.shape[-1])
        self.free_mask = jnp.asarray(free)
        self.essential_mask = jnp.asarray(essential)
        self.contact_multiplier_capacity = contact
        self.rigid_dof_capacity = rigid
        self.layout_id = canonical_fingerprint(
            {
                "kind": "mpm-implicit-unknown-layout",
                "shape": free.shape,
                "free_count": int(np.sum(free)),
                "essential_count": int(np.sum(essential)),
                "contact_multiplier_capacity": contact,
                "rigid_dof_capacity": rigid,
            }
        )

    @property
    def velocity_shape(self):
        return self.free_mask.shape

    @property
    def variable_count(self):
        return (
            int(np.prod(self.velocity_shape))
            + self.contact_multiplier_capacity
            + self.rigid_dof_capacity
        )

    def pack(self, velocity, contact_multipliers=None, rigid_dofs=None, /):
        velocity_ = jnp.asarray(velocity)
        if velocity_.shape != self.velocity_shape:
            raise ValueError("Implicit MPM velocity shape changed.")
        contact = (
            jnp.zeros((self.contact_multiplier_capacity,), dtype=velocity_.dtype)
            if contact_multipliers is None
            else jnp.asarray(contact_multipliers, dtype=velocity_.dtype)
        )
        rigid = (
            jnp.zeros((self.rigid_dof_capacity,), dtype=velocity_.dtype)
            if rigid_dofs is None
            else jnp.asarray(rigid_dofs, dtype=velocity_.dtype)
        )
        if contact.shape != (self.contact_multiplier_capacity,) or rigid.shape != (
            self.rigid_dof_capacity,
        ):
            raise ValueError("Implicit multiplier/rigid DOF capacity changed.")
        return jnp.concatenate((velocity_.reshape(-1), contact, rigid))

    def unpack(self, state, /):
        value = jnp.asarray(state)
        if value.shape != (self.variable_count,):
            raise ValueError("Implicit MPM packed unknown size changed.")
        velocity_size = int(np.prod(self.velocity_shape))
        velocity = value[:velocity_size].reshape(self.velocity_shape)
        contact_end = velocity_size + self.contact_multiplier_capacity
        return velocity, value[velocity_size:contact_end], value[contact_end:]


class MPMImplicitTopologyPlan(StrictModule, NonTrainableState):
    layout: MPMImplicitUnknownLayout
    route_digest: int = eqx.field(static=True)
    block_digest: int = eqx.field(static=True)
    field_digest: int = eqx.field(static=True)
    contact_digest: int = eqx.field(static=True)
    material_branch_digest: int = eqx.field(static=True)
    topology_generation: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        layout: MPMImplicitUnknownLayout,
        /,
        *,
        route_digest: int,
        block_digest: int,
        field_digest: int,
        contact_digest: int,
        material_branch_digest: int,
        topology_generation: int,
    ):
        if not isinstance(layout, MPMImplicitUnknownLayout):
            raise TypeError("layout must be MPMImplicitUnknownLayout.")
        self.layout = layout
        self.route_digest = int(route_digest)
        self.block_digest = int(block_digest)
        self.field_digest = int(field_digest)
        self.contact_digest = int(contact_digest)
        self.material_branch_digest = int(material_branch_digest)
        self.topology_generation = int(topology_generation)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mpm-implicit-topology",
                "layout": layout.layout_id,
                "route_digest": self.route_digest,
                "block_digest": self.block_digest,
                "field_digest": self.field_digest,
                "contact_digest": self.contact_digest,
                "material_branch_digest": self.material_branch_digest,
                "topology_generation": self.topology_generation,
            }
        )


class MPMRouteSupersetState(StrictModule):
    base_routes: ParticleGridSplatState
    minimum_weight_margin: Array
    minimum_domain_margin: Array
    topology_generation: Array
    superset_id: str = eqx.field(static=True)


class MPMMovingDomainDerivative(StrictModule):
    primal_weights: Array
    primal_gradients: Array
    primal_offsets: Array
    weight_jvp: Array
    gradient_jvp: Array
    offset_jvp: Array
    position_transpose: Array
    deformation_transpose: Array
    input_transpose: Any
    route_topology_stable: Array
    successful: Array


class MPMRouteSupersetPlan(StrictModule, NonTrainableState):
    prepared: PreparedParticleGridSplat
    minimum_margin: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self, prepared: PreparedParticleGridSplat, /, *, minimum_margin: float = 1.0e-8
    ):
        if not isinstance(prepared, PreparedParticleGridSplat):
            raise TypeError("prepared must be PreparedParticleGridSplat.")
        margin = float(minimum_margin)
        if not np.isfinite(margin) or margin <= 0.0:
            raise ValueError("Route superset margin must be finite and positive.")
        self.prepared = prepared
        self.minimum_margin = margin
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mpm-route-superset",
                "prepared": prepared.prepared_id,
                "minimum_margin": margin,
            }
        )

    def build(self, position, assignment_input=None, /, *, topology_generation=0):
        routes = self.prepared.build(position, assignment_input=assignment_input)
        margin = jnp.min(
            jnp.where(routes.stencil.valid, jnp.abs(routes.stencil.weights), jnp.inf)
        )
        return MPMRouteSupersetState(
            routes,
            margin,
            routes.minimum_domain_margin,
            jnp.asarray(topology_generation, dtype=jnp.int32),
            canonical_fingerprint(
                {
                    "kind": "mpm-route-superset-state",
                    "plan": self.plan_id,
                    "topology_generation": int(topology_generation),
                    "indices_shape": routes.stencil.indices.shape,
                }
            ),
        )

    def linearize(
        self,
        state: MPMRouteSupersetState,
        position,
        deformation,
        assignment_input,
        position_direction,
        deformation_direction,
        input_direction,
        cotangents,
        /,
    ):
        assignment = self.prepared.plan.assignment

        def floating_outputs(current_position, current_deformation, current_input):
            updated = assignment.update_input(
                current_position, current_deformation, current_input
            )
            routes = self.prepared.build(current_position, assignment_input=updated)
            return (
                routes.stencil.weights,
                routes.weight_gradients,
                routes.route_offsets,
            )

        primal, tangent = jax.jvp(
            floating_outputs,
            (position, deformation, assignment_input),
            (position_direction, deformation_direction, input_direction),
        )
        _, pullback = jax.vjp(floating_outputs, position, deformation, assignment_input)
        transpose = pullback(cotangents)
        candidate_input = assignment.update_input(position, deformation, assignment_input)
        candidate = self.prepared.build(position, assignment_input=candidate_input)
        stable = jnp.array_equal(
            candidate.stencil.indices, state.base_routes.stencil.indices
        ) & jnp.array_equal(candidate.stencil.valid, state.base_routes.stencil.valid)
        margin = jnp.minimum(state.minimum_weight_margin, state.minimum_domain_margin)
        successful = stable & (margin >= self.minimum_margin) & candidate.successful
        return MPMMovingDomainDerivative(
            primal[0],
            primal[1],
            primal[2],
            tangent[0],
            tangent[1],
            tangent[2],
            transpose[0],
            transpose[1],
            transpose[2],
            stable,
            successful,
        )


class MPMCompactOperatorResult(StrictModule):
    residual: Array
    jvp: Array
    transpose: Array
    dense_compact_residual_defect: Array
    dense_compact_jvp_defect: Array
    dense_compact_transpose_defect: Array
    successful: Array


class MPMCompactImplicitOperator(StrictModule, NonTrainableState):
    storage: BlockSparseMPMNodalStoragePlan
    active: MPMActiveBlockState
    operator_id: str = eqx.field(static=True)

    def __init__(
        self,
        storage: BlockSparseMPMNodalStoragePlan,
        active: MPMActiveBlockState,
        /,
    ):
        if not isinstance(storage, BlockSparseMPMNodalStoragePlan):
            raise TypeError("storage must be BlockSparseMPMNodalStoragePlan.")
        if not isinstance(active, MPMActiveBlockState) or not bool(active.successful):
            raise ValueError("Compact implicit operator requires successful block state.")
        self.storage = storage
        self.active = active
        self.operator_id = canonical_fingerprint(
            {
                "kind": "mpm-compact-implicit-operator",
                "storage": storage.storage_id,
                "active_count": int(active.active_block_count),
            }
        )

    def apply(
        self,
        dense_operator: Callable[[Array], Array],
        compact_state: ArrayLike,
        compact_direction: ArrayLike,
        compact_cotangent: ArrayLike,
        /,
    ):
        if not callable(dense_operator):
            raise TypeError("dense_operator must be callable.")
        state = jnp.asarray(compact_state)
        direction = jnp.asarray(compact_direction, dtype=state.dtype)
        cotangent = jnp.asarray(compact_cotangent, dtype=state.dtype)

        def compact_operator(value):
            dense = self.storage.unpack(value, self.active)
            return self.storage.pack(dense_operator(dense), self.active)

        residual, jvp = jax.jvp(compact_operator, (state,), (direction,))
        _, pullback = jax.vjp(compact_operator, state)
        transpose = pullback(cotangent)[0]
        dense_state = self.storage.unpack(state, self.active)
        dense_direction = self.storage.unpack(direction, self.active)
        dense_cotangent = self.storage.unpack(cotangent, self.active)
        dense_residual = dense_operator(dense_state)
        _, dense_jvp = jax.jvp(dense_operator, (dense_state,), (dense_direction,))
        _, dense_pullback = jax.vjp(dense_operator, dense_state)
        dense_transpose = dense_pullback(dense_cotangent)[0]
        packed_residual = self.storage.pack(dense_residual, self.active)
        packed_jvp = self.storage.pack(dense_jvp, self.active)
        packed_transpose = self.storage.pack(dense_transpose, self.active)
        residual_defect = jnp.linalg.norm(residual - packed_residual)
        jvp_defect = jnp.linalg.norm(jvp - packed_jvp)
        transpose_defect = jnp.linalg.norm(transpose - packed_transpose)
        successful = (
            jnp.all(jnp.isfinite(residual))
            & jnp.all(jnp.isfinite(jvp))
            & jnp.all(jnp.isfinite(transpose))
            & (residual_defect <= 1.0e-10)
            & (jvp_defect <= 1.0e-10)
            & (transpose_defect <= 1.0e-10)
        )
        return MPMCompactOperatorResult(
            residual,
            jvp,
            transpose,
            residual_defect,
            jvp_defect,
            transpose_defect,
            successful,
        )


class MPMBlockJacobiPreconditioner(StrictModule, NonTrainableState):
    diagonal: Array
    minimum_diagonal: float = eqx.field(static=True)

    def __init__(self, diagonal: ArrayLike, /, *, minimum_diagonal=1.0e-12):
        value = jnp.asarray(diagonal)
        minimum = float(minimum_diagonal)
        if value.ndim < 1 or minimum <= 0.0:
            raise ValueError("Block-Jacobi diagonal/preconditioner tolerance invalid.")
        self.diagonal = value
        self.minimum_diagonal = minimum

    def apply(self, value: ArrayLike, /):
        array = jnp.asarray(value)
        return array / jnp.where(
            jnp.abs(self.diagonal) >= self.minimum_diagonal,
            self.diagonal,
            jnp.copysign(self.minimum_diagonal, self.diagonal + 1.0e-30),
        )


class MPMTwoLevelMultigrid(StrictModule, NonTrainableState):
    restriction: Callable = eqx.field(static=True)
    prolongation: Callable = eqx.field(static=True)
    coarse_solve: Callable = eqx.field(static=True)
    smoother: MPMBlockJacobiPreconditioner

    def __init__(self, restriction, prolongation, coarse_solve, smoother, /):
        if not all(
            callable(value) for value in (restriction, prolongation, coarse_solve)
        ):
            raise TypeError("Multigrid transfer/coarse solve must be callable.")
        if not isinstance(smoother, MPMBlockJacobiPreconditioner):
            raise TypeError("smoother must be MPMBlockJacobiPreconditioner.")
        self.restriction = restriction
        self.prolongation = prolongation
        self.coarse_solve = coarse_solve
        self.smoother = smoother

    def apply(self, residual: ArrayLike, /):
        value = jnp.asarray(residual)
        pre = self.smoother.apply(value)
        coarse = self.restriction(value)
        correction = self.prolongation(self.coarse_solve(coarse))
        return pre + correction


class MPMImplicitContactLinearization(StrictModule):
    residual: Array
    jvp: Array
    transpose: Array
    successful: Array


def linearize_kway_contact(
    plan: KWayMPMContactPlan,
    mass: ArrayLike,
    velocity: ArrayLike,
    graph: MPMContactGraph,
    step_size: ArrayLike,
    direction: ArrayLike,
    cotangent: ArrayLike,
    /,
    *,
    epsilon: float = 1.0e-6,
):
    velocity_ = jnp.asarray(velocity)
    direction_ = jnp.asarray(direction, dtype=velocity_.dtype)
    cotangent_ = jnp.asarray(cotangent, dtype=velocity_.dtype)
    epsilon_ = jnp.asarray(epsilon, dtype=velocity_.dtype)
    size = velocity_.size

    def solved(flattened):
        current = flattened.reshape(velocity_.shape)
        return plan.solve(mass, current, graph, step_size).velocity.reshape((-1,))

    flat = velocity_.reshape((-1,))
    basis = jnp.eye(size, dtype=velocity_.dtype)
    columns = jax.vmap(
        lambda value: (
            (solved(flat + epsilon_ * value) - solved(flat - epsilon_ * value))
            / (2.0 * epsilon_)
        )
    )(basis)
    jacobian = columns.T
    contact = plan.solve(mass, velocity_, graph, step_size)
    primal = contact.velocity
    jvp = (jacobian @ direction_.reshape((-1,))).reshape(velocity_.shape)
    transpose = (jacobian.T @ cotangent_.reshape((-1,))).reshape(velocity_.shape)
    successful = (
        contact.successful
        & jnp.all(jnp.isfinite(jacobian))
        & jnp.all(jnp.isfinite(jvp))
        & jnp.all(jnp.isfinite(transpose))
    )
    return MPMImplicitContactLinearization(primal, jvp, transpose, successful)


class MPMSparseContactOperator(StrictModule, NonTrainableState):
    storage: BlockSparseMPMNodalStoragePlan
    active: MPMActiveBlockState
    contact: KWayMPMContactPlan

    def __init__(self, storage, active, contact, /):
        if not isinstance(storage, BlockSparseMPMNodalStoragePlan):
            raise TypeError("storage must be BlockSparseMPMNodalStoragePlan.")
        if not isinstance(active, MPMActiveBlockState):
            raise TypeError("active must be MPMActiveBlockState.")
        if not isinstance(contact, KWayMPMContactPlan):
            raise TypeError("contact must be KWayMPMContactPlan.")
        self.storage = storage
        self.active = active
        self.contact = contact

    def apply(
        self,
        compact_mass,
        compact_velocity,
        compact_mass_gradient,
        step_size,
        /,
    ):
        mass = jnp.stack(
            tuple(
                self.storage.unpack(value, self.active)
                for value in jnp.asarray(compact_mass)
            )
        )
        velocity = jnp.stack(
            tuple(
                self.storage.unpack(value, self.active)
                for value in jnp.asarray(compact_velocity)
            )
        )
        gradient = jnp.stack(
            tuple(
                self.storage.unpack(value, self.active)
                for value in jnp.asarray(compact_mass_gradient)
            )
        )
        graph = self.contact.build_graph(mass, gradient)
        result = self.contact.solve(mass, velocity, graph, step_size)
        compact = jnp.stack(
            tuple(
                self.storage.pack(result.velocity[field], self.active)
                for field in range(self.contact.field_count)
            )
        )
        return compact, result


class MPMSparsePhaseFieldOperator(StrictModule, NonTrainableState):
    storage: BlockSparseMPMNodalStoragePlan
    active: MPMActiveBlockState
    spacing: tuple[float, ...] = eqx.field(static=True)
    periodic: tuple[bool, ...] = eqx.field(static=True)

    def __init__(self, storage, active, spacing, periodic, /):
        if not isinstance(storage, BlockSparseMPMNodalStoragePlan):
            raise TypeError("storage must be BlockSparseMPMNodalStoragePlan.")
        self.storage = storage
        self.active = active
        self.spacing = tuple(float(value) for value in spacing)
        self.periodic = tuple(bool(value) for value in periodic)
        if len(self.spacing) != len(self.storage.blocks.grid_shape):
            raise ValueError("Sparse phase-field spacing dimension changed.")

    def apply(self, compact_damage, compact_history, gc, length_scale, /):
        damage = self.storage.unpack(compact_damage, self.active)
        history = self.storage.unpack(compact_history, self.active)
        laplacian = jnp.zeros_like(damage)
        for axis, spacing in enumerate(self.spacing):
            if self.periodic[axis]:
                lower = jnp.roll(damage, 1, axis=axis)
                upper = jnp.roll(damage, -1, axis=axis)
            else:
                indices = jnp.arange(damage.shape[axis])
                lower = jnp.take(damage, jnp.maximum(indices - 1, 0), axis=axis)
                upper = jnp.take(
                    damage,
                    jnp.minimum(indices + 1, damage.shape[axis] - 1),
                    axis=axis,
                )
            laplacian = laplacian + (upper - 2.0 * damage + lower) / (spacing**2)
        residual = (
            gc / length_scale * damage
            - gc * length_scale * laplacian
            - 2.0 * (1.0 - damage) * history
        )
        return self.storage.pack(residual, self.active)


__all__ = [
    "MPMBlockJacobiPreconditioner",
    "MPMCompactImplicitOperator",
    "MPMCompactOperatorResult",
    "MPMImplicitContactLinearization",
    "MPMImplicitTopologyPlan",
    "MPMImplicitUnknownLayout",
    "MPMMovingDomainDerivative",
    "MPMRouteSupersetPlan",
    "MPMRouteSupersetState",
    "MPMSparseContactOperator",
    "MPMSparsePhaseFieldOperator",
    "MPMTwoLevelMultigrid",
    "linearize_kway_contact",
]
