#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from itertools import combinations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import SmallLinearSolvePlan, solve_small_linear
from ._contact import AbstractMPMFrictionPlan, SharpCoulombMPMFrictionPlan


class MPMContactGraph(StrictModule):
    field_pairs: Array
    normals: Array
    gaps: Array
    valid: Array
    occupied_fields: Array
    graph_digest: Array
    successful: Array


class MPMRigidActorState(StrictModule):
    mass: Array
    inertia: Array
    center: Array
    linear_velocity: Array
    angular_velocity: Array
    active: Array


class MPMKWayContactResult(StrictModule):
    velocity: Array
    normal_impulses: Array
    tangential_impulses: Array
    essential_multipliers: Array
    active_pairs: Array
    modes: Array
    complementarity_residual: Array
    cone_residual: Array
    equality_residual: Array
    action_reaction_defect: Array
    dissipation: Array
    iterations: Array
    converged: Array
    successful: Array


class KWayMPMContactPlan(StrictModule, NonTrainableState):
    field_count: int = eqx.field(static=True)
    maximum_pairs: int = eqx.field(static=True)
    friction: AbstractMPMFrictionPlan
    maximum_steps: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    smoothing: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        field_count: int,
        /,
        *,
        friction: AbstractMPMFrictionPlan | None = None,
        maximum_steps: int = 25,
        tolerance: float = 1.0e-10,
        smoothing: float = 0.0,
    ):
        fields = int(field_count)
        maximum = fields * (fields - 1) // 2
        steps = int(maximum_steps)
        tolerance_ = float(tolerance)
        smoothing_ = float(smoothing)
        friction_ = SharpCoulombMPMFrictionPlan(0.0) if friction is None else friction
        if (
            fields < 2
            or fields > 3
            or steps <= 0
            or not np.isfinite(tolerance_)
            or tolerance_ <= 0.0
            or not np.isfinite(smoothing_)
            or smoothing_ < 0.0
            or not isinstance(friction_, AbstractMPMFrictionPlan)
        ):
            raise ValueError("K-way MPM contact plan is invalid.")
        self.field_count = fields
        self.maximum_pairs = maximum
        self.friction = friction_
        self.maximum_steps = steps
        self.tolerance = tolerance_
        self.smoothing = smoothing_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "k-way-mpm-contact",
                "field_count": fields,
                "maximum_pairs": maximum,
                "friction": friction_.plan_id,
                "maximum_steps": steps,
                "tolerance": tolerance_,
                "smoothing": smoothing_,
                "normal_complementarity": "fischer-burmeister",
                "tangent_projection": "simultaneous-jacobi",
            }
        )

    def build_graph(
        self,
        mass: ArrayLike,
        mass_gradient: ArrayLike,
        /,
        *,
        gap: ArrayLike | None = None,
        mass_tolerance: float = 0.0,
    ) -> MPMContactGraph:
        mass_ = jnp.asarray(mass)
        gradient = jnp.asarray(mass_gradient)
        if mass_.shape[0] != self.field_count or gradient.shape != mass_.shape + (
            gradient.shape[-1],
        ):
            raise ValueError("K-way contact mass/gradient field shape changed.")
        if gradient.shape[:-1] != mass_.shape:
            raise ValueError("K-way contact gradient layout differs from mass.")
        pairs = jnp.asarray(
            tuple(combinations(range(self.field_count), 2)), dtype=jnp.int32
        )
        first = pairs[:, 0]
        second = pairs[:, 1]
        occupied = mass_ > float(mass_tolerance)
        normalized = gradient / jnp.where(occupied[..., None], mass_[..., None], 1.0)
        normal_raw = normalized[first] - normalized[second]
        norm = jnp.sqrt(jnp.sum(normal_raw * normal_raw, axis=-1))
        reliable = jnp.isfinite(norm) & (norm > 1.0e-12)
        normals = normal_raw / jnp.where(reliable, norm, 1.0)[..., None]
        valid = occupied[first] & occupied[second] & reliable
        gaps = (
            jnp.zeros(valid.shape, dtype=mass_.dtype)
            if gap is None
            else jnp.broadcast_to(jnp.asarray(gap, dtype=mass_.dtype), valid.shape)
        )
        digest = jnp.sum(
            jnp.where(valid, (first[:, None] + 1) * 1009 + second[:, None] + 1, 0),
            dtype=jnp.int64,
        )
        return MPMContactGraph(
            pairs,
            normals,
            gaps,
            valid,
            occupied,
            digest,
            jnp.all(~valid | jnp.isfinite(gaps)),
        )

    def solve(
        self,
        mass: ArrayLike,
        velocity: ArrayLike,
        graph: MPMContactGraph,
        step_size: ArrayLike,
        /,
        *,
        essential_mask: ArrayLike | None = None,
        essential_values: ArrayLike | None = None,
    ) -> MPMKWayContactResult:
        mass_ = jnp.asarray(mass)
        velocity_ = jnp.asarray(velocity)
        dt = jnp.asarray(step_size, dtype=velocity_.dtype)
        if mass_.shape != velocity_.shape[:-1] or mass_.shape[0] != self.field_count:
            raise ValueError("K-way contact velocity/mass layout changed.")
        if not isinstance(graph, MPMContactGraph):
            raise TypeError("graph must be MPMContactGraph.")
        spatial_shape = mass_.shape[1:]
        dimension = velocity_.shape[-1]
        node_count = int(np.prod(spatial_shape))
        flat_mass = mass_.reshape((self.field_count, node_count)).T
        flat_velocity = velocity_.reshape(
            (self.field_count, node_count, dimension)
        ).transpose((1, 0, 2))
        normals = graph.normals.reshape(
            (self.maximum_pairs, node_count, dimension)
        ).transpose((1, 0, 2))
        gaps = graph.gaps.reshape((self.maximum_pairs, node_count)).T
        valid = graph.valid.reshape((self.maximum_pairs, node_count)).T
        pairs = graph.field_pairs
        mask = (
            jnp.zeros_like(velocity_, dtype=bool)
            if essential_mask is None
            else jnp.broadcast_to(
                jnp.asarray(essential_mask, dtype=bool), velocity_.shape
            )
        )
        values = (
            jnp.zeros_like(velocity_)
            if essential_values is None
            else jnp.broadcast_to(
                jnp.asarray(essential_values, dtype=velocity_.dtype), velocity_.shape
            )
        )
        flat_mask = mask.reshape((self.field_count, node_count, dimension)).transpose(
            (1, 0, 2)
        )
        flat_values = values.reshape((self.field_count, node_count, dimension)).transpose(
            (1, 0, 2)
        )

        def solve_node(
            node_mass,
            node_velocity,
            node_normals,
            node_gaps,
            node_valid,
            node_mask,
            node_values,
        ):
            first = pairs[:, 0]
            second = pairs[:, 1]
            inverse_mass = jnp.where(node_mass > 0.0, 1.0 / node_mass, 0.0)

            def contact_velocity(lambdas):
                impulse = lambdas[:, None] * node_normals
                delta = jnp.zeros_like(node_velocity)
                delta = delta.at[first].add(-impulse * inverse_mass[first, None])
                delta = delta.at[second].add(impulse * inverse_mass[second, None])
                return node_velocity + delta

            def residual(lambdas):
                current = contact_velocity(lambdas)
                relative = current[first] - current[second]
                normal_speed = jnp.sum(relative * node_normals, axis=-1)
                separation = node_gaps / jnp.maximum(dt, 1.0e-30) - normal_speed
                root = (
                    jnp.sqrt(lambdas**2 + separation**2 + self.smoothing**2)
                    - lambdas
                    - separation
                )
                return jnp.where(node_valid, root, lambdas)

            def iteration(_, lambdas):
                value = residual(lambdas)
                jacobian = jax.jacfwd(residual)(lambdas)
                identity = jnp.eye(self.maximum_pairs, dtype=lambdas.dtype)
                solve = solve_small_linear(
                    SmallLinearSolvePlan(self.maximum_pairs), jacobian, identity
                )
                delta = solve.value @ value
                candidate = (
                    lambdas - delta
                    if self.smoothing > 0.0
                    else jnp.maximum(lambdas - delta, 0.0)
                )
                return jnp.where(
                    jnp.linalg.norm(value) <= self.tolerance, lambdas, candidate
                )

            normal_impulse = jax.lax.fori_loop(
                0,
                self.maximum_steps,
                iteration,
                jnp.zeros((self.maximum_pairs,), dtype=node_velocity.dtype),
            )
            current = contact_velocity(normal_impulse)
            relative = current[first] - current[second]
            normal_speed = jnp.sum(relative * node_normals, axis=-1)
            tangent_velocity = relative - normal_speed[:, None] * node_normals
            tangent_speed = jnp.linalg.norm(tangent_velocity, axis=-1)
            tangent_direction = (
                tangent_velocity
                / jnp.where(tangent_speed > 0.0, tangent_speed, 1.0)[:, None]
            )
            reduced_mass = (
                node_mass[first]
                * node_mass[second]
                / jnp.where(
                    node_mass[first] + node_mass[second] > 0.0,
                    node_mass[first] + node_mass[second],
                    1.0,
                )
            )
            tangential_magnitude = self.friction.impulse_magnitude(
                tangent_speed, normal_impulse, reduced_mass
            )
            tangential = -tangential_magnitude[:, None] * tangent_direction
            tangential = jnp.where(node_valid[:, None], tangential, 0.0)
            tangent_delta = jnp.zeros_like(current)
            tangent_delta = tangent_delta.at[first].add(
                tangential * inverse_mass[first, None]
            )
            tangent_delta = tangent_delta.at[second].add(
                -tangential * inverse_mass[second, None]
            )
            current = current + tangent_delta
            unconstrained = current
            current = jnp.where(node_mask, node_values, current)
            essential_multiplier = node_mass[:, None] * (current - unconstrained)
            value = residual(normal_impulse)
            complementarity = jnp.max(jnp.abs(value))
            cone = jnp.max(
                jnp.maximum(
                    jnp.linalg.norm(tangential, axis=-1)
                    - self.friction.coefficient * normal_impulse,
                    0.0,
                )
            )
            equality = jnp.max(jnp.abs(jnp.where(node_mask, current - node_values, 0.0)))
            mode = jnp.where(
                node_valid & (normal_impulse > self.tolerance),
                jnp.where(
                    tangent_speed * reduced_mass
                    <= self.friction.coefficient * normal_impulse,
                    1,
                    2,
                ),
                0,
            ).astype(jnp.int32)
            active = node_valid & (normal_impulse > self.tolerance)
            dissipation = jnp.sum(tangential_magnitude * tangent_speed)
            converged = (
                (complementarity <= self.tolerance)
                & (cone <= self.tolerance)
                & (equality <= self.tolerance)
                & jnp.all(jnp.isfinite(current))
            )
            action_reaction = jnp.linalg.norm(
                jnp.sum(
                    jnp.zeros_like(current)
                    .at[first]
                    .add(-normal_impulse[:, None] * node_normals + tangential)
                    .at[second]
                    .add(normal_impulse[:, None] * node_normals - tangential),
                    axis=0,
                )
            )
            return (
                current,
                normal_impulse,
                tangential,
                essential_multiplier,
                active,
                mode,
                complementarity,
                cone,
                equality,
                action_reaction,
                dissipation,
                converged,
            )

        outputs = jax.vmap(solve_node)(
            flat_mass,
            flat_velocity,
            normals,
            gaps,
            valid,
            flat_mask,
            flat_values,
        )
        (
            next_velocity,
            normal_impulse,
            tangential,
            essential_multiplier,
            active_pairs,
            modes,
            complementarity,
            cone,
            equality,
            action_reaction,
            dissipation,
            converged,
        ) = outputs
        velocity_out = next_velocity.transpose((1, 0, 2)).reshape(velocity_.shape)
        pair_shape = (self.maximum_pairs,) + spatial_shape
        return MPMKWayContactResult(
            velocity_out,
            normal_impulse.T.reshape(pair_shape),
            tangential.transpose((1, 0, 2)).reshape(pair_shape + (dimension,)),
            essential_multiplier.transpose((1, 0, 2)).reshape(velocity_.shape),
            active_pairs.T.reshape(pair_shape),
            modes.T.reshape(pair_shape),
            jnp.max(complementarity),
            jnp.max(cone),
            jnp.max(equality),
            jnp.max(action_reaction),
            jnp.sum(dissipation),
            jnp.asarray(self.maximum_steps, dtype=jnp.int32),
            jnp.all(converged),
            graph.successful & jnp.all(converged),
        )


def apply_rigid_actor_reactions(
    actors: MPMRigidActorState,
    contact_positions: ArrayLike,
    impulses: ArrayLike,
    actor_indices: ArrayLike,
    /,
) -> MPMRigidActorState:
    positions = jnp.asarray(contact_positions)
    impulses_ = jnp.asarray(impulses)
    indices = jnp.asarray(actor_indices, dtype=jnp.int32)
    if (
        positions.shape != impulses_.shape
        or positions.shape[-1] != actors.center.shape[-1]
    ):
        raise ValueError("Rigid actor contact position/impulse shape changed.")
    linear_impulse = jnp.zeros_like(actors.linear_velocity).at[indices].add(impulses_)
    relative = positions - actors.center[indices]
    if positions.shape[-1] == 2:
        torque_values = (
            relative[:, 0] * impulses_[:, 1] - relative[:, 1] * impulses_[:, 0]
        )
        torque = jnp.zeros_like(actors.angular_velocity).at[indices].add(torque_values)
    else:
        torque_values = jnp.cross(relative, impulses_)
        torque = jnp.zeros_like(actors.angular_velocity).at[indices].add(torque_values)
    linear = actors.linear_velocity + linear_impulse / jnp.where(
        actors.active[:, None], actors.mass[:, None], 1.0
    )
    if positions.shape[-1] == 2:
        angular = actors.angular_velocity + torque / jnp.where(
            actors.active, actors.inertia, 1.0
        )
    else:
        identity = jnp.broadcast_to(
            jnp.eye(3, dtype=actors.inertia.dtype), actors.inertia.shape
        )
        solve = solve_small_linear(SmallLinearSolvePlan(3), actors.inertia, identity)
        angular = actors.angular_velocity + ein.contract(
            "aij,aj->ai", solve.value, torque
        )
    return MPMRigidActorState(
        actors.mass,
        actors.inertia,
        actors.center,
        linear,
        angular,
        actors.active,
    )


__all__ = [
    "KWayMPMContactPlan",
    "MPMContactGraph",
    "MPMKWayContactResult",
    "MPMRigidActorState",
    "apply_rigid_actor_reactions",
]
