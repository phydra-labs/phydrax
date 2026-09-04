#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PyTree

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.contact import (
    contact_tangent_basis,
    ContactCandidateEpoch,
    ContactStencilKind,
    evaluate_contact_stencils,
    PreparedCollisionScene,
)
from ._barrier import (
    clamped_log_barrier_first_derivative,
    physical_barrier_scale,
)
from ._potential import PreparedConvergentContactPotential


def smooth_coulomb_potential(speed: ArrayLike, threshold: ArrayLike, /) -> Array:
    magnitude = jnp.asarray(speed)
    epsilon = jnp.asarray(threshold, dtype=magnitude.dtype)
    interior = (
        epsilon / 3.0
        + magnitude * magnitude / epsilon
        - magnitude * magnitude * magnitude / (3.0 * epsilon * epsilon)
    )
    return jnp.where(magnitude < epsilon, interior, magnitude)


class LaggedCoulombFrictionPlan(StrictModule, NonTrainableState):
    coefficient: Array
    velocity_threshold: float = eqx.field(static=True)
    maximum_lag_iterations: int = eqx.field(static=True)
    lag_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        coefficient: ArrayLike,
        velocity_threshold: float,
        /,
        *,
        maximum_lag_iterations: int = 4,
        lag_tolerance: float = 1.0e-8,
    ):
        coefficient_ = jnp.asarray(coefficient)
        threshold = float(velocity_threshold)
        iterations = int(maximum_lag_iterations)
        tolerance = float(lag_tolerance)
        if coefficient_.ndim not in (0, 2) or (
            coefficient_.ndim == 2 and coefficient_.shape[0] != coefficient_.shape[1]
        ):
            raise ValueError(
                "Friction coefficient must be scalar or one square body-pair table."
            )
        if not bool(jnp.all(jnp.isfinite(coefficient_) & (coefficient_ >= 0.0))):
            raise ValueError("Friction coefficients must be finite and nonnegative.")
        if not np.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("velocity_threshold must be finite and positive.")
        if iterations <= 0:
            raise ValueError("maximum_lag_iterations must be positive.")
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("lag_tolerance must be finite and nonnegative.")
        self.coefficient = coefficient_
        self.velocity_threshold = threshold
        self.maximum_lag_iterations = iterations
        self.lag_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "lagged-coulomb-friction-plan",
                "coefficient": array_tree_fingerprint(np.asarray(coefficient_)),
                "velocity_threshold": threshold.hex(),
                "maximum_lag_iterations": iterations,
                "lag_tolerance": tolerance.hex(),
            }
        )

    def prepare(
        self,
        scene: PreparedCollisionScene,
        contact: PreparedConvergentContactPotential,
        /,
    ) -> PreparedLaggedCoulombFriction:
        return PreparedLaggedCoulombFriction(self, scene, contact)


class ContactFrictionState(StrictModule, NonTrainableState):
    vertex_indices: Array
    coefficients: Array
    tangent_basis: Array
    normal_force: Array
    friction_coefficient: Array
    route_keys: Array
    valid: Array
    state_version: Array
    prepared_id: str = eqx.field(static=True)


class ContactFrictionEvaluation(StrictModule):
    energy: Array
    surface_force: Array
    state_force: PyTree[Array]
    dissipation_rate: Array
    maximum_force: Array
    active_contacts: Array
    finite: Array
    dissipative: Array
    successful: Array
    prepared_id: str = eqx.field(static=True)


class PreparedLaggedCoulombFriction(StrictModule, NonTrainableState):
    plan: LaggedCoulombFrictionPlan
    scene: PreparedCollisionScene
    contact: PreparedConvergentContactPotential
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: LaggedCoulombFrictionPlan,
        scene: PreparedCollisionScene,
        contact: PreparedConvergentContactPotential,
        /,
    ):
        if not isinstance(plan, LaggedCoulombFrictionPlan):
            raise TypeError("plan must be LaggedCoulombFrictionPlan.")
        if not isinstance(scene, PreparedCollisionScene):
            raise TypeError("scene must be PreparedCollisionScene.")
        if (
            not isinstance(contact, PreparedConvergentContactPotential)
            or contact.scene.scene_id != scene.scene_id
        ):
            raise ValueError("contact must be prepared for the friction collision scene.")
        if plan.coefficient.ndim == 2:
            maximum_body = int(jnp.max(scene.vertex_body_ids, initial=0))
            if maximum_body >= plan.coefficient.shape[0]:
                raise ValueError(
                    "Friction body-pair table does not cover scene body IDs."
                )
        self.plan = plan
        self.scene = scene
        self.contact = contact
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-lagged-coulomb-friction",
                "plan": plan.plan_id,
                "scene": scene.scene_id,
                "contact": contact.prepared_id,
            }
        )

    def _coefficient(self, left_body: Array, right_body: Array, /) -> Array:
        if self.plan.coefficient.ndim == 0:
            return jnp.full(left_body.shape, self.plan.coefficient)
        return self.plan.coefficient[left_body, right_body]

    def build_state(
        self,
        positions: ArrayLike,
        epoch: ContactCandidateEpoch,
        /,
        *,
        rest_positions: ArrayLike | None = None,
        stiffness: ArrayLike | None = None,
        state_version: ArrayLike = 0,
    ) -> ContactFrictionState:
        current = jnp.asarray(
            positions, dtype=self.scene.surfaces[0].precision.geometry_dtype
        )
        rest = (
            self.contact._rest_positions()
            if rest_positions is None
            else jnp.asarray(rest_positions, dtype=current.dtype)
        )
        stiffness_ = (
            self.contact.plan.stiffness
            if stiffness is None
            else jnp.asarray(stiffness, dtype=current.dtype)
        )
        vertex_measure, edge_measure = self.contact._measures(rest)
        vertex_indices = []
        coefficients = []
        tangent_basis = []
        normal_force = []
        friction_coefficient = []
        route_keys = []
        valid_values = []
        body_ids = self.scene.vertex_body_ids
        d_hat = jnp.asarray(self.contact.plan.activation_distance, dtype=current.dtype)
        edge_offset = self.scene.vertex_count
        face_offset = self.scene.vertex_count + self.scene.edge_count
        del face_offset
        for batch in epoch.active_batches:
            evaluation = evaluate_contact_stencils(
                batch,
                current,
                rest,
                tolerance=self.contact.plan.geometry_tolerance,
            )
            squared = evaluation.distance.squared_distance
            separation = evaluation.minimum_separation
            active = evaluation.valid & (squared < (separation + d_hat) ** 2)
            distance = jnp.sqrt(jnp.maximum(squared, jnp.finfo(current.dtype).tiny))
            shifted = jnp.maximum(
                squared - separation * separation,
                jnp.finfo(current.dtype).tiny,
            )
            threshold = (2.0 * separation + d_hat) * d_hat
            derivative = clamped_log_barrier_first_derivative(shifted, threshold)
            if batch.kind in (
                ContactStencilKind.EDGE_VERTEX,
                ContactStencilKind.FACE_VERTEX,
            ):
                query = batch.vertex_indices[:, 0]
                weight = 0.5 * vertex_measure[query]
                right_endpoint = batch.vertex_indices[:, 1]
            elif batch.kind == ContactStencilKind.EDGE_EDGE:
                left_edge = jnp.clip(
                    batch.left_feature_indices - edge_offset,
                    0,
                    self.scene.edge_count - 1,
                ).astype(jnp.int32)
                right_edge = jnp.clip(
                    batch.right_feature_indices - edge_offset,
                    0,
                    self.scene.edge_count - 1,
                ).astype(jnp.int32)
                weight = 0.25 * (edge_measure[left_edge] + edge_measure[right_edge])
                right_endpoint = batch.vertex_indices[:, 2]
            else:
                weight = jnp.ones((batch.capacity,), dtype=current.dtype)
                right_endpoint = batch.vertex_indices[:, 1]
            force = (
                -stiffness_.astype(current.dtype)
                * weight
                * evaluation.mollifier
                * physical_barrier_scale(d_hat, separation)
                * derivative
                * 2.0
                * distance
            )
            left_endpoint = batch.vertex_indices[:, 0]
            left_body = body_ids[jnp.clip(left_endpoint, 0, self.scene.vertex_count - 1)]
            right_body = body_ids[
                jnp.clip(right_endpoint, 0, self.scene.vertex_count - 1)
            ]
            vertex_indices.append(batch.vertex_indices)
            coefficients.append(evaluation.distance.coefficients)
            tangent_basis.append(contact_tangent_basis(evaluation.distance.normal))
            normal_force.append(jnp.where(active, jnp.maximum(force, 0.0), 0.0))
            friction_coefficient.append(
                self._coefficient(left_body, right_body).astype(current.dtype)
            )
            route_keys.append(batch.route_keys)
            valid_values.append(active)
        if not vertex_indices:
            raise ValueError(
                "Friction preparation requires positive contact candidate capacity."
            )
        return ContactFrictionState(
            jnp.concatenate(tuple(vertex_indices), axis=0),
            jnp.concatenate(tuple(coefficients), axis=0),
            jnp.concatenate(tuple(tangent_basis), axis=0),
            jnp.concatenate(tuple(normal_force), axis=0),
            jnp.concatenate(tuple(friction_coefficient), axis=0),
            jnp.concatenate(tuple(route_keys), axis=0),
            jnp.concatenate(tuple(valid_values), axis=0),
            jnp.asarray(state_version, dtype=jnp.int32),
            self.prepared_id,
        )

    def energy(
        self,
        velocities: ArrayLike,
        state: ContactFrictionState,
        /,
    ) -> Array:
        if (
            not isinstance(state, ContactFrictionState)
            or state.prepared_id != self.prepared_id
        ):
            raise ValueError("Friction state belongs to another prepared friction plan.")
        velocity = jnp.asarray(
            velocities, dtype=self.scene.surfaces[0].precision.geometry_dtype
        )
        expected = (self.scene.vertex_count, self.scene.ambient_dimension)
        if velocity.shape != expected:
            raise ValueError(f"velocities must have shape {expected}.")
        safe = jnp.clip(state.vertex_indices, 0, self.scene.vertex_count - 1)
        gathered = velocity[safe]
        relative = jnp.sum(state.coefficients[..., None] * gathered, axis=1)
        tangent_velocity = jnp.sum(state.tangent_basis * relative[:, :, None], axis=1)
        speed_regularization = jnp.finfo(velocity.dtype).eps * jnp.asarray(
            self.plan.velocity_threshold, dtype=velocity.dtype
        )
        speed = jnp.sqrt(
            jnp.sum(tangent_velocity * tangent_velocity, axis=-1)
            + speed_regularization * speed_regularization
        )
        potential = smooth_coulomb_potential(
            speed,
            jnp.asarray(self.plan.velocity_threshold, dtype=velocity.dtype),
        )
        value = state.friction_coefficient * state.normal_force * potential
        return jnp.sum(jnp.where(state.valid, value, 0.0))

    def evaluate(
        self,
        velocities: ArrayLike,
        state: ContactFrictionState,
        /,
    ) -> ContactFrictionEvaluation:
        velocity = jnp.asarray(
            velocities, dtype=self.scene.surfaces[0].precision.geometry_dtype
        )
        energy, gradient = jax.value_and_grad(lambda value: self.energy(value, state))(
            velocity
        )
        surface_effort = -gradient
        state_force = self.scene.effort_pullback(surface_effort)
        dissipation = -jnp.sum(surface_effort * velocity)
        force_norm = jnp.sqrt(jnp.sum(surface_effort * surface_effort, axis=-1))
        finite = (
            jnp.isfinite(energy)
            & jnp.all(jnp.isfinite(surface_effort))
            & jnp.isfinite(dissipation)
        )
        tolerance = jnp.finfo(velocity.dtype).eps * max(64, 8 * state.valid.size)
        dissipative = dissipation >= -tolerance * jnp.maximum(1.0, jnp.abs(dissipation))
        return ContactFrictionEvaluation(
            energy,
            surface_effort,
            state_force,
            dissipation,
            jnp.max(force_norm, initial=0.0),
            jnp.sum(state.valid, dtype=jnp.int32),
            finite,
            dissipative,
            finite & dissipative,
            self.prepared_id,
        )

    def lag_residual(
        self,
        previous: ContactFrictionState,
        candidate: ContactFrictionState,
        /,
    ) -> Array:
        if (
            previous.prepared_id != self.prepared_id
            or candidate.prepared_id != self.prepared_id
        ):
            raise ValueError("Friction lag states belong to another prepared plan.")
        same_route = previous.valid == candidate.valid
        keys = jnp.where(
            previous.valid, previous.route_keys == candidate.route_keys, True
        )
        route_valid = jnp.all(same_route & keys)
        force = jnp.max(
            jnp.abs(candidate.normal_force - previous.normal_force), initial=0.0
        )
        basis = jnp.max(
            jnp.abs(candidate.tangent_basis - previous.tangent_basis), initial=0.0
        )
        return jnp.where(route_valid, jnp.maximum(force, basis), jnp.inf)


__all__ = [
    "ContactFrictionEvaluation",
    "ContactFrictionState",
    "LaggedCoulombFrictionPlan",
    "PreparedLaggedCoulombFriction",
    "smooth_coulomb_potential",
]
