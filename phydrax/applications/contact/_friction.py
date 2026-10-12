#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import PyTree

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.contact._distance import contact_tangent_basis
from ...discretization.contact._participant import ContactParticipantScene
from ...discretization.contact._search import ContactCandidateEpoch
from ...discretization.contact._stencils import (
    ContactStencilKind,
    evaluate_contact_stencils,
)
from ...discretization.contact._surface import PreparedCollisionScene
from ...typing import checked
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


@final
class LaggedCoulombFrictionPlan(StrictModule, NonTrainableState):
    coefficient: Array
    velocity_threshold: float = eqx.field(static=True)
    maximum_lag_iterations: int = eqx.field(static=True)
    lag_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        coefficient: ArrayLike,
        velocity_threshold: float,
        /,
        *,
        maximum_lag_iterations: int = 4,
        lag_tolerance: float = 1.0e-8,
    ) -> None:
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

    @checked
    def prepare(
        self,
        scene: PreparedCollisionScene | ContactParticipantScene,
        contact: PreparedConvergentContactPotential,
        /,
    ) -> PreparedLaggedCoulombFriction:
        return PreparedLaggedCoulombFriction(self, scene, contact)


@final
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


@final
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


@final
class PreparedLaggedCoulombFriction(StrictModule, NonTrainableState):
    """Frozen Coulomb frame with effort pullback at the supplied current states.

    ``build_state`` refreshes closest coefficients, tangent frames, compressive
    normal forces and route evidence from the fixed candidate epoch. These remain
    frozen during ``evaluate``; only the participant transpose follows the current
    geometry state.
    """

    plan: LaggedCoulombFrictionPlan
    scene: PreparedCollisionScene | ContactParticipantScene
    contact: PreparedConvergentContactPotential
    prepared_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        plan: LaggedCoulombFrictionPlan,
        scene: PreparedCollisionScene | ContactParticipantScene,
        contact: PreparedConvergentContactPotential,
        /,
    ) -> None:
        if contact.scene.scene_id != scene.scene_id:
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

    @checked
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
        current = jnp.asarray(positions, dtype=self.contact.reference_positions.dtype)
        expected = (self.scene.vertex_count, self.scene.ambient_dimension)
        if current.shape != expected:
            raise ValueError(f"positions must have shape {expected}.")
        rest = (
            self.contact._rest_positions()
            if rest_positions is None
            else jnp.asarray(rest_positions, dtype=current.dtype)
        )
        if rest.shape != expected:
            raise ValueError(f"rest_positions must have shape {expected}.")
        stiffness_ = (
            self.contact.plan.stiffness
            if stiffness is None
            else jnp.asarray(stiffness, dtype=current.dtype)
        )
        if stiffness_.shape != ():
            raise ValueError("stiffness must be scalar.")
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

    @checked
    def energy(
        self,
        velocities: ArrayLike,
        state: ContactFrictionState,
        /,
    ) -> Array:
        if state.prepared_id != self.prepared_id:
            raise ValueError("Friction state belongs to another prepared friction plan.")
        velocity = jnp.asarray(velocities, dtype=self.contact.reference_positions.dtype)
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

    @checked
    def evaluate(
        self,
        velocities: ArrayLike,
        state: ContactFrictionState,
        /,
        *,
        participant_states: Sequence[PyTree[Array]] | None = None,
    ) -> ContactFrictionEvaluation:
        if isinstance(self.scene, ContactParticipantScene):
            if participant_states is None:
                raise ValueError("Participant scenes require participant_states.")
        elif participant_states is not None:
            raise ValueError(
                "Shared-state collision scenes do not accept participant_states."
            )
        velocity = jnp.asarray(velocities, dtype=self.contact.reference_positions.dtype)
        energy, gradient = jax.value_and_grad(lambda value: self.energy(value, state))(
            velocity
        )
        surface_effort = -gradient
        if isinstance(self.scene, ContactParticipantScene):
            if participant_states is None:
                raise ValueError("Participant scenes require participant_states.")
            state_force = self.scene.effort_pullback(participant_states, surface_effort)
        else:
            state_force = self.scene.effort_pullback(surface_effort)
        dissipation = -jnp.sum(surface_effort * velocity)
        force_norm = jnp.sqrt(jnp.sum(surface_effort * surface_effort, axis=-1))
        finite = (
            jnp.isfinite(energy)
            & jnp.all(jnp.isfinite(surface_effort))
            & jnp.isfinite(dissipation)
        )
        for leaf in jax.tree.leaves(state_force):
            finite = finite & jnp.all(jnp.isfinite(leaf))
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
        coefficients = jnp.max(
            jnp.where(
                previous.valid[:, None],
                jnp.abs(candidate.coefficients - previous.coefficients),
                0.0,
            ),
            initial=0.0,
        )
        return jnp.where(
            route_valid, jnp.maximum(jnp.maximum(force, basis), coefficients), jnp.inf
        )


class AxisymmetricInterfaceGeometry(StrictModule):
    """Current matching-trace geometry in physical (r,z) coordinates."""

    normal: Array
    tangent: Array
    normal_gap: Array
    current_measure_ratio: Array
    admissible: Array


def axisymmetric_interface_geometry(
    left_positions: ArrayLike,
    right_positions: ArrayLike,
    left_axial_tangent: ArrayLike,
    right_axial_tangent: ArrayLike,
    reference_radius: ArrayLike,
    /,
) -> AxisymmetricInterfaceGeometry:
    left, right = jnp.asarray(left_positions), jnp.asarray(right_positions)
    tangent_raw = (jnp.asarray(left_axial_tangent) + jnp.asarray(right_axial_tangent)) / 2
    magnitude = jnp.linalg.norm(tangent_raw, axis=-1)
    tangent = tangent_raw / magnitude[..., None]
    normal = jnp.stack((tangent[..., 1], -tangent[..., 0]), axis=-1)
    gap = jnp.sum((right - left) * normal, axis=-1)
    ratio = (
        left[..., 0]
        / jnp.asarray(reference_radius)
        * jnp.linalg.norm(jnp.asarray(left_axial_tangent), axis=-1)
    )
    valid = (
        jnp.all(jnp.isfinite(normal), axis=-1)
        & (magnitude > 0)
        & (ratio > 0)
        & (tangent_raw[..., 1] > 0)
        & jnp.isfinite(gap)
    )
    return AxisymmetricInterfaceGeometry(normal, tangent, gap, ratio, valid)


class CoulombInterfaceResponse(StrictModule):
    """Monolithic friction residual and dissipated work for a scalar trace.

    Force is exerted on the left body by the right. Slip is left minus right,
    so force * slip <= 0. The natural map is covariant under reversal of the
    tangent orientation; the paired traction is exactly equal and opposite.
    """

    residual: Array
    friction_bound: Array
    dissipated_work: Array
    admissible: Array


def coulomb_interface_response(
    tangential_force: ArrayLike,
    relative_slip: ArrayLike,
    normal_force: ArrayLike,
    coefficient: ArrayLike,
    /,
    *,
    force_scale: ArrayLike,
    slip_scale: ArrayLike,
) -> CoulombInterfaceResponse:
    """Exact stick/slip graph, with scales used only to condition its natural map."""
    force, slip, normal, mu, fs, ss = jnp.broadcast_arrays(
        jnp.asarray(tangential_force),
        jnp.asarray(relative_slip),
        jnp.asarray(normal_force),
        jnp.asarray(coefficient),
        jnp.asarray(force_scale),
        jnp.asarray(slip_scale),
    )
    bound = mu * jnp.maximum(normal, 0.0)
    projection = jnp.clip(force - fs * slip / ss, -bound, bound)
    residual = (force - projection) / fs
    dissipated = -force * slip
    admissible = (
        jnp.isfinite(residual)
        & (normal >= 0)
        & (mu >= 0)
        & (fs > 0)
        & (ss > 0)
        & (jnp.abs(force) <= bound + fs * 1e-10)
        & (dissipated >= -fs * ss * 1e-10)
    )
    return CoulombInterfaceResponse(residual, bound, dissipated, admissible)


__all__ = [
    "ContactFrictionEvaluation",
    "CoulombInterfaceResponse",
    "AxisymmetricInterfaceGeometry",
    "axisymmetric_interface_geometry",
    "coulomb_interface_response",
    "ContactFrictionState",
    "LaggedCoulombFrictionPlan",
    "PreparedLaggedCoulombFriction",
    "smooth_coulomb_potential",
]
