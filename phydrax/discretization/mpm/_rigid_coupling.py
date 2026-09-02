#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._tree_math import tree_allfinite, tree_where
from ..particle._rigid_body import (
    PreparedRigidBodySet,
    quaternion_rotation_matrix,
    RigidBodyKinematics,
    RigidBodyLoad,
)
from ._dynamics import PreparedMPMDynamics
from ._types import MPMRuntimeState, MPMStepResult


class RigidMPMCouplingMode(IntEnum):
    """Distinct fixed-route rigid/material-point coupling laws."""

    WELD = 0
    PENALTY = 1
    IMPULSE = 2


def _mode_array(
    value: ArrayLike | str | RigidMPMCouplingMode, count: int, /
) -> np.ndarray:
    names = {
        "weld": int(RigidMPMCouplingMode.WELD),
        "penalty": int(RigidMPMCouplingMode.PENALTY),
        "impulse": int(RigidMPMCouplingMode.IMPULSE),
    }
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized not in names:
            raise ValueError("mode must be 'weld', 'penalty', or 'impulse'.")
        return np.full((count,), names[normalized], dtype=np.int32)
    if isinstance(value, RigidMPMCouplingMode):
        return np.full((count,), int(value), dtype=np.int32)
    array = np.asarray(value)
    if array.ndim == 0:
        array = np.full((count,), int(array), dtype=np.int32)
    if array.shape != (count,) or not np.issubdtype(array.dtype, np.integer):
        raise TypeError("mode must be a mode name/code or a route-count integer array.")
    allowed = np.asarray([int(item) for item in RigidMPMCouplingMode])
    if np.any(~np.isin(array, allowed)):
        raise ValueError("mode contains an unknown coupling mode.")
    return array.astype(np.int32, copy=False)


def _route_scalars(name: str, value: ArrayLike, count: int, /) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full((count,), float(array), dtype=float)
    if array.shape != (count,) or np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be finite scalar or route-count data.")
    return array


def _stable_keys(
    particle: np.ndarray,
    body: np.ndarray,
    modes: np.ndarray,
    /,
) -> np.ndarray:
    mask = (1 << 64) - 1
    keys = np.empty(particle.shape, dtype=np.int64)
    for slot in range(particle.size):
        value = 1469598103934665603
        for item in (slot, int(particle[slot]), int(body[slot]), int(modes[slot])):
            value ^= (int(item) + 0x9E3779B97F4A7C15) & mask
            value = (value * 1099511628211) & mask
        keys[slot] = value & ((1 << 63) - 1)
    return keys


def _route_digest(route_state, /) -> Array:
    slots = jnp.arange(route_state.stencil.indices.shape[1], dtype=jnp.int64)[None, :]
    values = jnp.where(
        route_state.stencil.valid,
        route_state.stencil.indices.astype(jnp.int64) + 1,
        0,
    )
    return jnp.sum(values * (slots + 17))


def _rotate_local(
    dimension: int,
    orientation: Array,
    local_vector: Array,
    /,
) -> Array:
    if dimension == 2:
        angle = orientation[:, 0]
        cosine = jnp.cos(angle)
        sine = jnp.sin(angle)
        return jnp.stack(
            (
                cosine * local_vector[:, 0] - sine * local_vector[:, 1],
                sine * local_vector[:, 0] + cosine * local_vector[:, 1],
            ),
            axis=-1,
        )
    rotation = quaternion_rotation_matrix(orientation)
    return contract("rij,rj->ri", rotation, local_vector)


def _point_spin_velocity(
    dimension: int,
    angular_velocity: Array,
    arm: Array,
    /,
) -> Array:
    if dimension == 2:
        omega = angular_velocity[:, 0]
        return jnp.stack((-omega * arm[:, 1], omega * arm[:, 0]), axis=-1)
    return jnp.cross(angular_velocity, arm)


def _point_torque(dimension: int, arm: Array, force: Array, /) -> Array:
    if dimension == 2:
        return (arm[:, 0] * force[:, 1] - arm[:, 1] * force[:, 0])[:, None]
    return jnp.cross(arm, force)


class RigidMPMCouplingPlan(StrictModule, NonTrainableState):
    """Fixed material-particle to rigid-feature coupling topology and law data."""

    particle_indices: Array
    body_indices: Array
    modes: Array
    local_anchors: Array
    local_normals: Array
    stiffness: Array
    damping: Array
    restitution: Array
    active_mask: Array
    coupling_keys: Array
    ambient_dimension: int = eqx.field(static=True)
    activation_distance: float = eqx.field(static=True)
    baumgarte_factor: float = eqx.field(static=True)
    geometry_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        particle_indices: ArrayLike,
        body_indices: ArrayLike,
        local_anchors: ArrayLike,
        /,
        *,
        ambient_dimension: int,
        mode: ArrayLike | str | RigidMPMCouplingMode,
        local_normals: ArrayLike | None = None,
        stiffness: ArrayLike = 0.0,
        damping: ArrayLike = 0.0,
        restitution: ArrayLike = 0.0,
        active_mask: ArrayLike | None = None,
        coupling_keys: ArrayLike | None = None,
        activation_distance: float = 0.0,
        baumgarte_factor: float = 0.1,
        geometry_tolerance: float = 1.0e-12,
        plan_id: str | None = None,
    ):
        particle = np.asarray(particle_indices)
        body = np.asarray(body_indices)
        if (
            particle.ndim != 1
            or particle.size == 0
            or not np.issubdtype(particle.dtype, np.integer)
        ):
            raise TypeError("particle_indices must be a nonempty rank-1 integer array.")
        count = int(particle.size)
        if body.shape != (count,) or not np.issubdtype(body.dtype, np.integer):
            raise TypeError("body_indices must be a route-count integer array.")
        dimension = int(ambient_dimension)
        if dimension not in (2, 3):
            raise ValueError("Rigid-MPM coupling requires dimension two or three.")
        anchors = np.asarray(local_anchors, dtype=float)
        if anchors.shape != (count, dimension) or np.any(~np.isfinite(anchors)):
            raise ValueError("local_anchors must be finite route vectors.")
        modes = _mode_array(mode, count)
        normals = (
            np.broadcast_to(
                np.asarray((1.0,) + (0.0,) * (dimension - 1)), (count, dimension)
            ).copy()
            if local_normals is None
            else np.asarray(local_normals, dtype=float)
        )
        if normals.shape != (count, dimension) or np.any(~np.isfinite(normals)):
            raise ValueError("local_normals must be finite route vectors.")
        tolerance = float(geometry_tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("geometry_tolerance must be finite and positive.")
        normal_norm = np.linalg.norm(normals, axis=-1)
        impulse = modes == int(RigidMPMCouplingMode.IMPULSE)
        if np.any(impulse & (normal_norm <= tolerance)):
            raise ValueError("Impulse routes require nonzero local normals.")
        normals = normals / np.where(
            normal_norm[:, None] > tolerance, normal_norm[:, None], 1.0
        )
        stiffness_ = _route_scalars("stiffness", stiffness, count)
        damping_ = _route_scalars("damping", damping, count)
        restitution_ = _route_scalars("restitution", restitution, count)
        penalty = modes == int(RigidMPMCouplingMode.PENALTY)
        if np.any(stiffness_ < 0.0):
            raise ValueError("stiffness must be nonnegative.")
        if np.any(penalty & (stiffness_ <= 0.0)):
            raise ValueError("Penalty routes require strictly positive stiffness.")
        if np.any(damping_ < 0.0):
            raise ValueError("damping must be nonnegative.")
        if np.any((restitution_ < 0.0) | (restitution_ > 1.0)):
            raise ValueError("restitution must lie in [0, 1].")
        active = (
            np.ones((count,), dtype=bool)
            if active_mask is None
            else np.asarray(active_mask, dtype=bool)
        )
        if active.shape != (count,):
            raise ValueError("active_mask must have route-count shape.")
        if np.any(active & ((particle < 0) | (body < 0))):
            raise ValueError("Active particle/body indices must be nonnegative.")
        activation = float(activation_distance)
        baumgarte = float(baumgarte_factor)
        if not np.isfinite(activation) or activation < 0.0:
            raise ValueError("activation_distance must be finite and nonnegative.")
        if not np.isfinite(baumgarte) or baumgarte < 0.0:
            raise ValueError("baumgarte_factor must be finite and nonnegative.")
        if coupling_keys is None:
            keys = _stable_keys(
                particle.astype(np.int64), body.astype(np.int64), modes.astype(np.int64)
            )
        else:
            keys = np.asarray(coupling_keys)
            if keys.shape != (count,) or not np.issubdtype(keys.dtype, np.integer):
                raise TypeError("coupling_keys must be a route-count integer array.")
            keys = keys.astype(np.int64, copy=False)
        if np.any(active & (keys < 0)) or np.unique(keys[active]).size != int(
            np.sum(active)
        ):
            raise ValueError("Active coupling keys must be unique and nonnegative.")
        generated = canonical_fingerprint(
            {
                "kind": "rigid-mpm-coupling-plan",
                "arrays": array_tree_fingerprint(
                    {
                        "particle_indices": particle,
                        "body_indices": body,
                        "modes": modes,
                        "local_anchors": anchors,
                        "local_normals": normals,
                        "stiffness": stiffness_,
                        "damping": damping_,
                        "restitution": restitution_,
                        "active_mask": active,
                        "coupling_keys": keys,
                    }
                ),
                "ambient_dimension": dimension,
                "activation_distance": activation,
                "baumgarte_factor": baumgarte,
                "geometry_tolerance": tolerance,
            }
        )
        self.particle_indices = jnp.asarray(particle, dtype=jnp.int32)
        self.body_indices = jnp.asarray(body, dtype=jnp.int32)
        self.modes = jnp.asarray(modes, dtype=jnp.int32)
        self.local_anchors = jnp.asarray(anchors)
        self.local_normals = jnp.asarray(normals)
        self.stiffness = jnp.asarray(stiffness_)
        self.damping = jnp.asarray(damping_)
        self.restitution = jnp.asarray(restitution_)
        self.active_mask = jnp.asarray(active)
        self.coupling_keys = jnp.asarray(keys, dtype=jnp.int64)
        self.ambient_dimension = dimension
        self.activation_distance = activation
        self.baumgarte_factor = baumgarte
        self.geometry_tolerance = tolerance
        self.plan_id = generated if plan_id is None else str(plan_id)
        if not self.plan_id:
            raise ValueError("plan_id must be nonempty.")

    def prepare(
        self,
        dynamics: PreparedMPMDynamics,
        bodies: PreparedRigidBodySet,
        /,
    ) -> PreparedRigidMPMCoupling:
        return PreparedRigidMPMCoupling(self, dynamics, bodies)


class RigidMPMRouteCacheCertificate(StrictModule):
    route_digest: Array
    route_weight_checksum: Array
    valid_route_count: Array
    minimum_route_weight: Array
    minimum_domain_margin: Array
    cache_hit: Array
    cache_coherent: Array
    particle_step: Array
    finite: Array
    successful: Array
    certificate_id: str = eqx.field(static=True)


class RigidMPMConstraintPayload(StrictModule):
    particle_indices: Array
    body_indices: Array
    grid_indices: Array
    grid_weights: Array
    gap_vector: Array
    gap: Array
    normal: Array
    relative_velocity: Array
    normal_velocity: Array
    coupling_keys: Array
    hard: Array
    unilateral: Array
    valid: Array
    validity_margin: Array
    feature_margin: Array
    finite: Array
    successful: Array
    prepared_id: str = eqx.field(static=True)


class RigidMPMCouplingEvaluation(StrictModule):
    payload: RigidMPMConstraintPayload
    route_force: Array
    route_impulse: Array
    particle_force: Array
    grid_force: Array
    rigid_load: RigidBodyLoad
    action_reaction_residual: Array
    angular_action_reaction_residual: Array
    grid_scatter_residual: Array
    action_reaction_valid: Array
    grid_scatter_valid: Array
    certificate: RigidMPMRouteCacheCertificate
    finite: Array
    successful: Array
    prepared_id: str = eqx.field(static=True)


class RigidMPMCouplingState(StrictModule):
    accumulated_impulse: Array
    route_digest: Array
    route_weight_checksum: Array
    cache_generation: Array
    accepted_step: Array
    cache_valid: Array
    last_successful: Array
    prepared_id: str = eqx.field(static=True)


class RigidMPMCouplingStepResult(StrictModule):
    candidate_state: RigidMPMCouplingState
    accepted_state: RigidMPMCouplingState
    candidate_mpm_state: MPMRuntimeState
    accepted_mpm_state: MPMRuntimeState
    mpm_result: MPMStepResult
    evaluation: RigidMPMCouplingEvaluation
    successful: Array
    stability_margin: Array
    stability_margin_valid: Array
    route_margin: Array
    route_margin_valid: Array
    prepared_id: str = eqx.field(static=True)


class PreparedRigidMPMCoupling(StrictModule, NonTrainableState):
    """Rigid coupling adapter that leaves MPM transfer/material authority intact."""

    plan: RigidMPMCouplingPlan
    dynamics: PreparedMPMDynamics
    bodies: PreparedRigidBodySet
    prepared_id: str = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: RigidMPMCouplingPlan,
        dynamics: PreparedMPMDynamics,
        bodies: PreparedRigidBodySet,
        /,
    ):
        if not isinstance(plan, RigidMPMCouplingPlan):
            raise TypeError("plan must be RigidMPMCouplingPlan.")
        if not isinstance(dynamics, PreparedMPMDynamics):
            raise TypeError("dynamics must be PreparedMPMDynamics.")
        if not isinstance(bodies, PreparedRigidBodySet):
            raise TypeError("bodies must be PreparedRigidBodySet.")
        if (
            dynamics.dimension != plan.ambient_dimension
            or bodies.ambient_dimension != plan.ambient_dimension
        ):
            raise ValueError("Coupling, MPM, and rigid-body dimensions must match.")
        active = np.asarray(plan.active_mask)
        particle = np.asarray(plan.particle_indices)
        body = np.asarray(plan.body_indices)
        if np.any(particle[active] >= dynamics.particles.capacity):
            raise ValueError("A coupling particle index exceeds MPM capacity.")
        if np.any(body[active] >= bodies.capacity):
            raise ValueError("A coupling body index exceeds rigid-body capacity.")
        active_particle = np.asarray(dynamics.particles.active_mask)[particle[active]]
        active_body = np.asarray(bodies.particles.active_mask)[body[active]]
        if np.any(~active_particle) or np.any(~active_body):
            raise ValueError(
                "Active coupling routes require active particle/body endpoints."
            )
        prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-rigid-mpm-coupling",
                "plan": plan.plan_id,
                "dynamics": dynamics.prepared_id,
                "bodies": bodies.prepared_id,
            }
        )
        self.plan = plan
        self.dynamics = dynamics
        self.bodies = bodies
        self.prepared_id = prepared_id
        self.certificate_id = canonical_fingerprint(
            {
                "kind": "rigid-mpm-route-cache-certificate",
                "coupling": prepared_id,
                "splat": dynamics.splat.prepared_id,
            }
        )

    @property
    def route_count(self) -> int:
        return int(self.plan.particle_indices.shape[0])

    @property
    def ambient_dimension(self) -> int:
        return self.plan.ambient_dimension

    def initialize_state(self, /) -> RigidMPMCouplingState:
        dtype = self.dynamics.particles.safe_masses.dtype
        return RigidMPMCouplingState(
            jnp.zeros((self.route_count, self.ambient_dimension), dtype=dtype),
            jnp.asarray(-1, dtype=jnp.int64),
            jnp.zeros((), dtype=dtype),
            jnp.zeros((), dtype=jnp.int32),
            jnp.zeros((), dtype=jnp.int32),
            jnp.asarray(False),
            jnp.asarray(True),
            self.prepared_id,
        )

    def _require_state(self, state: RigidMPMCouplingState, /) -> None:
        if not isinstance(state, RigidMPMCouplingState):
            raise TypeError("coupling_state must be RigidMPMCouplingState.")
        if state.prepared_id != self.prepared_id:
            raise ValueError("coupling_state belongs to a different prepared coupling.")

    def _require_kinematics(self, kinematics: RigidBodyKinematics, /) -> None:
        if not isinstance(kinematics, RigidBodyKinematics):
            raise TypeError("rigid_kinematics must be RigidBodyKinematics.")
        expected_vector = (self.bodies.capacity, self.ambient_dimension)
        if (
            kinematics.position.shape != expected_vector
            or kinematics.velocity.shape != expected_vector
        ):
            raise ValueError("Rigid position/velocity shapes do not match the coupling.")
        if kinematics.orientation.shape != (
            self.bodies.capacity,
            self.bodies.orientation_dimension,
        ) or kinematics.angular_velocity.shape != (
            self.bodies.capacity,
            self.bodies.angular_dimension,
        ):
            raise ValueError("Rigid orientation/angular-velocity shapes do not match.")

    def _certificate(
        self,
        route_state,
        mpm_state: MPMRuntimeState,
        coupling_state: RigidMPMCouplingState,
        /,
    ) -> RigidMPMRouteCacheCertificate:
        digest = _route_digest(route_state)
        dtype = mpm_state.particles.position.dtype
        source = jnp.arange(route_state.stencil.indices.shape[0], dtype=dtype)[:, None]
        slot = jnp.arange(route_state.stencil.indices.shape[1], dtype=dtype)[None, :]
        weight_checksum = jnp.sum(
            jnp.where(
                route_state.stencil.valid,
                route_state.stencil.weights.astype(dtype),
                0.0,
            )
            * (source + 1.0)
            * (slot + 1.0)
        )
        domain_margin = jnp.where(
            jnp.isfinite(route_state.minimum_domain_margin),
            route_state.minimum_domain_margin,
            jnp.zeros((), dtype=dtype),
        )
        minimum_weight = jnp.where(
            jnp.isfinite(route_state.minimum_route_weight),
            route_state.minimum_route_weight,
            jnp.zeros((), dtype=dtype),
        )
        cache_hit = (
            coupling_state.cache_valid
            & (coupling_state.route_digest == digest)
            & (coupling_state.route_weight_checksum == weight_checksum)
        )
        cache_coherent = (~coupling_state.cache_valid) | cache_hit
        finite = (
            jnp.isfinite(weight_checksum)
            & jnp.isfinite(domain_margin)
            & jnp.isfinite(minimum_weight)
        )
        successful = route_state.successful & finite
        return RigidMPMRouteCacheCertificate(
            digest,
            weight_checksum,
            route_state.valid_route_count,
            minimum_weight,
            domain_margin,
            cache_hit,
            cache_coherent,
            mpm_state.accepted_step,
            finite,
            successful,
            self.certificate_id,
        )

    def evaluate(
        self,
        mpm_state: MPMRuntimeState,
        rigid_kinematics: RigidBodyKinematics,
        coupling_state: RigidMPMCouplingState,
        step_size: ArrayLike,
        /,
    ) -> RigidMPMCouplingEvaluation:
        if not isinstance(mpm_state, MPMRuntimeState):
            raise TypeError("mpm_state must be MPMRuntimeState.")
        self._require_kinematics(rigid_kinematics)
        self._require_state(coupling_state)
        position = mpm_state.particles.position
        dtype = position.dtype
        dt = jnp.asarray(step_size, dtype=dtype).reshape(())
        routes = self.dynamics.splat.build(position)
        certificate = self._certificate(routes, mpm_state, coupling_state)
        plan = self.plan
        safe_particle = jnp.clip(
            plan.particle_indices, 0, self.dynamics.particles.capacity - 1
        )
        safe_body = jnp.clip(plan.body_indices, 0, self.bodies.capacity - 1)
        particle_position = position[safe_particle]
        particle_velocity = mpm_state.particles.velocity[safe_particle]
        body_position = rigid_kinematics.position[safe_body]
        body_velocity = rigid_kinematics.velocity[safe_body]
        body_orientation = rigid_kinematics.orientation[safe_body]
        body_angular_velocity = rigid_kinematics.angular_velocity[safe_body]
        arm = _rotate_local(
            self.ambient_dimension,
            body_orientation,
            plan.local_anchors.astype(dtype),
        )
        anchor = body_position + arm
        anchor_velocity = body_velocity + _point_spin_velocity(
            self.ambient_dimension, body_angular_velocity, arm
        )
        gap_vector = particle_position - anchor
        relative_velocity = particle_velocity - anchor_velocity
        local_normal = _rotate_local(
            self.ambient_dimension,
            body_orientation,
            plan.local_normals.astype(dtype),
        )
        gap_norm = jnp.sqrt(jnp.sum(gap_vector * gap_vector, axis=-1))
        gap_direction = (
            gap_vector
            / jnp.where(gap_norm > plan.geometry_tolerance, gap_norm, 1.0)[:, None]
        )
        normal = jnp.where(
            (gap_norm > plan.geometry_tolerance)[:, None], gap_direction, local_normal
        )
        signed_gap = jnp.sum(gap_vector * local_normal, axis=-1)
        normal_velocity = jnp.sum(relative_velocity * local_normal, axis=-1)

        weld = plan.modes == int(RigidMPMCouplingMode.WELD)
        penalty = plan.modes == int(RigidMPMCouplingMode.PENALTY)
        impulse = plan.modes == int(RigidMPMCouplingMode.IMPULSE)
        finite_route = (
            jnp.all(jnp.isfinite(gap_vector), axis=-1)
            & jnp.all(jnp.isfinite(relative_velocity), axis=-1)
            & jnp.all(jnp.isfinite(local_normal), axis=-1)
            & jnp.isfinite(signed_gap)
            & jnp.isfinite(normal_velocity)
        )
        route_valid = (
            plan.active_mask
            & self.dynamics.particles.active_mask[safe_particle]
            & self.bodies.particles.active_mask[safe_body]
            & finite_route
            & certificate.successful
            & jnp.isfinite(dt)
            & (dt > 0.0)
        )
        route_authority = jnp.all((~plan.active_mask) | route_valid)
        penalty_force = (
            -plan.stiffness.astype(dtype)[:, None] * gap_vector
            - plan.damping.astype(dtype)[:, None] * relative_velocity
        )
        activation_margin = (
            jnp.asarray(plan.activation_distance, dtype=dtype) - signed_gap
        )
        payload_valid = route_valid & ((~impulse) | (activation_margin >= 0.0))
        impulse_active = (
            route_valid
            & impulse
            & (activation_margin >= 0.0)
            & ((signed_gap < 0.0) | (normal_velocity < 0.0))
        )
        particle_inverse_mass = 1.0 / self.dynamics.particles.safe_masses[
            safe_particle
        ].astype(dtype)
        body_inverse_mass = self.bodies.inverse_masses[safe_body].astype(dtype)
        if self.ambient_dimension == 2:
            rotational_direction = (
                arm[:, 0] * local_normal[:, 1] - arm[:, 1] * local_normal[:, 0]
            )
            rotational_inverse_mass = (
                rotational_direction
                * rotational_direction
                * self.bodies.inverse_inertia_body[safe_body].astype(dtype)
            )
        else:
            rotation = quaternion_rotation_matrix(body_orientation)
            inverse_body = self.bodies.inverse_inertia_body[safe_body].astype(dtype)
            inverse_world = contract("rij,rjk,rlk->ril", rotation, inverse_body, rotation)
            rotational_direction = jnp.cross(arm, local_normal)
            rotational_inverse_mass = contract(
                "ri,rij,rj->r", rotational_direction, inverse_world, rotational_direction
            )
        effective_inverse_mass = (
            particle_inverse_mass + body_inverse_mass + rotational_inverse_mass
        )
        safe_inverse_mass = jnp.where(
            effective_inverse_mass > 0.0, effective_inverse_mass, 1.0
        )
        separating_velocity = -(
            1.0 + plan.restitution.astype(dtype)
        ) * normal_velocity - jnp.asarray(
            plan.baumgarte_factor, dtype=dtype
        ) * jnp.minimum(signed_gap, 0.0) / jnp.where(dt > 0.0, dt, 1.0)
        impulse_magnitude = jnp.where(
            impulse_active,
            jnp.maximum(separating_velocity, 0.0) / safe_inverse_mass,
            0.0,
        )
        route_impulse = impulse_magnitude[:, None] * local_normal
        impulse_force = route_impulse / jnp.where(dt > 0.0, dt, 1.0)
        route_force = jnp.where(
            (route_valid & penalty)[:, None],
            penalty_force,
            jnp.where(impulse_active[:, None], impulse_force, 0.0),
        )
        route_impulse = jnp.where(impulse_active[:, None], route_impulse, 0.0)

        particle_force = jnp.zeros_like(position).at[safe_particle].add(route_force)
        body_route_force = -route_force
        rigid_force = (
            jnp.zeros_like(rigid_kinematics.position).at[safe_body].add(body_route_force)
        )
        reaction_arm = particle_position - body_position
        route_torque = _point_torque(
            self.ambient_dimension, reaction_arm, body_route_force
        )
        rigid_torque = (
            jnp.zeros_like(rigid_kinematics.angular_velocity)
            .at[safe_body]
            .add(route_torque)
        )
        stencil_indices = routes.stencil.indices[safe_particle]
        stencil_weights = routes.stencil.weights[safe_particle].astype(dtype)
        stencil_valid = routes.stencil.valid[safe_particle] & route_valid[:, None]
        payload_weights = jnp.where(stencil_valid, stencil_weights, 0.0)
        particle_grid_payload = jnp.where(
            routes.stencil.valid[:, :, None],
            routes.stencil.weights.astype(dtype)[:, :, None] * particle_force[:, None, :],
            0.0,
        )

        def scatter_grid_force(_):
            result = self.dynamics.splat.scatter_route_payload(
                routes, particle_grid_payload
            )
            return result.values, result.successful

        def reject_grid_force(_):
            output_dtype = self.dynamics.splat.plan.precision.output_dtype
            return (
                jnp.zeros(
                    self.dynamics.splat.target_shape + (self.ambient_dimension,),
                    dtype=output_dtype,
                ),
                jnp.asarray(False),
            )

        grid_force, grid_scatter_successful = jax.lax.cond(
            routes.successful,
            scatter_grid_force,
            reject_grid_force,
            operand=None,
        )
        flat_grid_force = grid_force.reshape(
            (self.dynamics.splat.target_size, self.ambient_dimension)
        )
        particle_total = jnp.sum(particle_force, axis=0)
        grid_total = jnp.sum(flat_grid_force, axis=0)
        rigid_total = jnp.sum(rigid_force, axis=0)
        action_reaction_residual = particle_total + rigid_total
        particle_moment = jnp.sum(
            _point_torque(self.ambient_dimension, position, particle_force), axis=0
        )
        rigid_moment = jnp.sum(
            _point_torque(self.ambient_dimension, rigid_kinematics.position, rigid_force)
            + rigid_torque,
            axis=0,
        )
        angular_action_reaction_residual = particle_moment + rigid_moment
        grid_scatter_residual = grid_total - particle_total
        force_scale = jnp.maximum(
            1.0,
            jnp.maximum(
                jnp.linalg.norm(particle_total),
                jnp.maximum(jnp.linalg.norm(grid_total), jnp.linalg.norm(rigid_total)),
            ),
        )
        balance_factor = jnp.finfo(dtype).eps * max(
            64, 8 * self.route_count * self.dynamics.splat.route_width
        )
        balance_tolerance = balance_factor * force_scale
        moment_scale = jnp.maximum(
            1.0,
            jnp.maximum(jnp.linalg.norm(particle_moment), jnp.linalg.norm(rigid_moment)),
        )
        moment_tolerance = balance_factor * moment_scale
        action_reaction_valid = (
            jnp.linalg.norm(action_reaction_residual) <= balance_tolerance
        ) & (jnp.linalg.norm(angular_action_reaction_residual) <= moment_tolerance)
        grid_scatter_valid = grid_scatter_successful & (
            jnp.linalg.norm(grid_scatter_residual) <= balance_tolerance
        )
        selected_weight_margin = jnp.min(
            jnp.where(
                routes.stencil.valid[safe_particle],
                stencil_weights,
                jnp.finfo(dtype).max,
            ),
            axis=-1,
        )
        selected_weight_margin = jnp.where(
            jnp.isfinite(selected_weight_margin), selected_weight_margin, 0.0
        )
        validity_margin = jnp.where(
            payload_valid,
            jnp.where(
                impulse, jnp.maximum(activation_margin, 0.0), selected_weight_margin
            ),
            0.0,
        )
        feature_margin = jnp.where(
            payload_valid,
            jnp.where(
                impulse,
                jnp.minimum(
                    jnp.abs(activation_margin),
                    jnp.abs(normal_velocity),
                ),
                selected_weight_margin,
            ),
            0.0,
        )
        payload_gap = jnp.where(impulse, signed_gap, gap_norm)
        hard = payload_valid & weld
        unilateral = payload_valid & impulse
        payload_finite = (
            jnp.all(jnp.isfinite(payload_weights))
            & jnp.all(jnp.isfinite(jnp.where(plan.active_mask[:, None], gap_vector, 0.0)))
            & jnp.all(jnp.isfinite(jnp.where(plan.active_mask, payload_gap, 0.0)))
            & jnp.all(jnp.isfinite(jnp.where(plan.active_mask[:, None], normal, 0.0)))
            & jnp.all(
                jnp.isfinite(jnp.where(plan.active_mask[:, None], relative_velocity, 0.0))
            )
            & jnp.all(jnp.isfinite(jnp.where(plan.active_mask, normal_velocity, 0.0)))
            & jnp.all(jnp.isfinite(validity_margin))
            & jnp.all(jnp.isfinite(feature_margin))
        )
        payload = RigidMPMConstraintPayload(
            plan.particle_indices,
            plan.body_indices,
            stencil_indices,
            payload_weights,
            jnp.where(route_valid[:, None], gap_vector, 0.0),
            jnp.where(route_valid, payload_gap, 0.0),
            jnp.where(
                route_valid[:, None],
                jnp.where(impulse[:, None], local_normal, normal),
                0.0,
            ),
            jnp.where(route_valid[:, None], relative_velocity, 0.0),
            jnp.where(route_valid, normal_velocity, 0.0),
            plan.coupling_keys,
            hard,
            unilateral,
            payload_valid,
            validity_margin,
            feature_margin,
            payload_finite,
            certificate.successful & route_authority & payload_finite,
            self.prepared_id,
        )
        rigid_load = RigidBodyLoad(rigid_force, rigid_torque)
        finite = (
            payload_finite
            & jnp.all(jnp.isfinite(route_force))
            & jnp.all(jnp.isfinite(route_impulse))
            & jnp.all(jnp.isfinite(particle_force))
            & jnp.all(jnp.isfinite(grid_force))
            & tree_allfinite(rigid_load)
            & jnp.all(jnp.isfinite(action_reaction_residual))
            & jnp.all(jnp.isfinite(angular_action_reaction_residual))
            & jnp.all(jnp.isfinite(grid_scatter_residual))
        )
        successful = (
            payload.successful & finite & action_reaction_valid & grid_scatter_valid
        )
        return RigidMPMCouplingEvaluation(
            payload,
            route_force,
            route_impulse,
            particle_force,
            grid_force,
            rigid_load,
            action_reaction_residual,
            angular_action_reaction_residual,
            grid_scatter_residual,
            action_reaction_valid,
            grid_scatter_valid,
            certificate,
            finite,
            successful,
            self.prepared_id,
        )

    def step_detailed(
        self,
        coupling_state: RigidMPMCouplingState,
        mpm_state: MPMRuntimeState,
        rigid_kinematics: RigidBodyKinematics,
        step_size: ArrayLike,
        arguments: Any,
        /,
    ) -> RigidMPMCouplingStepResult:
        """Advance MPM authority, then transactionally refresh coupling state/output."""
        self._require_state(coupling_state)
        self._require_kinematics(rigid_kinematics)
        if not isinstance(mpm_state, MPMRuntimeState):
            raise TypeError("mpm_state must be MPMRuntimeState.")
        mpm_result = self.dynamics.step_detailed(mpm_state, step_size, arguments)
        evaluation = self.evaluate(
            mpm_result.candidate_state,
            rigid_kinematics,
            coupling_state,
            step_size,
        )
        candidate = RigidMPMCouplingState(
            coupling_state.accumulated_impulse + evaluation.route_impulse,
            evaluation.certificate.route_digest,
            evaluation.certificate.route_weight_checksum,
            coupling_state.cache_generation + 1,
            coupling_state.accepted_step + 1,
            evaluation.certificate.successful,
            evaluation.successful,
            self.prepared_id,
        )
        successful = mpm_result.successful & evaluation.successful
        accepted = tree_where(successful, candidate, coupling_state)
        coupling_failure = mpm_result.successful & ~evaluation.successful
        accepted_mpm = tree_where(coupling_failure, mpm_state, mpm_result.accepted_state)
        stability_margin_valid = jnp.isfinite(mpm_result.stability_margin)
        stability_margin = jnp.where(
            stability_margin_valid,
            mpm_result.stability_margin,
            jnp.zeros((), dtype=mpm_state.particles.position.dtype),
        )
        route_margin = jnp.min(
            jnp.where(
                evaluation.payload.valid,
                evaluation.payload.validity_margin,
                jnp.finfo(mpm_state.particles.position.dtype).max,
            )
        )
        route_margin_valid = jnp.any(evaluation.payload.valid)
        route_margin = jnp.where(jnp.any(evaluation.payload.valid), route_margin, 0.0)
        return RigidMPMCouplingStepResult(
            candidate,
            accepted,
            mpm_result.candidate_state,
            accepted_mpm,
            mpm_result,
            evaluation,
            successful,
            stability_margin,
            stability_margin_valid,
            route_margin,
            route_margin_valid,
            self.prepared_id,
        )


__all__ = [
    "PreparedRigidMPMCoupling",
    "RigidMPMConstraintPayload",
    "RigidMPMCouplingEvaluation",
    "RigidMPMCouplingMode",
    "RigidMPMCouplingPlan",
    "RigidMPMCouplingState",
    "RigidMPMCouplingStepResult",
    "RigidMPMRouteCacheCertificate",
]
