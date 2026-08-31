#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._tree_math import tree_allfinite, tree_where
from ...linalg import (
    DenseLinearOperator,
    DenseLU,
    LinearSolvePolicy,
    LinearSystem,
    solve as solve_linear,
)
from ._rigid_body import (
    PreparedRigidBodySet,
    rigid_body_world_inertia,
    RigidBodyKinematics,
    RigidBodyLoad,
)
from ._rigid_contact import RigidContactGeometry
from ._rigid_sphere import sphere_lever_torque, sphere_spin_velocity
from ._rigid_unilateral import (
    FixedCapacityUnilateralPlan,
    PreparedUnilateralRows,
    UnilateralEvaluation,
    UnilateralState,
)


class CoulombConeProjection(StrictModule):
    """Exact Euclidean projection onto ``||t|| <= mu*n, n >= 0``."""

    normal_impulse: Array
    tangent_impulse: Array
    trial_tangent_norm: Array
    cone_radius: Array
    inside_cone: Array
    polar_region: Array
    boundary: Array
    generalized_jacobian: Array
    branch_margin: Array
    finite: Array
    successful: Array


class FrictionBallProjection(StrictModule):
    """Exact fixed-normal projection and an explicit generalized derivative."""

    tangent_impulse: Array
    trial_norm: Array
    cone_radius: Array
    sticking: Array
    sliding: Array
    radial_scale: Array
    derivative_tangent: Array
    derivative_normal: Array
    branch_margin: Array
    finite: Array
    successful: Array


def _broadcast_contact_parameter(value, shape, dtype, /):
    array = jnp.asarray(value, dtype=dtype)
    if array.ndim == 0:
        return jnp.broadcast_to(array, shape)
    if array.shape != shape:
        raise ValueError("Contact parameter must be scalar or match the batch shape.")
    return array


def project_isotropic_coulomb_impulse(
    trial_normal: ArrayLike,
    trial_tangent: ArrayLike,
    friction_coefficient: ArrayLike,
    /,
) -> CoulombConeProjection:
    """Project planar wedges or spatial impulses onto an isotropic cone.

    One tangent coordinate gives the exact planar polyhedral wedge. Two tangent
    coordinates give the rotationally invariant second-order cone. The selected
    Clarke derivative is returned explicitly; no component clipping is used.
    """

    tangent = jnp.asarray(trial_tangent)
    normal = jnp.asarray(trial_normal, dtype=tangent.dtype)
    if tangent.ndim < 1 or normal.shape != tangent.shape[:-1]:
        raise ValueError("trial_normal must match trial_tangent batch dimensions.")
    tangent_dimension = int(tangent.shape[-1])
    if tangent_dimension not in (1, 2):
        raise ValueError("Coulomb tangent coordinates must have dimension one or two.")
    coefficient = _broadcast_contact_parameter(
        friction_coefficient, normal.shape, tangent.dtype
    )
    finite_input = (
        jnp.isfinite(normal)
        & jnp.isfinite(coefficient)
        & (coefficient >= 0.0)
        & jnp.all(jnp.isfinite(tangent), axis=-1)
    )
    tangent_norm = jnp.sqrt(jnp.sum(tangent * tangent, axis=-1))
    safe_norm = jnp.where(tangent_norm > 0.0, tangent_norm, 1.0)
    direction = tangent / safe_norm[..., None]
    zero_friction = coefficient == 0.0
    inside = (
        finite_input
        & ~zero_friction
        & (normal >= 0.0)
        & (tangent_norm <= coefficient * normal)
    )
    polar = finite_input & ~zero_friction & (normal + coefficient * tangent_norm <= 0.0)
    boundary = finite_input & ~(zero_friction | inside | polar)
    denominator = 1.0 + coefficient * coefficient
    boundary_normal = (normal + coefficient * tangent_norm) / denominator
    boundary_tangent = (coefficient * boundary_normal)[..., None] * direction
    projected_normal = jnp.where(
        zero_friction,
        jnp.maximum(normal, 0.0),
        jnp.where(inside, normal, jnp.where(polar, 0.0, boundary_normal)),
    )
    projected_tangent = jnp.where(
        zero_friction[..., None],
        0.0,
        jnp.where(
            inside[..., None],
            tangent,
            jnp.where(polar[..., None], 0.0, boundary_tangent),
        ),
    )

    event_dimension = tangent_dimension + 1
    tangent_identity = jnp.broadcast_to(
        jnp.eye(tangent_dimension, dtype=tangent.dtype),
        tangent.shape[:-1] + (tangent_dimension, tangent_dimension),
    )
    identity = jnp.broadcast_to(
        jnp.eye(event_dimension, dtype=tangent.dtype),
        tangent.shape[:-1] + (event_dimension, event_dimension),
    )
    zero_jacobian = jnp.zeros_like(identity)
    outer = direction[..., :, None] * direction[..., None, :]
    dn_dn = 1.0 / denominator
    dn_dt = coefficient[..., None] / denominator[..., None] * direction
    transverse_scale = jnp.where(
        tangent_norm > 0.0,
        coefficient * boundary_normal / safe_norm,
        0.0,
    )
    dt_dt = coefficient[..., None, None] ** 2 / denominator[
        ..., None, None
    ] * outer + transverse_scale[..., None, None] * (tangent_identity - outer)
    boundary_jacobian = jnp.zeros_like(identity)
    boundary_jacobian = boundary_jacobian.at[..., 0, 0].set(dn_dn)
    boundary_jacobian = boundary_jacobian.at[..., 0, 1:].set(dn_dt)
    boundary_jacobian = boundary_jacobian.at[..., 1:, 0].set(dn_dt)
    boundary_jacobian = boundary_jacobian.at[..., 1:, 1:].set(dt_dt)
    zero_mu_jacobian = (
        jnp.zeros_like(identity).at[..., 0, 0].set((normal > 0.0).astype(tangent.dtype))
    )
    generalized = jnp.where(
        zero_friction[..., None, None],
        zero_mu_jacobian,
        jnp.where(
            inside[..., None, None],
            identity,
            jnp.where(polar[..., None, None], zero_jacobian, boundary_jacobian),
        ),
    )
    inside_margin = coefficient * normal - tangent_norm
    polar_margin = -(normal + coefficient * tangent_norm)
    boundary_margin = jnp.minimum(
        normal + coefficient * tangent_norm,
        tangent_norm - coefficient * normal,
    )
    branch_margin = jnp.where(
        zero_friction,
        jnp.abs(normal),
        jnp.where(inside, inside_margin, jnp.where(polar, polar_margin, boundary_margin)),
    )
    cone_radius = coefficient * projected_normal
    finite_output = (
        finite_input
        & jnp.isfinite(projected_normal)
        & jnp.all(jnp.isfinite(projected_tangent), axis=-1)
        & jnp.all(jnp.isfinite(generalized), axis=(-2, -1))
    )
    successful = (
        finite_output
        & (projected_normal >= 0.0)
        & (
            jnp.sqrt(jnp.sum(projected_tangent * projected_tangent, axis=-1))
            <= cone_radius + 8.0 * jnp.finfo(tangent.dtype).eps
        )
    )
    return CoulombConeProjection(
        projected_normal,
        projected_tangent,
        tangent_norm,
        cone_radius,
        inside | (zero_friction & (normal >= 0.0) & (tangent_norm == 0.0)),
        polar | (zero_friction & (normal < 0.0)),
        boundary,
        generalized,
        branch_margin,
        finite_output,
        successful,
    )


def project_friction_ball(
    normal_impulse: ArrayLike,
    trial_tangent: ArrayLike,
    friction_coefficient: ArrayLike,
    /,
) -> FrictionBallProjection:
    """Project onto the exact fixed-normal Coulomb disk or interval."""

    tangent = jnp.asarray(trial_tangent)
    normal = jnp.asarray(normal_impulse, dtype=tangent.dtype)
    if tangent.ndim < 1 or normal.shape != tangent.shape[:-1]:
        raise ValueError("normal_impulse must match trial_tangent batch dimensions.")
    coefficient = _broadcast_contact_parameter(
        friction_coefficient, normal.shape, tangent.dtype
    )
    norm = jnp.sqrt(jnp.sum(tangent * tangent, axis=-1))
    radius = coefficient * jnp.maximum(normal, 0.0)
    safe_norm = jnp.where(norm > 0.0, norm, 1.0)
    scale = jnp.minimum(1.0, radius / safe_norm)
    projected = scale[..., None] * tangent
    sticking = norm <= radius
    sliding = ~sticking
    direction = tangent / safe_norm[..., None]
    tangent_dimension = int(tangent.shape[-1])
    identity = jnp.broadcast_to(
        jnp.eye(tangent_dimension, dtype=tangent.dtype),
        tangent.shape[:-1] + (tangent_dimension, tangent_dimension),
    )
    outer = direction[..., :, None] * direction[..., None, :]
    sliding_derivative = scale[..., None, None] * (identity - outer)
    derivative_tangent = jnp.where(
        sticking[..., None, None], identity, sliding_derivative
    )
    derivative_normal = jnp.where(
        sliding[..., None] & (normal[..., None] > 0.0),
        coefficient[..., None] * direction,
        0.0,
    )
    finite = (
        jnp.isfinite(normal)
        & (normal >= 0.0)
        & jnp.isfinite(coefficient)
        & (coefficient >= 0.0)
        & jnp.all(jnp.isfinite(tangent), axis=-1)
        & jnp.all(jnp.isfinite(projected), axis=-1)
        & jnp.all(jnp.isfinite(derivative_tangent), axis=(-2, -1))
    )
    projected_norm = jnp.sqrt(jnp.sum(projected * projected, axis=-1))
    successful = finite & (projected_norm <= radius + 8.0 * jnp.finfo(tangent.dtype).eps)
    return FrictionBallProjection(
        projected,
        norm,
        radius,
        sticking,
        sliding,
        scale,
        derivative_tangent,
        derivative_normal,
        jnp.abs(radius - norm),
        finite,
        successful,
    )


class HardContactRoutePlan(StrictModule, NonTrainableState):
    """Fixed contact routes, material response, and hard-solve policy."""

    left_body: Array
    right_body: Array
    route_keys: Array
    valid: Array
    friction_coefficient: Array
    restitution_coefficient: Array
    normal_rows: FixedCapacityUnilateralPlan
    activation_distance: float = eqx.field(static=True)
    impact_velocity: float = eqx.field(static=True)
    release_velocity: float = eqx.field(static=True)
    position_stabilization: float = eqx.field(static=True)
    geometry_tolerance: float = eqx.field(static=True)
    energy_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        left_body: ArrayLike,
        right_body: ArrayLike,
        route_keys: ArrayLike,
        /,
        *,
        valid: ArrayLike | None = None,
        friction_coefficient: ArrayLike = 0.0,
        restitution_coefficient: ArrayLike = 0.0,
        activation_distance: float = 1.0e-8,
        impact_velocity: float = 1.0e-6,
        release_velocity: float = 1.0e-10,
        position_stabilization: float = 0.2,
        complementarity_tolerance: float = 1.0e-8,
        geometry_tolerance: float = 1.0e-8,
        energy_tolerance: float = 1.0e-9,
        plan_id: str | None = None,
    ):
        left = np.asarray(left_body)
        right = np.asarray(right_body)
        keys = np.asarray(route_keys)
        if (
            left.ndim != 1
            or left.size == 0
            or right.shape != left.shape
            or not np.issubdtype(left.dtype, np.integer)
            or not np.issubdtype(right.dtype, np.integer)
        ):
            raise TypeError(
                "left_body/right_body must be nonempty rank-1 integer arrays."
            )
        if (
            keys.ndim not in (1, 2)
            or keys.shape[0] != left.size
            or not np.issubdtype(keys.dtype, np.integer)
        ):
            raise TypeError("route_keys must be an integer array with contact capacity.")
        valid_ = (
            np.ones(left.shape, dtype=bool)
            if valid is None
            else np.asarray(valid, dtype=bool)
        )
        if valid_.shape != left.shape:
            raise ValueError("valid must have contact-capacity shape.")
        if np.any(left < -1) or np.any(right < -1):
            raise ValueError("Body routes may use only -1 for a world endpoint.")
        if np.any(valid_ & (left == -1) & (right == -1)):
            raise ValueError("A valid contact route must have at least one rigid body.")
        valid_count = int(np.count_nonzero(valid_))
        if valid_count:
            key_rows = keys[valid_].reshape((valid_count, -1))
            if np.unique(key_rows, axis=0).shape[0] != valid_count:
                raise ValueError("Valid hard-contact route keys must be unique.")
        friction = np.asarray(friction_coefficient)
        restitution = np.asarray(restitution_coefficient)
        if friction.ndim == 0:
            friction = np.full(left.shape, friction, dtype=friction.dtype)
        if restitution.ndim == 0:
            restitution = np.full(left.shape, restitution, dtype=restitution.dtype)
        if friction.shape != left.shape or restitution.shape != left.shape:
            raise ValueError("Contact coefficients must be scalar or capacity-shaped.")
        if np.any(~np.isfinite(friction)) or np.any(friction < 0.0):
            raise ValueError("Friction coefficients must be finite and nonnegative.")
        if (
            np.any(~np.isfinite(restitution))
            or np.any(restitution < 0.0)
            or np.any(restitution > 1.0)
        ):
            raise ValueError("Restitution coefficients must lie in [0, 1].")
        scales = tuple(
            float(value)
            for value in (
                activation_distance,
                impact_velocity,
                release_velocity,
                complementarity_tolerance,
                geometry_tolerance,
                energy_tolerance,
            )
        )
        stabilization = float(position_stabilization)
        if any(not isfinite(value) or value < 0.0 for value in scales):
            raise ValueError("Hard-contact tolerances must be finite and nonnegative.")
        if scales[1] <= 0.0 or scales[3] <= 0.0 or scales[4] <= 0.0 or scales[5] <= 0.0:
            raise ValueError(
                "Impact, complementarity, geometry, and energy scales must be positive."
            )
        if not isfinite(stabilization) or stabilization < 0.0 or stabilization > 1.0:
            raise ValueError("position_stabilization must lie in [0, 1].")
        normal_rows = FixedCapacityUnilateralPlan(
            np.arange(left.size, dtype=np.int32),
            valid=valid_,
            complementarity_tolerance=scales[3],
            plan_id=canonical_fingerprint(
                {
                    "kind": "hard-contact-normal-rows",
                    "routes": array_tree_fingerprint(keys),
                    "valid": valid_.tolist(),
                }
            ),
        )
        generated = canonical_fingerprint(
            {
                "kind": "hard-contact-route-plan",
                "topology": array_tree_fingerprint(
                    {"left": left, "right": right, "keys": keys, "valid": valid_}
                ),
                "response": array_tree_fingerprint(
                    {"friction": friction, "restitution": restitution}
                ),
                "normal_rows": normal_rows.plan_id,
                "scales": scales + (stabilization,),
            }
        )
        identifier = generated if plan_id is None else str(plan_id)
        if not identifier:
            raise ValueError("plan_id must be nonempty.")
        self.left_body = jnp.asarray(left, dtype=jnp.int32)
        self.right_body = jnp.asarray(right, dtype=jnp.int32)
        self.route_keys = jnp.asarray(keys)
        self.valid = jnp.asarray(valid_)
        self.friction_coefficient = jnp.asarray(friction)
        self.restitution_coefficient = jnp.asarray(restitution)
        self.normal_rows = normal_rows
        (
            self.activation_distance,
            self.impact_velocity,
            self.release_velocity,
            _,
            self.geometry_tolerance,
            self.energy_tolerance,
        ) = scales
        self.position_stabilization = stabilization
        self.plan_id = identifier

    @property
    def capacity(self) -> int:
        return int(self.left_body.shape[0])

    def prepare(self, bodies: PreparedRigidBodySet, /) -> PreparedHardContact:
        return PreparedHardContact(self, bodies)


class HardContactState(StrictModule):
    normal_impulse: Array
    tangent_impulse: Array
    active: Array
    impacting: Array
    numeric_version: Array


class RestitutionClassification(StrictModule):
    newly_active: Array
    approaching: Array
    impacting: Array
    resting: Array
    separating: Array
    target_velocity: Array
    impact_margin: Array
    resting_margin: Array


class HardContactCertificate(StrictModule):
    position_gap_violation: Array
    position_complementarity: Array
    velocity_primal_violation: Array
    velocity_complementarity: Array
    cone_violation: Array
    tangency_defect: Array
    finite: Array
    position_certified: Array
    velocity_certified: Array
    friction_certified: Array
    certified: Array


class HardContactEnergy(StrictModule):
    kinetic_before: Array
    kinetic_after: Array
    kinetic_change: Array
    normal_work: Array
    friction_dissipation: Array
    stabilization_work: Array
    energy_margin: Array
    finite: Array
    noncreating: Array


class HardContactEvaluation(StrictModule):
    normal_impulse: Array
    tangent_impulse: Array
    total_impulse: Array
    body_impulse: RigidBodyLoad
    normal_velocity_before: Array
    normal_velocity_after: Array
    tangential_velocity_before: Array
    tangential_velocity_after: Array
    active: Array
    impacting: Array
    sticking: Array
    sliding: Array
    restitution: RestitutionClassification
    friction_projection: FrictionBallProjection
    normal_unilateral: UnilateralEvaluation
    certificate: HardContactCertificate
    energy: HardContactEnergy
    active_branch_margin: Array
    impact_branch_margin: Array
    stick_branch_margin: Array
    geometry_valid: Array
    routes_match: Array
    corrected_kinematics: RigidBodyKinematics
    successful: Array
    prepared_id: str = eqx.field(static=True)


class HardContactStepResult(StrictModule):
    candidate_state: HardContactState
    accepted_state: HardContactState
    candidate_kinematics: RigidBodyKinematics
    accepted_kinematics: RigidBodyKinematics
    evaluation: HardContactEvaluation
    successful: Array


class _ImpulseResponse(StrictModule):
    linear_velocity: Array
    angular_velocity: Array
    relative_velocity: Array
    linear_impulse: Array
    angular_impulse: Array


class PreparedHardContact(StrictModule, NonTrainableState):
    plan: HardContactRoutePlan
    bodies: PreparedRigidBodySet
    left_index: Array
    right_index: Array
    left_present: Array
    right_present: Array
    normal_rows: PreparedUnilateralRows
    tangent_policy: LinearSolvePolicy
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: HardContactRoutePlan, bodies: PreparedRigidBodySet, /):
        if not isinstance(plan, HardContactRoutePlan):
            raise TypeError("plan must be a HardContactRoutePlan.")
        if not isinstance(bodies, PreparedRigidBodySet):
            raise TypeError("bodies must be a PreparedRigidBodySet.")
        left = np.asarray(plan.left_body)
        right = np.asarray(plan.right_body)
        valid = np.asarray(plan.valid)
        capacity = bodies.capacity
        left_present = left >= 0
        right_present = right >= 0
        if np.any(left_present & (left >= capacity)) or np.any(
            right_present & (right >= capacity)
        ):
            raise ValueError("Hard-contact body route exceeds rigid-body capacity.")
        active_bodies = np.asarray(bodies.particles.active_mask, dtype=bool)
        safe_left = np.where(left_present, left, 0)
        safe_right = np.where(right_present, right, 0)
        endpoints_active = (~left_present | active_bodies[safe_left]) & (
            ~right_present | active_bodies[safe_right]
        )
        if np.any(valid & ~endpoints_active):
            raise ValueError("Valid hard-contact routes must reference active bodies.")
        mobile = active_bodies & ~np.asarray(bodies.fixed_mask, dtype=bool)
        responds = (left_present & mobile[safe_left]) | (
            right_present & mobile[safe_right]
        )
        if np.any(valid & ~responds):
            raise ValueError("Every valid hard-contact route needs a mobile endpoint.")
        prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-hard-contact",
                "plan": plan.plan_id,
                "bodies": bodies.prepared_id,
            }
        )
        self.plan = plan
        self.bodies = bodies
        self.left_index = jnp.asarray(safe_left, dtype=jnp.int32)
        self.right_index = jnp.asarray(safe_right, dtype=jnp.int32)
        self.left_present = jnp.asarray(left_present)
        self.right_present = jnp.asarray(right_present)
        self.normal_rows = plan.normal_rows.prepare(prepared_scope_id=prepared_id)
        self.tangent_policy = LinearSolvePolicy(DenseLU())
        self.prepared_id = prepared_id

    @property
    def capacity(self) -> int:
        return self.plan.capacity

    @property
    def ambient_dimension(self) -> int:
        return self.bodies.ambient_dimension

    def initial_state(self, /) -> HardContactState:
        dtype = self.bodies.particles.safe_masses.dtype
        return HardContactState(
            jnp.zeros((self.capacity,), dtype=dtype),
            jnp.zeros((self.capacity, self.ambient_dimension), dtype=dtype),
            jnp.zeros((self.capacity,), dtype=bool),
            jnp.zeros((self.capacity,), dtype=bool),
            jnp.asarray(0, dtype=jnp.int32),
        )

    def evaluate(
        self,
        state: HardContactState,
        kinematics: RigidBodyKinematics,
        geometry: RigidContactGeometry,
        time_step: ArrayLike,
        /,
    ) -> HardContactStepResult:
        self._validate_inputs(state, kinematics, geometry)
        dtype = kinematics.position.dtype
        dt = jnp.asarray(time_step, dtype=dtype)
        if dt.ndim != 0:
            raise ValueError("time_step must be scalar.")
        row_finite = self._geometry_row_finite(geometry)
        route_mask = self.plan.valid.reshape(
            (self.capacity,) + (1,) * (self.plan.route_keys.ndim - 1)
        )
        routes_match = jnp.all(
            jnp.where(route_mask, geometry.contact_keys == self.plan.route_keys, True)
        )
        tolerance = jnp.asarray(self.plan.geometry_tolerance, dtype=dtype)
        normal_norm = jnp.sqrt(jnp.sum(geometry.normal * geometry.normal, axis=-1))
        normal_valid = jnp.abs(normal_norm - 1.0) <= tolerance
        reconstructed_normal_velocity = jnp.sum(
            geometry.relative_velocity * geometry.normal, axis=-1
        )
        reconstructed_tangent = (
            geometry.relative_velocity
            - geometry.normal_velocity[:, None] * geometry.normal
        )
        velocity_scale = jnp.maximum(
            jnp.sqrt(jnp.sum(geometry.relative_velocity**2, axis=-1)), 1.0
        )
        normal_velocity_valid = (
            jnp.abs(reconstructed_normal_velocity - geometry.normal_velocity)
            <= tolerance * velocity_scale
        )
        tangent_error = jnp.sqrt(
            jnp.sum(
                (reconstructed_tangent - geometry.tangential_velocity) ** 2,
                axis=-1,
            )
        )
        tangent_orthogonality = jnp.abs(
            jnp.sum(geometry.tangential_velocity * geometry.normal, axis=-1)
        )
        tangent_valid = (tangent_error <= tolerance * velocity_scale) & (
            tangent_orthogonality <= tolerance * velocity_scale
        )
        overlap_valid = jnp.abs(
            geometry.overlap - jnp.maximum(-geometry.gap, 0.0)
        ) <= tolerance * jnp.maximum(jnp.abs(geometry.gap), 1.0)
        expected_left_velocity = (
            kinematics.velocity[self.left_index]
            + sphere_spin_velocity(
                kinematics.angular_velocity[self.left_index],
                geometry.left_owner_arm,
                self.ambient_dimension,
            )
        ) * self.left_present[:, None]
        expected_right_velocity = (
            kinematics.velocity[self.right_index]
            + sphere_spin_velocity(
                kinematics.angular_velocity[self.right_index],
                geometry.right_owner_arm,
                self.ambient_dimension,
            )
        ) * self.right_present[:, None]
        kinematic_error = jnp.sqrt(
            jnp.sum(
                (
                    expected_left_velocity
                    - expected_right_velocity
                    - geometry.relative_velocity
                )
                ** 2,
                axis=-1,
            )
        )
        kinematics_valid = kinematic_error <= tolerance * velocity_scale
        geometry_consistent = (
            normal_valid
            & normal_velocity_valid
            & tangent_valid
            & overlap_valid
            & kinematics_valid
        )
        geometry_valid = (
            self.plan.valid
            & geometry.valid
            & row_finite
            & geometry_consistent
            & routes_match
        )
        geometry_successful = (
            jnp.asarray(geometry.successful, dtype=bool)
            & routes_match
            & jnp.all(
                jnp.where(
                    self.plan.valid & geometry.valid,
                    row_finite & geometry_consistent,
                    True,
                )
            )
        )
        normal = jnp.where(geometry_valid[:, None], geometry.normal, 0.0)
        left_arm = jnp.where(geometry_valid[:, None], geometry.left_owner_arm, 0.0)
        right_arm = jnp.where(geometry_valid[:, None], geometry.right_owner_arm, 0.0)
        gap = jnp.where(geometry_valid, geometry.gap, 0.0)
        normal_velocity = jnp.where(geometry_valid, geometry.normal_velocity, 0.0)
        relative_velocity = jnp.where(
            geometry_valid[:, None], geometry.relative_velocity, 0.0
        )
        activation = jnp.asarray(self.plan.activation_distance, dtype=dtype)
        release = jnp.asarray(self.plan.release_velocity, dtype=dtype)
        active = (
            geometry_valid
            & (gap <= activation)
            & ((normal_velocity < release) | (gap < 0.0) | (state.normal_impulse > 0.0))
        )
        newly_active = active & ~state.active
        approaching = active & (normal_velocity < 0.0)
        impact_threshold = jnp.asarray(self.plan.impact_velocity, dtype=dtype)
        impacting = newly_active & (normal_velocity < -impact_threshold)
        resting = (
            active
            & ~impacting
            & (state.active | (jnp.abs(normal_velocity) <= impact_threshold))
        )
        separating = geometry_valid & (normal_velocity > release)
        restitution_target = jnp.where(
            impacting,
            -self.plan.restitution_coefficient.astype(dtype) * normal_velocity,
            0.0,
        )
        stabilization_target = jnp.where(
            active,
            self.plan.position_stabilization
            * jnp.maximum(-gap, 0.0)
            / jnp.maximum(dt, jnp.finfo(dtype).tiny),
            0.0,
        )
        target_velocity = jnp.maximum(restitution_target, stabilization_target)
        vector_delassus = self._vector_delassus(
            kinematics, left_arm, right_arm, geometry_valid
        )
        normal_map = self._normal_map(normal)
        normal_delassus = contract(
            "ki,kl,lj->ij", normal_map, vector_delassus, normal_map
        )
        normal_free = normal_velocity - target_velocity
        warm_normal = UnilateralState(
            state.normal_impulse, state.active, state.numeric_version
        )
        normal_step = self.normal_rows.evaluate(
            warm_normal, normal_delassus, normal_free, active
        )
        normal_impulse = normal_step.candidate_state.impulses
        normal_world_impulse = normal_impulse[:, None] * normal
        normal_response = self._impulse_response(
            kinematics, normal_world_impulse, left_arm, right_arm, geometry_valid
        )
        relative_after_normal = relative_velocity + normal_response.relative_velocity
        normal_after_normal = jnp.sum(relative_after_normal * normal, axis=-1)
        tangent_after_normal = (
            relative_after_normal - normal_after_normal[:, None] * normal
        )
        tangent_trial, tangent_solve_successful = self._tangent_trial(
            vector_delassus,
            normal,
            tangent_after_normal,
            state.tangent_impulse,
            active,
        )
        tangent_trial = (
            tangent_trial
            - jnp.sum(tangent_trial * normal, axis=-1, keepdims=True) * normal
        )
        friction = project_friction_ball(
            normal_impulse,
            tangent_trial,
            self.plan.friction_coefficient.astype(dtype),
        )
        tangent_impulse = jnp.where(active[:, None], friction.tangent_impulse, 0.0)
        total_impulse = normal_world_impulse + tangent_impulse
        total_response = self._impulse_response(
            kinematics, total_impulse, left_arm, right_arm, geometry_valid
        )
        corrected = RigidBodyKinematics(
            kinematics.position,
            kinematics.velocity + total_response.linear_velocity,
            kinematics.orientation,
            kinematics.angular_velocity + total_response.angular_velocity,
        )
        relative_after = relative_velocity + total_response.relative_velocity
        normal_after = jnp.sum(relative_after * normal, axis=-1)
        tangent_after = relative_after - normal_after[:, None] * normal
        sticking = active & friction.sticking
        sliding = active & friction.sliding
        certificate = self._certificate(
            gap,
            normal_impulse,
            tangent_impulse,
            normal,
            normal_after,
            target_velocity,
            active,
            geometry_valid,
        )
        energy = self._energy(
            kinematics,
            corrected,
            normal_impulse,
            normal_velocity,
            normal_after_normal,
            tangent_impulse,
            tangent_after_normal,
            tangent_after,
            stabilization_target,
        )
        finite = (
            certificate.finite
            & energy.finite
            & tree_allfinite((normal_delassus, vector_delassus, corrected))
            & jnp.isfinite(dt)
        )
        successful = (
            geometry_successful
            & (dt > 0.0)
            & normal_step.successful
            & tangent_solve_successful
            & jnp.all(friction.successful | ~active)
            & certificate.velocity_certified
            & certificate.friction_certified
            & energy.noncreating
            & finite
        )
        candidate = HardContactState(
            normal_impulse,
            tangent_impulse,
            active,
            impacting,
            state.numeric_version + jnp.asarray(1, dtype=jnp.int32),
        )
        accepted = accept_hard_contact_candidate(state, candidate, successful)
        accepted_kinematics = tree_where(successful, corrected, kinematics)
        restitution = RestitutionClassification(
            newly_active,
            approaching,
            impacting,
            resting,
            separating,
            target_velocity,
            -normal_velocity - impact_threshold,
            impact_threshold - jnp.abs(normal_velocity),
        )
        evaluation = HardContactEvaluation(
            normal_impulse,
            tangent_impulse,
            total_impulse,
            RigidBodyLoad(total_response.linear_impulse, total_response.angular_impulse),
            normal_velocity,
            normal_after,
            geometry.tangential_velocity,
            tangent_after,
            active,
            impacting,
            sticking,
            sliding,
            restitution,
            friction,
            normal_step.evaluation,
            certificate,
            energy,
            activation - gap,
            -normal_velocity - impact_threshold,
            friction.branch_margin,
            geometry_valid,
            routes_match,
            corrected,
            successful,
            self.prepared_id,
        )
        return HardContactStepResult(
            candidate,
            accepted,
            corrected,
            accepted_kinematics,
            evaluation,
            successful,
        )

    def _validate_inputs(self, state, kinematics, geometry, /) -> None:
        if not isinstance(state, HardContactState):
            raise TypeError("state must be HardContactState.")
        if not isinstance(kinematics, RigidBodyKinematics):
            raise TypeError("kinematics must be RigidBodyKinematics.")
        if not isinstance(geometry, RigidContactGeometry):
            raise TypeError("geometry must be RigidContactGeometry.")
        contact = (self.capacity,)
        vector = (self.capacity, self.ambient_dimension)
        angular = (self.capacity, self.bodies.angular_dimension)
        if (
            state.normal_impulse.shape != contact
            or state.tangent_impulse.shape != vector
            or state.active.shape != contact
            or state.impacting.shape != contact
            or state.numeric_version.ndim != 0
        ):
            raise ValueError("Hard-contact state does not match prepared capacity.")
        if (
            kinematics.position.shape != (self.bodies.capacity, self.ambient_dimension)
            or kinematics.velocity.shape != kinematics.position.shape
            or kinematics.orientation.shape
            != (self.bodies.capacity, self.bodies.orientation_dimension)
            or kinematics.angular_velocity.shape
            != (self.bodies.capacity, self.bodies.angular_dimension)
        ):
            raise ValueError("Hard-contact kinematics do not match prepared bodies.")
        scalar_fields = (
            geometry.gap,
            geometry.overlap,
            geometry.effective_radius,
            geometry.normal_velocity,
            geometry.left_feature,
            geometry.right_feature,
            geometry.valid,
            geometry.degeneracy_code,
            geometry.feature_margin,
        )
        vector_fields = (
            geometry.normal,
            geometry.contact_point,
            geometry.left_owner_arm,
            geometry.right_owner_arm,
            geometry.left_contact_arm,
            geometry.right_contact_arm,
            geometry.relative_velocity,
            geometry.tangential_velocity,
        )
        if any(value.shape != contact for value in scalar_fields) or any(
            value.shape != vector for value in vector_fields
        ):
            raise ValueError(
                "Rigid contact geometry does not match route capacity/dimension."
            )
        if (
            geometry.left_angular_velocity.shape != angular
            or geometry.right_angular_velocity.shape != angular
            or geometry.contact_keys.shape != self.plan.route_keys.shape
            or jnp.asarray(geometry.successful).ndim != 0
        ):
            raise ValueError("Rigid contact metadata does not match the prepared routes.")

    @staticmethod
    def _geometry_row_finite(geometry: RigidContactGeometry, /) -> Array:
        scalars = jnp.stack(
            (
                geometry.gap,
                geometry.overlap,
                geometry.effective_radius,
                geometry.normal_velocity,
                geometry.feature_margin,
            ),
            axis=-1,
        )
        vectors = jnp.concatenate(
            (
                geometry.normal,
                geometry.contact_point,
                geometry.left_owner_arm,
                geometry.right_owner_arm,
                geometry.relative_velocity,
                geometry.tangential_velocity,
            ),
            axis=-1,
        )
        return jnp.all(jnp.isfinite(scalars), axis=-1) & jnp.all(
            jnp.isfinite(vectors), axis=-1
        )

    def _normal_map(self, normal: Array, /) -> Array:
        row_count = self.capacity * self.ambient_dimension
        mapping = jnp.zeros((row_count, self.capacity), dtype=normal.dtype)
        contacts = jnp.arange(self.capacity, dtype=jnp.int32)
        components = jnp.arange(self.ambient_dimension, dtype=jnp.int32)
        flat_rows = contacts[:, None] * self.ambient_dimension + components[None, :]
        return mapping.at[flat_rows, contacts[:, None]].set(normal)

    def _vector_delassus(self, kinematics, left_arm, right_arm, valid, /):
        size = self.capacity * self.ambient_dimension
        basis = jnp.eye(size, dtype=kinematics.position.dtype).reshape(
            (size, self.capacity, self.ambient_dimension)
        )

        def response(impulse):
            return self._impulse_response(
                kinematics, impulse, left_arm, right_arm, valid
            ).relative_velocity.reshape((-1,))

        return jax.vmap(response)(basis).T

    def _impulse_response(self, kinematics, impulse, left_arm, right_arm, valid, /):
        left_mask = valid & self.left_present
        right_mask = valid & self.right_present
        linear_impulse = jnp.zeros_like(kinematics.velocity)
        linear_impulse = linear_impulse.at[self.left_index].add(
            impulse * left_mask[:, None]
        )
        linear_impulse = linear_impulse.at[self.right_index].add(
            -impulse * right_mask[:, None]
        )
        left_torque = sphere_lever_torque(left_arm, impulse, self.ambient_dimension)
        right_torque = sphere_lever_torque(right_arm, -impulse, self.ambient_dimension)
        angular_impulse = jnp.zeros_like(kinematics.angular_velocity)
        angular_impulse = angular_impulse.at[self.left_index].add(
            left_torque * left_mask[:, None]
        )
        angular_impulse = angular_impulse.at[self.right_index].add(
            right_torque * right_mask[:, None]
        )
        linear_velocity = self.bodies.inverse_masses[:, None] * linear_impulse
        if self.ambient_dimension == 2:
            angular_velocity = self.bodies.inverse_inertia_body[:, None] * angular_impulse
        else:
            _, inverse_world = rigid_body_world_inertia(
                self.bodies, kinematics.orientation
            )
            angular_velocity = contract("bij,bj->bi", inverse_world, angular_impulse)
        left_point = linear_velocity[self.left_index] + sphere_spin_velocity(
            angular_velocity[self.left_index], left_arm, self.ambient_dimension
        )
        right_point = linear_velocity[self.right_index] + sphere_spin_velocity(
            angular_velocity[self.right_index], right_arm, self.ambient_dimension
        )
        relative = left_point * left_mask[:, None] - right_point * right_mask[:, None]
        return _ImpulseResponse(
            linear_velocity,
            angular_velocity,
            relative,
            linear_impulse,
            angular_impulse,
        )

    def _tangent_trial(
        self,
        vector_delassus,
        normal,
        tangent_velocity,
        warm_tangent,
        active,
        /,
    ):
        dimension = self.ambient_dimension
        contact_identity = jnp.eye(self.capacity, dtype=normal.dtype)
        space_identity = jnp.eye(dimension, dtype=normal.dtype)
        projector = space_identity[None] - normal[:, :, None] * normal[:, None, :]
        projected = contract("ij,iab->iajb", contact_identity, projector).reshape(
            (self.capacity * dimension, self.capacity * dimension)
        )
        normal_blocks = contract(
            "ij,ia,ib->iajb", contact_identity, normal, normal
        ).reshape((self.capacity * dimension, self.capacity * dimension))
        inactive_blocks = contract(
            "ij,ab,i->iajb",
            contact_identity,
            space_identity,
            (~active).astype(normal.dtype),
        ).reshape((self.capacity * dimension, self.capacity * dimension))
        active_mask = jnp.repeat(active, dimension).astype(normal.dtype)
        projected = projected * active_mask[:, None] * active_mask[None, :]
        matrix = projected @ vector_delassus @ projected + normal_blocks + inactive_blocks
        warm = projected @ warm_tangent.reshape((-1,))
        right_hand_side = -(
            projected @ (tangent_velocity.reshape((-1,)) + vector_delassus @ warm)
        )
        result = solve_linear(
            LinearSystem(DenseLinearOperator(matrix)),
            right_hand_side,
            policy=self.tangent_policy,
        )
        trial = warm + projected @ result.value
        return trial.reshape((self.capacity, dimension)), result.successful

    def _certificate(
        self,
        gap,
        normal_impulse,
        tangent_impulse,
        normal,
        normal_velocity_after,
        target_velocity,
        active,
        geometry_valid,
        /,
    ):
        tolerance = jnp.asarray(
            self.plan.normal_rows.complementarity_tolerance,
            dtype=normal_impulse.dtype,
        )
        gap_violation = jnp.max(
            jnp.where(geometry_valid, jnp.maximum(-gap, 0.0), 0.0), initial=0.0
        )
        position_product = jnp.max(
            jnp.where(geometry_valid, jnp.abs(gap * normal_impulse), 0.0),
            initial=0.0,
        )
        velocity_slack = normal_velocity_after - target_velocity
        velocity_primal = jnp.max(
            jnp.where(active, jnp.maximum(-velocity_slack, 0.0), 0.0), initial=0.0
        )
        velocity_product = jnp.max(
            jnp.where(active, jnp.abs(normal_impulse * velocity_slack), 0.0),
            initial=0.0,
        )
        tangent_norm = jnp.sqrt(jnp.sum(tangent_impulse * tangent_impulse, axis=-1))
        cone_violation = jnp.max(
            jnp.where(
                active,
                jnp.maximum(
                    tangent_norm
                    - self.plan.friction_coefficient.astype(normal_impulse.dtype)
                    * normal_impulse,
                    0.0,
                ),
                0.0,
            ),
            initial=0.0,
        )
        tangency = jnp.max(
            jnp.where(
                active,
                jnp.abs(jnp.sum(tangent_impulse * normal, axis=-1)),
                0.0,
            ),
            initial=0.0,
        )
        finite = tree_allfinite(
            (gap, normal_impulse, tangent_impulse, normal_velocity_after, target_velocity)
        )
        position_certified = (
            finite & (gap_violation <= tolerance) & (position_product <= tolerance)
        )
        velocity_certified = (
            finite & (velocity_primal <= tolerance) & (velocity_product <= tolerance)
        )
        friction_certified = (
            finite & (cone_violation <= tolerance) & (tangency <= tolerance)
        )
        return HardContactCertificate(
            gap_violation,
            position_product,
            velocity_primal,
            velocity_product,
            cone_violation,
            tangency,
            finite,
            position_certified,
            velocity_certified,
            friction_certified,
            position_certified & velocity_certified & friction_certified,
        )

    def _kinetic_energy(self, kinematics, /):
        mobile = self.bodies.particles.active_mask & ~self.bodies.fixed_mask
        translational = 0.5 * jnp.sum(
            jnp.where(
                mobile,
                self.bodies.particles.safe_masses
                * jnp.sum(kinematics.velocity * kinematics.velocity, axis=-1),
                0.0,
            )
        )
        if self.ambient_dimension == 2:
            rotational_density = self.bodies.inertia_body * jnp.sum(
                kinematics.angular_velocity * kinematics.angular_velocity, axis=-1
            )
        else:
            inertia_world, _ = rigid_body_world_inertia(
                self.bodies, kinematics.orientation
            )
            rotational_density = contract(
                "bi,bij,bj->b",
                kinematics.angular_velocity,
                inertia_world,
                kinematics.angular_velocity,
            )
        rotational = 0.5 * jnp.sum(jnp.where(mobile, rotational_density, 0.0))
        return translational + rotational

    def _energy(
        self,
        before,
        after,
        normal_impulse,
        normal_velocity_before,
        normal_velocity_after_normal,
        tangent_impulse,
        tangent_velocity_before,
        tangent_velocity_after,
        stabilization_target,
        /,
    ):
        kinetic_before = self._kinetic_energy(before)
        kinetic_after = self._kinetic_energy(after)
        kinetic_change = kinetic_after - kinetic_before
        normal_work = jnp.sum(
            0.5 * normal_impulse * (normal_velocity_before + normal_velocity_after_normal)
        )
        friction_work = jnp.sum(
            0.5 * tangent_impulse * (tangent_velocity_before + tangent_velocity_after)
        )
        friction_dissipation = -friction_work
        stabilization_work = jnp.sum(
            normal_impulse * jnp.maximum(stabilization_target, 0.0)
        )
        tolerance = jnp.asarray(self.plan.energy_tolerance, dtype=kinetic_before.dtype)
        allowed_gain = stabilization_work + tolerance * jnp.maximum(kinetic_before, 1.0)
        energy_margin = allowed_gain - kinetic_change
        finite = tree_allfinite(
            (
                kinetic_before,
                kinetic_after,
                normal_work,
                friction_dissipation,
                stabilization_work,
                energy_margin,
            )
        )
        noncreating = (
            finite
            & (kinetic_change <= allowed_gain)
            & (friction_dissipation >= -tolerance * jnp.maximum(kinetic_before, 1.0))
        )
        return HardContactEnergy(
            kinetic_before,
            kinetic_after,
            kinetic_change,
            normal_work,
            friction_dissipation,
            stabilization_work,
            energy_margin,
            finite,
            noncreating,
        )


def hard_contact_candidate(
    state: HardContactState,
    normal_impulse: ArrayLike,
    tangent_impulse: ArrayLike,
    active: ArrayLike,
    impacting: ArrayLike,
    /,
) -> HardContactState:
    """Build a complete versioned hard-contact warm-start candidate."""

    if not isinstance(state, HardContactState):
        raise TypeError("state must be HardContactState.")
    normal = jnp.asarray(normal_impulse, dtype=state.normal_impulse.dtype)
    tangent = jnp.asarray(tangent_impulse, dtype=state.tangent_impulse.dtype)
    active_ = jnp.asarray(active, dtype=bool)
    impacting_ = jnp.asarray(impacting, dtype=bool)
    if (
        normal.shape != state.normal_impulse.shape
        or tangent.shape != state.tangent_impulse.shape
        or active_.shape != state.active.shape
        or impacting_.shape != state.impacting.shape
    ):
        raise ValueError("Hard-contact candidate must preserve all capacities.")
    return HardContactState(
        normal,
        tangent,
        active_,
        impacting_,
        state.numeric_version + jnp.asarray(1, dtype=jnp.int32),
    )


def accept_hard_contact_candidate(
    current: HardContactState,
    candidate: HardContactState,
    accepted: ArrayLike,
    /,
) -> HardContactState:
    """Atomically commit or roll back normal/tangent warm starts and masks."""

    if not isinstance(current, HardContactState) or not isinstance(
        candidate, HardContactState
    ):
        raise TypeError("current and candidate must be HardContactState values.")
    predicate = jnp.asarray(accepted, dtype=bool)
    if predicate.ndim != 0:
        raise ValueError("accepted must be scalar.")
    return tree_where(predicate, candidate, current)


__all__ = [
    "CoulombConeProjection",
    "FrictionBallProjection",
    "HardContactCertificate",
    "HardContactEnergy",
    "HardContactEvaluation",
    "HardContactRoutePlan",
    "HardContactState",
    "HardContactStepResult",
    "PreparedHardContact",
    "RestitutionClassification",
    "accept_hard_contact_candidate",
    "hard_contact_candidate",
    "project_friction_ball",
    "project_isotropic_coulomb_impulse",
]
