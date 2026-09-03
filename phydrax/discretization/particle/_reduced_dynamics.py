#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from enum import IntEnum

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._strict import StrictModule
from ...linalg import (
    DenseCholesky,
    DenseLinearOperator,
    FailurePolicy,
    HermitianSpectrum,
    LinearSolvePolicy,
    LinearSolveStatus,
    LinearSystem,
    OperatorProperties,
    solve,
)
from ._reduced_articulation import (
    ArticulationDualityEvidence,
    PreparedReducedArticulation,
    ReducedArticulationState,
)
from ._rigid_body import RigidBodyLoad
from ._rigid_joints import RigidJointKind


class ReducedDynamicsStatus(IntEnum):
    SUCCESS = 0
    NONFINITE_INPUT = 1
    CAPABILITY_REJECTED = 2
    MASS_NOT_POSITIVE_DEFINITE = 3
    NONFINITE_OUTPUT = 4
    RESIDUAL_TOO_LARGE = 5
    LINEAR_SOLVE_FAILED = 6
    STEP_SIZE_REJECTED = 7
    ENERGY_BOUND_EXCEEDED = 8


class ReducedEnergyResult(StrictModule):
    kinetic: Array
    potential: Array
    total: Array
    finite: Array
    successful: Array
    status: Array


class ReducedMassMatrixResult(StrictModule):
    matrix: Array
    operator: DenseLinearOperator
    symmetry_residual: Array
    minimum_eigenvalue: Array
    finite: Array
    positive_definite: Array
    successful: Array
    status: Array


class ReducedBiasTermsResult(StrictModule):
    velocity: Array
    gravity: Array
    total: Array
    finite: Array
    successful: Array
    status: Array


class ReducedInverseDynamicsResult(StrictModule):
    candidate_effort: Array
    generalized_effort: Array
    mass_effort: Array
    bias_effort: Array
    gravity_effort: Array
    external_effort: Array
    external_duality: ArticulationDualityEvidence
    external_power_residual: Array
    decomposition_residual: Array
    finite: Array
    successful: Array
    status: Array


class ReducedForwardDynamicsResult(StrictModule):
    candidate_acceleration: Array
    acceleration: Array
    reconstructed_effort: Array
    inverse_forward_residual: Array
    relative_inverse_forward_residual: Array
    external_power_residual: Array
    minimum_articulated_inertia: Array
    finite: Array
    successful: Array
    status: Array


class ReducedSemiImplicitVelocityEulerStepPolicy(StrictModule):
    maximum_step_size: float = eqx.field(static=True)
    absolute_energy_tolerance: float = eqx.field(static=True)
    relative_energy_tolerance: float = eqx.field(static=True)
    inverse_forward_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_step_size: float = 0.1,
        absolute_energy_tolerance: float = 1.0e-6,
        relative_energy_tolerance: float = 1.0e-4,
        inverse_forward_tolerance: float = 1.0e-6,
    ):
        maximum = float(maximum_step_size)
        absolute = float(absolute_energy_tolerance)
        relative = float(relative_energy_tolerance)
        residual = float(inverse_forward_tolerance)
        if not math.isfinite(maximum) or maximum <= 0.0:
            raise ValueError("maximum_step_size must be finite and positive.")
        if not math.isfinite(absolute) or absolute < 0.0:
            raise ValueError(
                "absolute_energy_tolerance must be finite and non-negative."
            )
        if not math.isfinite(relative) or relative < 0.0:
            raise ValueError(
                "relative_energy_tolerance must be finite and non-negative."
            )
        if not math.isfinite(residual) or residual < 0.0:
            raise ValueError(
                "inverse_forward_tolerance must be finite and non-negative."
            )
        self.maximum_step_size = maximum
        self.absolute_energy_tolerance = absolute
        self.relative_energy_tolerance = relative
        self.inverse_forward_tolerance = residual


class ReducedSemiImplicitVelocityEulerStepDiagnostics(StrictModule):
    initial_energy: Array
    candidate_energy: Array
    applied_work: Array
    energy_defect: Array
    allowed_energy_defect: Array
    inverse_forward_residual: Array
    step_size: Array
    step_size_within_bound: Array
    finite: Array


class ReducedSemiImplicitVelocityEulerStepResult(StrictModule):
    candidate_state: ReducedArticulationState
    accepted_state: ReducedArticulationState
    dynamics: ReducedForwardDynamicsResult
    diagnostics: ReducedSemiImplicitVelocityEulerStepDiagnostics
    successful: Array
    status: Array


def _require_articulation(
    articulation: PreparedReducedArticulation, /
) -> PreparedReducedArticulation:
    if not isinstance(articulation, PreparedReducedArticulation):
        raise TypeError("articulation must be PreparedReducedArticulation.")
    if articulation.graph.bodies.ambient_dimension != 3:
        raise ValueError("Reduced rigid dynamics supports three-dimensional bodies only.")
    if articulation.nq != articulation.nv:
        raise ValueError("Reduced rigid dynamics requires fixed-base nq == nv state.")
    return articulation


def _configuration(
    articulation: PreparedReducedArticulation, value: ArrayLike, /
) -> Array:
    result = jnp.asarray(value, dtype=articulation.graph.bodies.particles.safe_masses.dtype)
    if result.shape != (articulation.nq,):
        raise ValueError("configuration must have articulation configuration shape.")
    return result


def _tangent(
    articulation: PreparedReducedArticulation,
    value: ArrayLike,
    /,
    *,
    name: str,
) -> Array:
    result = jnp.asarray(value, dtype=articulation.graph.bodies.particles.safe_masses.dtype)
    if result.shape != (articulation.nv,):
        raise ValueError(f"{name} must have articulation velocity shape.")
    return result


def _gravity(articulation: PreparedReducedArticulation, value: ArrayLike, /) -> Array:
    result = jnp.asarray(value, dtype=articulation.graph.bodies.particles.safe_masses.dtype)
    if result.shape != (3,):
        raise ValueError("gravity must be one explicit three-vector acceleration.")
    return result


def _body_load(
    articulation: PreparedReducedArticulation,
    load: RigidBodyLoad | None,
    /,
) -> RigidBodyLoad:
    capacity = articulation.graph.bodies.capacity
    dtype = articulation.graph.bodies.particles.safe_masses.dtype
    if load is None:
        return RigidBodyLoad(
            jnp.zeros((capacity, 3), dtype=dtype),
            jnp.zeros((capacity, 3), dtype=dtype),
        )
    if not isinstance(load, RigidBodyLoad):
        raise TypeError("external_load must be RigidBodyLoad or None.")
    force = jnp.asarray(load.force, dtype=dtype)
    torque = jnp.asarray(load.torque, dtype=dtype)
    if force.shape != (capacity, 3) or torque.shape != force.shape:
        raise ValueError("external_load must have body-capacity force/torque shape.")
    return RigidBodyLoad(force, torque)


def _capability_supported(articulation: PreparedReducedArticulation, /) -> Array:
    kinds = articulation.joint_kinds
    supported_kind = (
        (kinds == int(RigidJointKind.FIXED))
        | (kinds == int(RigidJointKind.HINGE))
        | (kinds == int(RigidJointKind.PRISMATIC))
    )
    root_fixed = articulation.graph.bodies.fixed_mask[articulation.root_index]
    return root_fixed & jnp.all(supported_kind)


def _status(
    *,
    input_finite: Array,
    capability_supported: Array,
    inertia_positive: ArrayLike = True,
    output_finite: ArrayLike = True,
    residual_accepted: ArrayLike = True,
) -> Array:
    inertia_positive_ = jnp.asarray(inertia_positive, dtype=bool)
    output_finite_ = jnp.asarray(output_finite, dtype=bool)
    residual_accepted_ = jnp.asarray(residual_accepted, dtype=bool)
    status = jnp.asarray(int(ReducedDynamicsStatus.SUCCESS), dtype=jnp.int32)
    status = jnp.where(
        ~residual_accepted_,
        int(ReducedDynamicsStatus.RESIDUAL_TOO_LARGE),
        status,
    )
    status = jnp.where(
        ~output_finite_,
        int(ReducedDynamicsStatus.NONFINITE_OUTPUT),
        status,
    )
    status = jnp.where(
        ~inertia_positive_,
        int(ReducedDynamicsStatus.MASS_NOT_POSITIVE_DEFINITE),
        status,
    )
    status = jnp.where(
        ~capability_supported,
        int(ReducedDynamicsStatus.CAPABILITY_REJECTED),
        status,
    )
    return jnp.where(
        ~input_finite,
        int(ReducedDynamicsStatus.NONFINITE_INPUT),
        status,
    ).astype(jnp.int32)


def _skew(value: Array, /) -> Array:
    x, y, z = value[..., 0], value[..., 1], value[..., 2]
    zero = jnp.zeros_like(x)
    return jnp.stack(
        (
            zero,
            -z,
            y,
            z,
            zero,
            -x,
            -y,
            x,
            zero,
        ),
        axis=-1,
    ).reshape(value.shape[:-1] + (3, 3))


def _world_body_inertia(
    articulation: PreparedReducedArticulation,
    body_transforms: Array,
    /,
) -> tuple[Array, Array]:
    properties = articulation.graph.bodies.mass_properties
    rotation = body_transforms[:, :3, :3]
    inertia = rotation @ properties.inertia_com @ jnp.swapaxes(rotation, -1, -2)
    mass = properties.masses.astype(rotation.dtype)
    return mass, inertia


def _edge_geometry(
    articulation: PreparedReducedArticulation,
    configuration: Array,
    /,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    kinematics = articulation.forward_kinematics(configuration)
    position = kinematics.bodies.position
    rotation = kinematics.body_transforms[:, :3, :3]
    parent = articulation.parent_indices
    child = articulation.child_indices
    parent_rotation = rotation[parent]
    axes = contract("eij,ej->ei", parent_rotation, articulation.parent_axes)
    anchors = position[parent] + contract(
        "eij,ej->ei", parent_rotation, articulation.parent_anchors
    )
    displacement = position[child] - position[parent]
    edge_count = articulation.child_indices.shape[0]
    identity = jnp.broadcast_to(
        jnp.eye(6, dtype=position.dtype), (edge_count, 6, 6)
    )
    motion_transform = identity.at[:, :3, 3:].set(-_skew(displacement))

    hinge = articulation.joint_kinds == int(RigidJointKind.HINGE)
    prismatic = articulation.joint_kinds == int(RigidJointKind.PRISMATIC)
    linear = jnp.where(
        hinge[:, None],
        jnp.cross(axes, position[child] - anchors),
        jnp.where(prismatic[:, None], axes, 0.0),
    )
    angular = jnp.where(hinge[:, None], axes, 0.0)
    motion_subspace = jnp.concatenate((linear, angular), axis=-1)
    moving = hinge | prismatic
    return (
        kinematics.body_transforms,
        motion_transform,
        motion_subspace,
        moving,
        parent,
        child,
    )


def _convective_body_acceleration(
    articulation: PreparedReducedArticulation,
    configuration: Array,
    velocity: Array,
    /,
) -> Array:
    return jax.jvp(
        lambda point: articulation.body_velocity_action(point, velocity),
        (configuration,),
        (velocity,),
    )[1]


def _external_power(
    articulation: PreparedReducedArticulation,
    configuration: Array,
    velocity: Array,
    load: RigidBodyLoad,
    external_effort: Array,
    /,
) -> tuple[Array, Array, Array]:
    body_velocity = articulation.body_velocity_action(configuration, velocity)
    body_power = jnp.sum(load.force * body_velocity[:, :3]) + jnp.sum(
        load.torque * body_velocity[:, 3:]
    )
    generalized_power = jnp.vdot(external_effort, velocity).real
    residual = body_power - generalized_power
    scale = jnp.maximum(
        jnp.maximum(jnp.abs(body_power), jnp.abs(generalized_power)), 1.0
    )
    return body_power, generalized_power, residual / scale


def reduced_energy(
    articulation: PreparedReducedArticulation,
    configuration: ArrayLike,
    velocity: ArrayLike,
    gravity: ArrayLike,
    /,
) -> ReducedEnergyResult:
    """Evaluate kinetic energy and V=-sum(m g·x) for explicit world gravity."""

    articulation = _require_articulation(articulation)
    q = _configuration(articulation, configuration)
    v = _tangent(articulation, velocity, name="velocity")
    gravity_ = _gravity(articulation, gravity)
    kinematics = articulation.forward_kinematics(q, v)
    selected = articulation.body_indices
    mass, world_inertia = _world_body_inertia(
        articulation, kinematics.body_transforms
    )
    linear = kinematics.bodies.velocity[selected]
    angular = kinematics.bodies.angular_velocity[selected]
    mass_ = mass[selected]
    inertia_ = world_inertia[selected]
    kinetic = 0.5 * jnp.sum(mass_ * jnp.sum(linear * linear, axis=-1))
    angular_momentum = contract("bij,bj->bi", inertia_, angular)
    kinetic = kinetic + 0.5 * jnp.sum(angular * angular_momentum)
    potential = -jnp.sum(
        mass_ * contract("bi,i->b", kinematics.bodies.position[selected], gravity_)
    )
    total = kinetic + potential
    input_finite = (
        jnp.all(jnp.isfinite(q))
        & jnp.all(jnp.isfinite(v))
        & jnp.all(jnp.isfinite(gravity_))
    )
    finite = input_finite & kinematics.finite & jnp.isfinite(total)
    capability = _capability_supported(articulation)
    successful = finite & capability
    return ReducedEnergyResult(
        kinetic,
        potential,
        total,
        finite,
        successful,
        _status(
            input_finite=input_finite,
            capability_supported=capability,
            output_finite=kinematics.finite & jnp.isfinite(total),
        ),
    )


def _dense_mass_matrix_reference(
    articulation: PreparedReducedArticulation,
    configuration: Array,
    /,
) -> Array:
    """Materialize JᵀGJ for diagnostics; ABA remains the forward dynamics path."""

    body_transforms = articulation.forward_kinematics(configuration).body_transforms
    mass, world_inertia = _world_body_inertia(articulation, body_transforms)
    jacobian = jax.jacfwd(
        lambda tangent: articulation.body_velocity_action(configuration, tangent)
    )(jnp.zeros((articulation.nv,), dtype=configuration.dtype))
    selected = articulation.body_indices
    linear = jacobian[selected, :3, :]
    angular = jacobian[selected, 3:, :]
    matrix = contract(
        "bai,b,baj->ij", linear, mass[selected], linear
    ) + contract(
        "bai,bac,bcj->ij", angular, world_inertia[selected], angular
    )
    return matrix


def reduced_mass_matrix(
    articulation: PreparedReducedArticulation,
    configuration: ArrayLike,
    /,
) -> ReducedMassMatrixResult:
    """Materialize the symmetric generalized mass operator and SPD evidence."""

    articulation = _require_articulation(articulation)
    q = _configuration(articulation, configuration)
    raw_matrix = _dense_mass_matrix_reference(articulation, q)
    symmetry_residual = jnp.max(
        jnp.abs(raw_matrix - raw_matrix.T), initial=0.0
    )
    matrix = 0.5 * (raw_matrix + raw_matrix.T)
    finite = jnp.all(jnp.isfinite(matrix))
    if articulation.nv == 0:
        minimum = jnp.asarray(jnp.inf, dtype=matrix.dtype)
        spectral_valid = jnp.asarray(True)
    else:
        spectrum = HermitianSpectrum(matrix)
        minimum = spectrum.minimum_eigenvalue
        spectral_valid = spectrum.valid
    positive = spectral_valid & (minimum > 0.0)
    input_finite = jnp.all(jnp.isfinite(q))
    capability = _capability_supported(articulation)
    successful = input_finite & finite & positive & capability
    safe_matrix = jnp.where(
        successful, matrix, jnp.eye(articulation.nv, dtype=matrix.dtype)
    )
    return ReducedMassMatrixResult(
        safe_matrix,
        DenseLinearOperator(
            safe_matrix,
            properties=OperatorProperties(
                self_adjoint=True,
                positive_definite=True,
                evidence={
                    "self_adjoint": "construction",
                    "positive_definite": "construction",
                },
            ),
        ),
        symmetry_residual,
        minimum,
        finite,
        positive,
        successful,
        _status(
            input_finite=input_finite,
            capability_supported=capability,
            inertia_positive=positive,
            output_finite=finite,
        ),
    )


def _rnea_effort(
    articulation: PreparedReducedArticulation,
    configuration: Array,
    velocity: Array,
    acceleration: Array,
    gravity: Array,
    /,
) -> Array:
    (
        transforms,
        motion_transform,
        motion_subspace,
        _,
        parent,
        child,
    ) = _edge_geometry(articulation, configuration)
    body_velocity = articulation.body_velocity_action(configuration, velocity)
    convective = _convective_body_acceleration(
        articulation, configuration, velocity
    )
    body_acceleration = convective + articulation.body_velocity_action(
        configuration, acceleration
    )
    mass, world_inertia = _world_body_inertia(articulation, transforms)
    angular_momentum = contract("bij,bj->bi", world_inertia, body_velocity[:, 3:])
    body_force = jnp.concatenate(
        (
            mass[:, None] * (body_acceleration[:, :3] - gravity[None, :]),
            contract("bij,bj->bi", world_inertia, body_acceleration[:, 3:])
            + jnp.cross(body_velocity[:, 3:], angular_momentum),
        ),
        axis=-1,
    )
    edge_effort = jnp.zeros((child.shape[0],), dtype=configuration.dtype)
    for edge in range(child.shape[0] - 1, -1, -1):
        child_force = body_force[child[edge]]
        edge_effort = edge_effort.at[edge].set(
            jnp.vdot(motion_subspace[edge], child_force).real
        )
        body_force = body_force.at[parent[edge]].add(
            motion_transform[edge].T @ child_force
        )
    return edge_effort[articulation.dof_joint_indices]


def reduced_bias_terms(
    articulation: PreparedReducedArticulation,
    configuration: ArrayLike,
    velocity: ArrayLike,
    gravity: ArrayLike,
    /,
) -> ReducedBiasTermsResult:
    """Return c(q,v), g(q), and c+g in M a+c+g=tau+Jᵀw."""

    articulation = _require_articulation(articulation)
    q = _configuration(articulation, configuration)
    v = _tangent(articulation, velocity, name="velocity")
    gravity_ = _gravity(articulation, gravity)
    zero = jnp.zeros_like(v)
    zero_gravity = jnp.zeros((3,), dtype=q.dtype)
    velocity_effort = _rnea_effort(articulation, q, v, zero, zero_gravity)
    gravity_effort = _rnea_effort(articulation, q, zero, zero, gravity_)
    total = velocity_effort + gravity_effort
    input_finite = (
        jnp.all(jnp.isfinite(q))
        & jnp.all(jnp.isfinite(v))
        & jnp.all(jnp.isfinite(gravity_))
    )
    finite = jnp.all(jnp.isfinite(total))
    capability = _capability_supported(articulation)
    successful = input_finite & finite & capability
    accepted_velocity = jnp.where(
        successful, velocity_effort, jnp.zeros_like(velocity_effort)
    )
    accepted_gravity = jnp.where(
        successful, gravity_effort, jnp.zeros_like(gravity_effort)
    )
    return ReducedBiasTermsResult(
        accepted_velocity,
        accepted_gravity,
        accepted_velocity + accepted_gravity,
        finite,
        successful,
        _status(
            input_finite=input_finite,
            capability_supported=capability,
            output_finite=finite,
        ),
    )


def reduced_inverse_dynamics(
    articulation: PreparedReducedArticulation,
    configuration: ArrayLike,
    velocity: ArrayLike,
    acceleration: ArrayLike,
    gravity: ArrayLike,
    /,
    *,
    external_load: RigidBodyLoad | None = None,
    power_tolerance: float = 1.0e-6,
) -> ReducedInverseDynamicsResult:
    """Compute actuator effort with topological world-frame RNEA."""

    articulation = _require_articulation(articulation)
    if not math.isfinite(float(power_tolerance)) or power_tolerance < 0.0:
        raise ValueError("power_tolerance must be finite and non-negative.")
    q = _configuration(articulation, configuration)
    v = _tangent(articulation, velocity, name="velocity")
    a = _tangent(articulation, acceleration, name="acceleration")
    gravity_ = _gravity(articulation, gravity)
    load = _body_load(articulation, external_load)
    zero = jnp.zeros_like(v)
    zero_gravity = jnp.zeros((3,), dtype=q.dtype)

    mass_effort = _rnea_effort(articulation, q, zero, a, zero_gravity)
    bias_effort = _rnea_effort(articulation, q, v, zero, zero_gravity)
    gravity_effort = _rnea_effort(articulation, q, zero, zero, gravity_)
    external_effort, duality = articulation.body_load_pullback(q, load, v)
    _, _, relative_power_residual = _external_power(
        articulation, q, v, load, external_effort
    )
    candidate = mass_effort + bias_effort + gravity_effort - external_effort
    direct = (
        _rnea_effort(articulation, q, v, a, gravity_) - external_effort
    )
    decomposition_scale = jnp.maximum(
        jnp.maximum(
            jnp.max(jnp.abs(candidate), initial=0.0),
            jnp.max(jnp.abs(direct), initial=0.0),
        ),
        1.0,
    )
    decomposition_residual = (
        jnp.max(jnp.abs(candidate - direct), initial=0.0)
        / decomposition_scale
    )
    input_finite = (
        jnp.all(jnp.isfinite(q))
        & jnp.all(jnp.isfinite(v))
        & jnp.all(jnp.isfinite(a))
        & jnp.all(jnp.isfinite(gravity_))
        & jnp.all(jnp.isfinite(load.force))
        & jnp.all(jnp.isfinite(load.torque))
    )
    finite = (
        jnp.all(jnp.isfinite(candidate))
        & jnp.isfinite(relative_power_residual)
        & jnp.isfinite(decomposition_residual)
        & duality.finite
    )
    residual_accepted = (
        duality.valid
        & (jnp.abs(relative_power_residual) <= power_tolerance)
        & (decomposition_residual <= power_tolerance)
    )
    capability = _capability_supported(articulation)
    successful = input_finite & finite & residual_accepted & capability
    accepted = jnp.where(successful, candidate, jnp.zeros_like(candidate))
    return ReducedInverseDynamicsResult(
        candidate,
        accepted,
        mass_effort,
        bias_effort,
        gravity_effort,
        external_effort,
        duality,
        relative_power_residual,
        decomposition_residual,
        finite,
        successful,
        _status(
            input_finite=input_finite,
            capability_supported=capability,
            output_finite=finite,
            residual_accepted=residual_accepted,
        ),
    )


def _aba_acceleration(
    articulation: PreparedReducedArticulation,
    configuration: Array,
    velocity: Array,
    generalized_effort: Array,
    gravity: Array,
    external_load: RigidBodyLoad,
    /,
) -> tuple[Array, Array, Array]:
    (
        transforms,
        motion_transform,
        motion_subspace,
        moving,
        parent,
        child,
    ) = _edge_geometry(articulation, configuration)
    body_velocity = articulation.body_velocity_action(configuration, velocity)
    convective = _convective_body_acceleration(
        articulation, configuration, velocity
    )
    mass, world_inertia = _world_body_inertia(articulation, transforms)
    capacity = articulation.graph.bodies.capacity
    spatial_inertia = jnp.zeros((capacity, 6, 6), dtype=configuration.dtype)
    spatial_inertia = spatial_inertia.at[:, :3, :3].set(
        mass[:, None, None] * jnp.eye(3, dtype=configuration.dtype)
    )
    spatial_inertia = spatial_inertia.at[:, 3:, 3:].set(world_inertia)
    angular_momentum = contract("bij,bj->bi", world_inertia, body_velocity[:, 3:])
    bias_force = jnp.concatenate(
        (
            -external_load.force - mass[:, None] * gravity[None, :],
            jnp.cross(body_velocity[:, 3:], angular_momentum)
            - external_load.torque,
        ),
        axis=-1,
    )
    edge_convective = convective[child] - contract(
        "eij,ej->ei", motion_transform, convective[parent]
    )
    edge_count = child.shape[0]
    edge_effort = jnp.zeros((edge_count,), dtype=configuration.dtype)
    edge_effort = edge_effort.at[articulation.dof_joint_indices].set(
        generalized_effort
    )
    articulated_inertia = spatial_inertia
    articulated_bias = bias_force
    projected_inertia = jnp.zeros((edge_count, 6), dtype=configuration.dtype)
    scalar_inertia = jnp.zeros((edge_count,), dtype=configuration.dtype)
    scalar_effort = jnp.zeros((edge_count,), dtype=configuration.dtype)
    inverse_scalar = jnp.zeros((edge_count,), dtype=configuration.dtype)
    minimum_inertia = jnp.asarray(jnp.inf, dtype=configuration.dtype)
    positive = jnp.asarray(True)
    pivot_tolerance = jnp.finfo(configuration.dtype).tiny

    for edge in range(edge_count - 1, -1, -1):
        child_index = child[edge]
        parent_index = parent[edge]
        inertia_child = articulated_inertia[child_index]
        bias_child = articulated_bias[child_index]
        projected = inertia_child @ motion_subspace[edge]
        pivot = jnp.vdot(motion_subspace[edge], projected).real
        effort = edge_effort[edge] - jnp.vdot(
            motion_subspace[edge], bias_child
        ).real
        safe_pivot = jnp.where(
            moving[edge] & (jnp.abs(pivot) > pivot_tolerance), pivot, 1.0
        )
        inverse = jnp.where(moving[edge], 1.0 / safe_pivot, 0.0)
        reduced_inertia = inertia_child - inverse * jnp.outer(projected, projected)
        reduced_bias = (
            bias_child
            + reduced_inertia @ edge_convective[edge]
            + projected * (inverse * effort)
        )
        articulated_inertia = articulated_inertia.at[parent_index].add(
            motion_transform[edge].T
            @ reduced_inertia
            @ motion_transform[edge]
        )
        articulated_bias = articulated_bias.at[parent_index].add(
            motion_transform[edge].T @ reduced_bias
        )
        projected_inertia = projected_inertia.at[edge].set(projected)
        scalar_inertia = scalar_inertia.at[edge].set(pivot)
        scalar_effort = scalar_effort.at[edge].set(effort)
        inverse_scalar = inverse_scalar.at[edge].set(inverse)
        minimum_inertia = jnp.where(
            moving[edge], jnp.minimum(minimum_inertia, pivot), minimum_inertia
        )
        positive = positive & (
            ~moving[edge] | (jnp.isfinite(pivot) & (pivot > pivot_tolerance))
        )

    body_acceleration = jnp.zeros((capacity, 6), dtype=configuration.dtype)
    edge_acceleration = jnp.zeros((edge_count,), dtype=configuration.dtype)
    for edge in range(edge_count):
        base = (
            motion_transform[edge] @ body_acceleration[parent[edge]]
            + edge_convective[edge]
        )
        acceleration = inverse_scalar[edge] * (
            scalar_effort[edge] - jnp.vdot(projected_inertia[edge], base).real
        )
        child_acceleration = base + motion_subspace[edge] * acceleration
        body_acceleration = body_acceleration.at[child[edge]].set(child_acceleration)
        edge_acceleration = edge_acceleration.at[edge].set(acceleration)
    generalized_acceleration = edge_acceleration[
        articulation.dof_joint_indices
    ]
    return generalized_acceleration, minimum_inertia, positive


def reduced_forward_dynamics(
    articulation: PreparedReducedArticulation,
    configuration: ArrayLike,
    velocity: ArrayLike,
    generalized_effort: ArrayLike,
    gravity: ArrayLike,
    /,
    *,
    external_load: RigidBodyLoad | None = None,
    residual_tolerance: float = 1.0e-6,
) -> ReducedForwardDynamicsResult:
    """Compute acceleration with ABA and certify it by inverse reconstruction."""

    articulation = _require_articulation(articulation)
    if not math.isfinite(float(residual_tolerance)) or residual_tolerance < 0.0:
        raise ValueError("residual_tolerance must be finite and non-negative.")
    q = _configuration(articulation, configuration)
    v = _tangent(articulation, velocity, name="velocity")
    effort = _tangent(
        articulation, generalized_effort, name="generalized_effort"
    )
    gravity_ = _gravity(articulation, gravity)
    load = _body_load(articulation, external_load)
    candidate, minimum_inertia, inertia_positive = _aba_acceleration(
        articulation, q, v, effort, gravity_, load
    )
    inverse = reduced_inverse_dynamics(
        articulation,
        q,
        v,
        candidate,
        gravity_,
        external_load=load,
        power_tolerance=residual_tolerance,
    )
    residual_vector = inverse.candidate_effort - effort
    residual = jnp.max(jnp.abs(residual_vector), initial=0.0)
    scale = jnp.maximum(jnp.max(jnp.abs(effort), initial=0.0), 1.0)
    relative_residual = residual / scale
    input_finite = (
        jnp.all(jnp.isfinite(q))
        & jnp.all(jnp.isfinite(v))
        & jnp.all(jnp.isfinite(effort))
        & jnp.all(jnp.isfinite(gravity_))
        & jnp.all(jnp.isfinite(load.force))
        & jnp.all(jnp.isfinite(load.torque))
    )
    finite = (
        jnp.all(jnp.isfinite(candidate))
        & jnp.isfinite(relative_residual)
        & jnp.isfinite(minimum_inertia)
        & inverse.finite
    )
    if articulation.nv == 0:
        finite = finite | (jnp.isinf(minimum_inertia) & inverse.finite)
    residual_accepted = relative_residual <= residual_tolerance
    capability = _capability_supported(articulation)
    successful = (
        input_finite
        & finite
        & inertia_positive
        & residual_accepted
        & inverse.successful
        & capability
    )
    accepted = jnp.where(successful, candidate, jnp.zeros_like(candidate))
    return ReducedForwardDynamicsResult(
        candidate,
        accepted,
        inverse.candidate_effort,
        residual,
        relative_residual,
        inverse.external_power_residual,
        minimum_inertia,
        finite,
        successful,
        _status(
            input_finite=input_finite,
            capability_supported=capability,
            inertia_positive=inertia_positive,
            output_finite=finite,
            residual_accepted=residual_accepted & inverse.successful,
        ),
    )


def _dense_reduced_forward_dynamics_reference(
    articulation: PreparedReducedArticulation,
    configuration: ArrayLike,
    velocity: ArrayLike,
    generalized_effort: ArrayLike,
    gravity: ArrayLike,
    /,
    *,
    external_load: RigidBodyLoad | None = None,
    residual_tolerance: float = 1.0e-6,
) -> ReducedForwardDynamicsResult:
    """Dense Cholesky reference; production forward dynamics uses ABA above."""

    articulation = _require_articulation(articulation)
    q = _configuration(articulation, configuration)
    v = _tangent(articulation, velocity, name="velocity")
    effort = _tangent(
        articulation, generalized_effort, name="generalized_effort"
    )
    gravity_ = _gravity(articulation, gravity)
    load = _body_load(articulation, external_load)
    mass = reduced_mass_matrix(articulation, q)
    bias = reduced_bias_terms(articulation, q, v, gravity_)
    external_effort, _ = articulation.body_load_pullback(q, load, v)
    rhs = effort + external_effort - bias.total
    solved = solve(
        LinearSystem(
            mass.operator,
            problem_id=f"{articulation.prepared_id}:dense-reduced-dynamics-reference",
        ),
        rhs,
        policy=LinearSolvePolicy(
            DenseCholesky(), failure=FailurePolicy("status")
        ),
    )
    candidate = solved.value
    inverse = reduced_inverse_dynamics(
        articulation,
        q,
        v,
        candidate,
        gravity_,
        external_load=load,
        power_tolerance=residual_tolerance,
    )
    residual = jnp.max(jnp.abs(inverse.candidate_effort - effort), initial=0.0)
    scale = jnp.maximum(jnp.max(jnp.abs(effort), initial=0.0), 1.0)
    relative = residual / scale
    solve_successful = solved.status == int(LinearSolveStatus.SUCCESS)
    input_finite = (
        jnp.all(jnp.isfinite(q))
        & jnp.all(jnp.isfinite(v))
        & jnp.all(jnp.isfinite(effort))
        & jnp.all(jnp.isfinite(gravity_))
        & jnp.all(jnp.isfinite(load.force))
        & jnp.all(jnp.isfinite(load.torque))
    )
    finite = jnp.all(jnp.isfinite(candidate)) & jnp.isfinite(relative)
    residual_accepted = relative <= residual_tolerance
    capability = _capability_supported(articulation)
    successful = (
        input_finite
        & finite
        & mass.successful
        & bias.successful
        & solve_successful
        & inverse.successful
        & residual_accepted
        & capability
    )
    status = _status(
        input_finite=input_finite,
        capability_supported=capability,
        inertia_positive=mass.positive_definite,
        output_finite=finite,
        residual_accepted=residual_accepted,
    )
    status = jnp.where(
        ~solve_successful,
        int(ReducedDynamicsStatus.LINEAR_SOLVE_FAILED),
        status,
    ).astype(jnp.int32)
    return ReducedForwardDynamicsResult(
        candidate,
        jnp.where(successful, candidate, jnp.zeros_like(candidate)),
        inverse.candidate_effort,
        residual,
        relative,
        inverse.external_power_residual,
        mass.minimum_eigenvalue,
        finite,
        successful,
        status,
    )


def reduced_semi_implicit_velocity_euler_step(
    articulation: PreparedReducedArticulation,
    state: ReducedArticulationState,
    generalized_effort: ArrayLike,
    gravity: ArrayLike,
    step_size: ArrayLike,
    /,
    *,
    external_load: RigidBodyLoad | None = None,
    policy: ReducedSemiImplicitVelocityEulerStepPolicy | None = None,
) -> ReducedSemiImplicitVelocityEulerStepResult:
    """Advance by bounded semi-implicit velocity Euler with atomic rollback."""

    articulation = _require_articulation(articulation)
    if not isinstance(state, ReducedArticulationState):
        raise TypeError("state must be ReducedArticulationState.")
    policy_ = (
        ReducedSemiImplicitVelocityEulerStepPolicy() if policy is None else policy
    )
    if not isinstance(policy_, ReducedSemiImplicitVelocityEulerStepPolicy):
        raise TypeError(
            "policy must be ReducedSemiImplicitVelocityEulerStepPolicy or None."
        )
    q = _configuration(articulation, state.configuration)
    v = _tangent(articulation, state.velocity, name="state.velocity")
    effort = _tangent(
        articulation, generalized_effort, name="generalized_effort"
    )
    gravity_ = _gravity(articulation, gravity)
    load = _body_load(articulation, external_load)
    dt = jnp.asarray(step_size, dtype=q.dtype).reshape(())
    dynamics = reduced_forward_dynamics(
        articulation,
        q,
        v,
        effort,
        gravity_,
        external_load=load,
        residual_tolerance=policy_.inverse_forward_tolerance,
    )
    candidate_velocity = v + dt * dynamics.candidate_acceleration
    candidate_configuration = articulation.integrate_configuration(
        q, candidate_velocity, dt
    )
    candidate = ReducedArticulationState(
        candidate_configuration, candidate_velocity
    )
    initial_energy = reduced_energy(articulation, q, v, gravity_)
    candidate_energy = reduced_energy(
        articulation, candidate_configuration, candidate_velocity, gravity_
    )
    external_effort, _ = articulation.body_load_pullback(q, load, candidate_velocity)
    applied_work = dt * jnp.vdot(
        effort + external_effort, candidate_velocity
    ).real
    energy_defect = candidate_energy.total - initial_energy.total - applied_work
    energy_scale = jnp.maximum(
        jnp.maximum(
            jnp.maximum(
                jnp.abs(initial_energy.kinetic),
                jnp.abs(candidate_energy.kinetic),
            ),
            jnp.abs(candidate_energy.potential - initial_energy.potential),
        ),
        1.0,
    )
    allowed_energy_defect = (
        policy_.absolute_energy_tolerance
        + policy_.relative_energy_tolerance * energy_scale
    )
    energy_accepted = jnp.abs(energy_defect) <= allowed_energy_defect
    step_size_within_bound = (
        jnp.isfinite(dt) & (dt > 0.0) & (dt <= policy_.maximum_step_size)
    )
    finite = (
        dynamics.finite
        & initial_energy.finite
        & candidate_energy.finite
        & jnp.all(jnp.isfinite(candidate.configuration))
        & jnp.all(jnp.isfinite(candidate.velocity))
        & jnp.isfinite(applied_work)
        & jnp.isfinite(energy_defect)
    )
    successful = (
        dynamics.successful
        & step_size_within_bound
        & finite
        & energy_accepted
    )
    accepted = ReducedArticulationState(
        jnp.where(successful, candidate.configuration, q),
        jnp.where(successful, candidate.velocity, v),
    )
    status = dynamics.status
    status = jnp.where(
        dynamics.successful & finite & ~energy_accepted,
        int(ReducedDynamicsStatus.ENERGY_BOUND_EXCEEDED),
        status,
    )
    status = jnp.where(
        dynamics.successful & ~finite,
        int(ReducedDynamicsStatus.NONFINITE_OUTPUT),
        status,
    )
    status = jnp.where(
        dynamics.successful & ~step_size_within_bound,
        int(ReducedDynamicsStatus.STEP_SIZE_REJECTED),
        status,
    ).astype(jnp.int32)
    diagnostics = ReducedSemiImplicitVelocityEulerStepDiagnostics(
        initial_energy.total,
        candidate_energy.total,
        applied_work,
        energy_defect,
        allowed_energy_defect,
        dynamics.inverse_forward_residual,
        dt,
        step_size_within_bound,
        finite,
    )
    return ReducedSemiImplicitVelocityEulerStepResult(
        candidate, accepted, dynamics, diagnostics, successful, status
    )


__all__ = [
    "ReducedBiasTermsResult",
    "ReducedDynamicsStatus",
    "ReducedEnergyResult",
    "ReducedForwardDynamicsResult",
    "ReducedInverseDynamicsResult",
    "ReducedMassMatrixResult",
    "ReducedSemiImplicitVelocityEulerStepDiagnostics",
    "ReducedSemiImplicitVelocityEulerStepPolicy",
    "ReducedSemiImplicitVelocityEulerStepResult",
    "reduced_bias_terms",
    "reduced_energy",
    "reduced_forward_dynamics",
    "reduced_inverse_dynamics",
    "reduced_mass_matrix",
    "reduced_semi_implicit_velocity_euler_step",
]
