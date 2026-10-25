#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from enum import IntEnum
from typing import final, overload

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from phydrax.ein import contract

from ..._strict import StrictModule
from ...backends._result_storage import evaluate_with_result_storage
from ...linalg import (
    DenseLinearOperator,
    HermitianSpectrum,
    OperatorProperties,
)
from ._articulated_mass import (
    _articulated_solve,
    _bind,
    _capability_supported,
    _edge_motion_geometry,
    _prepare_mass_from_kinematics,
    _require_articulation,
    _world_body_inertia,
    ArticulatedInertiaFactor,
    ArticulatedMassPolicy,
    PreparedArticulatedMass,
)
from ._articulated_sweeps import reduce_articulated_forces
from ._reduced_articulation import (
    ArticulationDualityEvidence,
    ArticulationKinematics,
    PreparedReducedArticulation,
    ReducedArticulationState,
)
from ._rigid_body import RigidBodyLoad


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
    """ABA forward dynamics certified by inverse-dynamics reconstruction.

    ``mass`` is the configuration-bound articulated factor that produced the
    acceleration; consumers reuse it instead of preparing ``M(q)`` again.
    """

    candidate_acceleration: Array
    acceleration: Array
    reconstructed_effort: Array
    inverse_forward_residual: Array
    relative_inverse_forward_residual: Array
    external_power_residual: Array
    minimum_articulated_inertia: Array
    mass: PreparedArticulatedMass
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
    ) -> None:
        maximum = float(maximum_step_size)
        absolute = float(absolute_energy_tolerance)
        relative = float(relative_energy_tolerance)
        residual = float(inverse_forward_tolerance)
        if not math.isfinite(maximum) or maximum <= 0.0:
            raise ValueError("maximum_step_size must be finite and positive.")
        if not math.isfinite(absolute) or absolute < 0.0:
            raise ValueError("absolute_energy_tolerance must be finite and non-negative.")
        if not math.isfinite(relative) or relative < 0.0:
            raise ValueError("relative_energy_tolerance must be finite and non-negative.")
        if not math.isfinite(residual) or residual < 0.0:
            raise ValueError("inverse_forward_tolerance must be finite and non-negative.")
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


def _configuration(
    articulation: PreparedReducedArticulation, value: ArrayLike, /
) -> Array:
    result = jnp.asarray(
        value, dtype=articulation.graph.bodies.particles.safe_masses.dtype
    )
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
    result = jnp.asarray(
        value, dtype=articulation.graph.bodies.particles.safe_masses.dtype
    )
    if result.shape != (articulation.nv,):
        raise ValueError(f"{name} must have articulation velocity shape.")
    return result


def _gravity(articulation: PreparedReducedArticulation, value: ArrayLike, /) -> Array:
    result = jnp.asarray(
        value, dtype=articulation.graph.bodies.particles.safe_masses.dtype
    )
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


def _status(
    *,
    input_finite: Array,
    capability_supported: Array,
    inertia_positive: ArrayLike = True,
    output_finite: ArrayLike = True,
    residual_accepted: ArrayLike = True,
) -> Array:
    inertia_positive_ = jnp.asarray(inertia_positive, dtype=jnp.bool_)
    output_finite_ = jnp.asarray(output_finite, dtype=jnp.bool_)
    residual_accepted_ = jnp.asarray(residual_accepted, dtype=jnp.bool_)
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


@final
class _BodyMotion(StrictModule):
    """Body twists ``J(q)u`` and convective accelerations ``(∂_q J(q)u)u``.

    ``rest_*`` are evaluated at ``u = 0`` rather than assumed zero, so
    nonfinite pose geometry reaches every reconstruction column exactly as it
    does through separate evaluations.
    """

    kinematics: ArticulationKinematics
    rest_velocity: Array
    rest_convective: Array
    velocity: Array
    convective: Array


@overload
def _body_motion(
    articulation: PreparedReducedArticulation,
    configuration: Array,
    velocity: Array,
    /,
) -> tuple[_BodyMotion, None]: ...


@overload
def _body_motion(
    articulation: PreparedReducedArticulation,
    configuration: Array,
    velocity: Array,
    acceleration: Array,
    /,
) -> tuple[_BodyMotion, Array]: ...


def _body_motion(
    articulation: PreparedReducedArticulation,
    configuration: Array,
    velocity: Array,
    acceleration: Array | None = None,
    /,
) -> tuple[_BodyMotion, Array | None]:
    """Evaluate independent rest and velocity motion using one shared pose pass.

    ``0`` and ``v`` retain the directional derivatives of the actual velocity
    action along themselves. An optional acceleration uses that same geometry
    for ``J(q)a`` without computing an unused zero-direction derivative.
    """

    reference_configuration = articulation._configuration(configuration, "Configuration")
    dtype = reference_configuration.dtype
    zero = jnp.zeros_like(velocity)
    rates = jnp.asarray(jnp.stack((zero, velocity)), dtype=dtype)
    position, orientation, twists, derivatives = articulation._motion(
        reference_configuration,
        rates,
        rates,
    )
    kinematics = articulation._kinematics_from_motion(
        reference_configuration,
        jnp.zeros((articulation.nv,), dtype=dtype),
        position,
        orientation,
        twists[0],
    )
    motion = _BodyMotion(kinematics, twists[0], derivatives[0], twists[1], derivatives[1])
    acceleration_twist = (
        None
        if acceleration is None
        else articulation._body_velocity_from_poses(
            articulation._velocity(acceleration, "Generalized velocity"),
            position,
            orientation,
        )
    )
    return motion, acceleration_twist


def _external_power(
    body_velocity: Array,
    velocity: Array,
    load: RigidBodyLoad,
    external_effort: Array,
    /,
) -> tuple[Array, Array, Array]:
    body_power = jnp.sum(load.force * body_velocity[:, :3]) + jnp.sum(
        load.torque * body_velocity[:, 3:]
    )
    generalized_power = jnp.vdot(external_effort, velocity).real
    residual = body_power - generalized_power
    scale = jnp.maximum(jnp.maximum(jnp.abs(body_power), jnp.abs(generalized_power)), 1.0)
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
        articulation, kinematics.body_transforms[:, :3, :3]
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
    mass, world_inertia = _world_body_inertia(articulation, body_transforms[:, :3, :3])
    jacobian = jax.jacfwd(
        lambda tangent: articulation.body_velocity_action(configuration, tangent)
    )(jnp.zeros((articulation.nv,), dtype=configuration.dtype))
    selected = articulation.body_indices
    linear = jacobian[selected, :3, :]
    angular = jacobian[selected, 3:, :]
    matrix = contract("bai,b,baj->ij", linear, mass[selected], linear) + contract(
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
    symmetry_residual = jnp.max(jnp.abs(raw_matrix - raw_matrix.T), initial=0.0)
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


def _rnea_body_force(
    mass: Array,
    world_inertia: Array,
    body_velocity: Array,
    body_acceleration: Array,
    gravity: Array,
    /,
) -> Array:
    angular_momentum = contract("bij,bj->bi", world_inertia, body_velocity[:, 3:])
    return jnp.concatenate(
        (
            mass[:, None] * (body_acceleration[:, :3] - gravity[None, :]),
            contract("bij,bj->bi", world_inertia, body_acceleration[:, 3:])
            + jnp.cross(body_velocity[:, 3:], angular_momentum),
        ),
        axis=-1,
    )


def _rnea_efforts(
    articulation: PreparedReducedArticulation,
    configuration: Array,
    kinematics: ArticulationKinematics,
    body_velocity: Array,
    body_acceleration: Array,
    gravity: Array,
    /,
    *,
    factor: ArticulatedInertiaFactor | None = None,
) -> Array:
    """Reconstruct ``(nv, K)`` topological world-frame RNEA effort columns.

    Column ``k`` uses twists ``body_velocity[k]``, accelerations
    ``body_acceleration[k]`` and gravity ``gravity[k]``. The configuration
    geometry is evaluated once and all columns share one reverse force
    reduction; columns never mix arithmetically.
    """

    if factor is None:
        rotation = kinematics.body_transforms[:, :3, :3]
        motion_transform, motion_subspace, _ = _edge_motion_geometry(
            articulation, kinematics.bodies.position, rotation
        )
        mass, world_inertia = _world_body_inertia(articulation, rotation)
    else:
        motion_transform = factor.motion_transforms
        motion_subspace = factor.motion_subspaces
        mass = factor.body_masses
        world_inertia = factor.world_inertia
    body_force = jax.vmap(_rnea_body_force, in_axes=(None, None, 0, 0, 0), out_axes=-1)(
        mass, world_inertia, body_velocity, body_acceleration, gravity
    )
    edge_effort = reduce_articulated_forces(
        motion_transform,
        motion_subspace,
        articulation.parent_indices,
        articulation.child_indices,
        body_force,
    )
    return edge_effort[articulation.dof_joint_indices].astype(configuration.dtype)


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
    zero_gravity = jnp.zeros((3,), dtype=q.dtype)
    motion, _ = _body_motion(articulation, q, v)
    efforts = _rnea_efforts(
        articulation,
        q,
        motion.kinematics,
        jnp.stack((motion.velocity, motion.rest_velocity)),
        jnp.stack(
            (
                motion.convective + motion.rest_velocity,
                motion.rest_convective + motion.rest_velocity,
            )
        ),
        jnp.stack((zero_gravity, gravity_)),
    )
    velocity_effort = efforts[:, 0]
    gravity_effort = efforts[:, 1]
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
    motion, acceleration_twist = _body_motion(articulation, q, v, a)
    return _inverse_dynamics(
        articulation,
        q,
        v,
        a,
        gravity_,
        load,
        power_tolerance,
        motion,
        acceleration_twist,
    )


def _inverse_dynamics(
    articulation: PreparedReducedArticulation,
    q: Array,
    v: Array,
    a: Array,
    gravity: Array,
    load: RigidBodyLoad,
    power_tolerance: float,
    motion: _BodyMotion,
    acceleration_twist: Array,
    /,
    *,
    factor: ArticulatedInertiaFactor | None = None,
) -> ReducedInverseDynamicsResult:
    """Certify validated inverse dynamics from precomputed body motion.

    Mass, bias, gravity and direct efforts remain four independent RNEA
    columns of one shared force reduction; the direct column certifies their
    decomposition.
    """

    zero_gravity = jnp.zeros((3,), dtype=q.dtype)
    efforts = _rnea_efforts(
        articulation,
        q,
        motion.kinematics,
        jnp.stack(
            (
                motion.rest_velocity,
                motion.velocity,
                motion.rest_velocity,
                motion.velocity,
            )
        ),
        jnp.stack(
            (
                motion.rest_convective + acceleration_twist,
                motion.convective + motion.rest_velocity,
                motion.rest_convective + motion.rest_velocity,
                motion.convective + acceleration_twist,
            )
        ),
        jnp.stack((zero_gravity, zero_gravity, gravity, gravity)),
        factor=factor,
    )
    mass_effort = efforts[:, 0]
    bias_effort = efforts[:, 1]
    gravity_effort = efforts[:, 2]
    reference_dtype = articulation.reference_position.dtype
    external_effort, duality = articulation._body_load_pullback_from_poses(
        jnp.asarray(q, dtype=reference_dtype),
        jnp.asarray(v, dtype=reference_dtype),
        jnp.asarray(load.force, dtype=reference_dtype),
        jnp.asarray(load.torque, dtype=reference_dtype),
        motion.kinematics.bodies.position,
        motion.kinematics.bodies.orientation,
    )
    _, _, relative_power_residual = _external_power(
        motion.velocity, v, load, external_effort
    )
    candidate = mass_effort + bias_effort + gravity_effort - external_effort
    direct = efforts[:, 3] - external_effort
    decomposition_scale = jnp.maximum(
        jnp.maximum(
            jnp.max(jnp.abs(candidate), initial=0.0),
            jnp.max(jnp.abs(direct), initial=0.0),
        ),
        1.0,
    )
    decomposition_residual = (
        jnp.max(jnp.abs(candidate - direct), initial=0.0) / decomposition_scale
    )
    input_finite = (
        jnp.all(jnp.isfinite(q))
        & jnp.all(jnp.isfinite(v))
        & jnp.all(jnp.isfinite(a))
        & jnp.all(jnp.isfinite(gravity))
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
    mass: PreparedArticulatedMass,
    motion: _BodyMotion,
    generalized_effort: Array,
    gravity: Array,
    external_load: RigidBodyLoad,
    /,
) -> Array:
    """Add velocity, gravity and load terms to one prepared ABA substitution."""

    factor = mass.factor
    body_velocity = motion.velocity
    convective = motion.convective
    angular_momentum = contract("bij,bj->bi", factor.world_inertia, body_velocity[:, 3:])
    bias_force = jnp.concatenate(
        (
            -external_load.force - factor.body_masses[:, None] * gravity[None, :],
            jnp.cross(body_velocity[:, 3:], angular_momentum) - external_load.torque,
        ),
        axis=-1,
    )
    edge_convective = convective[factor.child_indices] - contract(
        "eij,ej->ei", factor.motion_transforms, convective[factor.parent_indices]
    )
    return _articulated_solve(
        factor,
        generalized_effort[:, None],
        body_bias=bias_force[..., None],
        edge_convective=edge_convective[..., None],
    )[:, 0]


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
    mass_policy: ArticulatedMassPolicy | None = None,
) -> ReducedForwardDynamicsResult:
    """Compute acceleration with ABA and certify it by inverse reconstruction.

    The configuration/inertia factor is prepared once by
    `prepare_articulated_mass` and returned as ``mass``. Without
    ``mass_policy`` a moving pivot is accepted when it is finite and exceeds
    the smallest normal number of the configuration dtype; with an explicit
    policy the factor's relative pivot certificate is required. ``mass``
    always carries its own policy evidence. A refused pivot reports
    ``MASS_NOT_POSITIVE_DEFINITE``.
    """

    articulation = _require_articulation(articulation)
    if not math.isfinite(float(residual_tolerance)) or residual_tolerance < 0.0:
        raise ValueError("residual_tolerance must be finite and non-negative.")
    if mass_policy is not None and not isinstance(mass_policy, ArticulatedMassPolicy):
        raise TypeError("mass_policy must be ArticulatedMassPolicy or None.")
    q = _configuration(articulation, configuration)
    v = _tangent(articulation, velocity, name="velocity")
    effort = _tangent(articulation, generalized_effort, name="generalized_effort")
    gravity_ = _gravity(articulation, gravity)
    load = _body_load(articulation, external_load)
    return evaluate_with_result_storage(
        _forward_dynamics,
        articulation,
        q,
        v,
        effort,
        gravity_,
        load,
        residual_tolerance,
        mass_policy,
    )


def _forward_dynamics(
    articulation: PreparedReducedArticulation,
    q: Array,
    v: Array,
    effort: Array,
    gravity_: Array,
    load: RigidBodyLoad,
    residual_tolerance: float,
    mass_policy: ArticulatedMassPolicy | None,
    /,
) -> ReducedForwardDynamicsResult:
    """Compute and certify a query inside its private result-storage lifetime."""

    motion, _ = _body_motion(articulation, q, v)
    policy = ArticulatedMassPolicy() if mass_policy is None else mass_policy
    mass = _prepare_mass_from_kinematics(
        articulation, policy, _bind(articulation, q), motion.kinematics
    )
    candidate = _aba_acceleration(mass, motion, effort, gravity_, load)
    minimum_inertia = mass.evidence.minimum_pivot
    if mass_policy is None:
        pivots = mass.evidence.pivots
        inertia_positive = jnp.all(
            ~mass.evidence.moving
            | (jnp.isfinite(pivots) & (pivots > jnp.finfo(q.dtype).tiny))
        )
    else:
        inertia_positive = mass.evidence.pivots_positive
    reconstructed = _tangent(articulation, candidate, name="acceleration")
    inverse = _inverse_dynamics(
        articulation,
        q,
        v,
        reconstructed,
        gravity_,
        load,
        residual_tolerance,
        motion,
        articulation._body_velocity_from_poses(
            articulation._velocity(reconstructed, "Generalized velocity"),
            motion.kinematics.bodies.position,
            motion.kinematics.bodies.orientation,
        ),
        factor=mass.factor,
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
        mass,
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
    mass_policy: ArticulatedMassPolicy | None = None,
) -> ReducedSemiImplicitVelocityEulerStepResult:
    """Advance by bounded semi-implicit velocity Euler with atomic rollback."""

    articulation = _require_articulation(articulation)
    if not isinstance(state, ReducedArticulationState):
        raise TypeError("state must be ReducedArticulationState.")
    policy_ = ReducedSemiImplicitVelocityEulerStepPolicy() if policy is None else policy
    if not isinstance(policy_, ReducedSemiImplicitVelocityEulerStepPolicy):
        raise TypeError(
            "policy must be ReducedSemiImplicitVelocityEulerStepPolicy or None."
        )
    q = _configuration(articulation, state.configuration)
    v = _tangent(articulation, state.velocity, name="state.velocity")
    effort = _tangent(articulation, generalized_effort, name="generalized_effort")
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
        mass_policy=mass_policy,
    )
    candidate_velocity = v + dt * dynamics.candidate_acceleration
    candidate_configuration = articulation.integrate_configuration(
        q, candidate_velocity, dt
    )
    candidate = ReducedArticulationState(candidate_configuration, candidate_velocity)
    initial_energy = reduced_energy(articulation, q, v, gravity_)
    candidate_energy = reduced_energy(
        articulation, candidate_configuration, candidate_velocity, gravity_
    )
    external_effort, _ = articulation.body_load_pullback(q, load, candidate_velocity)
    applied_work = dt * jnp.vdot(effort + external_effort, candidate_velocity).real
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
    successful = dynamics.successful & step_size_within_bound & finite & energy_accepted
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
