#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""One first-order direct-forcing/pressure/DEM attempt, committed atomically.

The geometry, not an independent miscible-density transport equation, owns
mixture inertia. Native donor-cell transport supplies the velocity predictor;
centered geometric face inertia supplies forcing and the single projection.
Packed fluid coordinates retain the native donor-momentum representation.
The external-fluid reaction excludes fictitious internal mixture inertia.
"""

from __future__ import annotations

from enum import IntFlag
from typing import Any, final

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from .._strict import StrictModule
from .._tree_math import tree_allfinite, tree_where
from ..discretization.finite_volume._incompressible import FaceVelocity
from ..discretization.particle._dem import DEMRuntimeState, DEMStepEvaluation
from ..discretization.particle._rigid_body import (
    quaternion_rotation_matrix,
    RigidBodyKinematics,
)
from ..equations._mac_volume_fraction_ib_cfd_dem import (
    centered_face_values,
    evaluate_mac_volume_fraction_ib_cfd_dem,
    MACSDFCutFractionEvaluation,
    MACVolumeFractionIBCFDEMCouplingPlan,
    MACVolumeFractionIBEvaluation,
)
from ..typing import checked
from ._mac_variable_density import MACVariableDensityProjectionResult


class MACVolumeFractionIBStepStatus(IntFlag):
    SUCCESS = 0
    INVALID_STEP_SIZE = 1
    GEOMETRY_FAILED = 2
    TRANSPORT_FAILED = 4
    COUPLING_FAILED = 8
    PROJECTION_FAILED = 16
    DEM_FAILED = 32
    NONFINITE = 64


@final
class MACVolumeFractionIBState(StrictModule):
    """Committed DEM, native density-momentum coordinates, pressure and ledgers."""

    dem_state: DEMRuntimeState
    fluid_state: Array
    cumulative_body_impulse: Array
    cumulative_body_angular_impulse: Array
    pressure: Array
    accepted_steps: Array

    @classmethod
    @checked
    def initialize(
        cls,
        coupling: MACVolumeFractionIBCFDEMCouplingPlan,
        dem_state: DEMRuntimeState,
        fluid_state: ArrayLike,
        /,
        *,
        pressure: ArrayLike | None = None,
    ) -> MACVolumeFractionIBState:
        coordinates = coupling.fluid.validate_state(fluid_state)
        operators = coupling.projection.operators
        bodies = coupling.dynamics.bodies
        if dem_state.kinematics.position.shape != (
            bodies.capacity,
            bodies.ambient_dimension,
        ):
            raise ValueError("DEM state does not match the coupling body capacity.")
        if not bool(_runtime_solid_mass_valid(coupling, dem_state)):
            raise ValueError(
                "Active DEM runtime mass and inverse mass must match "
                "solid_density times the scaled shape volume."
            )
        p = (
            jnp.zeros(operators.discretization.cell_shape, dtype=coordinates.dtype)
            if pressure is None
            else operators.validate_pressure(pressure)
        )
        return cls(
            dem_state,
            coordinates,
            jnp.zeros_like(dem_state.kinematics.position),
            jnp.zeros_like(dem_state.kinematics.angular_velocity),
            p,
            jnp.zeros((), dtype=jnp.int32),
        )


@final
class MACVolumeFractionIBStepResult(StrictModule):
    """The uncommitted candidate and all native evidence survive a rollback."""

    candidate_state: MACVolumeFractionIBState
    accepted_state: MACVolumeFractionIBState
    evaluation: MACVolumeFractionIBEvaluation
    pressure_projection: MACVariableDensityProjectionResult
    dem_step: DEMStepEvaluation
    candidate_cut_fraction: MACSDFCutFractionEvaluation
    transport_successful: Array
    status: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


def _runtime_solid_mass_valid(
    coupling: MACVolumeFractionIBCFDEMCouplingPlan, state: DEMRuntimeState, /
) -> Array:
    """Admit the same solid mass for geometric mixture and rigid-body inertia."""
    properties = state.body_properties
    scale = properties.shape_scale
    expected = (
        coupling.solid_density
        * coupling.dynamics.shapes.volumes
        * scale**coupling.dynamics.bodies.ambient_dimension
    )
    tolerance = 128.0 * jnp.finfo(properties.masses.dtype).eps
    mobile = ~coupling.dynamics.bodies.fixed_mask
    mass_valid = (
        jnp.isfinite(scale)
        & (scale > 0.0)
        & jnp.isfinite(expected)
        & (expected > 0.0)
        & jnp.isfinite(properties.masses)
        & (jnp.abs(properties.masses - expected) <= tolerance * expected)
        & jnp.isfinite(properties.inverse_masses)
        & (
            jnp.abs(properties.inverse_masses * expected - mobile.astype(expected.dtype))
            <= tolerance
        )
    )
    return jnp.all(~properties.active | mass_valid)


def _native_momentum(
    coupling: MACVolumeFractionIBCFDEMCouplingPlan,
    density: Array,
    velocity: FaceVelocity,
    /,
) -> FaceVelocity:
    """Preserve velocity exactly across centered and native donor coordinates."""
    variable = coupling.fluid.variable_density
    donor = variable.face_density(density, velocity)
    return tuple(rho * u for rho, u in zip(donor, velocity, strict=True))


def _predict_velocity(
    coupling: MACVolumeFractionIBCFDEMCouplingPlan,
    state: MACVolumeFractionIBState,
    cut: MACSDFCutFractionEvaluation,
    time: Array,
    step: Array,
    args: Any,
    /,
) -> tuple[FaceVelocity, FaceVelocity, Array]:
    fluid = coupling.fluid
    variable = fluid.variable_density
    boundaries = variable.momentum.boundaries
    physical = fluid.physical_state(time, state.fluid_state, args)
    density = cut.fluid_fraction * coupling.fluid_density + (
        cut.solid_fraction * coupling.solid_density
    )
    rho_face = centered_face_values(variable.operators, density)
    momentum = _native_momentum(coupling, density, physical.velocity)
    stage = boundaries.evaluate(time, args)
    transport = variable.transport(density, momentum, stage=stage)
    density_rate = variable.face_density_rate(transport.density_rate, momentum)
    viscous = tuple(
        fluid.problem.dynamic_viscosity * value
        for value in variable.momentum.laplacian(physical.velocity, stage=stage)
    )
    physical = eqx.tree_at(
        lambda value: (
            value.density,
            value.face_momentum,
            value.face_density,
            value.minimum_density,
            value.positive,
            value.finite,
        ),
        physical,
        (
            density,
            transport.face_momentum,
            transport.face_density,
            transport.minimum_density,
            transport.positive,
            transport.finite,
        ),
    )
    acceleration = fluid._body_acceleration(time, physical, args)
    pressure_gradient = variable.operators.gradient(state.pressure)
    predictor = tuple(
        u + step * ((-advection - u * drho) / donor + (stress - pressure) / rho + a)
        for u, advection, drho, donor, stress, pressure, rho, a in zip(
            physical.velocity,
            transport.momentum_advection,
            density_rate,
            transport.face_density,
            viscous,
            pressure_gradient,
            rho_face,
            acceleration,
            strict=True,
        )
    )
    return boundaries.homogeneous_rate(predictor), viscous, transport.successful


def _hydrodynamic_kick(
    coupling: MACVolumeFractionIBCFDEMCouplingPlan,
    state: DEMRuntimeState,
    evaluation: MACVolumeFractionIBEvaluation,
    step: Array,
    /,
) -> DEMRuntimeState:
    """First-order load kick; native body kernels own posed rotational inertia."""
    dynamics = coupling.dynamics
    mobile = (state.body_properties.active & ~dynamics.bodies.fixed_mask)[:, None]
    velocity = state.kinematics.velocity + (
        step * state.body_properties.inverse_masses[:, None] * evaluation.body_force
    )
    properties = state.body_properties
    if dynamics.bodies.ambient_dimension == 2:
        angular_increment = (
            properties.inverse_inertia_com[:, None] * evaluation.body_torque
        )
    else:
        rotation = quaternion_rotation_matrix(state.kinematics.orientation)
        body_torque = jnp.swapaxes(rotation, -1, -2) @ evaluation.body_torque[..., None]
        angular_increment = (rotation @ (properties.inverse_inertia_com @ body_torque))[
            ..., 0
        ]
    # This kick applies only the hydraulic impulse. The native DEM step owns
    # gyroscopic free rotation; adding its angular acceleration here counts it twice.
    angular = state.kinematics.angular_velocity + step * angular_increment
    kinematics = RigidBodyKinematics(
        state.kinematics.position,
        jnp.where(mobile, velocity, state.kinematics.velocity),
        state.kinematics.orientation,
        jnp.where(mobile, angular, state.kinematics.angular_velocity),
    )
    updated = eqx.tree_at(lambda value: value.kinematics, state, kinematics)
    kinetic = dynamics._kinetic_energy(kinematics, state.body_properties)
    return eqx.tree_at(lambda value: value.energy.kinetic_energy, updated, kinetic)


@checked
def advance_mac_volume_fraction_ib_cfd_dem(
    coupling: MACVolumeFractionIBCFDEMCouplingPlan,
    state: MACVolumeFractionIBState,
    time: ArrayLike,
    step_size: ArrayLike,
    /,
    *,
    args: Any = None,
) -> MACVolumeFractionIBStepResult:
    """Evaluate exactly one projection and atomically accept both phase states.

    Hydrodynamic loading is a full kick before the native DEM Verlet step;
    pressure/viscous forces use the accepted-pose fraction and new pressure.
    No geometry, solver or DEM failure can commit either participant or ledgers.
    """
    dtype = state.fluid_state.dtype
    time_value = jnp.asarray(time, dtype=dtype).reshape(())
    dt = jnp.asarray(step_size, dtype=dtype).reshape(())
    step_valid = jnp.isfinite(dt) & (dt > 0.0) & jnp.isfinite(time_value)
    # A rejected invalid-step attempt still produces bounded native evidence.
    step = jnp.where(step_valid, dt, jnp.ones((), dtype=dtype))
    dem = state.dem_state
    cut = coupling.cut_fraction.evaluate(
        dem.kinematics,
        time_value,
        shape_scale=dem.body_properties.shape_scale,
        active_mask=dem.body_properties.active,
    )
    predictor, viscous, transport_ok = _predict_velocity(
        coupling, state, cut, time_value, step, args
    )
    forcing = evaluate_mac_volume_fraction_ib_cfd_dem(
        coupling, dem.kinematics, cut, predictor, viscous, step
    )
    boundaries = coupling.fluid.variable_density.momentum.boundaries
    momentum = boundaries.homogeneous_rate(
        tuple(
            rho * u + step * force
            for rho, u, force in zip(
                forcing.face_mixture_density,
                predictor,
                forcing.fluid_ib_force_density,
                strict=True,
            )
        )
    )
    projection = coupling.projection.project(
        momentum,
        tuple(1.0 / rho for rho in forcing.face_mixture_density),
        step,
        pressure=state.pressure,
    )
    gradient = coupling.projection.operators.gradient(projection.pressure)
    stress = tuple(v - p for v, p in zip(viscous, gradient, strict=True))
    evaluation = evaluate_mac_volume_fraction_ib_cfd_dem(
        coupling, dem.kinematics, cut, predictor, stress, step
    )
    kicked = _hydrodynamic_kick(coupling, dem, evaluation, step)
    dem_step = coupling.dynamics.step_detailed(
        state.accepted_steps, time_value, kicked, step, args
    )
    next_dem = dem_step.candidate_state
    candidate_cut = coupling.cut_fraction.evaluate(
        next_dem.kinematics,
        time_value + step,
        shape_scale=next_dem.body_properties.shape_scale,
        active_mask=next_dem.body_properties.active,
    )
    candidate_density = (
        candidate_cut.fluid_fraction * coupling.fluid_density
        + candidate_cut.solid_fraction * coupling.solid_density
    )
    packed = coupling.fluid.pack_state(
        candidate_density,
        _native_momentum(coupling, candidate_density, projection.velocity),
    )
    candidate = MACVolumeFractionIBState(
        next_dem,
        packed,
        state.cumulative_body_impulse + step * evaluation.body_force,
        state.cumulative_body_angular_impulse + step * evaluation.body_torque,
        projection.pressure,
        state.accepted_steps + jnp.ones((), dtype=jnp.int32),
    )
    finite = tree_allfinite(candidate) & tree_allfinite(evaluation)
    status = jnp.zeros((), dtype=jnp.int32)
    for admitted, reason in (
        (step_valid, MACVolumeFractionIBStepStatus.INVALID_STEP_SIZE),
        (
            cut.successful
            & candidate_cut.successful
            & _runtime_solid_mass_valid(coupling, dem)
            & _runtime_solid_mass_valid(coupling, next_dem),
            MACVolumeFractionIBStepStatus.GEOMETRY_FAILED,
        ),
        (transport_ok, MACVolumeFractionIBStepStatus.TRANSPORT_FAILED),
        (evaluation.successful, MACVolumeFractionIBStepStatus.COUPLING_FAILED),
        (projection.successful, MACVolumeFractionIBStepStatus.PROJECTION_FAILED),
        (dem_step.successful, MACVolumeFractionIBStepStatus.DEM_FAILED),
        (finite, MACVolumeFractionIBStepStatus.NONFINITE),
    ):
        status = status | jnp.where(admitted, 0, int(reason)).astype(jnp.int32)
    successful = status == 0
    return MACVolumeFractionIBStepResult(
        candidate,
        tree_where(successful, candidate, state),
        evaluation,
        projection,
        dem_step,
        candidate_cut,
        transport_ok,
        status,
        successful,
        coupling.plan_id,
    )


__all__ = [
    "MACVolumeFractionIBState",
    "MACVolumeFractionIBStepResult",
    "MACVolumeFractionIBStepStatus",
    "advance_mac_volume_fraction_ib_cfd_dem",
]
