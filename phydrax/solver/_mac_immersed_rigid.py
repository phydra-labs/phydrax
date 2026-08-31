#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.finite_volume._incompressible import FaceVelocity
from ..discretization.finite_volume._mac_marker_transfer import (
    PreparedMACMarkerTransfer,
)
from ..discretization.particle._rigid_body import (
    rigid_body_drift,
    rigid_body_world_inertia,
    RigidBodyKinematics,
    RigidBodyLoad,
)
from ..discretization.particle._rigid_marker import (
    PreparedRigidMarkerMap,
    RigidGeneralizedVelocity,
)
from ..equations._mac_incompressible import CompiledMACIncompressibleDynamics
from ..linalg import (
    BlockSpace,
    DifferentiationPolicy,
    FunctionLinearOperator,
    GMRES,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSystem,
    OperatorProperties,
    solve,
    TolerancePolicy,
)
from ._mac_viscous import MACHelmholtzResult, MACHelmholtzSolvePlan
from ._structured_incompressible import MACPressureClosureReport


class MACRigidImmersedProjectionResult(StrictModule):
    fluid_velocity: FaceVelocity
    body_kinematics: RigidBodyKinematics
    pressure: Array
    marker_force_density: Array
    marker_slip: Array
    divergence: Array
    kkt_residual_norm: Array
    linear: LinearSolveResult
    closure: MACPressureClosureReport
    finite: Array
    converged: Array
    plan_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.converged


class MACRigidImmersedProjectionPlan(StrictModule, NonTrainableState):
    """Simultaneous unit-density MAC and generic rigid-body velocity projection."""

    dynamics: CompiledMACIncompressibleDynamics
    rigid_markers: PreparedRigidMarkerMap
    transfer: PreparedMACMarkerTransfer
    constraint_length: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    linear_policy: LinearSolvePolicy
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        dynamics: CompiledMACIncompressibleDynamics,
        rigid_markers: PreparedRigidMarkerMap,
        transfer: PreparedMACMarkerTransfer,
        /,
        *,
        constraint_length: float,
        tolerance: float = 1.0e-9,
        maximum_iterations: int = 500,
        linear_policy: LinearSolvePolicy | None = None,
    ):

        if not isinstance(dynamics, CompiledMACIncompressibleDynamics):
            raise TypeError("dynamics must be CompiledMACIncompressibleDynamics.")
        if not isinstance(rigid_markers, PreparedRigidMarkerMap):
            raise TypeError("rigid_markers must be PreparedRigidMarkerMap.")
        if not isinstance(transfer, PreparedMACMarkerTransfer):
            raise TypeError("transfer must be PreparedMACMarkerTransfer.")
        if transfer.markers.prepared_id != rigid_markers.markers.prepared_id:
            raise ValueError("Rigid map and transfer must share markers.")
        if transfer.operators.prepared_id != dynamics.momentum.operators.prepared_id:
            raise ValueError("Rigid projection and fluid must share MAC operators.")
        length = float(constraint_length)
        tolerance_ = float(tolerance)
        if length <= 0.0 or tolerance_ <= 0.0:
            raise ValueError("constraint_length and tolerance must be positive.")
        policy = (
            LinearSolvePolicy(
                GMRES(restart=min(50, int(maximum_iterations))),
                tolerance=TolerancePolicy(
                    relative=tolerance_,
                    absolute=tolerance_,
                    max_steps=int(maximum_iterations),
                ),
                differentiation=DifferentiationPolicy("mathematical"),
            )
            if linear_policy is None
            else linear_policy
        )
        self.dynamics = dynamics
        self.rigid_markers = rigid_markers
        self.transfer = transfer
        self.constraint_length = length
        self.tolerance = tolerance_
        self.linear_policy = policy
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mac-rigid-immersed-projection",
                "dynamics": dynamics.compilation_id,
                "rigid_markers": rigid_markers.prepared_id,
                "transfer": transfer.prepared_id,
                "constraint_length": length,
                "linear_method": policy.method.name,
            }
        )

    def project(
        self,
        fluid_velocity: FaceVelocity,
        body_kinematics: RigidBodyKinematics,
        inverse_momentum_coefficient: ArrayLike,
        /,
        *,
        pressure: ArrayLike | None = None,
        marker_force_density: ArrayLike | None = None,
        time: ArrayLike = 0.0,
        args: Any = None,
    ) -> MACRigidImmersedProjectionResult:
        operators = self.dynamics.momentum.operators
        boundaries = self.dynamics.momentum.boundaries
        coefficient = jnp.asarray(
            inverse_momentum_coefficient, dtype=operators.pressure_space.dtype
        ).reshape(())
        coefficient = eqx.error_if(
            coefficient,
            ~jnp.isfinite(coefficient) | (coefficient <= 0.0),
            "Rigid immersed coefficient must be positive and finite.",
        )
        stage = boundaries.evaluate(time, args)
        bounded = boundaries.enforce(
            operators.validate_velocity(fluid_velocity), stage
        )
        marker_state = self.rigid_markers.evaluate(body_kinematics)
        relation = self.transfer.relation(marker_state.position)
        marker_before = self.transfer.gather(relation, bounded)
        rigid_velocity = self.rigid_markers.generalized_velocity(body_kinematics)
        k_operator = self.rigid_markers.velocity_operator(body_kinematics)
        rigid_marker_velocity = k_operator.mv(rigid_velocity)
        ell = jnp.asarray(
            self.constraint_length, dtype=operators.pressure_space.dtype
        )
        incoming_pressure = (
            jnp.zeros(operators.discretization.cell_shape, dtype=ell.dtype)
            if pressure is None
            else operators.validate_pressure(pressure)
        )
        if boundaries.closure_kind == "neumann":
            incoming_pressure = operators.gauge_project(incoming_pressure)
        initial_multiplier = (
            jnp.zeros_like(marker_before)
            if marker_force_density is None
            else self.transfer.markers.active_velocity_space.validate(
                jnp.asarray(marker_force_density)
            )
        )
        dual_space = BlockSpace(
            (operators.pressure_space, self.transfer.markers.active_velocity_space),
            names=("pressure", "marker"),
        )
        face_inverse = tuple(
            jnp.full(layout.shape, coefficient, dtype=ell.dtype)
            for layout in operators.discretization.face_layouts
        )
        mobile = self.rigid_markers.mobile_indices
        inverse_mass = self.rigid_markers.bodies.inverse_masses[mobile]
        if self.rigid_markers.bodies.ambient_dimension == 2:
            inverse_inertia_mobile = self.rigid_markers.bodies.inverse_inertia_body[
                mobile
            ]

            def angular_inverse(rotation):
                return inverse_inertia_mobile[:, None] * rotation

        else:
            _, inverse_inertia = rigid_body_world_inertia(
                self.rigid_markers.bodies, body_kinematics.orientation
            )
            inverse_inertia_mobile = inverse_inertia[mobile]

            def angular_inverse(rotation):
                return contract(
                    "...ij,...j->...i",
                    inverse_inertia_mobile,
                    rotation,
                )

        def body_inverse(load: RigidGeneralizedVelocity):
            return RigidGeneralizedVelocity(
                coefficient * inverse_mass[:, None] * load.translation,
                coefficient * angular_inverse(load.rotation),
            )

        def response(dual):
            scaled_pressure, multiplier = dual
            if boundaries.closure_kind == "neumann":
                projected_pressure = operators.gauge_project(scaled_pressure)
                volumes = operators.discretization.cell_volumes.astype(ell.dtype)
                mean = jnp.sum(volumes * scaled_pressure) / jnp.sum(volumes)
            else:
                projected_pressure = scaled_pressure
                mean = jnp.asarray(0.0, dtype=ell.dtype)
            gradient = boundaries.pressure_gradient(
                ell * projected_pressure,
                stage,
                homogeneous=boundaries.closure_kind == "neumann",
            )
            spread = self.transfer.spread(relation, multiplier)
            fluid_mass_image = boundaries.homogeneous_rate(
                tuple(
                    inverse * (derivative - force)
                    for inverse, derivative, force in zip(
                        face_inverse, gradient, spread, strict=True
                    )
                )
            )
            body_load = k_operator.adjoint_mv(multiplier)
            body_mass_image = body_inverse(body_load)
            return fluid_mass_image, body_mass_image, mean

        def action(dual):
            fluid_image, body_image, mean = response(dual)
            pressure_image = -ell * operators.divergence(fluid_image)
            if boundaries.closure_kind == "neumann":
                pressure_image = pressure_image + mean
            marker_image = k_operator.mv(body_image) - self.transfer.gather(
                relation, fluid_image
            )
            return pressure_image, marker_image

        operator = FunctionLinearOperator(
            action,
            source=dual_space,
            target=dual_space,
            properties=OperatorProperties(
                self_adjoint=True, evidence={"self_adjoint": "construction"}
            ),
            operator_id=f"mac-rigid-immersed-dual/{self.plan_id}",
        )
        pressure_rhs = -ell * operators.divergence(bounded)
        if boundaries.closure_kind == "neumann":
            pressure_rhs = operators.compatibility_project(pressure_rhs)
        marker_rhs = rigid_marker_velocity - marker_before
        linear = solve(
            LinearSystem(operator, problem_id=f"mac-rigid-immersed/{self.plan_id}"),
            (pressure_rhs, marker_rhs),
            policy=self.linear_policy,
            initial_guess=(incoming_pressure / ell, initial_multiplier),
        )
        fluid_image, body_image, _ = response(linear.value)
        candidate_fluid = tuple(
            value - image
            for value, image in zip(bounded, fluid_image, strict=True)
        )
        candidate_generalized = RigidGeneralizedVelocity(
            rigid_velocity.translation - body_image.translation,
            rigid_velocity.rotation - body_image.rotation,
        )
        candidate_body = self.rigid_markers.with_generalized_velocity(
            body_kinematics, candidate_generalized
        )
        pressure_candidate = ell * linear.value[0]
        if boundaries.closure_kind == "neumann":
            pressure_candidate = operators.gauge_project(
                incoming_pressure + pressure_candidate
            )
        marker_after = self.transfer.gather(relation, candidate_fluid)
        body_marker_after = k_operator.mv(candidate_generalized)
        slip = marker_after - body_marker_after
        divergence = operators.divergence(candidate_fluid)
        residual = operator(linear.value)
        residual_value = (
            residual[0] - pressure_rhs,
            residual[1] - marker_rhs,
        )
        residual_norm = jnp.sqrt(
            jnp.real(dual_space.inner(residual_value, residual_value))
        )
        volumes = operators.discretization.cell_volumes.astype(ell.dtype)
        scale = jnp.maximum(
            1.0,
            jnp.sqrt(
                jnp.sum(volumes * pressure_rhs**2)
                + jnp.real(
                    self.transfer.markers.active_velocity_space.inner(
                        marker_rhs, marker_rhs
                    )
                )
            ),
        )
        tolerance = self.tolerance * scale
        finite = (
            stage.finite
            & jnp.all(
                jnp.stack(
                    tuple(jnp.all(jnp.isfinite(value)) for value in candidate_fluid)
                )
            )
            & jnp.all(jnp.isfinite(candidate_body.velocity))
            & jnp.all(jnp.isfinite(candidate_body.angular_velocity))
            & jnp.all(jnp.isfinite(pressure_candidate))
            & jnp.all(jnp.isfinite(linear.value[1]))
            & jnp.isfinite(residual_norm)
        )
        divergence_norm = jnp.sqrt(jnp.sum(volumes * divergence**2))
        slip_norm = jnp.sqrt(
            jnp.real(
                self.transfer.markers.active_velocity_space.inner(slip, slip)
            )
        )
        converged = (
            stage.successful
            & relation.successful
            & linear.successful
            & finite
            & (residual_norm <= tolerance)
            & (divergence_norm <= tolerance)
            & (slip_norm <= tolerance)
        )
        accepted_fluid = tuple(
            jnp.where(converged, candidate, original)
            for candidate, original in zip(candidate_fluid, bounded, strict=True)
        )
        accepted_body = jax_tree_where(converged, candidate_body, body_kinematics)
        pressure_value = jnp.where(converged, pressure_candidate, incoming_pressure)
        multiplier_value = jnp.where(
            converged, linear.value[1], initial_multiplier
        )
        closure = MACPressureClosureReport(
            kind=boundaries.closure_kind,
            gauge="zero-mean" if boundaries.closure_kind == "neumann" else "none",
            compatibility="projected"
            if boundaries.closure_kind == "neumann"
            else "unprojected",
            integrated_mass_flux=jnp.sum(volumes * operators.divergence(accepted_fluid)),
            mass_defect=jnp.abs(
                jnp.sum(volumes * operators.divergence(accepted_fluid))
            ),
            gauge_defect=jnp.abs(jnp.sum(volumes * pressure_value))
            if boundaries.closure_kind == "neumann"
            else jnp.asarray(0.0, dtype=ell.dtype),
            finite=finite,
            successful=converged,
            closure_id=self.plan_id,
        )
        return MACRigidImmersedProjectionResult(
            accepted_fluid,
            accepted_body,
            pressure_value,
            multiplier_value,
            slip,
            operators.divergence(accepted_fluid),
            residual_norm,
            linear,
            closure,
            finite,
            converged,
            self.plan_id,
        )


class MACRigidImmersedStepResult(StrictModule):
    fluid_state: Array
    body_kinematics: RigidBodyKinematics
    pressure: Array
    marker_force_density: Array
    helmholtz: MACHelmholtzResult
    projection: MACRigidImmersedProjectionResult
    accepted: Array
    method_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.accepted


class MACRigidImmersedEulerMethod(StrictModule, NonTrainableState):
    """Contact-free synchronized fluid/rigid Euler step with exact velocity coupling."""

    dynamics: CompiledMACIncompressibleDynamics
    projection: MACRigidImmersedProjectionPlan
    helmholtz: MACHelmholtzSolvePlan
    step_size: float = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        dynamics: CompiledMACIncompressibleDynamics,
        projection: MACRigidImmersedProjectionPlan,
        step_size: float,
        /,
    ):
        step = float(step_size)
        if step <= 0.0:
            raise ValueError("step_size must be positive.")
        if projection.dynamics.compilation_id != dynamics.compilation_id:
            raise ValueError("Rigid projection and method must share dynamics.")
        viscosity = float(jnp.asarray(dynamics.problem.viscosity))
        self.dynamics = dynamics
        self.projection = projection
        self.helmholtz = MACHelmholtzSolvePlan(
            dynamics.momentum,
            fixed_mass_coefficient=1.0,
            fixed_diffusion_coefficient=step * viscosity,
        )
        self.step_size = step
        self.method_id = canonical_fingerprint(
            {
                "kind": "mac-rigid-immersed-euler",
                "dynamics": dynamics.compilation_id,
                "projection": projection.plan_id,
                "step_size": step,
            }
        )

    def step(
        self,
        time: ArrayLike,
        fluid_state: ArrayLike,
        body_kinematics: RigidBodyKinematics,
        /,
        *,
        body_load: RigidBodyLoad | None = None,
        pressure: ArrayLike | None = None,
        marker_force_density: ArrayLike | None = None,
        args: Any = None,
    ) -> MACRigidImmersedStepResult:
        step = jnp.asarray(
            self.step_size,
            dtype=self.dynamics.momentum.operators.pressure_space.dtype,
        )
        current = self.dynamics.validate_state(fluid_state)
        velocity = self.dynamics.unpack_velocity(current)
        _, convection, _, forcing = self.dynamics.rate_components(time, current, args)
        explicit = tuple(
            -advective + source
            for advective, source in zip(convection, forcing, strict=True)
        )
        rhs = tuple(
            value + step * rate
            for value, rate in zip(velocity, explicit, strict=True)
        )
        attempted_time = jnp.asarray(time, dtype=step.dtype) + step
        stage = self.dynamics.momentum.boundaries.evaluate(attempted_time, args)
        helmholtz = self.helmholtz.solve(rhs, stage, initial_guess=velocity)
        predicted_pose = rigid_body_drift(
            self.projection.rigid_markers.bodies, body_kinematics, step
        )
        if body_load is not None:
            bodies = self.projection.rigid_markers.bodies
            mobile = self.projection.rigid_markers.mobile_indices
            if bodies.ambient_dimension == 2:
                angular_increment = (
                    step
                    * bodies.inverse_inertia_body[mobile, None]
                    * body_load.torque[mobile]
                )
            else:
                _, inverse_inertia = rigid_body_world_inertia(
                    bodies, predicted_pose.orientation
                )
                angular_increment = step * contract(
                    "...ij,...j->...i",
                    inverse_inertia[mobile],
                    body_load.torque[mobile],
                )
            predicted_pose = RigidBodyKinematics(
                predicted_pose.position,
                predicted_pose.velocity.at[mobile].add(
                    step
                    * bodies.inverse_masses[mobile][:, None]
                    * body_load.force[mobile]
                ),
                predicted_pose.orientation,
                predicted_pose.angular_velocity.at[mobile].add(
                    angular_increment
                ),
            )
        projected = self.projection.project(
            helmholtz.value,
            predicted_pose,
            step,
            pressure=pressure,
            marker_force_density=marker_force_density,
            time=attempted_time,
            args=args,
        )
        candidate_state = self.dynamics.momentum.operators.velocity_space.flatten(
            projected.fluid_velocity
        )
        accepted = helmholtz.converged & projected.converged
        return MACRigidImmersedStepResult(
            jnp.where(accepted, candidate_state, current),
            jax_tree_where(accepted, projected.body_kinematics, body_kinematics),
            projected.pressure,
            projected.marker_force_density,
            helmholtz,
            projected,
            accepted,
            self.method_id,
        )


def jax_tree_where(condition: Array, candidate, fallback):
    import jax

    return jax.tree.map(
        lambda accepted, rejected: jnp.where(condition, accepted, rejected),
        candidate,
        fallback,
    )


__all__ = [
    "MACRigidImmersedEulerMethod",
    "MACRigidImmersedProjectionPlan",
    "MACRigidImmersedProjectionResult",
    "MACRigidImmersedStepResult",
]
